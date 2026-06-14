"""Evidential Deep Learning helpers for YOLOv5 binary detection (nc=1, K=2).

Implements the EDL loss from Sensoy et al. (2018) with the digamma Bayes-risk
variant recommended by Gao et al. (2025) and softplus evidence activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def softplus_evidence(logits):
    """Softplus activation for non-negative evidence (Gao et al. 2025 modern standard)."""
    return F.softplus(logits)


def kl_divergence_dirichlet(alpha_tilde, K=2):
    """KL[Dir(p|alpha_tilde) || Dir(p|1)] computed analytically.

    Equivalent to KL from a Dirichlet with params alpha_tilde toward the
    uniform Dirichlet (all params = 1). Used as the regularisation term in
    the EDL loss after removing evidence for the correct class.

    Args:
        alpha_tilde: Dirichlet params with correct-class evidence zeroed out,
                     shape (..., K), all values >= 1.
        K: number of classes.
    Returns:
        KL per sample, shape (...,).
    """
    S_tilde = alpha_tilde.sum(dim=-1)  # (...,)
    kl = (torch.lgamma(S_tilde)
          - torch.lgamma(torch.tensor(float(K), device=alpha_tilde.device))
          - torch.lgamma(alpha_tilde).sum(dim=-1)
          + ((alpha_tilde - 1.0) * (digamma(alpha_tilde)
             - digamma(S_tilde.unsqueeze(-1)))).sum(dim=-1))
    return kl  # (...,)


class ComputeLossEDL(nn.Module):
    """EDL variant of ComputeLoss for binary detection (nc=1, K=2).

    Keeps box regression (CIoU) and build_targets unchanged.
    Replaces BCEobj + BCEcls with the EDL Bayes-risk + KL loss (Eq. 7,
    methodology Section 3.2), weighted by hyp['obj'].

    Usage:
        compute_loss = ComputeLossEDL(model, t_anneal=10)
        compute_loss.set_epoch(epoch)  # call once per epoch
        loss, loss_items = compute_loss(predictions, targets)
    """

    sort_obj_iou = False

    def __init__(self, model, t_anneal=10, autobalance=False):
        super().__init__()
        device = next(model.parameters()).device
        h = model.hyp

        m = de_parallel(model).model[-1]  # DetectEDL module
        assert m.nc == 1, "ComputeLossEDL only supports nc=1 (binary detection)"

        self.K = 2  # Dirichlet classes: background (0) and UAV (1)
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m.stride).index(16) if autobalance else 0
        self.hyp = h
        self.autobalance = autobalance
        self.na = m.na
        self.nc = m.nc
        self.nl = m.nl
        self.anchors = m.anchors
        self.device = device
        self.gr = 1.0

        # Annealing: lambda_t = min(1, epoch / t_anneal)
        self.t_anneal = max(t_anneal, 1)
        self.lambda_t = 0.0

    def set_epoch(self, epoch):
        """Update KL annealing coefficient. Call once at the start of each epoch."""
        self.lambda_t = min(1.0, epoch / self.t_anneal)

    def __call__(self, p, targets):
        lbox = torch.zeros(1, device=self.device)
        ledl = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)  # kept for logging shape compat

        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):  # layer i, raw logits (bs, na, ny, nx, no)
            b, a, gj, gi = indices[i]
            bs, na, ny, nx, _ = pi.shape
            n = b.shape[0]  # number of positive anchors this layer

            if n:
                # Box regression — identical to ComputeLoss
                pxy, pwh, _ = pi[b, a, gj, gi].split((2, 2, self.K), 1)
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

            # EDL loss on ALL anchor positions (positives + background)
            # channels 4 and 5 are the K=2 evidence logits
            evid_all = pi[..., 4:4 + self.K].reshape(-1, self.K)  # (bs*na*ny*nx, K)
            evidence = softplus_evidence(evid_all)
            alpha = evidence + 1.0          # Dirichlet params, shape (N, K)
            S = alpha.sum(dim=-1, keepdim=True)  # (N, 1)

            # Binary class labels: 0=background, 1=UAV
            tclass = torch.zeros(bs, na, ny, nx, dtype=torch.long, device=self.device)
            if n:
                tclass[b, a, gj, gi] = 1
            tclass_flat = tclass.reshape(-1)  # (N,)

            # One-hot: y[..., 0]=1 for background, y[..., 1]=1 for UAV
            y = torch.zeros_like(evid_all)
            y.scatter_(1, tclass_flat.unsqueeze(1), 1.0)

            # Bayes risk: Σ_k y_k * (ψ(S) − ψ(α_k))
            bayes_risk = (y * (digamma(S) - digamma(alpha))).sum(dim=-1)  # (N,)

            # KL regularisation with annealing
            if self.lambda_t > 0:
                # alpha_tilde removes evidence for the correct class before KL
                alpha_tilde = y + (1.0 - y) * alpha  # (N, K)
                kl = kl_divergence_dirichlet(alpha_tilde, K=self.K)  # (N,)
                edl_per_anchor = bayes_risk + self.lambda_t * kl
            else:
                edl_per_anchor = bayes_risk

            obji = edl_per_anchor.mean() * self.balance[i]
            ledl += obji
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        ledl *= self.hyp['obj']   # EDL uses the objectness loss budget
        bs_val = p[0].shape[0]

        return (lbox + ledl) * bs_val, torch.cat((lbox, ledl, lcls)).detach()

    def build_targets(self, p, targets):
        """Identical to ComputeLoss.build_targets — finds matched anchor positions."""
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g = 0.5
        off = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            device=self.device).float() * g

        for i in range(self.nl):
            anchor_i, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchor_i[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchor_i[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
