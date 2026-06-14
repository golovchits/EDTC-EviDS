"""
Compute ECE (Expected Calibration Error) from per-sequence .npz diagnostics.

Confidence construction per frame:
  - Detector active (switch=1): p = det_confidences (YOLO sigmoid); 0.0 if nan (no detection)
  - Tracker active  (switch=0): p = 1 - trk_uncertainties (EDL vacuity → presence confidence)

Ground truth: gt_present (bool).

Outputs:
  - Overall ECE, MCE, Brier score
  - Per-source ECE (detector frames vs tracker frames)
  - Reliability diagram saved as {out_dir}/reliability_diagram.png
  - Summary saved as {out_dir}/ece_summary.json
"""
import os
import sys
import json
import argparse
import numpy as np
import glob

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)


def build_confidence(npz):
    """Construct per-frame presence probability in [0,1]."""
    switch = npz['switch_decisions']   # 0=tracker, 1=detector
    det_c  = npz['det_confidences']    # nan when tracker active
    trk_u  = npz['trk_uncertainties']  # nan when detector active

    conf = np.zeros(len(switch), dtype=np.float32)
    det_mask = switch == 1
    trk_mask = switch == 0

    # detector frames: use YOLO confidence; treat nan (no detection) as 0
    det_vals = det_c[det_mask]
    det_vals = np.where(np.isnan(det_vals), 0.0, det_vals)
    conf[det_mask] = det_vals

    # tracker frames: 1 - vacuity uncertainty
    trk_vals = trk_u[trk_mask]
    trk_vals = np.where(np.isnan(trk_vals), 0.5, trk_vals)  # fallback to 0.5 if missing
    conf[trk_mask] = 1.0 - trk_vals

    conf = np.clip(conf, 0.0, 1.0)
    return conf, det_mask, trk_mask


def ece_from_arrays(confs, labels, n_bins=10):
    """
    ECE with equal-width bins. Returns (ece, mce, bin_data).
    bin_data: list of (mean_conf, mean_acc, n) per non-empty bin.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0
    N = len(confs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        n = mask.sum()
        mean_conf = confs[mask].mean()
        mean_acc  = labels[mask].mean()
        cal_err   = abs(mean_conf - mean_acc)
        ece      += (n / N) * cal_err
        mce       = max(mce, cal_err)
        bin_data.append((float(mean_conf), float(mean_acc), int(n)))
    return float(ece), float(mce), bin_data


def brier(confs, labels):
    return float(np.mean((confs - labels.astype(np.float32)) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', default='/gpfs/work5/0/prjs1970/output/EDTC/results/val')
    parser.add_argument('--out_dir', default='/gpfs/work5/0/prjs1970/output/EDTC/ece')
    parser.add_argument('--n_bins', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
    print(f'Found {len(npz_files)} sequences in {args.npz_dir}')

    all_conf, all_gt = [], []
    all_det_conf, all_det_gt = [], []
    all_trk_conf, all_trk_gt = [], []

    for f in npz_files:
        npz = np.load(f)
        conf, det_mask, trk_mask = build_confidence(npz)
        gt = npz['gt_present'].astype(bool)

        all_conf.append(conf)
        all_gt.append(gt)
        all_det_conf.append(conf[det_mask])
        all_det_gt.append(gt[det_mask])
        all_trk_conf.append(conf[trk_mask])
        all_trk_gt.append(gt[trk_mask])

    all_conf = np.concatenate(all_conf)
    all_gt   = np.concatenate(all_gt)
    all_det_conf = np.concatenate(all_det_conf)
    all_det_gt   = np.concatenate(all_det_gt)
    all_trk_conf = np.concatenate(all_trk_conf)
    all_trk_gt   = np.concatenate(all_trk_gt)

    ece,  mce,  bins_all = ece_from_arrays(all_conf,     all_gt,     args.n_bins)
    ece_d, mce_d, bins_d = ece_from_arrays(all_det_conf, all_det_gt, args.n_bins)
    ece_t, mce_t, bins_t = ece_from_arrays(all_trk_conf, all_trk_gt, args.n_bins)
    bs  = brier(all_conf,     all_gt)
    bs_d = brier(all_det_conf, all_det_gt)
    bs_t = brier(all_trk_conf, all_trk_gt)

    print(f'\n=== ECE Results ({args.n_bins} bins) ===')
    print(f'Overall   ECE={ece:.4f}  MCE={mce:.4f}  Brier={bs:.4f}  N={len(all_conf)}')
    print(f'Detector  ECE={ece_d:.4f}  MCE={mce_d:.4f}  Brier={bs_d:.4f}  N={len(all_det_conf)}')
    print(f'Tracker   ECE={ece_t:.4f}  MCE={mce_t:.4f}  Brier={bs_t:.4f}  N={len(all_trk_conf)}')

    # reliability diagram
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        if bins_all:
            cx, cy = zip(*[(b[0], b[1]) for b in bins_all])
            ax.plot(cx, cy, 'o-', color='steelblue', label=f'Overall ECE={ece:.3f}')
        if bins_d:
            cx, cy = zip(*[(b[0], b[1]) for b in bins_d])
            ax.plot(cx, cy, 's--', color='tomato', label=f'Detector ECE={ece_d:.3f}')
        if bins_t:
            cx, cy = zip(*[(b[0], b[1]) for b in bins_t])
            ax.plot(cx, cy, '^--', color='seagreen', label=f'Tracker ECE={ece_t:.3f}')
        ax.set_xlabel('Mean confidence'); ax.set_ylabel('Fraction present (accuracy)')
        cond_name = os.path.basename(os.path.normpath(args.out_dir))
        ax.set_title(f'Reliability Diagram — {cond_name}')
        ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, 'reliability_diagram.png')
        fig.savefig(fig_path, dpi=150)
        print(f'\nReliability diagram → {fig_path}')
    except ImportError:
        print('matplotlib not available — skipping plot')

    summary = {
        'n_sequences': len(npz_files),
        'n_bins': args.n_bins,
        'overall': {'ece': ece, 'mce': mce, 'brier': bs, 'n_frames': int(len(all_conf))},
        'detector': {'ece': ece_d, 'mce': mce_d, 'brier': bs_d, 'n_frames': int(len(all_det_conf))},
        'tracker':  {'ece': ece_t, 'mce': mce_t, 'brier': bs_t, 'n_frames': int(len(all_trk_conf))},
        'bin_data': {'overall': bins_all, 'detector': bins_d, 'tracker': bins_t},
    }
    json_path = os.path.join(args.out_dir, 'ece_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary JSON → {json_path}')


if __name__ == '__main__':
    main()
