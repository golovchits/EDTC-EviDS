"""
Post-hoc temperature scaling calibration for EDTC detector (YOLO sigmoid head).

Procedure:
  1. Load detector confidences from tune-split .npz files (30 seqs, from training data).
     These are used as the calibration set — separate from the reporting set.
  2. Fit scalar temperature T by minimising NLL on tune-split detector frames.
  3. Apply T to val-split detector confidences (50 seqs).
  4. Report ECE before/after T scaling for detector frames only.
  5. Save results to {out_dir}/temp_scaling_report.md and temp_scaling.json.

Prerequisites:
  - Run tune-split inference first:
      sbatch jobs/run_tune_inference.sh
  - Val-split .npz files already exist from E0/sigmoid-270 run.

Usage:
    python tracking/calibrate_temperature_edtc.py \\
        --tune_npz_dir /gpfs/work5/0/prjs1970/output/EDTC/results/sigmoid_270_tune \\
        --val_npz_dir  /gpfs/work5/0/prjs1970/output/EDTC/results/sigmoid_270 \\
        --out_dir      /gpfs/work5/0/prjs1970/output/EDTC/ece/sigmoid_270
"""
import os
import sys
import json
import argparse
import glob
import numpy as np
from scipy.optimize import minimize_scalar


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1.0 - p))


def nll_loss(T, logits, labels):
    scaled = sigmoid(logits / T)
    scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
    return -np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled))


def fit_temperature(probs, labels):
    lgs = logit(probs)
    result = minimize_scalar(
        lambda T: nll_loss(T, lgs, labels),
        bounds=(0.05, 200.0),
        method='bounded',
    )
    return float(result.x)


def apply_temperature(probs, T):
    return sigmoid(logit(probs) / T)


def ece(confs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    mce = 0.0
    N = len(confs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        n = mask.sum()
        cal_err = abs(confs[mask].mean() - labels[mask].mean())
        e += (n / N) * cal_err
        mce = max(mce, cal_err)
    return float(e), float(mce)


def load_detector_frames(npz_dir):
    """Extract detector-active frames: (confidences, gt_present) arrays."""
    confs, gts = [], []
    for f in sorted(glob.glob(os.path.join(npz_dir, '*.npz'))):
        npz = np.load(f)
        switch = npz['switch_decisions']
        det_c  = npz['det_confidences']
        gt     = npz['gt_present'].astype(bool)
        det_mask = switch == 1
        c = det_c[det_mask]
        c = np.where(np.isnan(c), 0.0, c)
        confs.append(c)
        gts.append(gt[det_mask])
    return np.concatenate(confs), np.concatenate(gts).astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_npz_dir', required=True,
                        help='Directory with .npz files from tune-split inference')
    parser.add_argument('--val_npz_dir', required=True,
                        help='Directory with .npz files from val-split inference')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_bins', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- fit T on tune set ---
    tune_confs, tune_labels = load_detector_frames(args.tune_npz_dir)
    print(f'Tune set detector frames: {len(tune_confs):,}')

    T = fit_temperature(tune_confs, tune_labels)
    print(f'Fitted temperature T = {T:.4f}')

    tune_ece_base, tune_mce_base = ece(tune_confs, tune_labels, args.n_bins)
    tune_cal = apply_temperature(tune_confs, T)
    tune_ece_cal, tune_mce_cal = ece(tune_cal, tune_labels, args.n_bins)
    print(f'Tune ECE: {tune_ece_base:.4f} → {tune_ece_cal:.4f}  (T={T:.4f})')

    # --- apply T to val set ---
    val_confs, val_labels = load_detector_frames(args.val_npz_dir)
    print(f'Val set detector frames: {len(val_confs):,}')

    val_ece_base, val_mce_base = ece(val_confs, val_labels, args.n_bins)
    val_cal = apply_temperature(val_confs, T)
    val_ece_cal, val_mce_cal = ece(val_cal, val_labels, args.n_bins)

    print(f'\n=== Temperature Scaling Results (val set, detector frames) ===')
    print(f'T = {T:.4f}  (>1 → over-confident, <1 → under-confident)')
    print(f'Baseline ECE: {val_ece_base:.4f}  MCE: {val_mce_base:.4f}')
    print(f'Temp-scaled ECE: {val_ece_cal:.4f}  MCE: {val_mce_cal:.4f}')
    print(f'ΔECE: {val_ece_cal - val_ece_base:+.4f}')

    results = {
        'T': T,
        'tune': {
            'n_frames': int(len(tune_confs)),
            'ece_base': tune_ece_base, 'mce_base': tune_mce_base,
            'ece_cal':  tune_ece_cal,  'mce_cal':  tune_mce_cal,
        },
        'val': {
            'n_frames': int(len(val_confs)),
            'ece_base': val_ece_base, 'mce_base': val_mce_base,
            'ece_cal':  val_ece_cal,  'mce_cal':  val_mce_cal,
            'delta_ece': val_ece_cal - val_ece_base,
        },
    }

    json_path = os.path.join(args.out_dir, 'temp_scaling.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults → {json_path}')

    # markdown report
    lines = [
        '# Temperature Scaling — EDTC Sigmoid-270 Detector',
        '',
        f'Calibration set: tune split (30 seqs, {len(tune_confs):,} detector frames)',
        f'Evaluation set: val split (50 seqs, {len(val_confs):,} detector frames)',
        '',
        f'**Fitted T = {T:.4f}** ({"over-confident → softened" if T > 1 else "under-confident → sharpened"})',
        '',
        '| Split | ECE (base) | ECE (T-scaled) | ΔECE |',
        '|-------|------------|----------------|------|',
        f'| Tune  | {tune_ece_base:.4f} | {tune_ece_cal:.4f} | {tune_ece_cal-tune_ece_base:+.4f} |',
        f'| Val   | {val_ece_base:.4f} | {val_ece_cal:.4f} | {val_ece_cal-val_ece_base:+.4f} |',
        '',
    ]
    md_path = os.path.join(args.out_dir, 'temp_scaling_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Report → {md_path}')


if __name__ == '__main__':
    main()
