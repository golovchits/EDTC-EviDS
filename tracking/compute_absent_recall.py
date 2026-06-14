"""
Compute Absent Recall and related operational metrics from per-sequence .npz files.

Metrics:
  - Absent Recall @ tau: fraction of gt_present=0 frames where confidence < tau
    (model correctly expresses uncertainty on absent frames)
  - Absent Precision @ tau: fraction of low-confidence (<tau) frames that are absent
  - Detector-trigger rate on absent frames: how often the detector activates
  - False positive initiation rate: absent frames where detector fires high-confidence

These metrics directly measure the operational value of the evidential head:
suppressing false-positive tracker initiations on hard negative frames.

Usage:
    python tracking/compute_absent_recall.py \
        --npz_dir /path/to/results/edl_270 \
        --label    edl_270
"""

import os
import sys
import json
import argparse
import glob
import numpy as np


def build_confidence(npz):
    """Replicates compute_ece.py confidence construction."""
    switch = npz['switch_decisions']
    det_c  = npz['det_confidences']
    trk_u  = npz['trk_uncertainties']

    conf = np.zeros(len(switch), dtype=np.float32)
    det_mask = switch == 1
    trk_mask = switch == 0

    det_vals = det_c[det_mask]
    det_vals = np.where(np.isnan(det_vals), 0.0, det_vals)
    conf[det_mask] = det_vals

    trk_vals = trk_u[trk_mask]
    trk_vals = np.where(np.isnan(trk_vals), 0.5, trk_vals)
    conf[trk_mask] = 1.0 - trk_vals

    conf = np.clip(conf, 0.0, 1.0)
    return conf, det_mask, trk_mask


def absent_metrics(confs, labels, det_mask, trk_mask, threshold=0.5):
    """
    Compute absent-frame metrics at a given confidence threshold.

    Returns dict with counts and rates.
    """
    absent_mask = ~labels.astype(bool)
    present_mask = labels.astype(bool)

    n_total = len(confs)
    n_absent = absent_mask.sum()
    n_present = present_mask.sum()

    # Absent Recall: fraction of absent frames with confidence < threshold
    # (model correctly identifies absence)
    if n_absent > 0:
        absent_low_conf = (confs[absent_mask] < threshold).sum()
        absent_recall = absent_low_conf / n_absent
    else:
        absent_recall = np.nan

    # Absent Precision: fraction of low-confidence frames that are absent
    low_conf_mask = confs < threshold
    n_low_conf = low_conf_mask.sum()
    if n_low_conf > 0:
        absent_precision = (absent_mask & low_conf_mask).sum() / n_low_conf
    else:
        absent_precision = np.nan

    # F1
    if absent_recall and absent_precision and not np.isnan(absent_recall) and not np.isnan(absent_precision):
        if absent_recall + absent_precision > 0:
            absent_f1 = 2 * absent_recall * absent_precision / (absent_recall + absent_precision)
        else:
            absent_f1 = 0.0
    else:
        absent_f1 = np.nan

    # Detector-trigger rate on absent frames
    if n_absent > 0:
        det_on_absent = (absent_mask & det_mask).sum()
        det_trigger_rate = det_on_absent / n_absent
    else:
        det_trigger_rate = np.nan

    # False positive initiation rate: absent frames where detector fires AND confidence > threshold
    if n_absent > 0:
        fp_init = (absent_mask & det_mask & (confs >= threshold)).sum()
        fp_init_rate = fp_init / n_absent
    else:
        fp_init_rate = np.nan

    # Absent recall broken down by active subsystem
    det_absent_mask = absent_mask & det_mask
    trk_absent_mask = absent_mask & trk_mask
    n_det_absent = det_absent_mask.sum()
    n_trk_absent = trk_absent_mask.sum()

    det_absent_recall = np.nan
    trk_absent_recall = np.nan
    if n_det_absent > 0:
        det_absent_recall = (confs[det_absent_mask] < threshold).sum() / n_det_absent
    if n_trk_absent > 0:
        trk_absent_recall = (confs[trk_absent_mask] < threshold).sum() / n_trk_absent

    # Also: Present Recall = fraction of present frames with confidence >= threshold
    # (standard detection recall — sanity check)
    if n_present > 0:
        present_recall = (confs[present_mask] >= threshold).sum() / n_present
    else:
        present_recall = np.nan

    return {
        'n_total': int(n_total),
        'n_absent': int(n_absent),
        'n_present': int(n_present),
        'n_det_absent': int(n_det_absent),
        'n_trk_absent': int(n_trk_absent),
        'absent_recall': float(absent_recall) if not np.isnan(absent_recall) else None,
        'absent_precision': float(absent_precision) if not np.isnan(absent_precision) else None,
        'absent_f1': float(absent_f1) if not np.isnan(absent_f1) else None,
        'det_trigger_rate': float(det_trigger_rate) if not np.isnan(det_trigger_rate) else None,
        'fp_init_rate': float(fp_init_rate) if not np.isnan(fp_init_rate) else None,
        'det_absent_recall': float(det_absent_recall) if not np.isnan(det_absent_recall) else None,
        'trk_absent_recall': float(trk_absent_recall) if not np.isnan(trk_absent_recall) else None,
        'present_recall': float(present_recall) if not np.isnan(present_recall) else None,
    }


def per_sequence_metrics(npz_dir, threshold=0.5):
    """Compute per-sequence absent recall for outlier detection."""
    results = {}
    for f in sorted(glob.glob(os.path.join(npz_dir, '*.npz'))):
        seq_name = os.path.splitext(os.path.basename(f))[0]
        npz = np.load(f)
        conf, det_mask, trk_mask = build_confidence(npz)
        gt = npz['gt_present'].astype(bool)
        m = absent_metrics(conf, gt, det_mask, trk_mask, threshold)
        results[seq_name] = m
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', required=True,
                        help='Directory with .npz diagnostic files')
    parser.add_argument('--label', required=True,
                        help='Human-readable label for this condition')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.3, 0.5, 0.7],
                        help='Confidence thresholds for absent recall')
    parser.add_argument('--out_dir', default=None,
                        help='Directory for output JSON (default: npz_dir/../absent_recall)')
    parser.add_argument('--per_sequence', action='store_true',
                        help='Also compute per-sequence metrics')
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
    print(f'Condition: {args.label}')
    print(f'Sequences: {len(npz_files)}')

    # Collect all confidences and labels
    all_conf, all_gt = [], []
    all_det_mask, all_trk_mask = [], []
    for f in npz_files:
        npz = np.load(f)
        conf, det_mask, trk_mask = build_confidence(npz)
        gt = npz['gt_present'].astype(bool)
        all_conf.append(conf)
        all_gt.append(gt)
        all_det_mask.append(det_mask)
        all_trk_mask.append(trk_mask)

    all_conf = np.concatenate(all_conf)
    all_gt = np.concatenate(all_gt)
    all_det = np.concatenate(all_det_mask)
    all_trk = np.concatenate(all_trk_mask)

    # Compute at each threshold
    print(f'\n{"="*70}')
    print(f'{"Threshold":>10}  {"Abs.Recall":>10}  {"Abs.Prec":>10}  '
          f'{"Abs.F1":>10}  {"DetTrig%":>10}  {"FPinit%":>10}  {"Pres.Recall":>12}')
    print(f'{"="*70}')

    results = {'label': args.label, 'n_sequences': len(npz_files),
               'thresholds': {}}

    for tau in args.thresholds:
        m = absent_metrics(all_conf, all_gt, all_det, all_trk, tau)
        results['thresholds'][tau] = m
        print(f'{tau:>10.1f}  {m["absent_recall"]:>10.4f}  {m["absent_precision"]:>10.4f}  '
              f'{m["absent_f1"]:>10.4f}  {m["det_trigger_rate"]*100:>9.1f}%  '
              f'{m["fp_init_rate"]*100:>9.1f}%  {m["present_recall"]:>12.4f}')

    print(f'\nFrame counts: {m["n_total"]:,} total | {m["n_absent"]:,} absent | '
          f'{m["n_present"]:,} present')
    print(f'Absent frames: {m["n_det_absent"]:,} detector-active | {m["n_trk_absent"]:,} tracker-active')

    # Per-sequence outlier detection
    if args.per_sequence:
        per_seq = per_sequence_metrics(args.npz_dir, 0.5)
        low_recall = [(s, m['absent_recall']) for s, m in per_seq.items()
                      if m['absent_recall'] is not None and m['absent_recall'] < 0.5 and m['n_absent'] > 10]
        if low_recall:
            print(f'\nSequences with Absent Recall < 0.5 and >10 absent frames:')
            for s, r in sorted(low_recall, key=lambda x: x[1]):
                m = per_seq[s]
                print(f'  {s}: recall={r:.3f}, absent={m["n_absent"]}, '
                      f'det_absent={m["n_det_absent"]}')
        results['per_sequence'] = {s: m['absent_recall'] for s, m in per_seq.items()}

    # Save JSON
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.npz_dir), '..', 'absent_recall')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f'{args.label}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved → {json_path}')


if __name__ == '__main__':
    main()
