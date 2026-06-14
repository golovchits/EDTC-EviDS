"""
SQ2: Correlate per-frame model uncertainty with challenge attributes.

For each sequence that HAS challenge attribute metadata (OC, FM, SV, DBC, TC),
compute the Spearman rank correlation between per-frame model confidence and each
attribute value. Sequences without attributes (32/51 original_antiuav in val) are
silently skipped.

Confidence construction (same as compute_ece.py):
  - Detector frames: det_confidences (YOLO), NaN → 0.0
  - Tracker frames: 1 - trk_uncertainties (EDL vacuity)

Attributes are expected in IR_label.json as per-frame lists. TC (Target Scale)
is cast to binary (> 0) before correlation — it is already 0/1 in this dataset
but the cast is applied as a safety measure.

Output: JSON with per-attribute Spearman rho, p-value, and n (valid frames).

Usage:
    python tracking/compute_sq2_correlations.py \
        --npz_dir      /path/to/results/edl_270 \
        --dataset_root /Volumes/My\ Passport/AntiUAV600/validation \
        --label        edl_270 \
        --out_dir      reports/sq2
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
from scipy.stats import spearmanr

ATTRIBUTE_KEYS = {'OC', 'FM', 'SV', 'DBC', 'TC'}
ATTRIBUTE_NAMES = {
    'OC': 'Occlusion', 'FM': 'Fast Motion', 'SV': 'Scale Variation',
    'DBC': 'Dynamic Background Clusters', 'TC': 'Target Scale',
}


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


def has_attributes(seq_dir):
    """Check if sequence IR_label.json contains challenge attribute keys."""
    anno_path = os.path.join(seq_dir, 'IR_label.json')
    if not os.path.exists(anno_path):
        return False
    with open(anno_path) as f:
        keys = set(json.load(f).keys())
    return bool(keys & ATTRIBUTE_KEYS)


def load_attributes(seq_dir, attr_key):
    """Load per-frame attribute values, cast to binary (> 0)."""
    anno_path = os.path.join(seq_dir, 'IR_label.json')
    with open(anno_path) as f:
        data = json.load(f)
    values = np.array(data.get(attr_key, []), dtype=np.float32)
    if len(values) == 0:
        return None
    # Binarize: categorical severity > 0 → 1
    return (values > 0).astype(np.float32)


def load_frame_count(seq_dir):
    """Count frames from jpg files (same order as npz arrays)."""
    jpgs = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')],
                  key=lambda x: int(x[:-4]))
    return len(jpgs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', required=True,
                        help='Directory with .npz diagnostic files')
    parser.add_argument('--dataset_root', required=True,
                        help='Path to AntiUAV600 val or train directory')
    parser.add_argument('--label', required=True,
                        help='Human-readable label for this condition')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory for JSON (default: reports/sq2)')
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
    print(f'Condition: {args.label}')
    print(f'NPZ files: {len(npz_files)}')

    # Collect per-sequence data first, then accumulate per attribute
    included = 0
    skipped = 0
    seq_data = []  # list of {conf, det_mask, trk_mask, attrs: {key: array}}

    for npz_path in npz_files:
        seq_name = os.path.splitext(os.path.basename(npz_path))[0]
        seq_name_clean = seq_name.replace('_time', '')
        seq_dir = os.path.join(args.dataset_root, seq_name_clean)

        if not os.path.isdir(seq_dir):
            seq_dir = os.path.join(args.dataset_root, seq_name)
        if not os.path.isdir(seq_dir):
            skipped += 1
            continue

        if not has_attributes(seq_dir):
            skipped += 1
            continue

        included += 1

        npz = np.load(npz_path)
        conf, det_mask, trk_mask = build_confidence(npz)

        attrs = {}
        for attr_key in ATTRIBUTE_KEYS:
            attr_vals = load_attributes(seq_dir, attr_key)
            if attr_vals is not None and len(attr_vals) > 0:
                n = min(len(conf), len(attr_vals))
                attrs[attr_key] = attr_vals[:n]

        if attrs:
            seq_data.append({
                'conf': conf, 'det_mask': det_mask, 'trk_mask': trk_mask,
                'attrs': attrs, 'n': min(len(conf), min(len(v) for v in attrs.values())),
            })

    print(f'Included: {included} sequences with attributes')
    print(f'Skipped:  {skipped} sequences without attributes')

    # Compute Spearman correlations per attribute
    results = {'label': args.label, 'n_sequences_with_attrs': included,
               'n_sequences_skipped': skipped, 'attributes': {}}

    print(f'\n{"Attribute":<30s} {"Overall rho":>12s} {"p-val":>8s} {"N":>6s}  '
          f'{"Det.rho":>10s} {"Trk.rho":>10s}')
    print('-' * 82)

    for attr_key in sorted(ATTRIBUTE_KEYS):
        # Collect frames that have this attribute
        confs, attrs = [], []
        det_confs, det_attrs = [], []
        trk_confs, trk_attrs = [], []

        for sd in seq_data:
            if attr_key not in sd['attrs']:
                continue
            a = sd['attrs'][attr_key]
            n = sd['n']
            c = sd['conf'][:n]
            dm = sd['det_mask'][:n]
            tm = sd['trk_mask'][:n]

            confs.append(c)
            attrs.append(a)
            if dm.sum() > 0:
                det_confs.append(c[dm])
                det_attrs.append(a[dm])
            if tm.sum() > 0:
                trk_confs.append(c[tm])
                trk_attrs.append(a[tm])

        if not confs:
            print(f'{ATTRIBUTE_NAMES[attr_key]:<30s}  -- no data --')
            continue

        conf_cat = np.concatenate(confs)
        attr_cat = np.concatenate(attrs)
        mask = ~np.isnan(conf_cat) & ~np.isnan(attr_cat)
        rho, pval = spearmanr(conf_cat[mask], attr_cat[mask])
        n_all = mask.sum()

        rho_det, rho_trk = np.nan, np.nan
        if det_confs:
            dc = np.concatenate(det_confs); da = np.concatenate(det_attrs)
            dm_ok = ~np.isnan(dc) & ~np.isnan(da)
            if dm_ok.sum() >= 3:
                rho_det, _ = spearmanr(dc[dm_ok], da[dm_ok])
        if trk_confs:
            tc = np.concatenate(trk_confs); ta = np.concatenate(trk_attrs)
            tm_ok = ~np.isnan(tc) & ~np.isnan(ta)
            if tm_ok.sum() >= 3:
                rho_trk, _ = spearmanr(tc[tm_ok], ta[tm_ok])

        results['attributes'][attr_key] = {
            'name': ATTRIBUTE_NAMES[attr_key],
            'spearman_rho': float(rho),
            'p_value': float(pval),
            'n_frames': int(n_all),
            'detector_rho': float(rho_det) if not np.isnan(rho_det) else None,
            'tracker_rho': float(rho_trk) if not np.isnan(rho_trk) else None,
        }

        print(f'{ATTRIBUTE_NAMES[attr_key]:<30s} {rho:>+12.4f} {pval:>8.4f} {n_all:>6d}  '
              f'{rho_det:>+10.4f} {rho_trk:>+10.4f}')

    # Save
    out_dir = args.out_dir or 'reports/sq2'
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f'{args.label}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults → {json_path}')


if __name__ == '__main__':
    main()
