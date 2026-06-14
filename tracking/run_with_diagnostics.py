"""
EDTC inference with full per-frame diagnostic saving for ECE and uncertainty analysis.

Runs the same pipeline as tracking/test.py but additionally saves per-sequence .npz
files containing all continuous values needed for E1 (EviDS-UAV thesis).

Saved arrays per sequence in {npz_dir}/{sequence_name}.npz:
  pred_boxes        (T, 4)  float  predicted bbox [x,y,w,h] per frame
  pred_present      (T,)    bool   True if system predicted a detection (not [0,0,0,0])
  det_confidences   (T,)    float  YOLO sigmoid/EDL p_uav confidence (nan when tracker active)
  det_vacuity       (T,)    float  EDL detector vacuity u = K/S (nan when tracker active or non-EDL)
  trk_uncertainties (T,)    float  EDL tracker vacuity (nan when detector was active)
  switch_decisions  (T,)    int8   0=tracker active this frame, 1=detector active
  gt_boxes          (T, 4)  float  ground-truth bbox from IR_label.json ([0,0,0,0] when absent)
  gt_present        (T,)    bool   target visibility from IR_label.json exist flags

Usage on Snellius:
    python tracking/run_with_diagnostics.py \\
        --tracker_name uavtrack_eh \\
        --tracker_param baseline \\
        --dataset_name antiuav \\
        --num_gpus 1 \\
        --params__model /gpfs/work5/0/prjs1970/code/EDTC/pretrained_models/UAVTrackEH.pth.tar \\
        --params__evidential_threshold 0.2 \\
        --npz_dir /gpfs/work5/0/prjs1970/output/EDTC/results/val

Local smoke test (single sequence, CPU):
    EDTC_DEVICE=cpu python tracking/run_with_diagnostics.py \\
        --tracker_name uavtrack_eh \\
        --tracker_param baseline \\
        --dataset_name antiuav \\
        --sequence 0 \\
        --num_gpus 0 \\
        --params__model pretrained_models/UAVTrackEH.pth.tar \\
        --params__evidential_threshold 0.2 \\
        --npz_dir /tmp/edtc_smoke
"""
import os
import sys
import argparse
import time
import numpy as np

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import cv2 as cv
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.running import _save_tracker_output
from lib.test.evaluation.environment import env_settings


def _is_empty_pred(bbox):
    """Matches the not_exist() logic in evaluate_antiuav_performance.py."""
    return (bbox[0] == 0 and bbox[2] == 0) or len(bbox) == 0


def run_sequence_with_diagnostics(seq, tracker_obj, npz_dir, debug=False):
    """
    Runs one AntiUAV sequence and saves both the standard .txt result and a .npz
    with all continuous diagnostic values.

    Mirrors _det_track_sequence() in running.py exactly — no logic changes.
    The tracker's initialize() and track() methods return the extra fields
    yolo_conf and trk_uncertainty after the patches in uavtrack_eh.py.
    """
    params = tracker_obj.params
    params.debug = debug

    tracker = tracker_obj.create_tracker(params)
    detector = tracker.initialize_yolo()

    T = len(seq.frames)

    # --- output buffers ---
    pred_boxes        = np.zeros((T, 4), dtype=np.float32)
    pred_present      = np.zeros(T, dtype=bool)
    det_confidences   = np.full(T, np.nan, dtype=np.float32)
    det_vacuity       = np.full(T, np.nan, dtype=np.float32)
    trk_uncertainties = np.full(T, np.nan, dtype=np.float32)
    switch_decisions  = np.zeros(T, dtype=np.int8)   # 0=tracker, 1=detector
    times             = np.zeros(T, dtype=np.float64)

    # GT from sequence object (populated by antiuavdataset.py changes)
    gt_boxes    = seq.ground_truth_rect.astype(np.float32)          # (T,4)
    gt_present  = seq.target_visible.astype(bool) if seq.target_visible is not None \
                  else np.array([not np.all(b == 0) for b in gt_boxes], dtype=bool)

    # also build the standard list output for _save_tracker_output
    std_output = {'target_bbox': [], 'time': []}

    detection_flag = True

    for frame_idx, frame_path in enumerate(seq.frames):
        image = tracker_obj._read_image(frame_path)
        t0 = time.time()

        if detection_flag:
            out = tracker.initialize(image, detector)
            switch_decisions[frame_idx] = 1
        else:
            out = tracker.track(image)
            switch_decisions[frame_idx] = 0

        elapsed = time.time() - t0
        detection_flag = out['detection_flag']

        bbox = out['target_bbox']
        pred_boxes[frame_idx]   = bbox
        pred_present[frame_idx] = not _is_empty_pred(bbox)

        # continuous diagnostic values
        yc = out.get('yolo_conf', np.nan)
        dv = out.get('yolo_vacuity', np.nan)
        tu = out.get('trk_uncertainty', np.nan)
        if not np.isnan(yc):
            det_confidences[frame_idx] = yc
        if not np.isnan(dv):
            det_vacuity[frame_idx] = dv
        if not np.isnan(tu):
            trk_uncertainties[frame_idx] = tu

        times[frame_idx] = elapsed
        std_output['target_bbox'].append(bbox)
        std_output['time'].append(elapsed)

    fps = T / times.sum()
    print(f'  {seq.name}  FPS: {fps:.1f}')

    # --- save standard .txt for evaluate_antiuav_performance.py ---
    _save_tracker_output(seq, tracker_obj, std_output)

    # --- save .npz ---
    os.makedirs(npz_dir, exist_ok=True)
    npz_path = os.path.join(npz_dir, f'{seq.name}.npz')
    np.savez_compressed(
        npz_path,
        pred_boxes=pred_boxes,
        pred_present=pred_present,
        det_confidences=det_confidences,
        det_vacuity=det_vacuity,
        trk_uncertainties=trk_uncertainties,
        switch_decisions=switch_decisions,
        gt_boxes=gt_boxes,
        gt_present=gt_present,
        times=times,
    )
    print(f'  Saved diagnostics → {npz_path}')

    return fps


def main():
    parser = argparse.ArgumentParser(description='EDTC inference with diagnostic saving.')
    parser.add_argument('--tracker_name',  default='uavtrack_eh')
    parser.add_argument('--tracker_param', default='baseline')
    parser.add_argument('--runid',         type=int, default=None)
    parser.add_argument('--dataset_name',  default='antiuav')
    parser.add_argument('--sequence',      default=None,
                        help='Sequence name or 0-based index. Omit to run all.')
    parser.add_argument('--debug',         type=int, default=0)
    parser.add_argument('--num_gpus',      type=int, default=1)
    parser.add_argument('--npz_dir',       type=str,
                        default='/gpfs/work5/0/prjs1970/output/EDTC/results/val',
                        help='Directory where per-sequence .npz files are saved.')

    parser.add_argument('--params__model',                  type=str,
                        default='/gpfs/work5/0/prjs1970/code/EDTC/pretrained_models/UAVTrackEH.pth.tar')
    parser.add_argument('--params__update_interval',        type=int,   default=None)
    parser.add_argument('--params__online_sizes',           type=int,   default=None)
    parser.add_argument('--params__search_area_scale',      type=float, default=4.55)
    parser.add_argument('--params__max_score_decay',        type=float, default=1.0)
    parser.add_argument('--params__evidential_threshold',   type=float, default=0.2)
    parser.add_argument('--params__vis_attn', type=int, choices=[0, 1], default=0)

    args = parser.parse_args()

    tracker_params = {}
    for param in [s for s in dir(args) if s.startswith('params__') and getattr(args, s) is not None]:
        tracker_params[param[len('params__'):]] = getattr(args, param)
    print('tracker_params:', tracker_params)

    dataset = get_dataset(args.dataset_name)

    # resolve --sequence: accept name string or 0-based integer index
    if args.sequence is not None:
        try:
            dataset = [dataset[int(args.sequence)]]
        except (ValueError, TypeError):
            dataset = [dataset[args.sequence]]

    tracker_obj = Tracker(args.tracker_name, args.tracker_param, args.dataset_name,
                          args.runid, tracker_params=tracker_params)

    print(f'Running {len(dataset)} sequences → npz_dir={args.npz_dir}')
    all_fps = []
    for i, seq in enumerate(dataset):
        print(f'\n[{i + 1}/{len(dataset)}] {seq.name}')
        fps = run_sequence_with_diagnostics(seq, tracker_obj, args.npz_dir,
                                            debug=bool(args.debug))
        all_fps.append(fps)

    print(f'\nDone. Mean FPS: {np.mean(all_fps):.1f}')


if __name__ == '__main__':
    main()
