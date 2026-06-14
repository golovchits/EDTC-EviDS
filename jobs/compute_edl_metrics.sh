#!/bin/bash
#SBATCH --job-name=edl_metrics
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/edl_metrics_%j.out
#SBATCH --error=logs/edl_metrics_%j.err

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p $TMPDIR
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system

CODE="${CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-${CODE}/results}"
GT="${GT:-${DATA}/validation}"

cd $CODE

# --- ECE for condition (b): EDL-270 ---
echo "=== ECE condition (b): EDL-270 ==="
mkdir -p $OUT/ece/edl_270
python tracking/compute_ece.py \
    --npz_dir $OUT/results/edl_270 \
    --out_dir $OUT/ece/edl_270

# --- ECE for condition (c): EDL-270 + tau_det ---
echo "=== ECE condition (c): EDL-270 + tau_det ==="
mkdir -p $OUT/ece/edl_270_taudet
python tracking/compute_ece.py \
    --npz_dir $OUT/results/edl_270_taudet \
    --out_dir $OUT/ece/edl_270_taudet

# --- Acc for condition (b): EDL-270 ---
echo "=== Acc condition (b): EDL-270 ==="
python - <<'PYEOF'
import os, json, numpy as np

gt_root  = '${DATA}/validation'
pred_root = '${OUT}/test/tracking_results/uavtrack_eh/edl_270'

def iou(a, b):
    ax1,ay1,aw,ah = a; bx1,by1,bw,bh = b
    ix = max(ax1,bx1); iy = max(ay1,by1)
    ix2 = min(ax1+aw,bx1+bw); iy2 = min(ay1+ah,by1+bh)
    inter = max(0,ix2-ix)*max(0,iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0

scores = []
for seq in sorted(os.listdir(gt_root)):
    gt_file   = os.path.join(gt_root,  seq, 'IR_label.json')
    pred_file = os.path.join(pred_root, seq+'.txt')
    if not os.path.exists(gt_file) or not os.path.exists(pred_file): continue
    with open(gt_file) as f: gt = json.load(f)
    gt_boxes  = gt['gt_rect']
    gt_exist  = gt['exist']
    preds = []
    with open(pred_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            vals = list(map(float, line.split(',')))
            preds.append(vals[:4] if len(vals) >= 4 else [0,0,0,0])
    n = min(len(gt_boxes), len(preds))
    mixed = []
    for i in range(n):
        if gt_exist[i]:
            mixed.append(iou(gt_boxes[i], preds[i]))
        else:
            px,py,pw,ph = preds[i]
            mixed.append(1.0 if pw <= 0 or ph <= 0 else 0.0)
    if mixed: scores.append(np.mean(mixed))

acc = np.mean(scores)
print(f'Condition (b) EDL-270: Acc={acc:.4f}  (n={len(scores)} sequences)')
PYEOF

# --- Acc for condition (c): EDL-270 + tau_det ---
echo "=== Acc condition (c): EDL-270 + tau_det ==="
python - <<'PYEOF'
import os, json, numpy as np

gt_root  = '${DATA}/validation'
pred_root = '${OUT}/test/tracking_results/uavtrack_eh/edl_270_taudet'

def iou(a, b):
    ax1,ay1,aw,ah = a; bx1,by1,bw,bh = b
    ix = max(ax1,bx1); iy = max(ay1,by1)
    ix2 = min(ax1+aw,bx1+bw); iy2 = min(ay1+ah,by1+bh)
    inter = max(0,ix2-ix)*max(0,iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0

scores = []
for seq in sorted(os.listdir(gt_root)):
    gt_file   = os.path.join(gt_root,  seq, 'IR_label.json')
    pred_file = os.path.join(pred_root, seq+'.txt')
    if not os.path.exists(gt_file) or not os.path.exists(pred_file): continue
    with open(gt_file) as f: gt = json.load(f)
    gt_boxes  = gt['gt_rect']
    gt_exist  = gt['exist']
    preds = []
    with open(pred_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            vals = list(map(float, line.split(',')))
            preds.append(vals[:4] if len(vals) >= 4 else [0,0,0,0])
    n = min(len(gt_boxes), len(preds))
    mixed = []
    for i in range(n):
        if gt_exist[i]:
            mixed.append(iou(gt_boxes[i], preds[i]))
        else:
            px,py,pw,ph = preds[i]
            mixed.append(1.0 if pw <= 0 or ph <= 0 else 0.0)
    if mixed: scores.append(np.mean(mixed))

acc = np.mean(scores)
print(f'Condition (c) EDL-270+tau_det: Acc={acc:.4f}  (n={len(scores)} sequences)')
PYEOF

echo "=== Done — sync with: rsync -avz snellius:${OUT}/ece/ reports/ece/ ==="
