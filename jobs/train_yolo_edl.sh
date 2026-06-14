#!/bin/bash
#SBATCH --job-name=yolo_edl
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=17:00:00
#SBATCH --output=logs/yolo_edl_%j.out
#SBATCH --error=logs/yolo_edl_%j.err

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
export CUDA_HOME=/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1
export TORCH_CUDA_ARCH_LIST="8.0"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p $TMPDIR
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system

CODE="${CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA="${DATA:-/path/to/AntiUAV600}"  # update for your system
OUT="${OUT:-${CODE}/results}"

mkdir -p $OUT/yolo_runs

# --- Step 1: generate image list files (idempotent — skips if already done) ---
echo "=== Preparing data lists ==="
python $CODE/tools/prepare_antiuav_yolo_data.py \
    --dataset_root $DATA/train \
    --split_json   $CODE/antiuav_train_val_split.json \
    --out_dir      $DATA/yolo_lists

# --- Step 2: train YOLOv5s_s with EDL head (20 epochs, mirrors sigmoid_270) ---
# --edl switches the loss to ComputeLossEDL and expects DetectEDL in the cfg.
# --t-anneal 10: KL annealing over first half of training (lambda_t = min(1, epoch/10)).
echo "=== Training YOLOv5s_s EDL head (20 epochs) ==="
cd $CODE/yolov5
python train.py \
    --weights yolov5s.pt \
    --cfg     models/yolov5s_s_edl.yaml \
    --data    data/antiuav_270.yaml \
    --hyp     data/hyps/hyp.scratch-low.yaml \
    --epochs  20 \
    --batch-size 32 \
    --imgsz   640 \
    --project $OUT/yolo_runs \
    --name    edl_270 \
    --exist-ok \
    --workers 8 \
    --edl \
    --t-anneal 10

# --- Step 3: copy best checkpoint ---
echo "=== Saving checkpoint ==="
cp $OUT/yolo_runs/edl_270/weights/best.pt \
   $CODE/pretrained_models/yolo_edl_270.pt
echo "Checkpoint saved → $CODE/pretrained_models/yolo_edl_270.pt"

echo "=== Done ==="
