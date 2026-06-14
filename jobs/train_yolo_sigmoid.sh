#!/bin/bash
#SBATCH --job-name=yolo_sigmoid
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=17:00:00
#SBATCH --output=logs/yolo_sigmoid_%j.out
#SBATCH --error=logs/yolo_sigmoid_%j.err

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

mkdir -p $OUT/results/sigmoid_270
mkdir -p $OUT/yolo_runs

# --- Step 1: generate image list files ---
echo "=== Preparing data lists ==="
python $CODE/tools/prepare_antiuav_yolo_data.py \
    --dataset_root $DATA/train \
    --split_json   $CODE/antiuav_train_val_split.json \
    --out_dir      $DATA/yolo_lists

# --- Step 2: train YOLOv5s_s sigmoid baseline ---
echo "=== Training YOLOv5s_s sigmoid (20 epochs) ==="
cd $CODE/yolov5
python train.py \
    --weights yolov5s.pt \
    --cfg     models/yolov5s_s.yaml \
    --data    data/antiuav_270.yaml \
    --hyp     data/hyps/hyp.scratch-low.yaml \
    --epochs  20 \
    --batch-size 32 \
    --imgsz   640 \
    --project $OUT/yolo_runs \
    --name    sigmoid_270 \
    --exist-ok \
    --workers 8

# --- Step 3: copy best checkpoint ---
echo "=== Saving checkpoint ==="
cp $OUT/yolo_runs/sigmoid_270/weights/best.pt \
   $CODE/pretrained_models/yolo_sigmoid_270.pt
echo "Checkpoint saved → $CODE/pretrained_models/yolo_sigmoid_270.pt"

# --- Step 4: run val inference with sigmoid_270 checkpoint ---
echo "=== Running val inference ==="
cd $CODE
python tracking/run_with_diagnostics.py \
    --tracker_name  uavtrack_eh \
    --tracker_param sigmoid_270 \
    --dataset_name  antiuav \
    --num_gpus      1 \
    --params__model $CODE/pretrained_models/UAVTrackEH.pth.tar \
    --params__evidential_threshold 0.2 \
    --npz_dir $OUT/results/sigmoid_270

echo "=== Done ==="
