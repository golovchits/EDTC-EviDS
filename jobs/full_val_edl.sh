#!/bin/bash
#SBATCH --job-name=edtc_val_edl
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/edtc_val_edl_%j.out
#SBATCH --error=logs/edtc_val_edl_%j.err

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
OUT="${OUT:-${CODE}/results}"

mkdir -p $OUT/results/edl_270
mkdir -p $OUT/results/edl_270_taudet

cd $CODE

# --- Condition (b): EDL-270, no uncertainty suppression ---
echo "=== Condition (b): EDL-270 inference (50 val sequences) ==="
python tracking/run_with_diagnostics.py \
    --tracker_name  uavtrack_eh \
    --tracker_param edl_270 \
    --dataset_name  antiuav \
    --num_gpus      1 \
    --params__model $CODE/pretrained_models/UAVTrackEH.pth.tar \
    --params__evidential_threshold 0.2 \
    --npz_dir $OUT/results/edl_270

# --- Condition (c): EDL-270 + tau_det=0.5 ---
echo "=== Condition (c): EDL-270 + tau_det=0.5 inference (50 val sequences) ==="
python tracking/run_with_diagnostics.py \
    --tracker_name  uavtrack_eh \
    --tracker_param edl_270_taudet \
    --dataset_name  antiuav \
    --num_gpus      1 \
    --params__model $CODE/pretrained_models/UAVTrackEH.pth.tar \
    --params__evidential_threshold 0.2 \
    --npz_dir $OUT/results/edl_270_taudet

# --- Tune split inference for EDL (needed for calibration comparison) ---
mkdir -p $OUT/results/edl_270_tune

echo "=== EDL-270 tune split inference (30 sequences) ==="
python tracking/run_with_diagnostics.py \
    --tracker_name  uavtrack_eh \
    --tracker_param edl_270 \
    --dataset_name  antiuav_tune \
    --num_gpus      1 \
    --params__model $CODE/pretrained_models/UAVTrackEH.pth.tar \
    --params__evidential_threshold 0.2 \
    --npz_dir $OUT/results/edl_270_tune

# --- Calibration metrics (ECE, reliability diagrams) ---
echo "=== EDL calibration (ECE) ==="
python tracking/calibrate_temperature_edtc.py \
    --tune_npz_dir $OUT/results/edl_270_tune \
    --val_npz_dir  $OUT/results/edl_270 \
    --out_dir      $OUT/ece/edl_270

echo "=== Done ==="
