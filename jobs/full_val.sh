#!/bin/bash
#SBATCH --job-name=edtc_val
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/edtc_val_%j.out
#SBATCH --error=logs/edtc_val_%j.err

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
export CUDA_HOME=/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1
export TORCH_CUDA_ARCH_LIST="8.0"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p $TMPDIR

source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system
mkdir -p ${OUT}/results/val
mkdir -p ${OUT}/test/tracking_results

cd "${CODE}"

python tracking/run_with_diagnostics.py \
    --tracker_name uavtrack_eh \
    --tracker_param baseline \
    --dataset_name antiuav \
    --num_gpus 1 \
    --params__model ${CODE}/pretrained_models/UAVTrackEH.pth.tar \
    --params__evidential_threshold 0.2 \
    --npz_dir ${OUT}/results/val
