#!/bin/bash
#SBATCH --job-name=MultiTRAM_Stage1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage1_camera_%j.out
#SBATCH --error=logs/stage1_camera_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx:1

# ============================================================================
# Stage 1: Camera Estimation (VGGT)
# Estimates camera poses, intrinsics, and scene depth from video
# Output: cameras.npz, depth_maps/, point_cloud.ply
# ============================================================================

# Set cache directories FIRST, before loading any modules
export TORCH_HOME=/scratch/user/divyanshu/.cache/torch
export HF_HOME=/scratch/user/divyanshu/.cache/huggingface
export XDG_CACHE_HOME=/scratch/user/divyanshu/.cache

mkdir -p $TORCH_HOME
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

# Load modules
module purge
module load WebProxy Anaconda3/2021.11

# Initial GPU check
echo "========== GPU STATUS =========="
nvidia-smi
echo "================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
cd /scratch/user/divyanshu/research/multi-tram
conda activate multi-tram

echo "========== ANACONDA ENVIRONMENT =========="
conda list | grep -E "numpy|torch|opencv"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Start background GPU monitoring
(
  while true; do
    sleep 300  # Every 5 minutes
    echo "========== GPU STATUS CHECK $(date) =========="
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    echo "=============================================="
  done
) &
MONITOR_PID=$!

# ============================================================================
# CONFIGURE THESE PATHS
# ============================================================================
VIDEO_PATH="./data/videos/example_video.mp4"
OUTPUT_BASE_DIR="./results/example_video"
CONFIG_FILE="./configs/vggt.yaml"
MAX_FRAMES=100  # Set to empty string for all frames

# Create output directory
mkdir -p ${OUTPUT_BASE_DIR}/1_camera_estimation

# ============================================================================
# RUN STAGE 1
# ============================================================================
echo "========== STARTING STAGE 1: CAMERA ESTIMATION =========="
echo "Video: ${VIDEO_PATH}"
echo "Output: ${OUTPUT_BASE_DIR}/1_camera_estimation"
echo "Config: ${CONFIG_FILE}"
echo "Max frames: ${MAX_FRAMES}"
echo "=========================================================="

python experiments/run_stage1_camera.py \
    --video "${VIDEO_PATH}" \
    --output_dir "${OUTPUT_BASE_DIR}/1_camera_estimation" \
    --config "${CONFIG_FILE}" \
    --max_frames ${MAX_FRAMES} \
    --method vggt

EXIT_CODE=$?

# Kill GPU monitor
kill $MONITOR_PID 2>/dev/null

echo "========== STAGE 1 COMPLETED =========="
echo "Exit code: ${EXIT_CODE}"
echo "======================================="

exit $EXIT_CODE
