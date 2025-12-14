#!/bin/bash
#SBATCH --job-name=MultiTRAM_Stage2
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/stage2_tracking_%j.out
#SBATCH --error=logs/stage2_tracking_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx:1

# ============================================================================
# Stage 2: Multi-Person Tracking
# Detects and tracks multiple people across frames
# Depends on: Stage 1 outputs (cameras.npz, depth_maps/)
# Output: tracks.npz, detections/*.pkl, masks/*.png
# ============================================================================

# Set cache directories FIRST
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
    sleep 300
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
STAGE1_DIR="${OUTPUT_BASE_DIR}/1_camera_estimation"
CONFIG_FILE="./configs/tracking.yaml"
MAX_FRAMES=100
CONFIDENCE_THRESHOLD=0.5

# Verify Stage 1 outputs exist
if [ ! -f "${STAGE1_DIR}/cameras.npz" ]; then
    echo "ERROR: Stage 1 output not found: ${STAGE1_DIR}/cameras.npz"
    echo "Please run Stage 1 first!"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_BASE_DIR}/2_tracking

# ============================================================================
# RUN STAGE 2
# ============================================================================
echo "========== STARTING STAGE 2: MULTI-PERSON TRACKING =========="
echo "Video: ${VIDEO_PATH}"
echo "Stage 1 input: ${STAGE1_DIR}"
echo "Output: ${OUTPUT_BASE_DIR}/2_tracking"
echo "Config: ${CONFIG_FILE}"
echo "Confidence threshold: ${CONFIDENCE_THRESHOLD}"
echo "=============================================================="

python experiments/run_stage2_tracking.py \
    --video "${VIDEO_PATH}" \
    --stage1_dir "${STAGE1_DIR}" \
    --output_dir "${OUTPUT_BASE_DIR}/2_tracking" \
    --config "${CONFIG_FILE}" \
    --max_frames ${MAX_FRAMES} \
    --confidence_threshold ${CONFIDENCE_THRESHOLD}

EXIT_CODE=$?

# Kill GPU monitor
kill $MONITOR_PID 2>/dev/null

echo "========== STAGE 2 COMPLETED =========="
echo "Exit code: ${EXIT_CODE}"
echo "======================================="

exit $EXIT_CODE
