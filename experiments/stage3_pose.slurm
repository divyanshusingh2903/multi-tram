#!/bin/bash
#SBATCH --job-name=MultiTRAM_Stage3
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/stage3_pose_%j.out
#SBATCH --error=logs/stage3_pose_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx:2

# ============================================================================
# Stage 3: Per-Person Pose Estimation (VIMO)
# Estimates 3D body pose and shape for each tracked person
# Depends on: Stage 1 (cameras.npz), Stage 2 (tracks.npz)
# Output: person_XXX/smpl_params_camera.npz, joints_camera.npy
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
STAGE2_DIR="${OUTPUT_BASE_DIR}/2_tracking"
CONFIG_FILE="./configs/vimo.yaml"
MAX_FRAMES=100
BATCH_SIZE=8

# Optional: Process specific person IDs only
# PERSON_IDS="001 002 003"  # Uncomment to process specific people
PERSON_IDS=""  # Empty = process all people

# Verify previous stage outputs exist
if [ ! -f "${STAGE1_DIR}/cameras.npz" ]; then
    echo "ERROR: Stage 1 output not found: ${STAGE1_DIR}/cameras.npz"
    exit 1
fi

if [ ! -f "${STAGE2_DIR}/tracks.npz" ]; then
    echo "ERROR: Stage 2 output not found: ${STAGE2_DIR}/tracks.npz"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_BASE_DIR}/3_pose_estimation

# ============================================================================
# RUN STAGE 3
# ============================================================================
echo "========== STARTING STAGE 3: POSE ESTIMATION (VIMO) =========="
echo "Video: ${VIDEO_PATH}"
echo "Stage 1 input: ${STAGE1_DIR}"
echo "Stage 2 input: ${STAGE2_DIR}"
echo "Output: ${OUTPUT_BASE_DIR}/3_pose_estimation"
echo "Config: ${CONFIG_FILE}"
echo "Batch size: ${BATCH_SIZE}"
if [ -n "$PERSON_IDS" ]; then
    echo "Processing specific persons: ${PERSON_IDS}"
fi
echo "==============================================================="

if [ -n "$PERSON_IDS" ]; then
    python experiments/run_stage3_pose.py \
        --video "${VIDEO_PATH}" \
        --stage1_dir "${STAGE1_DIR}" \
        --stage2_dir "${STAGE2_DIR}" \
        --output_dir "${OUTPUT_BASE_DIR}/3_pose_estimation" \
        --config "${CONFIG_FILE}" \
        --max_frames ${MAX_FRAMES} \
        --batch_size ${BATCH_SIZE} \
        --person_ids ${PERSON_IDS}
else
    python experiments/run_stage3_pose.py \
        --video "${VIDEO_PATH}" \
        --stage1_dir "${STAGE1_DIR}" \
        --stage2_dir "${STAGE2_DIR}" \
        --output_dir "${OUTPUT_BASE_DIR}/3_pose_estimation" \
        --config "${CONFIG_FILE}" \
        --max_frames ${MAX_FRAMES} \
        --batch_size ${BATCH_SIZE}
fi

EXIT_CODE=$?

# Kill GPU monitor
kill $MONITOR_PID 2>/dev/null

echo "========== STAGE 3 COMPLETED =========="
echo "Exit code: ${EXIT_CODE}"
echo "======================================="

exit $EXIT_CODE
