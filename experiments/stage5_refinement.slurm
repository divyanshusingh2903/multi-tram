#!/bin/bash
#SBATCH --job-name=MultiTRAM_Stage5
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/stage5_refinement_%j.out
#SBATCH --error=logs/stage5_refinement_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx:2

# ============================================================================
# Stage 5: Optional Refinement (SLAHMR-based Optimization)
# Refines SMPL parameters through multi-person optimization
# Depends on: All previous stage outputs (1-4)
# Output: person_XXX/smpl_params_refined.npz, optimization_metrics.json
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
conda list | grep -E "numpy|torch|scipy"
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
STAGE3_DIR="${OUTPUT_BASE_DIR}/3_pose_estimation"
STAGE4_DIR="${OUTPUT_BASE_DIR}/4_world_space"
CONFIG_FILE="./configs/refinement.yaml"

# Optimization parameters
NUM_ITERATIONS=100
LAMBDA_DEPTH=5.0
LAMBDA_SCALE=2.0

# Optional: Process specific person IDs only
# PERSON_IDS="001 002 003"
PERSON_IDS=""

# Verify all previous stage outputs exist
REQUIRED_FILES=(
    "${STAGE1_DIR}/cameras.npz"
    "${STAGE2_DIR}/tracks.npz"
)

for FILE in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "ERROR: Required file not found: $FILE"
        echo "Please run all previous stages first!"
        exit 1
    fi
done

if [ ! -d "${STAGE3_DIR}" ]; then
    echo "ERROR: Stage 3 output directory not found: ${STAGE3_DIR}"
    exit 1
fi

if [ ! -d "${STAGE4_DIR}" ]; then
    echo "ERROR: Stage 4 output directory not found: ${STAGE4_DIR}"
    exit 1
fi

# Check if person directories exist
PERSON_COUNT=$(find ${STAGE4_DIR} -maxdepth 1 -type d -name "person_*" | wc -l)
if [ $PERSON_COUNT -eq 0 ]; then
    echo "ERROR: No person directories found in ${STAGE4_DIR}"
    exit 1
fi

echo "Found ${PERSON_COUNT} person directories for refinement"

# Create output directory
mkdir -p ${OUTPUT_BASE_DIR}/5_refinement

# ============================================================================
# RUN STAGE 5
# ============================================================================
echo "========== STARTING STAGE 5: REFINEMENT (SLAHMR) =========="
echo "Video: ${VIDEO_PATH}"
echo "Stage 1 input: ${STAGE1_DIR}"
echo "Stage 2 input: ${STAGE2_DIR}"
echo "Stage 3 input: ${STAGE3_DIR}"
echo "Stage 4 input: ${STAGE4_DIR}"
echo "Output: ${OUTPUT_BASE_DIR}/5_refinement"
echo "Config: ${CONFIG_FILE}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Lambda depth: ${LAMBDA_DEPTH}"
echo "Lambda scale: ${LAMBDA_SCALE}"
if [ -n "$PERSON_IDS" ]; then
    echo "Processing specific persons: ${PERSON_IDS}"
fi
echo "============================================================"

if [ -n "$PERSON_IDS" ]; then
    python experiments/run_stage5_refinement.py \
        --video "${VIDEO_PATH}" \
        --stage1_dir "${STAGE1_DIR}" \
        --stage2_dir "${STAGE2_DIR}" \
        --stage3_dir "${STAGE3_DIR}" \
        --stage4_dir "${STAGE4_DIR}" \
        --output_dir "${OUTPUT_BASE_DIR}/5_refinement" \
        --config "${CONFIG_FILE}" \
        --num_iterations ${NUM_ITERATIONS} \
        --lambda_depth ${LAMBDA_DEPTH} \
        --lambda_scale ${LAMBDA_SCALE} \
        --person_ids ${PERSON_IDS}
else
    python experiments/run_stage5_refinement.py \
        --video "${VIDEO_PATH}" \
        --stage1_dir "${STAGE1_DIR}" \
        --stage2_dir "${STAGE2_DIR}" \
        --stage3_dir "${STAGE3_DIR}" \
        --stage4_dir "${STAGE4_DIR}" \
        --output_dir "${OUTPUT_BASE_DIR}/5_refinement" \
        --config "${CONFIG_FILE}" \
        --num_iterations ${NUM_ITERATIONS} \
        --lambda_depth ${LAMBDA_DEPTH} \
        --lambda_scale ${LAMBDA_SCALE}
fi

EXIT_CODE=$?

# Kill GPU monitor
kill $MONITOR_PID 2>/dev/null

echo "========== STAGE 5 COMPLETED =========="
echo "Exit code: ${EXIT_CODE}"
echo "======================================="

exit $EXIT_CODE
