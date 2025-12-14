#!/bin/bash
#SBATCH --job-name=MultiTRAM_Stage4
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/stage4_world_%j.out
#SBATCH --error=logs/stage4_world_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx:1

# ============================================================================
# Stage 4: World-Space Transformation
# Transforms all people from camera frame to shared world coordinates
# Depends on: Stage 1 (cameras.npz), Stage 3 (person_XXX/)
# Output: person_XXX/smpl_params_world.npz, trajectory_world.npy
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
conda list | grep -E "numpy|torch"
echo "=========================================="

# Create logs directory
mkdir -p logs

# ============================================================================
# CONFIGURE THESE PATHS
# ============================================================================
OUTPUT_BASE_DIR="./results/example_video"
STAGE1_DIR="${OUTPUT_BASE_DIR}/1_camera_estimation"
STAGE3_DIR="${OUTPUT_BASE_DIR}/3_pose_estimation"
CONFIG_FILE="./configs/world_transform.yaml"

# Optional: Process specific person IDs only
# PERSON_IDS="001 002 003"
PERSON_IDS=""

# Verify previous stage outputs exist
if [ ! -f "${STAGE1_DIR}/cameras.npz" ]; then
    echo "ERROR: Stage 1 output not found: ${STAGE1_DIR}/cameras.npz"
    exit 1
fi

if [ ! -d "${STAGE3_DIR}" ]; then
    echo "ERROR: Stage 3 output directory not found: ${STAGE3_DIR}"
    exit 1
fi

# Check if person directories exist
PERSON_COUNT=$(find ${STAGE3_DIR} -maxdepth 1 -type d -name "person_*" | wc -l)
if [ $PERSON_COUNT -eq 0 ]; then
    echo "ERROR: No person directories found in ${STAGE3_DIR}"
    exit 1
fi

echo "Found ${PERSON_COUNT} person directories"

# Create output directory
mkdir -p ${OUTPUT_BASE_DIR}/4_world_space

# ============================================================================
# RUN STAGE 4
# ============================================================================
echo "========== STARTING STAGE 4: WORLD TRANSFORMATION =========="
echo "Stage 1 input: ${STAGE1_DIR}"
echo "Stage 3 input: ${STAGE3_DIR}"
echo "Output: ${OUTPUT_BASE_DIR}/4_world_space"
echo "Config: ${CONFIG_FILE}"
if [ -n "$PERSON_IDS" ]; then
    echo "Processing specific persons: ${PERSON_IDS}"
fi
echo "============================================================"

if [ -n "$PERSON_IDS" ]; then
    python experiments/run_stage4_world.py \
        --stage1_dir "${STAGE1_DIR}" \
        --stage3_dir "${STAGE3_DIR}" \
        --output_dir "${OUTPUT_BASE_DIR}/4_world_space" \
        --config "${CONFIG_FILE}" \
        --person_ids ${PERSON_IDS}
else
    python experiments/run_stage4_world.py \
        --stage1_dir "${STAGE1_DIR}" \
        --stage3_dir "${STAGE3_DIR}" \
        --output_dir "${OUTPUT_BASE_DIR}/4_world_space" \
        --config "${CONFIG_FILE}"
fi

EXIT_CODE=$?

echo "========== STAGE 4 COMPLETED =========="
echo "Exit code: ${EXIT_CODE}"
echo "======================================="

exit $EXIT_CODE
