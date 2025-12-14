#!/bin/bash
#SBATCH --job-name=MultiTRAM_Full
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/multi_tram_full_%j.out
#SBATCH --error=logs/multi_tram_full_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx:2

# ============================================================================
# Multi-TRAM Complete Pipeline
# Runs all 5 stages sequentially:
#   Stage 1: Camera Estimation (VGGT)
#   Stage 2: Multi-Person Tracking
#   Stage 3: Per-Person Pose Estimation (VIMO)
#   Stage 4: World-Space Transformation
#   Stage 5: Optional Refinement (SLAHMR)
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
conda list | grep -E "numpy|torch|opencv|scipy"
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
# CONFIGURATION
# ============================================================================
VIDEO_PATH="./data/videos/example_video.mp4"
OUTPUT_BASE_DIR="./results/example_video"
MAX_FRAMES=100  # Set to empty for all frames

# Stage-specific configs
CONFIG_STAGE1="./configs/vggt.yaml"
CONFIG_STAGE2="./configs/tracking.yaml"
CONFIG_STAGE3="./configs/vimo.yaml"
CONFIG_STAGE4="./configs/world_transform.yaml"
CONFIG_STAGE5="./configs/refinement.yaml"

# Optional: Skip specific stages (e.g., SKIP_STAGES="5" to skip refinement)
SKIP_STAGES=""

# Refinement parameters
NUM_ITERATIONS=100
LAMBDA_DEPTH=5.0
LAMBDA_SCALE=2.0

# Create all output directories
mkdir -p ${OUTPUT_BASE_DIR}/{1_camera_estimation,2_tracking,3_pose_estimation,4_world_space,5_refinement}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
function check_exit_code() {
    if [ $1 -ne 0 ]; then
        echo "ERROR: Stage $2 failed with exit code $1"
        kill $MONITOR_PID 2>/dev/null
        exit $1
    fi
}

function should_skip_stage() {
    if [[ $SKIP_STAGES =~ (^|[[:space:]])$1($|[[:space:]]) ]]; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# STAGE 1: CAMERA ESTIMATION
# ============================================================================
if should_skip_stage "1"; then
    echo "Skipping Stage 1 (Camera Estimation)"
else
    echo ""
    echo "========================================================================"
    echo "STAGE 1: CAMERA ESTIMATION (VGGT)"
    echo "========================================================================"
    echo "Started: $(date)"
    echo ""

    python experiments/run_stage1_camera.py \
        --video "${VIDEO_PATH}" \
        --output_dir "${OUTPUT_BASE_DIR}/1_camera_estimation" \
        --config "${CONFIG_STAGE1}" \
        --max_frames ${MAX_FRAMES} \
        --method vggt

    check_exit_code $? "1 (Camera Estimation)"

    echo ""
    echo "Stage 1 completed: $(date)"
    echo "========================================================================"
fi

# ============================================================================
# STAGE 2: MULTI-PERSON TRACKING
# ============================================================================
if should_skip_stage "2"; then
    echo "Skipping Stage 2 (Multi-Person Tracking)"
else
    echo ""
    echo "========================================================================"
    echo "STAGE 2: MULTI-PERSON TRACKING"
    echo "========================================================================"
    echo "Started: $(date)"
    echo ""

    python experiments/run_stage2_tracking.py \
        --video "${VIDEO_PATH}" \
        --stage1_dir "${OUTPUT_BASE_DIR}/1_camera_estimation" \
        --output_dir "${OUTPUT_BASE_DIR}/2_tracking" \
        --config "${CONFIG_STAGE2}" \
        --max_frames ${MAX_FRAMES} \
        --confidence_threshold 0.5

    check_exit_code $? "2 (Multi-Person Tracking)"

    echo ""
    echo "Stage 2 completed: $(date)"
    echo "========================================================================"
fi

# ============================================================================
# STAGE 3: PER-PERSON POSE ESTIMATION
# ============================================================================
if should_skip_stage "3"; then
    echo "Skipping Stage 3 (Pose Estimation)"
else
    echo ""
    echo "========================================================================"
    echo "STAGE 3: PER-PERSON POSE ESTIMATION (VIMO)"
    echo "========================================================================"
    echo "Started: $(date)"
    echo ""

    python experiments/run_stage3_pose.py \
        --video "${VIDEO_PATH}" \
        --stage1_dir "${OUTPUT_BASE_DIR}/1_camera_estimation" \
        --stage2_dir "${OUTPUT_BASE_DIR}/2_tracking" \
        --output_dir "${OUTPUT_BASE_DIR}/3_pose_estimation" \
        --config "${CONFIG_STAGE3}" \
        --max_frames ${MAX_FRAMES} \
        --batch_size 8

    check_exit_code $? "3 (Pose Estimation)"

    echo ""
    echo "Stage 3 completed: $(date)"
    echo "========================================================================"
fi

# ============================================================================
# STAGE 4: WORLD-SPACE TRANSFORMATION
# ============================================================================
if should_skip_stage "4"; then
    echo "Skipping Stage 4 (World Transformation)"
else
    echo ""
    echo "========================================================================"
    echo "STAGE 4: WORLD-SPACE TRANSFORMATION"
    echo "========================================================================"
    echo "Started: $(date)"
    echo ""

    python experiments/run_stage4_world.py \
        --stage1_dir "${OUTPUT_BASE_DIR}/1_camera_estimation" \
        --stage3_dir "${OUTPUT_BASE_DIR}/3_pose_estimation" \
        --output_dir "${OUTPUT_BASE_DIR}/4_world_space" \
        --config "${CONFIG_STAGE4}"

    check_exit_code $? "4 (World Transformation)"

    echo ""
    echo "Stage 4 completed: $(date)"
    echo "========================================================================"
fi

# ============================================================================
# STAGE 5: OPTIONAL REFINEMENT
# ============================================================================
if should_skip_stage "5"; then
    echo "Skipping Stage 5 (Refinement) - OPTIONAL"
else
    echo ""
    echo "========================================================================"
    echo "STAGE 5: OPTIONAL REFINEMENT (SLAHMR-BASED)"
    echo "========================================================================"
    echo "Started: $(date)"
    echo ""

    python experiments/run_stage5_refinement.py \
        --video "${VIDEO_PATH}" \
        --stage1_dir "${OUTPUT_BASE_DIR}/1_camera_estimation" \
        --stage2_dir "${OUTPUT_BASE_DIR}/2_tracking" \
        --stage3_dir "${OUTPUT_BASE_DIR}/3_pose_estimation" \
        --stage4_dir "${OUTPUT_BASE_DIR}/4_world_space" \
        --output_dir "${OUTPUT_BASE_DIR}/5_refinement" \
        --config "${CONFIG_STAGE5}" \
        --num_iterations ${NUM_ITERATIONS} \
        --lambda_depth ${LAMBDA_DEPTH} \
        --lambda_scale ${LAMBDA_SCALE}

    check_exit_code $? "5 (Refinement)"

    echo ""
    echo "Stage 5 completed: $(date)"
    echo "========================================================================"
fi

# Kill GPU monitor
kill $MONITOR_PID 2>/dev/null

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "========================================================================"
echo "MULTI-TRAM PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================================"
echo "Video: ${VIDEO_PATH}"
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo "Completed: $(date)"
echo ""
echo "Results structure:"
echo "  ${OUTPUT_BASE_DIR}/"
echo "  ├── 1_camera_estimation/    # Cameras, depth maps, point cloud"
echo "  ├── 2_tracking/              # Track IDs, detections, masks"
echo "  ├── 3_pose_estimation/       # Per-person SMPL (camera frame)"
echo "  ├── 4_world_space/           # Per-person SMPL (world frame)"
echo "  └── 5_refinement/            # Refined SMPL parameters (optional)"
echo "========================================================================"

exit 0
