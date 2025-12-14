#!/bin/bash
# ============================================================================
# Multi-TRAM Setup Checker
# Verifies environment and dependencies before running pipeline
# Usage: bash experiments/check_setup.sh
# ============================================================================

echo "============================================================================"
echo "Multi-TRAM Setup Verification"
echo "============================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# ============================================================================
# 1. Check Python version
# ============================================================================
echo "1. Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION (>= 3.8 required)"
else
    echo -e "${RED}✗${NC} Python $PYTHON_VERSION (>= 3.8 required)"
    ERRORS=$((ERRORS + 1))
fi

# ============================================================================
# 2. Check conda environment
# ============================================================================
echo ""
echo "2. Checking conda environment..."
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}⚠${NC} No conda environment activated"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✓${NC} Conda environment: $CONDA_DEFAULT_ENV"
fi

# ============================================================================
# 3. Check required Python packages
# ============================================================================
echo ""
echo "3. Checking required Python packages..."

REQUIRED_PACKAGES=("torch" "numpy" "opencv-python" "scipy" "yaml")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import ${pkg}" 2>/dev/null; then
        VERSION=$(python -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✓${NC} ${pkg} (${VERSION})"
    else
        echo -e "${RED}✗${NC} ${pkg} not found"
        ERRORS=$((ERRORS + 1))
    fi
done

# ============================================================================
# 4. Check CUDA/GPU availability
# ============================================================================
echo ""
echo "4. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        echo -e "${GREEN}✓${NC} Found $GPU_COUNT GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
            echo "  - $line"
        done
    else
        echo -e "${RED}✗${NC} No GPUs detected"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}✗${NC} nvidia-smi not found"
    ERRORS=$((ERRORS + 1))
fi

# Check PyTorch CUDA
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    echo -e "${GREEN}✓${NC} PyTorch CUDA ${CUDA_VERSION} available"
else
    echo -e "${RED}✗${NC} PyTorch CUDA not available"
    ERRORS=$((ERRORS + 1))
fi

# ============================================================================
# 5. Check directory structure
# ============================================================================
echo ""
echo "5. Checking directory structure..."

REQUIRED_DIRS=("src" "src/stages" "src/models" "src/utils" "configs" "experiments")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $dir/"
    else
        echo -e "${RED}✗${NC} $dir/ not found"
        ERRORS=$((ERRORS + 1))
    fi
done

# ============================================================================
# 6. Check stage files
# ============================================================================
echo ""
echo "6. Checking stage implementation files..."

STAGE_FILES=(
    "src/stages/stage1_camera.py"
    "src/stages/stage2_tracking.py"
    "src/stages/stage3_pose.py"
    "src/stages/stage4_world_transform.py"
    "src/stages/stage5_refinement.py"
)

for file in "${STAGE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file not found"
        ERRORS=$((ERRORS + 1))
    fi
done

# ============================================================================
# 7. Check experiment scripts
# ============================================================================
echo ""
echo "7. Checking experiment scripts..."

EXPERIMENT_SCRIPTS=(
    "experiments/run_stage1_camera.py"
    "experiments/run_stage2_tracking.py"
    "experiments/run_stage3_pose.py"
    "experiments/run_stage4_world.py"
    "experiments/run_stage5_refinement.py"
)

for script in "${EXPERIMENT_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo -e "${GREEN}✓${NC} $script"
    else
        echo -e "${RED}✗${NC} $script not found"
        ERRORS=$((ERRORS + 1))
    fi
done

# ============================================================================
# 8. Check SLURM scripts
# ============================================================================
echo ""
echo "8. Checking SLURM job scripts..."

SLURM_SCRIPTS=(
    "experiments/stage1_camera.slurm"
    "experiments/stage2_tracking.slurm"
    "experiments/stage3_pose.slurm"
    "experiments/stage4_world.slurm"
    "experiments/stage5_refinement.slurm"
    "experiments/run_all_stages.slurm"
)

for script in "${SLURM_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo -e "${GREEN}✓${NC} $script"
    else
        echo -e "${RED}✗${NC} $script not found"
        ERRORS=$((ERRORS + 1))
    fi
done

# ============================================================================
# 9. Check configuration files
# ============================================================================
echo ""
echo "9. Checking configuration files..."

CONFIG_FILES=(
    "configs/vggt.yaml"
)

for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        echo -e "${GREEN}✓${NC} $config"
    else
        echo -e "${YELLOW}⚠${NC} $config not found (will use defaults)"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# ============================================================================
# 10. Check pretrained models (if applicable)
# ============================================================================
echo ""
echo "10. Checking pretrained models..."

MODEL_DIRS=(
    "data/pretrained_models/vggt"
    "data/pretrained_models/vimo"
    "data/pretrained_models/phalp"
)

for dir in "${MODEL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $dir/"
    else
        echo -e "${YELLOW}⚠${NC} $dir/ not found (models will be downloaded on first run)"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# ============================================================================
# 11. Check cache directories
# ============================================================================
echo ""
echo "11. Checking cache directories..."

CACHE_DIRS=(
    "$TORCH_HOME"
    "$HF_HOME"
    "$XDG_CACHE_HOME"
)

for dir in "${CACHE_DIRS[@]}"; do
    if [ -n "$dir" ]; then
        if [ -d "$dir" ]; then
            echo -e "${GREEN}✓${NC} $dir (writable: $([ -w "$dir" ] && echo "yes" || echo "no"))"
        else
            echo -e "${YELLOW}⚠${NC} $dir does not exist (will be created)"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo -e "${YELLOW}⚠${NC} Cache directory not set"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "Summary"
echo "============================================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed! ✓${NC}"
    echo "You're ready to run the Multi-TRAM pipeline."
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Checks completed with $WARNINGS warning(s) ⚠${NC}"
    echo "Pipeline should work, but check warnings above."
    exit 0
else
    echo -e "${RED}Checks failed with $ERRORS error(s) and $WARNINGS warning(s) ✗${NC}"
    echo ""
    echo "Please fix the errors above before running the pipeline."
    echo ""
    echo "Common fixes:"
    echo "  - Install missing packages: pip install torch numpy opencv-python scipy pyyaml"
    echo "  - Activate conda environment: conda activate multi-tram"
    echo "  - Check GPU drivers: nvidia-smi"
    echo "  - Create missing directories: mkdir -p data/pretrained_models"
    exit 1
fi
