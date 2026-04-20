#!/bin/bash
# Install all pip-only dependencies for multi-tram.
# Run after: conda env create -f environment.yml && conda activate multi-tram
#
# Usage:
#   bash install_dependencies.sh
#   bash install_dependencies.sh --skip-slahmr   # skip Stage 5 deps

set -e

SKIP_SLAHMR=false
for arg in "$@"; do
  [[ "$arg" == "--skip-slahmr" ]] && SKIP_SLAHMR=true
done

# CUDA_HOME is required to build extensions (neural-renderer-pytorch, detectron2, etc.)
if [ -z "$CUDA_HOME" ]; then
  for candidate in /sw/eb/sw/CUDA/13.0.1 /usr/local/cuda /usr/local/cuda-12* /usr/local/cuda-11*; do
    if [ -d "$candidate" ]; then
      export CUDA_HOME="$candidate"
      echo "Auto-detected CUDA_HOME=$CUDA_HOME"
      break
    fi
  done
fi
if [ -z "$CUDA_HOME" ]; then
  echo "ERROR: CUDA_HOME is not set and could not be auto-detected."
  echo "  Set it manually before running this script:"
  echo "    export CUDA_HOME=/usr/local/cuda"
  exit 1
fi

echo "=========================================="
echo " multi-tram: pip dependency installer"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1: VGGT checkpoint (informational — downloaded separately)
# ---------------------------------------------------------------------------
echo "Step 1: VGGT checkpoint"
if [ ! -f "data/pretrained_models/vggt/model.pt" ]; then
  echo "  Not found. Download with:"
  echo "    huggingface-cli login"
  echo "    python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('facebook/VGGT-1B', 'model.pt', local_dir='data/pretrained_models/vggt')\""
else
  echo "  data/pretrained_models/vggt/model.pt already present."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: YOLOv8x weights
# ---------------------------------------------------------------------------
echo "Step 2: YOLOv8x weights..."
if [ ! -f "yolov8x.pt" ]; then
  python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')" \
    && echo "  yolov8x.pt downloaded." \
    || echo "  WARNING: YOLOv8x download failed — download manually."
else
  echo "  yolov8x.pt already present, skipping."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: SAM 2 (pixel-accurate dynamic masks, Stage 1)
# ---------------------------------------------------------------------------
echo "Step 3: SAM 2..."
if ! python -c "import sam2" 2>/dev/null; then
  pip install "git+https://github.com/facebookresearch/sam2.git"
else
  echo "  SAM 2 already installed, skipping."
fi

echo "  Downloading SAM 2 checkpoint..."
mkdir -p checkpoints/sam2
if [ ! -f "checkpoints/sam2/sam2.1_hiera_large.pt" ]; then
  wget -q --show-progress -P checkpoints/sam2/ \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
else
  echo "  Checkpoint already present, skipping."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: detectron2 (PHALP dependency — needs --no-build-isolation)
# ---------------------------------------------------------------------------
echo "Step 4: detectron2..."
if ! python -c "import detectron2" 2>/dev/null; then
  pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git"
  # detectron2 pins iopath<0.1.10 but sam-2 needs >=0.1.10; upgrade it back
  pip install "iopath>=0.1.10" --upgrade
else
  echo "  detectron2 already installed, skipping."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 5: PHALP+ (multi-person tracking, Stage 2)
# ---------------------------------------------------------------------------
echo "Step 5: PHALP+..."
if ! python -c "import phalp" 2>/dev/null; then
  # --no-build-isolation needed because PHALP's detectron2 dep also requires torch at build time
  pip install --no-build-isolation "phalp[all]@git+https://github.com/brjathu/PHALP.git"
else
  echo "  PHALP already installed, skipping."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 6: SLAHMR (Stage 5 refinement — optional)
# ---------------------------------------------------------------------------
if [ "$SKIP_SLAHMR" = true ]; then
  echo "Step 6: SLAHMR — skipped (--skip-slahmr)."
else
  echo "Step 6: SLAHMR (Stage 5 refinement)..."
  if [ ! -d "thirdparty/slahmr" ] || [ -z "$(ls -A thirdparty/slahmr)" ]; then
    echo "  ERROR: thirdparty/slahmr not found. Run first:"
    echo "    git submodule update --init --recursive"
    exit 1
  fi
  pip install -e thirdparty/slahmr
  pip install \
    "git+https://github.com/nghorbani/configer" \
    "git+https://github.com/mattloper/chumpy"
fi
echo ""

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo "=========================================="
echo " Verifying installations..."
echo "=========================================="
python -c "import sam2;       print('  SAM 2      OK')" 2>/dev/null || echo "  SAM 2      FAILED"
python -c "import detectron2; print('  detectron2 OK')" 2>/dev/null || echo "  detectron2 FAILED"
python -c "import phalp;      print('  PHALP      OK')" 2>/dev/null || echo "  PHALP      FAILED"
if [ "$SKIP_SLAHMR" = false ]; then
  python -c "import slahmr;   print('  SLAHMR     OK')" 2>/dev/null || echo "  SLAHMR     FAILED"
fi
echo ""
echo "Done. Model weights that require manual download:"
echo "  - VGGT:  data/pretrained_models/vggt/model.pt  (see Step 1 above)"
echo "  - PHALP: auto-downloaded on first run"
