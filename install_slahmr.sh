#!/bin/bash
# Install SLAHMR and PHALP for multi-person tracking

echo "=========================================="
echo "Installing PHALP and SLAHMR"
echo "=========================================="
echo ""

# IMPORTANT: PHALP is a separate package, NOT part of SLAHMR!

# Install PHALP first (separate package)
echo "Step 1: Installing PHALP (separate package)..."
pip install "phalp[all]@git+https://github.com/brjathu/PHALP.git"

echo ""
echo "Step 2: Installing SLAHMR (for Stage 5 optimization)..."
cd thirdparty/slahmr
pip install -e .
cd ../..

# Install key dependencies
echo ""
echo "Step 3: Installing key dependencies..."
pip install smplx pyrender open3d imageio-ffmpeg
pip install mmcv==1.3.9
pip install timm==0.4.9
pip install motmetrics
pip install dill
pip install xtcocotools
pip install pandas==1.4.0
pip install tensorboard

# Install git dependencies
echo ""
echo "Step 4: Installing git dependencies..."
pip install git+https://github.com/nghorbani/configer
pip install git+https://github.com/mattloper/chumpy

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "PHALP and SLAHMR are now installed:"
echo "  - PHALP: Person tracking (Stage 2)"
echo "  - SLAHMR: Multi-person optimization (Stage 5)"
echo ""
echo "To verify installation:"
echo "  python -c 'import phalp; print(\"PHALP OK\")'"
echo "  python -c 'import slahmr; print(\"SLAHMR OK\")'"
