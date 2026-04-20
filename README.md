# TRAM-MP: Multi-Person 3D Reconstruction from Monocular Video

A modular pipeline for reconstructing multiple people in 3D from regular monocular videos, extending single-person TRAM to multi-person scenarios with fast feed-forward geometry estimation.

## Project Overview

TRAM-MP reconstructs full 3D body shapes, poses, and metric-scale trajectories for multiple people from in-the-wild monocular videos. The system handles challenging scenarios including:
- Unconstrained camera motion (handheld, moving cameras)
- Multiple people with occlusions and interactions
- Real-world lighting and backgrounds
- No depth sensors or multi-view cameras required

**Key Innovation**: Combines VGGT's fast feed-forward geometry estimation with VIMO's temporal pose estimation, achieving 20-30x speedup over previous multi-person methods while maintaining or improving accuracy.

## Pipeline Architecture

The system consists of 5 modular stages:

```
Video Input
    �
[1] Camera Estimation (VGGT/DROID-SLAM)
    � Outputs: Camera poses, intrinsics, depth maps
    �
[2] Multi-Person Tracking (PHALP+)
    � Outputs: Track IDs, bounding boxes, masks
    �
[3] Per-Person Pose Estimation (VIMO)
    � Outputs: SMPL parameters in camera frame
    �
[4] World-Space Transformation
    � Outputs: SMPL parameters in shared world coordinates
    �
[5] Optional Refinement (SLAHMR-based)
    � Outputs: Refined SMPL with novel multi-person losses
    �
3D Humans in Metric Space
```

### Stage Details

| Stage | Component | Status |
|-------|-----------|--------|
| 1 | Camera Estimation (VGGT) | Tested
| 2 | Multi-Person Tracking | Tested
| 3 | Pose Estimation (VIMO) |  Tested | 
| 4 | World Transformation | Tested |
| 5 | Refinement (SLAHMR) | In Development |


## Current Development Status

### Completed & Tested
- **Stage 1:** Camera estimation with VGGT integration and DROID-SLAM fallback
- **Stage 2:** Multi-person detection and tracking with Kalman filtering and Hungarian matching
- **Stage 3:** Per-person pose estimation using VIMO transformer
- **Stage 4:** Coordinate transformation to shared world space
- **Experiment Scripts:** Complete SLURM job files and Python runners for HPC execution

### In Development (Not Yet Tested)
- **Stage 5:** Multi-person refinement with novel losses
  - Depth consistency loss (using VGGT depth maps)
  - Multi-person scale loss (aggregated HuMoR likelihood)
  - SLAHMR-based L-BFGS optimization

## Quick Start

### Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA
- GPU with 16GB+ memory (2 GPUs recommended for Stages 3 & 5)

### Installation

**1. Install Miniconda** (skip if already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Restart your shell or run: source ~/.bashrc
```

**2. Create the environment**

```bash
git clone https://github.com/your-username/multi-tram.git
cd multi-tram

conda env create -f environment.yml
conda activate multi-tram
```

**3. Initialize submodules**

```bash
git submodule update --init --recursive
```

**4. Download VGGT model checkpoint**

Stage 1 requires the VGGT-1B checkpoint at `data/pretrained_models/vggt/model.pt`. Download it from Hugging Face (you may need to accept the license and log in first):

```bash
# Log in if the model is gated
huggingface-cli login

mkdir -p data/pretrained_models/vggt

# Option A: via huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='facebook/VGGT-1B', filename='model.pt', local_dir='data/pretrained_models/vggt')
"

# Option B: direct download
wget -P data/pretrained_models/vggt/ "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
```

The checkpoint path can be changed in `configs/vggt.yaml` via `vggt_model_path`. If the checkpoint is not found, Stage 1 automatically falls back to DROID-SLAM.

**5. Download YOLOv8x weights**

Stage 1 requires `yolov8x.pt` in the project root. Download it once:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
# or manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

The path can be customized in `configs/vggt.yaml` via the `yolo_model_path` key.

### Running the Pipeline

**Option 1: Run individual stages (SLURM)**
```bash
sbatch experiments/stage1_camera.slurm
sbatch experiments/stage2_tracking.slurm
sbatch experiments/stage3_pose.slurm
sbatch experiments/stage4_world.slurm
# sbatch experiments/stage5_refinement.slurm  # Not yet tested
```

**Option 2: Run stages directly**
```bash
python experiments/run_stage1_camera.py --video data/my_video.mp4 --output_dir ./output/
```

## Project Structure

```
multi-tram/
|-- src/
|   |-- stages/              # 5 pipeline stages
|   |-- models/              # Model wrappers (VGGT, VIMO, PHALP, DROID)
|   |-- utils/               # Utilities (Kalman, Hungarian, SMPL)
|-- configs/                 # Configuration files
|-- experiments/             # SLURM job files and run scripts
|-- thirdparty/              # External dependencies (git submodules)
|   |-- vggt/                # Visual Geometry Grounded Transformer
|   |-- tram/                # VIMO model + single-person TRAM
|   |-- slahmr/              # Multi-person optimization
|-- docs/                    # Documentation and guides
```

## Performance



## Novel Contributions

1. **VGGT Integration:** First application of VGGT to multi-person 3D reconstruction
2. **Depth Consistency Loss:** Novel constraint using scene depth maps
3. **Multi-Person Scale Loss:** Enforces consistent human dimensions across people
4. **Scene-Centric Architecture:** Decouples scene geometry from human motion
5. **Feed-Forward Path:** Complete system works without slow optimization

## AI Assistance & Transparency

This project was developed with assistance from **Claude (Anthropic)** for:
- Code architecture and implementation planning
- Documentation generation (README, technical presentation, experiment guides)
- SLURM job script creation and HPC optimization
- Debugging and code review (integration with existing frameworks)

We believe in transparent AI collaboration - Claude served as a development assistant while all research decisions, experimental validation, and scientific contributions remain human-driven.

## Citation (future : unpublished)

If you use TRAM-MP in your research, please cite:

```bibtex
@article{tram-mp2025,
  title={TRAM-MP: Multi-Person 3D Reconstruction from Monocular Video},
  author={Divyanshu Singh},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

This project builds upon:
- **TRAM:** Single-person reconstruction and VIMO model
- **VGGT:** Visual Geometry Grounded Transformer for camera estimation
- **SLAHMR:** Multi-person optimization framework
- **PHALP:** Multi-person tracking system

## License



## Contact

- **Author:** Divyanshu Singh
- **Email:** diyanshu@tamu.edu
---

