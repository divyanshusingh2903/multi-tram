# Multi-TRAM Experiment Scripts

This directory contains standalone Python scripts and SLURM job files for running each stage of the Multi-TRAM pipeline on HPC clusters.

## Directory Structure

```
experiments/
├── README.md                      # This file
├── run_stage1_camera.py           # Python runner for Stage 1
├── run_stage2_tracking.py         # Python runner for Stage 2
├── run_stage3_pose.py             # Python runner for Stage 3
├── run_stage4_world.py            # Python runner for Stage 4
├── run_stage5_refinement.py       # Python runner for Stage 5
├── stage1_camera.slurm            # SLURM job for Stage 1
├── stage2_tracking.slurm          # SLURM job for Stage 2
├── stage3_pose.slurm              # SLURM job for Stage 3
├── stage4_world.slurm             # SLURM job for Stage 4
├── stage5_refinement.slurm        # SLURM job for Stage 5
└── run_all_stages.slurm           # Master SLURM job (runs all stages)
```

## Pipeline Overview

The Multi-TRAM pipeline consists of 5 stages:

1. **Stage 1: Camera Estimation** - VGGT/DROID-SLAM for camera poses and scene depth
2. **Stage 2: Multi-Person Tracking** - PHALP+ detection and tracking
3. **Stage 3: Pose Estimation** - VIMO for per-person SMPL parameters
4. **Stage 4: World Transformation** - Transform to shared world coordinates
5. **Stage 5: Refinement** (Optional) - SLAHMR-based optimization

**Dependencies between stages:**
- Stage 2 requires Stage 1 outputs (cameras.npz)
- Stage 3 requires Stage 1 + Stage 2 outputs
- Stage 4 requires Stage 1 + Stage 3 outputs
- Stage 5 requires all previous stage outputs

## Quick Start

### Option 1: Run Complete Pipeline

Submit the master job that runs all stages sequentially:

```bash
cd /scratch/user/divyanshu/research/multi-tram

# Edit run_all_stages.slurm to set VIDEO_PATH and OUTPUT_BASE_DIR
nano experiments/run_all_stages.slurm

# Submit job
sbatch experiments/run_all_stages.slurm
```

### Option 2: Run Individual Stages

Run stages separately for debugging or to resume from a specific stage:

```bash
# Stage 1: Camera Estimation (~30 min, 1 GPU)
sbatch experiments/stage1_camera.slurm

# Stage 2: Multi-Person Tracking (~1 hour, 1 GPU)
sbatch experiments/stage2_tracking.slurm

# Stage 3: Pose Estimation (~2 hours, 2 GPUs)
sbatch experiments/stage3_pose.slurm

# Stage 4: World Transformation (~15 min, 1 GPU)
sbatch experiments/stage4_world.slurm

# Stage 5: Refinement (~2 hours, 2 GPUs, OPTIONAL)
sbatch experiments/stage5_refinement.slurm
```

## Configuration

### Environment Setup

Before running, ensure your conda environment is set up:

```bash
# Create environment
conda env create -f environment.yml
conda activate multi-tram

# Or create manually
conda create -n multi-tram python=3.10
conda activate multi-tram
pip install torch torchvision numpy opencv-python scipy pyyaml
```

### Modifying SLURM Scripts

Each SLURM script has a **CONFIGURATION** section near the top. Edit these variables:

```bash
# In each stage*.slurm file:
VIDEO_PATH="./data/videos/example_video.mp4"      # Input video
OUTPUT_BASE_DIR="./results/example_video"         # Output directory
MAX_FRAMES=100                                     # Process first N frames (or "" for all)
```

**Important paths to update:**
- Line ~10: `TORCH_HOME`, `HF_HOME`, `XDG_CACHE_HOME` - Set to your cache directories
- Line ~15: `cd /scratch/user/divyanshu/research/multi-tram` - Set to your project path
- Line ~17: `conda activate multi-tram` - Set to your conda environment name

## Running Individual Python Scripts

You can also run the Python scripts directly (without SLURM) for testing:

```bash
# Stage 1
python experiments/run_stage1_camera.py \
    --video ./data/videos/example.mp4 \
    --output_dir ./results/example/1_camera_estimation \
    --config ./configs/vggt.yaml \
    --max_frames 100

# Stage 2
python experiments/run_stage2_tracking.py \
    --video ./data/videos/example.mp4 \
    --stage1_dir ./results/example/1_camera_estimation \
    --output_dir ./results/example/2_tracking \
    --max_frames 100

# Stage 3
python experiments/run_stage3_pose.py \
    --video ./data/videos/example.mp4 \
    --stage1_dir ./results/example/1_camera_estimation \
    --stage2_dir ./results/example/2_tracking \
    --output_dir ./results/example/3_pose_estimation \
    --max_frames 100 \
    --batch_size 8

# Stage 4
python experiments/run_stage4_world.py \
    --stage1_dir ./results/example/1_camera_estimation \
    --stage3_dir ./results/example/3_pose_estimation \
    --output_dir ./results/example/4_world_space

# Stage 5 (Optional)
python experiments/run_stage5_refinement.py \
    --video ./data/videos/example.mp4 \
    --stage1_dir ./results/example/1_camera_estimation \
    --stage2_dir ./results/example/2_tracking \
    --stage3_dir ./results/example/3_pose_estimation \
    --stage4_dir ./results/example/4_world_space \
    --output_dir ./results/example/5_refinement \
    --num_iterations 100
```

## Resource Requirements

### GPU Memory

| Stage | GPUs | Memory | Time (100 frames, 10 people) |
|-------|------|--------|-------------------------------|
| 1     | 1    | 16GB   | ~30 minutes                   |
| 2     | 1    | 32GB   | ~1 hour                       |
| 3     | 2    | 64GB   | ~2 hours                      |
| 4     | 1    | 16GB   | ~15 minutes                   |
| 5     | 2    | 64GB   | ~2 hours                      |

**Total (all stages):** ~6-7 hours on 2 GPUs

### Adjusting Resources

Edit the `#SBATCH` directives at the top of each `.slurm` file:

```bash
#SBATCH --time=2:00:00        # Increase if needed
#SBATCH --mem=32G             # Increase for more people/longer videos
#SBATCH --gres=gpu:rtx:2      # Change GPU count/type
```

## Advanced Usage

### Processing Specific People Only

To process only certain tracked people (useful for debugging):

```bash
# In stage3_pose.slurm or stage4_world.slurm:
PERSON_IDS="001 002 003"  # Process only persons 001, 002, 003
```

### Skipping Stages

To skip certain stages in `run_all_stages.slurm`:

```bash
SKIP_STAGES="5"  # Skip refinement (Stage 5)
# or
SKIP_STAGES="3 5"  # Skip multiple stages
```

### Fast Mode (No Refinement)

For fastest results, skip Stage 5:

```bash
# Edit run_all_stages.slurm:
SKIP_STAGES="5"
```

This reduces total time from ~7 hours to ~4 hours.

### Adjusting Optimization Parameters

Stage 5 refinement can be tuned:

```bash
# In stage5_refinement.slurm:
NUM_ITERATIONS=100      # L-BFGS iterations (default: 100)
LAMBDA_DEPTH=5.0        # Depth consistency weight (default: 5.0)
LAMBDA_SCALE=2.0        # Multi-person scale weight (default: 2.0)
```

## Monitoring Jobs

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific job
squeue -j <job_id>
```

### View Logs

Logs are saved to `logs/` directory:

```bash
# View output log
tail -f logs/stage1_camera_<job_id>.out

# View error log
tail -f logs/stage1_camera_<job_id>.err
```

### GPU Monitoring

Each script includes automatic GPU monitoring (updates every 5 minutes in the log).

## Output Structure

After running all stages, your output directory will look like:

```
results/example_video/
├── 1_camera_estimation/
│   ├── cameras.npz                 # SE(3) poses, intrinsics
│   ├── depth_maps/                 # Per-frame depth predictions
│   │   ├── depth_0000.npy
│   │   └── ...
│   └── point_cloud.ply             # 3D scene reconstruction
├── 2_tracking/
│   ├── tracks.npz                  # Track metadata (IDs, lifetimes)
│   ├── detections/                 # Per-frame detections
│   │   ├── frame_0000.pkl
│   │   └── ...
│   └── masks/                      # Segmentation masks
│       ├── person_001_frame_0000.png
│       └── ...
├── 3_pose_estimation/
│   ├── person_001/
│   │   ├── smpl_params_camera.npz  # θ, β, T (camera frame)
│   │   └── joints_camera.npy       # 3D joint positions
│   ├── person_002/
│   └── ...
├── 4_world_space/
│   ├── person_001/
│   │   ├── smpl_params_world.npz   # θ, β, T (world frame)
│   │   └── trajectory_world.npy    # Root trajectory
│   ├── person_002/
│   ├── ...
│   └── all_people_world.npz        # Combined scene
└── 5_refinement/                   # Optional
    ├── person_001/
    │   └── smpl_params_refined.npz # Optimized parameters
    ├── person_002/
    ├── ...
    └── optimization_metrics.json   # Loss history
```

## Troubleshooting

### Stage Fails with "Output not found"

**Cause:** Previous stage didn't complete or output path is wrong

**Solution:**
1. Check previous stage completed successfully
2. Verify `OUTPUT_BASE_DIR` is consistent across all stages
3. Check paths in SLURM script match your setup

### Out of Memory Errors

**Cause:** Video too long or too many people

**Solutions:**
1. Reduce `MAX_FRAMES` (process fewer frames)
2. Increase `#SBATCH --mem` in SLURM script
3. Reduce `--batch_size` in Stage 3
4. Process specific `PERSON_IDS` instead of all people

### CUDA Out of Memory

**Solutions:**
1. Request more/larger GPUs in `#SBATCH --gres`
2. Reduce batch size in Stage 3
3. Process people sequentially (set `PERSON_IDS`)

### Module/Import Errors

**Cause:** Conda environment not activated or packages missing

**Solutions:**
1. Verify conda environment name in SLURM script
2. Check `conda list` shows required packages
3. Reinstall environment: `conda env create -f environment.yml`

### Slow Camera Estimation (Stage 1)

**Cause:** VGGT might be slower than expected on some GPUs

**Solution:** Use DROID-SLAM fallback:
```bash
# In stage1_camera.slurm:
--method droid
```

## Example Workflow

Complete example for processing a new video:

```bash
# 1. Prepare your video
mkdir -p data/videos
cp /path/to/your/video.mp4 data/videos/my_video.mp4

# 2. Edit configuration in run_all_stages.slurm
nano experiments/run_all_stages.slurm
# Change:
#   VIDEO_PATH="./data/videos/my_video.mp4"
#   OUTPUT_BASE_DIR="./results/my_video"

# 3. Submit job
sbatch experiments/run_all_stages.slurm

# 4. Monitor progress
tail -f logs/multi_tram_full_*.out

# 5. Check results
ls -lh results/my_video/
```

## Citation

If you use Multi-TRAM in your research, please cite:

```bibtex
@article{multitram2024,
  title={Multi-TRAM: Multi-Person 3D Reconstruction from Monocular Video},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Contact

For issues or questions:
- GitHub Issues: [your-repo-url]
- Email: your.email@institution.edu
