# Multi-TRAM Implementation Guide

This document describes the implementation of Stages 2, 3, and 4 of the Multi-TRAM pipeline for multi-person 3D human reconstruction from video.

## Overview

The Multi-TRAM pipeline consists of 7 stages, with this implementation covering stages 2-4:

- **Stage 1**: Camera Estimation (VGGT) - Already implemented
- **Stage 2**: Multi-Person Detection, Segmentation & Tracking (NEW)
- **Stage 3**: Per-Person 3D Pose & Shape Estimation (NEW)
- **Stage 4**: World-Space Transformation (NEW)
- **Stage 5**: Optional refinement via SLAHMR
- **Stage 6**: Post-processing & Quality Metrics
- **Stage 7**: Visualization & Export

## Implementation Details

### Stage 2: Multi-Person Detection, Segmentation & Tracking

**File**: `src/stages/stage2_tracking.py`

#### Key Components:

1. **TrackingManager**: Manages track IDs across frames
   - Associates detections with existing tracks using IoU-based similarity
   - Assigns new IDs to unmatched detections
   - Supports spatial association using bounding boxes

2. **PersonSegmenter**: Handles person segmentation
   - Supports simple bounding box segmentation
   - Fallback for SAM (Segment Anything Model) integration
   - Can be enhanced with actual SAM implementation

3. **PHALP_Tracker**: Wrapper around PHALP+ detector/tracker
   - Placeholder for actual PHALP+ model
   - Can be integrated with real PHALP+ weights
   - Includes 2D keypoint extraction

#### Data Structures:

```python
@dataclass
class Detection:
    frame_id: int
    track_id: Optional[int]
    bbox: np.ndarray  # [x, y, w, h]
    mask: Optional[np.ndarray]  # Binary mask
    keypoints_2d: np.ndarray  # (K, 3) with confidence
    confidence: float
    visibility: float

@dataclass
class Track:
    track_id: int
    detections: List[Detection]
    start_frame: int
    end_frame: int
```

#### Output:
- `detections/frame_XXXXXX.pkl`: Per-frame detection results
- `masks/frame_XXXXXX_person_XXX.npy`: Segmentation masks
- `metadata.json`: Pipeline metadata

### Stage 3: Per-Person 3D Pose & Shape Estimation (VIMO)

**File**: `src/stages/stage3_pose.py`

#### Key Components:

1. **VIMOPredictor**: Video Transformer for pose estimation
   - Frozen ViT-Huge backbone for feature extraction
   - Temporal transformer for propagating appearance cues
   - Motion temporal transformer for smooth pose sequences
   - Outputs SMPL parameters in camera frame

2. **PoseEstimator**: High-level manager for per-person pose estimation
   - Processes each tracked person independently
   - Handles frame extraction and preprocessing
   - Manages input/output for VIMO

#### Data Structures:

```python
@dataclass
class SMPLOutput:
    frame_id: int
    poses: np.ndarray  # (23, 3) body pose in axis-angle
    betas: np.ndarray  # (10,) body shape
    global_orient: np.ndarray  # (3,) root orientation
    transl: np.ndarray  # (3,) root translation in camera frame
    joints: Optional[np.ndarray]  # (24, 3) 3D joint positions
    vertices: Optional[np.ndarray]  # (6890, 3) mesh vertices
    keypoints_2d: Optional[np.ndarray]  # (24, 3) 2D projections
    confidence: float

@dataclass
class PersonPose:
    track_id: int
    start_frame: int
    end_frame: int
    frames: List[int]
    poses: List[SMPLOutput]
```

#### Processing Pipeline:
1. For each tracked person:
   - Extract frames where person is visible
   - Crop to bounding box with padding
   - Apply mask to isolate person
   - Resize to 256×256 maintaining aspect ratio
2. Feed sequence to VIMO
3. Get SMPL parameters per frame

#### Output:
```
person_XXX/
  ├── smpl_params_camera.npz  # SMPL in camera frame
  │   ├── poses: (T, 23, 3)
  │   ├── betas: (10,)
  │   ├── global_orient: (T, 3)
  │   └── transl: (T, 3)
  ├── joints_camera.npy  # (T, 24, 3)
  ├── vertices_camera.npy  # (T, 6890, 3)
  ├── keypoints_2d.npy  # (T, 24, 3)
  └── metadata.json
```

### Stage 4: World-Space Transformation

**File**: `src/stages/stage4_world_transform.py`

#### Key Components:

1. **WorldTransformer**: Handles coordinate frame transformations
   - Manages camera poses from Stage 1
   - Transforms person positions from camera to world frame
   - Ensures gravity alignment

2. **Transformation Mathematics**:
   ```
   For each person i and timestep t:

   r_W(i,t) = R_t^(-1) · r_C(i,t)
   π_W(i,t) = R_t^(-1) · (π_C(i,t) - T_t)
   θ_W(i,t) = θ_C(i,t)  (unchanged)
   β_W(i) = β_C(i)      (unchanged)
   ```

   Where:
   - R_t: Camera rotation matrix at frame t
   - T_t: Camera translation at frame t
   - r_C: Root orientation in camera frame
   - π_C: Root translation in camera frame

#### Coordinate Frame Convention:
- **World origin**: First camera position (t=0)
- **World axes**: Gravity-aligned (Y-up, ground plane XZ)
- **Metric scale**: From VGGT or TRAM's scale estimation
- **All people share this common reference frame**

#### Output:
```
person_XXX/
  ├── smpl_params_world.npz  # SMPL in world coordinates
  │   ├── poses: (T, 23, 3)
  │   ├── betas: (10,)
  │   ├── global_orient: (T, 3)
  │   └── transl: (T, 3)
  ├── trajectory_world.npy  # (T, 3) root positions
  ├── joints_world.npy  # (T, 24, 3) joints in world
  └── metadata.json

all_people_world.npz  # Combined multi-person data
metadata.json
```

## Pipeline Architecture

### Main Pipeline Script

**File**: `pipeline.py`

The `MultiTRAMPipeline` class orchestrates all stages:

```python
pipeline = MultiTRAMPipeline(
    config_path="configs/vggt.yaml",
    output_base_dir="results/my_video"
)

results = pipeline.run_full_pipeline(
    video_path="input.mp4",
    max_frames=None,  # None = process all frames
    skip_stages=[5]   # Skip optional refinement
)
```

### Output Directory Structure

```
results/my_video/
├── 1_camera_estimation/
│   ├── cameras.npz
│   ├── depth_maps/
│   ├── point_cloud.ply
│   └── metadata.json
├── 2_tracking/
│   ├── detections/
│   ├── masks/
│   └── metadata.json
├── 3_pose_estimation/
│   ├── person_001/
│   ├── person_002/
│   └── metadata.json
├── 4_world_space/
│   ├── person_001/
│   ├── person_002/
│   ├── all_people_world.npz
│   └── metadata.json
└── pipeline_summary.json
```

## Usage

### Running the Full Pipeline

```bash
python pipeline.py input_video.mp4 \
    --output results \
    --config configs/vggt.yaml \
    --stages 1 2 3 4
```

### Running Individual Stages

```bash
# Stage 2 only
python pipeline.py input_video.mp4 --stages 2

# Stages 2 and 3 (skip camera estimation and world transform)
python pipeline.py input_video.mp4 --stages 2 3

# Skip optional refinement
python pipeline.py input_video.mp4 --skip_stages 5
```

### Testing Stages Independently

```bash
# Test Stage 2
python src/stages/stage2_tracking.py \
    --video input.mp4 \
    --output results/stage2

# Test Stage 3
python src/stages/stage3_pose.py \
    --video input.mp4 \
    --tracking_results tracking_results.pkl \
    --output results/stage3

# Test Stage 4
python src/stages/stage4_world_transform.py \
    --camera_poses results/stage1/cameras.npz \
    --stage3_dir results/stage3 \
    --output results/stage4
```

## Configuration

The pipeline is configured via YAML files (e.g., `configs/vggt.yaml`):

```yaml
# Stage 1: Camera Estimation
stage1:
  method: 'vggt'  # or 'droid'
  device: 'cuda'
  max_frames: 100
  image_size: 512
  fallback_to_droid: true

# Stage 2: Tracking
stage2:
  device: 'cuda'
  max_age: 30
  similarity_threshold: 0.5

# Stage 3: Pose Estimation
stage3:
  device: 'cuda'
  num_frames_per_batch: 16
  vimo_model_path: 'models/vimo_pretrained.pth'

# Stage 4: World Transform
stage4:
  up_axis: 1  # Y-up
```

## Key Features

### Stage 2: Tracking
- ✓ Multi-person detection and tracking
- ✓ Unique ID assignment across frames
- ✓ Simple IoU-based association
- ✓ Segmentation mask extraction
- ⭐ Can be enhanced with: Real PHALP+ model, SAM integration, depth-guided tracking

### Stage 3: Pose Estimation
- ✓ Per-person SMPL parameter prediction
- ✓ Independent processing per tracked person
- ✓ 256×256 crop preprocessing
- ✓ Temporal coherence via transformer
- ⭐ Can be enhanced with: Actual VIMO weights, ground truth comparison, optimization

### Stage 4: World Transform
- ✓ Camera-to-world coordinate transformation
- ✓ Metric-scale handling
- ✓ Per-person trajectory extraction
- ✓ Multi-person spatial composition
- ✓ Gravity-aligned output
- ⭐ Can be enhanced with: Ground plane fitting, interaction graphs, relative distance metrics

## Integration Points for External Models

### PHALP+ Integration (Stage 2)
```python
from slahmr.models.phalp.phalp import PHALP

# In PHALP_Tracker._init_model()
self.model = PHALP(model_path=model_path, device=device)

# In PHALP_Tracker._detect_with_phalp()
detections = self.model(frame)
```

### VIMO Integration (Stage 3)
```python
# Load pretrained VIMO model
model = torch.load(vimo_model_path)

# Prepare input: (B, T, 3, 256, 256)
with torch.no_grad():
    smpl_params = model(frames)
```

### SAM Integration (Stage 2)
```python
from segment_anything import sam_model_registry

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
masks = sam(image=frame, input_box=bbox)
```

## Performance Expectations

Based on the documentation:

| Component | Time | Notes |
|-----------|------|-------|
| Stage 1 (VGGT) | ~0.5s | For 100 frames |
| Stage 2 (PHALP+) | ~3s | Detection + tracking |
| Stage 3 (VIMO×N) | ~8s | For 10 people, 100 frames |
| Stage 4 (Transform) | <1s | Coordinate transformation |
| **Total** | **~11.5s** | Feed-forward path |

## Known Limitations & Future Work

1. **PHALP+ Model**: Currently a placeholder. Integration requires:
   - Download pretrained weights
   - Implement proper PHALP forward pass
   - Add 2D keypoint extraction

2. **VIMO Model**: Dummy predictions for now. Requires:
   - Actual model weights
   - Proper ViT-Huge backbone
   - Temporal transformer implementation

3. **Segmentation**: Currently uses bounding box. Can improve with:
   - SAM integration
   - Better mask refinement
   - Erosion/dilation for clean boundaries

4. **Tracking Association**: Simple IoU-based. Can enhance with:
   - Feature-based matching (VGGT tracking features)
   - Kalman filtering
   - Hungarian algorithm for optimal assignment

5. **World Transformation**: Basic camera-to-world. Can add:
   - Ground plane fitting
   - Gravity alignment validation
   - Multi-person consistency constraints

## References

- **TRAM**: Scene-centric 3D human motion reconstruction
- **SLAHMR**: SLAM + multi-person human motion recovery
- **PHALP+**: Parametric model of human body in video
- **VGGT**: Vision Transformer for geometric scene understanding
- **VIMO**: Video transformer for temporally coherent pose estimation

## Citation

If you use this implementation, please cite the original papers:

```bibtex
@inproceedings{weng2023tram,
  title={TRAM: 3D Human Body Mesh Recovery with Implicit Video Volume Rendering},
  author={Weng, Chung-Yi and ...},
  booktitle={ICCV},
  year={2023}
}
```

## Support

For issues, questions, or contributions, please refer to the project repository.
