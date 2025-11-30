"""
Stage implementations for multi-TRAM pipeline

Stages:
1. Scene Understanding & Camera Estimation (VGGT + optional DROID-SLAM fallback)
2. Multi-Person Detection, Segmentation & Tracking
3. Per-Person 3D Pose & Shape Estimation (VIMO)
4. World-Space Transformation
5. (Optional) Multi-Person Refinement via SLAHMR Optimization
6. Post-Processing & Quality Metrics
7. Visualization & Export
"""

from .stage1_camera import (
    Stage1Pipeline,
    Stage1Config,
    VGGTModel,
    create_stage1_pipeline,
)

__all__ = [
    "Stage1Pipeline",
    "Stage1Config",
    "VGGTModel",
    "create_stage1_pipeline",
]
