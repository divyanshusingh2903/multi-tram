"""
Stage implementations for multi-TRAM pipeline.

Stage 1: Camera estimation (VGGT primary, DROID-SLAM fallback)
Stage 2: Multi-person detection & tracking (PHALP+ + world-frame correction)
Stage 3: Per-person pose estimation (VIMO)
Stage 4: World-frame transformation
Stage 5: Optional multi-person refinement (SLAHMR)
"""

from .stage1_camera import run_camera_estimation, load_video, save_results
from .stage2_tracking import run_tracking, Track

__all__ = [
    "run_camera_estimation",
    "load_video",
    "save_results",
    "run_tracking",
    "Track",
]
