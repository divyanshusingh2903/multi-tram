"""
Rendering and visualization utilities for multi-TRAM.

Currently stubs — full rendering requires pyrender or Open3D.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def render_smpl_overlay(
    frames: np.ndarray,
    smpl_params_list: List[Dict],
    camera_intrinsics: np.ndarray,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Render SMPL meshes overlaid on video frames.

    Args:
        frames:           (T, H, W, 3) uint8 RGB.
        smpl_params_list: List of per-person dicts (theta, beta, transl).
        camera_intrinsics:(T, 3, 3).
        output_path:      If provided, save rendered video to this path.

    Returns:
        rendered: (T, H, W, 3) uint8 frames with SMPL overlay.
    """
    # TODO: implement with pyrender
    return frames.copy()


def draw_tracks_on_frames(
    frames: np.ndarray,
    tracks_per_frame: List[List],
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and track IDs on video frames.

    Args:
        frames:           (T, H, W, 3) uint8 RGB.
        tracks_per_frame: List (T) of lists of Track objects with .bbox and .id.
        output_path:      If provided, save annotated video.

    Returns:
        annotated: (T, H, W, 3) uint8.
    """
    import cv2
    annotated = frames.copy()
    colors = {}

    for t, tracks in enumerate(tracks_per_frame):
        for track in tracks:
            tid = int(track.id) if hasattr(track, "id") else int(track.get("track_id", 0))
            if tid not in colors:
                rng = np.random.default_rng(tid)
                colors[tid] = tuple(int(c) for c in rng.integers(80, 255, 3))

            bbox = track.bbox if hasattr(track, "bbox") else track.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated[t], (x1, y1), (x2, y2), colors[tid], 2)
            cv2.putText(annotated[t], f"ID {tid}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[tid], 1)

    if output_path is not None:
        save_video(annotated, output_path)

    return annotated


def save_video(frames: np.ndarray, output_path: Path, fps: float = 30.0) -> None:
    """Save (T, H, W, 3) uint8 RGB array as an MP4."""
    import cv2
    output_path = Path(output_path)
    T, H, W, _ = frames.shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
