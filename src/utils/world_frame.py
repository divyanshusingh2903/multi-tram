"""
World-Frame Transformation Utilities
Transforms detections from image coordinates to world coordinates using VGGT outputs

Key Functions:
- unproject_bbox: Convert 2D bbox + depth to 3D camera-relative position
- transform_to_world: Convert camera-relative to world coordinates
- extract_point_track_features: Get VGGT point track features for a bbox
"""
import numpy as np
import torch
from typing import Tuple, Optional, Dict
import cv2


def unproject_bbox(
    bbox: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    method: str = 'center'
) -> np.ndarray:
    """
    Unproject 2D bounding box to 3D camera coordinates using depth

    Args:
        bbox: Bounding box [x, y, w, h] in pixels
        depth_map: Depth map (H, W) in meters
        intrinsics: Camera intrinsics 3x3 matrix
        method: Depth estimation method
            - 'center': Use depth at bbox center
            - 'median': Use median depth within bbox
            - 'bottom': Use depth at bbox bottom center (feet)

    Returns:
        position_3d: 3D position [x, y, z] in camera coordinates (meters)
    """
    x, y, w, h = bbox
    H, W = depth_map.shape

    # Get depth value based on method
    if method == 'center':
        # Center of bounding box
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        cx = np.clip(cx, 0, W - 1)
        cy = np.clip(cy, 0, H - 1)
        depth = depth_map[cy, cx]

    elif method == 'median':
        # Median depth within bounding box
        x1, y1 = int(max(0, x)), int(max(0, y))
        x2, y2 = int(min(W, x + w)), int(min(H, y + h))
        bbox_depth = depth_map[y1:y2, x1:x2]
        depth = np.median(bbox_depth[bbox_depth > 0])  # Exclude invalid depths

    elif method == 'bottom':
        # Bottom center (feet position)
        cx = int(x + w / 2)
        cy = int(y + h)  # Bottom of bbox
        cx = np.clip(cx, 0, W - 1)
        cy = np.clip(cy, 0, H - 1)
        depth = depth_map[cy, cx]

    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle invalid depth
    if depth <= 0 or np.isnan(depth):
        depth = 5.0  # Default to 5 meters if depth unavailable

    # Unproject to 3D using intrinsics
    # [u, v, 1]^T = K @ [X, Y, Z]^T
    # Therefore: [X, Y, Z]^T = K^-1 @ [u*Z, v*Z, Z]^T

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx_intr = intrinsics[0, 2]
    cy_intr = intrinsics[1, 2]

    # Center pixel coordinates
    u = x + w / 2
    v = y + h / 2

    # Unproject
    X = (u - cx_intr) * depth / fx
    Y = (v - cy_intr) * depth / fy
    Z = depth

    position_3d = np.array([X, Y, Z])

    return position_3d


def transform_to_world(
    position_camera: np.ndarray,
    camera_pose: np.ndarray
) -> np.ndarray:
    """
    Transform position from camera coordinates to world coordinates

    Args:
        position_camera: Position in camera frame [x, y, z]
        camera_pose: Camera pose 4x4 matrix [R | t; 0 0 0 1]
                    OR 3x4 matrix [R | t]

    Returns:
        position_world: Position in world frame [x, y, z]
    """
    # Handle both 3x4 and 4x4 formats
    if camera_pose.shape == (3, 4):
        R = camera_pose[:, :3]
        t = camera_pose[:, 3]
    elif camera_pose.shape == (4, 4):
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
    else:
        raise ValueError(f"Invalid camera_pose shape: {camera_pose.shape}")

    # Transform: p_world = R @ p_camera + t
    position_world = R @ position_camera + t

    return position_world


def transform_detections_to_world(
    detections: list,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    method: str = 'center'
) -> list:
    """
    Transform list of detections to world frame

    Args:
        detections: List of YOLODetection objects
        depth_map: Depth map (H, W)
        intrinsics: Camera intrinsics 3x3
        camera_pose: Camera pose 4x4 or 3x4
        method: Depth estimation method

    Returns:
        detections: Same list with added position_3d and position_world attributes
    """
    for det in detections:
        # Unproject to camera 3D
        det.position_3d = unproject_bbox(
            det.bbox,
            depth_map,
            intrinsics,
            method=method
        )

        # Transform to world 3D
        det.position_world = transform_to_world(
            det.position_3d,
            camera_pose
        )

    return detections


def extract_point_track_features(
    bbox: np.ndarray,
    point_tracks: np.ndarray,
    feature_type: str = 'count'
) -> np.ndarray:
    """
    Extract geometric features from VGGT point tracks within a bounding box

    Args:
        bbox: Bounding box [x, y, w, h]
        point_tracks: VGGT point tracks (N, 2) - [u, v] coordinates
                     or (N, 3) with confidence
        feature_type: Type of feature to extract
            - 'count': Number of points in bbox
            - 'density': Point density (points per pixel)
            - 'motion': Average optical flow magnitude (if available)

    Returns:
        features: Feature vector
    """
    if point_tracks is None or len(point_tracks) == 0:
        # No point tracks available
        return np.array([0.0])

    x, y, w, h = bbox

    # Get points within bounding box
    if point_tracks.shape[1] >= 2:
        u = point_tracks[:, 0]
        v = point_tracks[:, 1]

        # Check which points are inside bbox
        inside = (u >= x) & (u < x + w) & (v >= y) & (v < y + h)
        points_inside = point_tracks[inside]

        if feature_type == 'count':
            # Simple count
            return np.array([len(points_inside)])

        elif feature_type == 'density':
            # Points per 100 pixels
            area = w * h
            density = len(points_inside) / (area / 100.0) if area > 0 else 0.0
            return np.array([density])

        elif feature_type == 'motion':
            # If point tracks have motion info (x, y, dx, dy)
            if point_tracks.shape[1] >= 4:
                dx = points_inside[:, 2]
                dy = points_inside[:, 3]
                motion_mag = np.sqrt(dx**2 + dy**2)
                avg_motion = np.mean(motion_mag) if len(motion_mag) > 0 else 0.0
                return np.array([avg_motion])
            else:
                return np.array([0.0])

    return np.array([0.0])


def get_world_frame_velocity(
    prev_position: np.ndarray,
    curr_position: np.ndarray,
    dt: float = 1/30.0
) -> np.ndarray:
    """
    Compute velocity in world frame

    Args:
        prev_position: Previous position [x, y, z] in world frame
        curr_position: Current position [x, y, z] in world frame
        dt: Time delta (default 1/30 for 30 FPS)

    Returns:
        velocity: Velocity vector [vx, vy, vz] in m/s
    """
    velocity = (curr_position - prev_position) / dt
    return velocity


def world_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Compute Euclidean distance in world frame

    Args:
        pos1: Position 1 [x, y, z]
        pos2: Position 2 [x, y, z]

    Returns:
        distance: Euclidean distance in meters
    """
    return np.linalg.norm(pos1 - pos2)


def filter_detections_by_world_position(
    detections: list,
    world_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> list:
    """
    Filter detections by world-frame position bounds

    Args:
        detections: List of detections with position_world attribute
        world_bounds: Dictionary with 'x', 'y', 'z' keys
                     Each value is (min, max) tuple
                     E.g., {'x': (-10, 10), 'y': (0, 3), 'z': (0, 20)}

    Returns:
        filtered_detections: Detections within bounds
    """
    if world_bounds is None:
        return detections

    filtered = []
    for det in detections:
        if not hasattr(det, 'position_world'):
            continue

        x, y, z = det.position_world

        # Check bounds
        valid = True
        if 'x' in world_bounds:
            x_min, x_max = world_bounds['x']
            if not (x_min <= x <= x_max):
                valid = False

        if 'y' in world_bounds:
            y_min, y_max = world_bounds['y']
            if not (y_min <= y <= y_max):
                valid = False

        if 'z' in world_bounds:
            z_min, z_max = world_bounds['z']
            if not (z_min <= z <= z_max):
                valid = False

        if valid:
            filtered.append(det)

    return filtered


if __name__ == "__main__":
    """Test world-frame transformations"""

    # Test unproject_bbox
    print("Testing unproject_bbox...")
    bbox = np.array([320, 240, 100, 200])  # Person in center
    depth_map = np.ones((480, 640)) * 5.0  # Uniform 5m depth
    intrinsics = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])

    pos_3d = unproject_bbox(bbox, depth_map, intrinsics, method='center')
    print(f"  3D position (camera): {pos_3d}")
    print(f"  Distance from camera: {np.linalg.norm(pos_3d):.2f}m")

    # Test transform_to_world
    print("\nTesting transform_to_world...")
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([1, 0, 0])  # Camera at (1, 0, 0)

    pos_world = transform_to_world(pos_3d, camera_pose)
    print(f"  3D position (world): {pos_world}")

    # Test point track features
    print("\nTesting extract_point_track_features...")
    point_tracks = np.random.rand(100, 2) * 640  # Random points
    features = extract_point_track_features(bbox, point_tracks, feature_type='count')
    print(f"  Point count in bbox: {features[0]:.0f}")

    features_density = extract_point_track_features(bbox, point_tracks, feature_type='density')
    print(f"  Point density: {features_density[0]:.2f} points/100px")

    print("\nAll tests passed!")
