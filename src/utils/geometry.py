"""
SE(3) utilities and ground-plane estimation for Stage 4.

Key functions:
- fit_ground_plane_ransac: RANSAC plane fit from VGGT point cloud
- align_to_yup: Rotate world frame so fitted plane normal → [0, 1, 0]
- axis_angle_to_matrix / matrix_to_axis_angle: rotation conversions
"""
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle (3,) to rotation matrix (3, 3).
    Uses Rodrigues' formula.
    """
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / angle
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [ axis[2],  0,       -axis[0]],
        [-axis[1],  axis[0],  0     ],
    ], dtype=np.float64)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (3, 3) to axis-angle (3,)."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64) / (2 * np.sin(angle))
    return (axis * angle).astype(np.float32)


# ---------------------------------------------------------------------------
# Ground plane fitting (spec §5.4)
# ---------------------------------------------------------------------------

def fit_ground_plane_ransac(
    points: np.ndarray,
    n_sample: int = 5000,
    inlier_threshold: float = 0.05,
    n_iters: int = 1000,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Fit a ground plane to a point cloud using RANSAC.

    Args:
        points:            (N, 3) world-frame points (merged across all frames).
        n_sample:          Number of points to subsample before RANSAC.
        inlier_threshold:  Inlier distance threshold in metres (spec: 0.05 m).
        n_iters:           Number of RANSAC iterations.
        seed:              Random seed.

    Returns:
        normal:  (3,) unit normal of the best-fit plane.
        offset:  Scalar d such that normal · x = d for points on the plane.
    """
    rng = np.random.default_rng(seed)

    if len(points) > n_sample:
        idx = rng.choice(len(points), n_sample, replace=False)
        pts = points[idx]
    else:
        pts = points

    best_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    best_offset = 0.0
    best_n_inliers = 0

    for _ in range(n_iters):
        # Sample 3 points
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        v0 = tri[1] - tri[0]
        v1 = tri[2] - tri[0]
        n = np.cross(v0, v1)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        d = float(n @ tri[0])

        # Count inliers
        dist = np.abs(pts @ n - d)
        n_inliers = int((dist < inlier_threshold).sum())

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_normal = n.astype(np.float32)
            best_offset = d

    # Ensure normal points upward
    if best_normal[1] < 0:
        best_normal = -best_normal
        best_offset = -best_offset

    return best_normal, best_offset


def align_to_yup(
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    """
    Build a 4×4 rigid transform that:
      1. Rotates world frame so `normal` aligns with [0, 1, 0].
      2. Translates so the plane passes through the origin.

    Apply this transform uniformly to all camera poses and person trajectories
    in Stage 4 to get a canonical Y-up world frame.

    Args:
        normal:  (3,) unit plane normal (pointing upward).
        offset:  Plane equation: normal · x = offset.

    Returns:
        T_align:  (4, 4) SE(3) alignment transform.
    """
    y_up = np.array([0.0, 1.0, 0.0])
    v = np.cross(normal, y_up)
    s = np.linalg.norm(v)
    c = float(np.dot(normal, y_up))

    if s < 1e-8:
        # Already aligned (or anti-aligned)
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, 1.0])
    else:
        v_x = np.array([
            [0,    -v[2],  v[1]],
            [ v[2],  0,   -v[0]],
            [-v[1],  v[0],  0  ],
        ])
        R = np.eye(3) + v_x + v_x @ v_x * ((1 - c) / (s ** 2))

    # Translation: move plane to y = 0
    t = np.array([0.0, -offset, 0.0])

    T_align = np.eye(4, dtype=np.float64)
    T_align[:3, :3] = R
    T_align[:3, 3] = t
    return T_align.astype(np.float32)
