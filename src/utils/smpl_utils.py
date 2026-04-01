"""
SMPL forward-kinematics helpers used across stages.

Currently stubs — full implementation requires smplx or TRAM's SMPL layer.
Import via: from src.utils.smpl_utils import smpl_forward, get_joints
"""
import numpy as np
from typing import Dict, Optional


def smpl_forward(
    poses: np.ndarray,
    betas: np.ndarray,
    root_orient: np.ndarray,
    transl: np.ndarray,
    smpl_model=None,
) -> Dict[str, np.ndarray]:
    """
    Run SMPL forward kinematics.

    Args:
        poses:       (T, 23, 3) body joint rotations (axis-angle).
        betas:       (10,) shape parameters.
        root_orient: (T, 3) root orientation (axis-angle).
        transl:      (T, 3) root translation (metres).
        smpl_model:  Optional SMPL model instance (smplx.SMPL or TRAM variant).
                     If None, returns dummy output with correct shapes.

    Returns:
        dict with keys:
          joints   (T, 24, 3) — 3D joint positions
          vertices (T, 6890, 3) — mesh vertices
    """
    T = len(poses)

    if smpl_model is None:
        return {
            "joints": np.zeros((T, 24, 3), dtype=np.float32),
            "vertices": np.zeros((T, 6890, 3), dtype=np.float32),
        }

    # Full implementation: call smpl_model with batched inputs
    import torch
    output = smpl_model(
        betas=torch.from_numpy(betas).float().unsqueeze(0).expand(T, -1),
        global_orient=torch.from_numpy(root_orient).float(),
        body_pose=torch.from_numpy(poses.reshape(T, -1)).float(),
        transl=torch.from_numpy(transl).float(),
    )
    return {
        "joints": output.joints[:, :24].detach().cpu().numpy(),
        "vertices": output.vertices.detach().cpu().numpy(),
    }


def reproject_joints(
    joints_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Project 3D joints to 2D image coordinates.

    Args:
        joints_3d:  (T, J, 3) joints in camera frame (or world frame if
                    extrinsics are provided).
        intrinsics: (T, 3, 3) or (3, 3) camera intrinsics.
        extrinsics: Optional (T, 4, 4) world-to-camera transforms.

    Returns:
        keypoints_2d: (T, J, 3) — [u, v, 1] per joint.
    """
    T, J, _ = joints_3d.shape
    kpts = np.zeros((T, J, 3), dtype=np.float32)

    K = intrinsics if intrinsics.ndim == 3 else np.tile(intrinsics[None], (T, 1, 1))

    for t in range(T):
        pts = joints_3d[t]  # (J, 3)

        if extrinsics is not None:
            R = extrinsics[t, :3, :3]
            tv = extrinsics[t, :3, 3]
            pts = (R @ pts.T + tv[:, None]).T

        # Perspective projection
        z = pts[:, 2:3]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        uv = (K[t, :2, :2] @ (pts[:, :2] / z).T).T + K[t, :2, 2]
        kpts[t, :, :2] = uv
        kpts[t, :, 2] = 1.0

    return kpts
