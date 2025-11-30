"""
DROID-SLAM Wrapper for Camera Estimation (Fallback)
Used when VGGT fails or for comparison

Usage Examples:
    # Load with default weights
    droid = DROIDSLAMWrapper()

    # Load with custom weights
    droid = DROIDSLAMWrapper(weights_path='path/to/droid.pth')

    # Process video frames (T, H, W, 3) in [0, 255]
    results = droid.estimate_cameras(frames)

    # Results dict contains:
    # - poses: (T, 4, 4) camera poses
    # - intrinsics: (T, 3, 3) camera intrinsics
    # - depths: (T, H, W) sparse depth maps
    # - point_cloud: (N, 3) 3D points
    # - point_colors: (N, 3) RGB colors
"""
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Tuple
import cv2
import torch

class DROIDSLAMWrapper:
    """Wrapper for DROID-SLAM as fallback camera estimator"""

    def __init__(self,
                 weights_path: str = None,
                 device: str = 'cuda',
                 buffer_size: int = 512):
        """
        Initialize DROID-SLAM wrapper

        Args:
            weights_path: Path to DROID-SLAM weights (.pth file).
                         If None, searches for weights in TRAM directory.
            device: Device to run on ('cuda' or 'cpu')
            buffer_size: SLAM buffer size for keyframe tracking
        """
        self.device = device
        self.buffer_size = buffer_size
        self.slam = None

        # Add DROID-SLAM to path
        tram_path = Path(__file__).parent.parent.parent / 'thirdparty' / 'tram'
        droid_path = tram_path / 'thirdparty' / 'DROID-SLAM' / 'droid_slam'
        sys.path.insert(0, str(droid_path))

        try:
            # Try to import DROID from TRAM
            try:
                from droid import Droid
                print("[DROID-SLAM] Successfully imported DROID from TRAM")
            except ImportError:
                # Fall back to looking for droid_slam module
                droid_slam_path = tram_path / 'thirdparty' / 'DROID-SLAM'
                sys.path.insert(0, str(droid_slam_path))
                from droid_slam.droid import Droid
                print("[DROID-SLAM] Successfully imported DROID from DROID-SLAM directory")

            print("[DROID-SLAM] Initializing SLAM system")

            # Determine weights path
            if weights_path is None:
                # Search for weights in common locations
                possible_paths = [
                    droid_path / 'droid.pth',
                    tram_path / 'thirdparty' / 'DROID-SLAM' / 'droid.pth',
                    Path.home() / '.cache' / 'droid' / 'droid.pth',
                ]

                for p in possible_paths:
                    if p.exists():
                        weights_path = str(p)
                        print(f"[DROID-SLAM] Found weights at: {weights_path}")
                        break

                if weights_path is None:
                    raise FileNotFoundError(
                        f"Could not find DROID-SLAM weights. Searched in: {[str(p) for p in possible_paths]}"
                    )
            else:
                # Verify provided path exists
                if not Path(weights_path).exists():
                    raise FileNotFoundError(f"Weights not found at: {weights_path}")
                print(f"[DROID-SLAM] Loading weights from: {weights_path}")

            # Initialize DROID-SLAM
            self.slam = Droid(weights_path, device=device, buffer=buffer_size)
            print("[DROID-SLAM] Initialized successfully")

        except Exception as e:
            print(f"[DROID-SLAM] Error initializing: {e}")
            print(f"[DROID-SLAM] Make sure TRAM/DROID-SLAM is properly installed in thirdparty/")
            raise

    def estimate_cameras(self, frames: np.ndarray) -> Dict:
        """
        Estimate camera poses using DROID-SLAM

        Args:
            frames: numpy array of shape (T, H, W, 3) in [0, 255]

        Returns:
            Dictionary containing:
                - 'poses': Camera poses (T, 4, 4) - camera-to-world transforms
                - 'intrinsics': Camera intrinsics (T, 3, 3)
                - 'depths': Estimated depth maps (T, H, W) - sparse
                - 'point_cloud': Sparse 3D points (N, 3)
                - 'point_colors': Point colors (N, 3)
        """
        if self.slam is None:
            raise RuntimeError("[DROID-SLAM] SLAM system not initialized")

        print(f"[DROID-SLAM] Processing {len(frames)} frames")

        T, H, W, C = frames.shape

        try:
            # Reset SLAM state
            self.slam.video.counter.value = 0

            # Feed frames to SLAM
            for t in range(T):
                frame = frames[t]

                # Convert to grayscale for SLAM
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame

                # Add frame to SLAM
                self.slam.track(t, gray)

                if (t + 1) % max(1, T // 10) == 0:
                    print(f"[DROID-SLAM] Processed {t+1}/{T} frames")

            # Finalize and extract results
            print("[DROID-SLAM] Finalizing reconstruction...")
            self.slam.terminate()

            # Get camera poses
            if hasattr(self.slam, 'poses') and self.slam.poses is not None:
                poses = self.slam.poses.detach().cpu().numpy()  # (T, 7) - quaternion + translation
                poses_matrix = self._se3_to_matrix(poses)  # Convert to (T, 4, 4)
            else:
                # Fallback to identity matrices if poses not available
                poses_matrix = np.tile(np.eye(4)[np.newaxis, :, :], (T, 1, 1))
                print("[DROID-SLAM] Warning: Could not extract poses, using identity matrices")

            # Get intrinsics (DROID uses fixed intrinsics)
            fx = W * 0.8  # Approximate focal length based on width
            fy = W * 0.8
            cx = W / 2.0
            cy = H / 2.0
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            intrinsics = np.stack([intrinsics] * T, axis=0)

            # Get 3D points (sparse point cloud from SLAM)
            if hasattr(self.slam, 'points') and self.slam.points is not None:
                points_3d = self.slam.points.detach().cpu().numpy()  # (N, 3)
                print(f"[DROID-SLAM] Extracted {len(points_3d)} 3D points")
            else:
                points_3d = np.zeros((0, 3), dtype=np.float32)
                print("[DROID-SLAM] Warning: No 3D points extracted")

            # Get point colors
            if len(points_3d) > 0:
                point_colors = self._get_point_colors(frames, points_3d, poses_matrix, intrinsics)
            else:
                point_colors = np.zeros((0, 3), dtype=np.uint8)

            # Create sparse depth maps from 3D points
            if len(points_3d) > 0:
                depths = self._estimate_depths_from_points(
                    points_3d, poses_matrix, intrinsics, (H, W)
                )
            else:
                depths = np.zeros((T, H, W), dtype=np.float32)

            results = {
                'poses': poses_matrix,
                'intrinsics': intrinsics,
                'depths': depths,
                'point_cloud': points_3d,
                'point_colors': point_colors,
                'original_size': (H, W)
            }

            print(f"[DROID-SLAM] Camera estimation complete")
            print(f"  - Poses: {poses_matrix.shape}")
            print(f"  - Point cloud: {points_3d.shape}")

            return results

        except Exception as e:
            print(f"[DROID-SLAM] Error during tracking: {e}")
            raise

    def _se3_to_matrix(self, poses: np.ndarray) -> np.ndarray:
        """
        Convert SE(3) representation to 4x4 matrices

        Args:
            poses: (T, 7) - [qw, qx, qy, qz, tx, ty, tz] or
                   (T, 4, 4) if already in matrix form

        Returns:
            poses_matrix: (T, 4, 4) transformation matrices
        """
        # If already in matrix form, return as is
        if poses.ndim == 3 and poses.shape[1:] == (4, 4):
            return poses

        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            print("[DROID-SLAM] Warning: scipy not available, using quaternion fallback")
            return self._se3_to_matrix_fallback(poses)

        T = len(poses)
        poses_matrix = np.zeros((T, 4, 4), dtype=np.float32)

        for t in range(T):
            # Extract quaternion and translation
            quat = poses[t, :4]  # [qw, qx, qy, qz]
            trans = poses[t, 4:]  # [tx, ty, tz]

            # Normalize quaternion
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 0:
                quat = quat / quat_norm

            # Convert quaternion to rotation matrix
            # scipy expects [x, y, z, w], but poses are [w, x, y, z]
            try:
                rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            except Exception as e:
                print(f"[DROID-SLAM] Warning: Could not convert quaternion {quat}: {e}")
                rot = np.eye(3)

            # Build 4x4 matrix
            poses_matrix[t, :3, :3] = rot
            poses_matrix[t, :3, 3] = trans
            poses_matrix[t, 3, 3] = 1.0

        return poses_matrix

    def _se3_to_matrix_fallback(self, poses: np.ndarray) -> np.ndarray:
        """
        Fallback conversion without scipy (basic quaternion to matrix)

        Args:
            poses: (T, 7) - [qw, qx, qy, qz, tx, ty, tz]

        Returns:
            poses_matrix: (T, 4, 4) transformation matrices
        """
        T = len(poses)
        poses_matrix = np.zeros((T, 4, 4), dtype=np.float32)

        for t in range(T):
            # Extract quaternion and translation
            qw, qx, qy, qz = poses[t, :4]
            tx, ty, tz = poses[t, 4:]

            # Quaternion to rotation matrix (basic implementation)
            # Normalize quaternion
            q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if q_norm > 0:
                qw, qx, qy, qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm

            # Build rotation matrix from quaternion
            rot = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
            ], dtype=np.float32)

            # Build 4x4 matrix
            poses_matrix[t, :3, :3] = rot
            poses_matrix[t, :3, 3] = [tx, ty, tz]
            poses_matrix[t, 3, 3] = 1.0

        return poses_matrix

    def _get_point_colors(self,
                          frames: np.ndarray,
                          points_3d: np.ndarray,
                          poses: np.ndarray,
                          intrinsics: np.ndarray) -> np.ndarray:
        """
        Get colors for 3D points by projecting to frames

        Args:
            frames: (T, H, W, 3)
            points_3d: (N, 3)
            poses: (T, 4, 4)
            intrinsics: (T, 3, 3)

        Returns:
            colors: (N, 3) RGB colors
        """
        N = len(points_3d)
        colors = np.zeros((N, 3), dtype=np.uint8)

        # Use first frame for coloring
        t = 0
        pose = poses[t]
        K = intrinsics[t]
        frame = frames[t]
        H, W = frame.shape[:2]

        # Transform points to camera frame
        points_homo = np.concatenate([points_3d, np.ones((N, 1))], axis=1)
        points_cam = (pose @ points_homo.T).T[:, :3]

        # Project to image
        points_img = (K @ points_cam.T).T
        u = points_img[:, 0] / (points_img[:, 2] + 1e-8)
        v = points_img[:, 1] / (points_img[:, 2] + 1e-8)

        # Sample colors
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (points_cam[:, 2] > 0)
        u_valid = u[valid].astype(int)
        v_valid = v[valid].astype(int)

        colors[valid] = frame[v_valid, u_valid]

        return colors

    def _estimate_depths_from_points(self,
                                     points_3d: np.ndarray,
                                     poses: np.ndarray,
                                     intrinsics: np.ndarray,
                                     img_size: tuple) -> np.ndarray:
        """
        Create sparse depth maps from 3D points

        Args:
            points_3d: (N, 3)
            poses: (T, 4, 4)
            intrinsics: (T, 3, 3)
            img_size: (H, W)

        Returns:
            depths: (T, H, W) sparse depth maps
        """
        T = len(poses)
        H, W = img_size
        N = len(points_3d)

        depths = np.zeros((T, H, W), dtype=np.float32)

        points_homo = np.concatenate([points_3d, np.ones((N, 1))], axis=1)

        for t in range(T):
            pose = poses[t]
            K = intrinsics[t]

            # Transform to camera frame
            points_cam = (pose @ points_homo.T).T[:, :3]

            # Project
            points_img = (K @ points_cam.T).T
            u = points_img[:, 0] / (points_img[:, 2] + 1e-8)
            v = points_img[:, 1] / (points_img[:, 2] + 1e-8)
            d = points_cam[:, 2]

            # Fill depth map
            valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (d > 0)
            u_valid = u[valid].astype(int)
            v_valid = v[valid].astype(int)
            d_valid = d[valid]

            depths[t, v_valid, u_valid] = d_valid

        return depths