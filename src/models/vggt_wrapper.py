"""
VGGT Wrapper for Camera Estimation
Wraps the VGGT model for feed-forward camera pose and depth estimation

Usage Examples:
    # Load default model from HuggingFace Hub
    vggt = VGGTWrapper()

    # Load custom checkpoint
    vggt = VGGTWrapper(model_path='path/to/checkpoint.pt')

    # Process video frames (T, H, W, 3) in [0, 255]
    results = vggt.estimate_cameras(frames)

    # Results dict contains:
    # - poses: (T, 4, 4) camera poses
    # - intrinsics: (T, 3, 3) camera intrinsics
    # - depths: (T, H, W) depth maps
    # - point_cloud: (N, 3) 3D points
    # - point_colors: (N, 3) RGB colors
"""
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Tuple
import cv2

class VGGTWrapper:
    """Wrapper for VGGT (Video Geometry Guidance Transformer)"""

    def __init__(self,  model_path: str = None, device: str = 'cuda', max_frames: int = 100, image_size: int = 518):
        """
        Initialize VGGT wrapper

        Args:
            model_path: Path to pretrained VGGT model checkpoint (.pt file).
                       If None, will use the default HuggingFace model.
            device: Device to run on ('cuda' or 'cpu')
            max_frames: Maximum number of frames to process at once
            image_size: Input image size for VGGT (should be 518 for default model)
        """
        self.device = device
        self.max_frames = max_frames
        self.image_size = image_size

        # Add VGGT to path
        thirdparty_path = Path(__file__).parent.parent.parent / 'thirdparty' / 'vggt'
        sys.path.insert(0, str(thirdparty_path))

        try:
            from vggt.models.vggt import VGGT

            print(f"[VGGT] Initializing VGGT model...")
            self.model = VGGT(img_size=image_size).to(device)

            # Load model weights
            if model_path is not None:
                # Load from local checkpoint file
                print(f"[VGGT] Loading model from local file: {model_path}")
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
            else:
                # Load from HuggingFace Hub
                print(f"[VGGT] Loading model from HuggingFace Hub...")
                hf_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
                state_dict = torch.hub.load_state_dict_from_url(hf_url, map_location=device)
                self.model.load_state_dict(state_dict)

            self.model.eval()
            print("[VGGT] Model loaded successfully")

        except Exception as e:
            print(f"[VGGT] Error loading model: {e}")
            raise

    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess video frames for VGGT

        Args:
            frames: numpy array of shape (T, H, W, 3) in [0, 255]

        Returns:
            Preprocessed frames as tensor (T, 3, H, W) normalized to [0, 1]
        """
        T, H, W, C = frames.shape

        # Resize to model input size
        resized_frames = []
        for i in range(T):
            frame = cv2.resize(frames[i], (self.image_size, self.image_size))
            resized_frames.append(frame)
        resized_frames = np.stack(resized_frames, axis=0)

        # Convert to tensor and normalize
        frames_tensor = torch.from_numpy(resized_frames).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
        frames_tensor = frames_tensor / 255.0  # Normalize to [0, 1]

        # VGGT expects images in [0, 1] range (no further normalization needed)
        # The model applies normalization internally if needed

        return frames_tensor.to(self.device)

    @torch.no_grad()
    def estimate_cameras(self, frames: np.ndarray) -> Dict:
        """
        Estimate camera poses and depths from video frames

        Args:
            frames: numpy array of shape (T, H, W, 3) in [0, 255]

        Returns:
            Dictionary containing:
                - 'poses': Camera poses (T, 4, 4) - camera-to-world transforms
                - 'intrinsics': Camera intrinsics (T, 3, 3)
                - 'depths': Predicted depth maps (T, H, W)
                - 'point_cloud': 3D point cloud (N, 3)
                - 'point_colors': Point colors (N, 3)
        """
        print(f"[VGGT] Processing {len(frames)} frames")

        # Split into chunks if too many frames
        if len(frames) > self.max_frames:
            print(f"[VGGT] Splitting into chunks of {self.max_frames} frames")
            return self._process_chunks(frames)

        # Preprocess
        frames_tensor = self.preprocess_frames(frames)

        # Run VGGT
        try:
            # Add sequence dimension if needed: (T, 3, H, W) -> (1, T, 3, H, W)
            if frames_tensor.dim() == 4:
                frames_tensor = frames_tensor.unsqueeze(0)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(frames_tensor)

            # Extract outputs from model
            # The model returns pose encoding and depth predictions
            # We need to decode these appropriately

            # For now, extract the raw outputs
            # Note: The exact output format depends on model configuration
            T = frames_tensor.shape[1]

            if isinstance(outputs, dict):
                # If model returns a dict with predictions
                pose_enc = outputs.get('pose_enc', outputs.get('poses', None))
                depths = outputs.get('depth', outputs.get('depths', None))

                if pose_enc is not None:
                    pose_enc = pose_enc.squeeze(0).cpu().numpy()  # (T, 9)
                if depths is not None:
                    depths = depths.squeeze(0).cpu().numpy()  # (T, H, W)
            else:
                raise ValueError(f"Unexpected model output format: {type(outputs)}")

            # Default intrinsic matrix (assuming no intrinsic changes)
            intrinsics = self._estimate_intrinsics(frames.shape[1:3])
            intrinsics = np.tile(intrinsics[np.newaxis, :, :], (T, 1, 1))

            # Default poses (identity matrices for now - refinement needed)
            poses = np.tile(np.eye(4)[np.newaxis, :, :], (T, 1, 1))

            # Generate point cloud
            if depths is not None:
                point_cloud, point_colors = self._generate_point_cloud(
                    frames, depths, poses, intrinsics
                )
            else:
                point_cloud = np.zeros((0, 3))
                point_colors = np.zeros((0, 3))

            results = {
                'poses': poses,
                'intrinsics': intrinsics,
                'depths': depths,
                'point_cloud': point_cloud,
                'point_colors': point_colors,
                'original_size': frames.shape[1:3]
            }

            print(f"[VGGT] Camera estimation complete")
            print(f"  - Poses: {poses.shape}")
            if depths is not None:
                print(f"  - Depths: {depths.shape}")
            print(f"  - Point cloud: {point_cloud.shape}")

            return results

        except Exception as e:
            print(f"[VGGT] Error during inference: {e}")
            raise

    def _estimate_intrinsics(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Estimate camera intrinsic matrix from frame shape
        Assumes perspective camera with square pixels and principal point at image center

        Args:
            frame_shape: Tuple of (height, width)

        Returns:
            intrinsic matrix (3, 3) in OpenCV format
        """
        H, W = frame_shape

        # Focal length estimation based on image size
        # Assuming 50-degree field of view as default
        focal_length = max(H, W) / (2 * np.tan(np.radians(25)))

        # Principal point at image center
        cx = W / 2.0
        cy = H / 2.0

        # Build intrinsic matrix
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def _process_chunks(self, frames: np.ndarray) -> Dict:
        """Process video in chunks for long sequences"""
        all_poses = []
        all_depths = []
        all_intrinsics = []

        for start_idx in range(0, len(frames), self.max_frames):
            end_idx = min(start_idx + self.max_frames, len(frames))
            chunk = frames[start_idx:end_idx]

            print(f"[VGGT] Processing frames {start_idx}-{end_idx}")
            chunk_results = self.estimate_cameras(chunk)

            all_poses.append(chunk_results['poses'])
            all_depths.append(chunk_results['depths'])
            all_intrinsics.append(chunk_results['intrinsics'])

        # Concatenate results
        poses = np.concatenate(all_poses, axis=0)
        depths = np.concatenate(all_depths, axis=0)
        intrinsics = np.concatenate(all_intrinsics, axis=0)

        # Generate combined point cloud
        point_cloud, point_colors = self._generate_point_cloud(
            frames, depths, poses, intrinsics
        )

        return {
            'poses': poses,
            'intrinsics': intrinsics,
            'depths': depths,
            'point_cloud': point_cloud,
            'point_colors': point_colors,
            'original_size': frames.shape[1:3]
        }

    def _generate_point_cloud(self, frames: np.ndarray, depths: np.ndarray, poses: np.ndarray, intrinsics: np.ndarray,
                              subsample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D point cloud from depth maps and camera poses

        Args:
            frames: RGB frames (T, H, W, 3)
            depths: Depth maps (T, H, W)
            poses: Camera poses (T, 4, 4)
            intrinsics: Camera intrinsics (T, 3, 3)
            subsample: Subsample factor for efficiency

        Returns:
            point_cloud: (N, 3) world coordinates
            point_colors: (N, 3) RGB colors
        """
        T, H, W = depths.shape

        all_points = []
        all_colors = []

        # Process every Nth frame for efficiency
        for t in range(0, T, max(1, T // 10)):
            depth = depths[t]
            pose = poses[t]
            K = intrinsics[t]
            frame = frames[t]

            # Resize depth to match frame size if needed
            if depth.shape != frame.shape[:2]:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

            # Create pixel grid
            h, w = depth.shape
            y, x = np.mgrid[0:h:subsample, 0:w:subsample]

            # Sample depth and colors
            z = depth[y, x]
            colors = frame[y, x]

            # Filter valid depths
            valid = (z > 0) & (z < 100)  # Reasonable depth range
            x, y, z = x[valid], y[valid], z[valid]
            colors = colors[valid]

            if len(x) == 0:
                continue

            # Unproject to camera space
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            Z = z

            # Stack to homogeneous coordinates
            points_cam = np.stack([X, Y, Z, np.ones_like(X)], axis=1)

            # Transform to world coordinates
            pose_inv = np.linalg.inv(pose)
            points_world = (pose_inv @ points_cam.T).T[:, :3]

            all_points.append(points_world)
            all_colors.append(colors)

        if len(all_points) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        point_cloud = np.concatenate(all_points, axis=0)
        point_colors = np.concatenate(all_colors, axis=0)

        return point_cloud, point_colors

    def visualize_cameras(self, poses: np.ndarray, output_path: str):
        """Visualize camera trajectory"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Extract camera centers
        centers = []
        for pose in poses:
            pose_inv = np.linalg.inv(pose)
            center = pose_inv[:3, 3]
            centers.append(center)
        centers = np.array(centers)

        # Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                'b-', linewidth=2, label='Camera trajectory')
        ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2],
                   c='g', s=100, marker='o', label='Start')
        ax.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2],
                   c='r', s=100, marker='x', label='End')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Camera Trajectory')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[VGGT] Camera visualization saved to {output_path}")