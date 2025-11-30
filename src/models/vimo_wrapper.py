"""
VIMO Wrapper for Per-Person Pose Estimation
Wraps VIMO (Video Transformer for Human Motion) for temporally coherent SMPL estimation

Usage Examples:
    # Load default model
    vimo = VIMOWrapper()

    # Load custom checkpoint
    vimo = VIMOWrapper(model_path='path/to/vimo_model.pth')

    # Process sequence of person frames
    smpl_params = vimo.predict_sequence(frames)

    # Results contain:
    # - poses: (T, 23, 3) body pose per frame
    # - betas: (10,) body shape (shared across sequence)
    # - global_orient: (T, 3) root orientation
    # - transl: (T, 3) root position
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import cv2
from dataclasses import dataclass


@dataclass
class VIMOOutput:
    """VIMO prediction output"""
    poses: np.ndarray  # (T, 23, 3) body pose
    betas: np.ndarray  # (10,) body shape
    global_orient: np.ndarray  # (T, 3) root rotation
    transl: np.ndarray  # (T, 3) root translation
    joints: Optional[np.ndarray] = None  # (T, 24, 3) 3D joints
    vertices: Optional[np.ndarray] = None  # (T, 6890, 3) mesh vertices
    confidence: Optional[np.ndarray] = None  # (T,) per-frame confidence


class VIMOWrapper:
    """
    Wrapper for VIMO (Video Transformer for Human Motion)

    Provides:
    - Temporally coherent SMPL parameter estimation
    - Per-frame pose refinement using temporal context
    - Video-level shape estimation
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        backbone: str = 'vit_huge',
        num_frames: int = 16,
        temporal_model: str = 'transformer'
    ):
        """
        Initialize VIMO wrapper

        Args:
            model_path: Path to pretrained VIMO model checkpoint
            device: Device to run on ('cuda' or 'cpu')
            backbone: ViT backbone size ('vit_huge', 'vit_large', 'vit_base')
            num_frames: Number of frames to process per window
            temporal_model: Temporal modeling approach ('transformer', 'rnn', etc)
        """
        self.device = device
        self.model_path = model_path
        self.backbone = backbone
        self.num_frames = num_frames
        self.temporal_model = temporal_model
        self.model = None

        # Image normalization parameters
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])

        # Add paths to system
        tram_path = Path(__file__).parent.parent.parent / 'thirdparty' / 'tram'
        sys.path.insert(0, str(tram_path))

        self._init_model()

    def _init_model(self):
        """Initialize VIMO model"""
        try:
            print(f"[VIMOWrapper] Initializing VIMO model...")
            print(f"  - Backbone: {self.backbone}")
            print(f"  - Temporal model: {self.temporal_model}")
            print(f"  - Device: {self.device}")

            # Try to import VIMO from thirdparty
            # Note: This is a placeholder - actual VIMO implementation may differ
            try:
                from tram.models.vimo import VIMO

                if self.model_path and Path(self.model_path).exists():
                    # Load custom checkpoint
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    self.model = VIMO(
                        backbone=self.backbone,
                        temporal_model=self.temporal_model,
                        device=self.device
                    )
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    print(f"[VIMOWrapper] Loaded checkpoint from {self.model_path}")
                else:
                    # Load default model
                    self.model = VIMO(
                        backbone=self.backbone,
                        temporal_model=self.temporal_model,
                        device=self.device
                    )
                    print(f"[VIMOWrapper] Loaded default VIMO model")

                self.model.eval()
                print(f"[VIMOWrapper] VIMO model ready")

            except ImportError:
                print("[VIMOWrapper] VIMO not found in thirdparty, using dummy model")
                self.model = None

        except Exception as e:
            print(f"[VIMOWrapper] Error loading VIMO: {e}")
            self.model = None

    def predict_sequence(
        self,
        frames: np.ndarray,
        masks: Optional[np.ndarray] = None,
        keypoints_2d: Optional[np.ndarray] = None,
        use_temporal_smoothing: bool = True
    ) -> VIMOOutput:
        """
        Predict SMPL parameters for a sequence of frames

        Args:
            frames: Person frames (T, 256, 256, 3) RGB normalized to [0, 1]
            masks: Optional binary masks (T, 256, 256)
            keypoints_2d: Optional 2D keypoints (T, K, 3) with confidence
            use_temporal_smoothing: Whether to apply temporal smoothing

        Returns:
            VIMOOutput with SMPL parameters for entire sequence
        """
        T = len(frames)

        if self.model is not None:
            return self._predict_with_model(
                frames, masks, keypoints_2d, use_temporal_smoothing
            )
        else:
            return self._predict_dummy(T)

    def _predict_with_model(
        self,
        frames: np.ndarray,
        masks: Optional[np.ndarray],
        keypoints_2d: Optional[np.ndarray],
        use_temporal_smoothing: bool
    ) -> VIMOOutput:
        """Prediction using actual VIMO model"""
        T = len(frames)

        try:
            # Prepare input tensors
            frames_tensor = self._prepare_frames(frames)  # (T, 3, 256, 256)

            if frames_tensor is None:
                return self._predict_dummy(T)

            frames_tensor = frames_tensor.to(self.device)

            # Prepare optional inputs
            masks_tensor = None
            if masks is not None:
                masks_tensor = torch.from_numpy(masks).float()
                masks_tensor = masks_tensor.unsqueeze(1).to(self.device)  # (T, 1, 256, 256)

            keypoints_tensor = None
            if keypoints_2d is not None:
                keypoints_tensor = torch.from_numpy(keypoints_2d).float()
                keypoints_tensor = keypoints_tensor.to(self.device)  # (T, K, 3)

            with torch.no_grad():
                # Run model on chunks to manage memory
                all_poses = []
                betas = None
                all_orients = []
                all_transl = []

                for start_idx in range(0, T, self.num_frames):
                    end_idx = min(start_idx + self.num_frames, T)
                    chunk_size = end_idx - start_idx

                    chunk_frames = frames_tensor[start_idx:end_idx]

                    chunk_masks = None
                    if masks_tensor is not None:
                        chunk_masks = masks_tensor[start_idx:end_idx]

                    chunk_kpts = None
                    if keypoints_tensor is not None:
                        chunk_kpts = keypoints_tensor[start_idx:end_idx]

                    # Run model
                    output = self.model(
                        chunk_frames,
                        masks=chunk_masks,
                        keypoints_2d=chunk_kpts
                    )

                    # Extract outputs
                    if isinstance(output, dict):
                        all_poses.extend(output.get('poses', [np.zeros((23, 3))] * chunk_size))
                        all_orients.extend(output.get('global_orient', [np.zeros(3)] * chunk_size))
                        all_transl.extend(output.get('transl', [np.zeros(3)] * chunk_size))

                        if betas is None and 'betas' in output:
                            betas = output['betas']

            if not all_poses:
                return self._predict_dummy(T)

            # Stack results
            poses = np.stack(all_poses)  # (T, 23, 3)
            global_orient = np.stack(all_orients)  # (T, 3)
            transl = np.stack(all_transl)  # (T, 3)

            if betas is None:
                betas = np.zeros(10)
            elif isinstance(betas, torch.Tensor):
                betas = betas.cpu().numpy()

            # Apply temporal smoothing if requested
            if use_temporal_smoothing:
                poses = self._temporal_smooth(poses)
                transl = self._temporal_smooth(transl)

            return VIMOOutput(
                poses=poses,
                betas=betas,
                global_orient=global_orient,
                transl=transl,
                confidence=np.ones(T)
            )

        except Exception as e:
            print(f"[VIMOWrapper] Model prediction failed: {e}")
            return self._predict_dummy(T)

    def _prepare_frames(self, frames: np.ndarray) -> Optional[torch.Tensor]:
        """
        Prepare frames for input to model

        Args:
            frames: (T, 256, 256, 3) in [0, 1] or [0, 255]

        Returns:
            Tensor (T, 3, 256, 256) normalized for ImageNet
        """
        try:
            T = len(frames)
            prepared = np.zeros((T, 256, 256, 3), dtype=np.float32)

            for t in range(T):
                frame = frames[t].astype(np.float32)

                # Normalize to [0, 1] if needed
                if frame.max() > 1.0:
                    frame = frame / 255.0

                # Normalize to ImageNet statistics
                frame = (frame - self.image_mean) / self.image_std

                prepared[t] = frame

            # Convert to tensor (T, 3, 256, 256)
            tensor = torch.from_numpy(prepared).permute(0, 3, 1, 2)
            return tensor

        except Exception as e:
            print(f"[VIMOWrapper] Frame preparation failed: {e}")
            return None

    @staticmethod
    def _temporal_smooth(
        sequence: np.ndarray,
        window_size: int = 5,
        poly_order: int = 2
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to temporal sequence

        Args:
            sequence: (T, D) temporal sequence
            window_size: Smoothing window size (must be odd)
            poly_order: Polynomial order

        Returns:
            Smoothed sequence (T, D)
        """
        try:
            from scipy.signal import savgol_filter

            T, D = sequence.shape

            # Ensure window size is odd and valid
            window_size = max(3, min(window_size, T))
            if window_size % 2 == 0:
                window_size -= 1
            poly_order = min(poly_order, window_size - 1)

            smoothed = np.zeros_like(sequence)

            for d in range(D):
                smoothed[:, d] = savgol_filter(sequence[:, d], window_size, poly_order)

            return smoothed

        except ImportError:
            print("[VIMOWrapper] scipy not available, skipping temporal smoothing")
            return sequence

    def _predict_dummy(self, num_frames: int) -> VIMOOutput:
        """Create dummy SMPL predictions"""
        return VIMOOutput(
            poses=np.zeros((num_frames, 23, 3)),
            betas=np.zeros(10),
            global_orient=np.zeros((num_frames, 3)),
            transl=np.zeros((num_frames, 3)),
            joints=None,
            vertices=None,
            confidence=np.zeros(num_frames)
        )


class VIMOBatchPredictor:
    """
    Batch predictor for multiple people sequences
    """

    def __init__(self, vimo_model: VIMOWrapper):
        """
        Initialize batch predictor

        Args:
            vimo_model: Initialized VIMOWrapper instance
        """
        self.vimo = vimo_model

    def predict_batch(
        self,
        person_frame_sequences: List[np.ndarray],
        person_masks: Optional[List[np.ndarray]] = None,
        person_keypoints: Optional[List[np.ndarray]] = None
    ) -> List[VIMOOutput]:
        """
        Predict SMPL for multiple people

        Args:
            person_frame_sequences: List of frame sequences per person
            person_masks: Optional masks per person
            person_keypoints: Optional keypoints per person

        Returns:
            List of VIMOOutput per person
        """
        results = []

        for idx, frames in enumerate(person_frame_sequences):
            masks = person_masks[idx] if person_masks else None
            keypoints = person_keypoints[idx] if person_keypoints else None

            output = self.vimo.predict_sequence(
                frames=frames,
                masks=masks,
                keypoints_2d=keypoints
            )

            results.append(output)

        return results


if __name__ == "__main__":
    """Test VIMO wrapper"""
    import argparse

    parser = argparse.ArgumentParser(description="Test VIMO wrapper")
    parser.add_argument("--video", type=str, help="Test video path")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")

    args = parser.parse_args()

    # Initialize wrapper
    vimo = VIMOWrapper(model_path=args.model, device=args.device)

    if args.video:
        # Test video
        cap = cv2.VideoCapture(args.video)
        frames = []

        while len(frames) < 30:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()
        frames = np.stack(frames)

        print(f"Input shape: {frames.shape}")

        # Predict
        output = vimo.predict_sequence(frames)

        print(f"\nPrediction Results:")
        print(f"  - Poses shape: {output.poses.shape}")
        print(f"  - Betas shape: {output.betas.shape}")
        print(f"  - Global orient shape: {output.global_orient.shape}")
        print(f"  - Translation shape: {output.transl.shape}")
