"""
PHALP+ Wrapper for Multi-person Detection and Tracking
Wraps the PHALP+ model for multi-person detection, pose estimation, and tracking

Usage Examples:
    # Load default model
    phalp = PHALPWrapper()

    # Load custom checkpoint
    phalp = PHALPWrapper(model_path='path/to/phalp_model.pth')

    # Process single frame
    detections = phalp.detect_frame(frame)

    # Process video sequence with tracking
    results = phalp.detect_and_track(frames)

    # Results dict contains:
    # - detections: List[Detection] per frame
    # - track_ids: Unique person IDs
    # - keypoints_2d: 2D joint positions (K, 2)
    # - bboxes: Bounding boxes (4,)
"""
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import cv2
from dataclasses import dataclass


@dataclass
class PHALPDetection:
    """Single detection from PHALP"""
    bbox: np.ndarray  # [x, y, w, h]
    track_id: Optional[int] = None
    keypoints_2d: Optional[np.ndarray] = None  # (K, 3) with confidence
    pose_smpl: Optional[Dict] = None  # SMPL pose estimate
    confidence: float = 0.0


class PHALPWrapper:
    """
    Wrapper for PHALP+ (Parametric Human Shape and Pose Lifting Plus)

    Provides:
    - Multi-person detection
    - 2D/3D keypoint estimation
    - Initial SMPL parameter prediction
    - Track ID assignment
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        detector_backbone: str = 'vit',
        tracker_type: str = 'flow',
        num_people: int = 20
    ):
        """
        Initialize PHALP wrapper

        Args:
            model_path: Path to pretrained PHALP model checkpoint
            device: Device to run on ('cuda' or 'cpu')
            detector_backbone: ViT or CNN backbone for detection
            tracker_type: 'flow' (optical flow) or 'box' (bbox only)
            num_people: Max number of people to track simultaneously
        """
        self.device = device
        self.model_path = model_path
        self.detector_backbone = detector_backbone
        self.tracker_type = tracker_type
        self.num_people = num_people
        self.model = None

        # Add SLAHMR to path for PHALP access
        thirdparty_path = Path(__file__).parent.parent.parent / 'thirdparty' / 'slahmr'
        sys.path.insert(0, str(thirdparty_path))

        self._init_model()

    def _init_model(self):
        """Initialize PHALP model"""
        try:
            # Try to import PHALP from thirdparty
            from slahmr.models.phalp.phalp import PHALP

            print(f"[PHALPWrapper] Loading PHALP model...")

            if self.model_path and Path(self.model_path).exists():
                # Load custom checkpoint
                self.model = PHALP(
                    backbone=self.detector_backbone,
                    tracker=self.tracker_type,
                    device=self.device
                )
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"[PHALPWrapper] Loaded checkpoint from {self.model_path}")
            else:
                # Load default model
                self.model = PHALP(
                    backbone=self.detector_backbone,
                    tracker=self.tracker_type,
                    device=self.device
                )
                print(f"[PHALPWrapper] Loaded default PHALP model")

            self.model.eval()
            print(f"[PHALPWrapper] PHALP model initialized on {self.device}")

        except ImportError as e:
            print(f"[PHALPWrapper] Warning: Could not import PHALP - {e}")
            print("[PHALPWrapper] Will use dummy detections. Install SLAHMR to enable PHALP.")
            self.model = None

    def detect_frame(
        self,
        frame: np.ndarray,
        return_confidence: bool = True
    ) -> List[PHALPDetection]:
        """
        Detect people in a single frame

        Args:
            frame: Input frame (H, W, 3) RGB in [0, 255]
            return_confidence: Whether to return confidence scores

        Returns:
            List of PHALPDetection objects
        """
        if self.model is None:
            return []

        try:
            # Prepare input
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
            frame_tensor = frame_tensor.to(self.device)

            # Normalize to [0, 1]
            frame_tensor = frame_tensor / 255.0

            with torch.no_grad():
                # Run detection
                # Output format depends on PHALP implementation
                # Typically: bboxes, keypoints, pose parameters, scores
                detections = self.model.detect_frame(frame_tensor)

            # Convert to PHALPDetection format
            results = []
            if detections:
                for det in detections:
                    phalp_det = PHALPDetection(
                        bbox=det.get('bbox', np.zeros(4)),
                        keypoints_2d=det.get('keypoints_2d', None),
                        pose_smpl=det.get('pose_smpl', None),
                        confidence=det.get('confidence', 0.0)
                    )
                    results.append(phalp_det)

            return results

        except Exception as e:
            print(f"[PHALPWrapper] Detection failed: {e}")
            return []

    def detect_and_track(
        self,
        frames: np.ndarray,
        return_pose_estimate: bool = True
    ) -> Dict:
        """
        Detect people and track across frames

        Args:
            frames: Video frames (T, H, W, 3) RGB in [0, 255]
            return_pose_estimate: Whether to return initial SMPL estimates

        Returns:
            Dictionary with:
            - detections: List[List[PHALPDetection]] per frame
            - track_ids: Set of unique track IDs
            - pose_estimates: Optional initial SMPL params per person
        """
        if self.model is None:
            print("[PHALPWrapper] Warning: Model not initialized, returning empty results")
            return {
                'detections': [[] for _ in range(len(frames))],
                'track_ids': set(),
                'pose_estimates': {}
            }

        T = len(frames)
        all_detections = []
        track_ids = set()
        pose_estimates = {}

        try:
            # Process frames in batches
            batch_size = 8  # Process 8 frames at a time
            for batch_start in range(0, T, batch_size):
                batch_end = min(batch_start + batch_size, T)
                batch_frames = frames[batch_start:batch_end]

                # Prepare batch
                batch_tensor = torch.from_numpy(batch_frames).float()
                batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # (B, 3, H, W)
                batch_tensor = batch_tensor.to(self.device) / 255.0

                with torch.no_grad():
                    # Run detection and tracking on batch
                    batch_results = self.model.detect_and_track(batch_tensor)

                # Process batch results
                for frame_idx, frame_dets in enumerate(batch_results):
                    frame_detections = []

                    if frame_dets:
                        for det in frame_dets:
                            track_id = det.get('track_id', None)
                            if track_id is not None:
                                track_ids.add(track_id)

                            phalp_det = PHALPDetection(
                                bbox=det.get('bbox', np.zeros(4)),
                                track_id=track_id,
                                keypoints_2d=det.get('keypoints_2d', None),
                                pose_smpl=det.get('pose_smpl', None),
                                confidence=det.get('confidence', 0.0)
                            )
                            frame_detections.append(phalp_det)

                            # Store pose estimate
                            if track_id is not None and return_pose_estimate:
                                if track_id not in pose_estimates:
                                    pose_estimates[track_id] = det.get('pose_smpl', None)

                    all_detections.append(frame_detections)

            return {
                'detections': all_detections,
                'track_ids': track_ids,
                'pose_estimates': pose_estimates,
                'num_frames': T,
                'num_people': len(track_ids)
            }

        except Exception as e:
            print(f"[PHALPWrapper] Tracking failed: {e}")
            return {
                'detections': [[] for _ in range(T)],
                'track_ids': set(),
                'pose_estimates': {}
            }

    def get_2d_keypoints(
        self,
        frame: np.ndarray,
        bbox: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 2D keypoints for a person region

        Args:
            frame: Input frame (H, W, 3)
            bbox: Bounding box [x, y, w, h]

        Returns:
            Tuple of (keypoints_2d, confidence)
            - keypoints_2d: (K, 2) 2D positions
            - confidence: (K,) confidence scores
        """
        if self.model is None:
            return np.zeros((17, 2)), np.zeros(17)

        try:
            x, y, w, h = bbox.astype(int)
            H, W = frame.shape[:2]

            # Crop region
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(W, x + w)
            y_max = min(H, y + h)

            crop = frame[y_min:y_max, x_min:x_max].copy()

            # Run keypoint detection
            frame_tensor = torch.from_numpy(crop).float().permute(2, 0, 1).unsqueeze(0)
            frame_tensor = (frame_tensor.to(self.device) / 255.0)

            with torch.no_grad():
                kpts_result = self.model.get_keypoints(frame_tensor)

            if kpts_result is not None:
                keypoints_2d = kpts_result.get('keypoints_2d', np.zeros((17, 2)))
                confidence = kpts_result.get('confidence', np.zeros(17))
                return keypoints_2d, confidence
            else:
                return np.zeros((17, 2)), np.zeros(17)

        except Exception as e:
            print(f"[PHALPWrapper] Keypoint extraction failed: {e}")
            return np.zeros((17, 2)), np.zeros(17)

    def estimate_smpl(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        keypoints_2d: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Estimate SMPL parameters for a person

        Args:
            frame: Input frame (H, W, 3)
            bbox: Bounding box [x, y, w, h]
            keypoints_2d: Optional 2D keypoints for better estimation

        Returns:
            Dictionary with SMPL parameters:
            - poses: (23, 3) body pose
            - betas: (10,) body shape
            - global_orient: (3,) root orientation
            - transl: (3,) root translation
        """
        if self.model is None:
            return {
                'poses': np.zeros((23, 3)),
                'betas': np.zeros(10),
                'global_orient': np.zeros(3),
                'transl': np.array([0, 0, 5])
            }

        try:
            x, y, w, h = bbox.astype(int)
            H, W = frame.shape[:2]

            # Crop region
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(W, x + w)
            y_max = min(H, y + h)

            crop = frame[y_min:y_max, x_min:x_max].copy()

            # Prepare input
            frame_tensor = torch.from_numpy(crop).float().permute(2, 0, 1).unsqueeze(0)
            frame_tensor = (frame_tensor.to(self.device) / 255.0)

            with torch.no_grad():
                smpl_result = self.model.estimate_smpl(
                    frame_tensor,
                    keypoints_2d=keypoints_2d
                )

            if smpl_result is not None:
                return smpl_result
            else:
                return {
                    'poses': np.zeros((23, 3)),
                    'betas': np.zeros(10),
                    'global_orient': np.zeros(3),
                    'transl': np.array([0, 0, 5])
                }

        except Exception as e:
            print(f"[PHALPWrapper] SMPL estimation failed: {e}")
            return {
                'poses': np.zeros((23, 3)),
                'betas': np.zeros(10),
                'global_orient': np.zeros(3),
                'transl': np.array([0, 0, 5])
            }


if __name__ == "__main__":
    """Test PHALP wrapper"""
    import argparse

    parser = argparse.ArgumentParser(description="Test PHALP wrapper")
    parser.add_argument("--image", type=str, help="Test image path")
    parser.add_argument("--video", type=str, help="Test video path")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")

    args = parser.parse_args()

    # Initialize wrapper
    phalp = PHALPWrapper(model_path=args.model, device=args.device)

    if args.image:
        # Test single frame
        img = cv2.imread(args.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = phalp.detect_frame(img)
        print(f"Detected {len(detections)} people")

    if args.video:
        # Test video
        import cv2
        cap = cv2.VideoCapture(args.video)
        frames = []
        while len(frames) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        frames = np.stack(frames)

        results = phalp.detect_and_track(frames)
        print(f"Processed {results['num_frames']} frames")
        print(f"Detected {results['num_people']} people")
        print(f"Track IDs: {results['track_ids']}")
