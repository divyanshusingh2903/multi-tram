"""
Stage 2: Multi-Person Detection, Segmentation & Tracking
Uses PHALP+ for tracking with optional VGGT feature enhancement
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import json
import pickle
from dataclasses import dataclass

from src.models.phalp_wrapper import PHALPWrapper
from src.utils.kalman_filter import KalmanFilter
from src.utils.hungarian_algorithm import (
    optimal_assignment,
    greedy_assignment,
    create_cost_matrix,
    SCIPY_AVAILABLE
)


# KalmanFilter imported from utils.kalman_filter

@dataclass
class Detection:
    """Detection result for a single person in a frame"""
    frame_id: int
    track_id: Optional[int]
    bbox: np.ndarray  # [x, y, w, h]
    mask: Optional[np.ndarray]  # Binary mask
    keypoints_2d: np.ndarray  # (K, 3) with confidence
    confidence: float
    visibility: float


@dataclass
class Track:
    """Multi-frame track for a single person"""
    track_id: int
    detections: List[Detection]
    start_frame: int
    end_frame: int

    def __post_init__(self):
        if self.detections:
            self.start_frame = min(d.frame_id for d in self.detections)
            self.end_frame = max(d.frame_id for d in self.detections)


class TrackingManager:
    """Manages tracking across frames with ID assignment"""

    def __init__(
        self,
        max_age: int = 30,
        similarity_threshold: float = 0.5,
        use_kalman_filter: bool = True,
        use_hungarian_algorithm: bool = True
    ):
        """
        Args:
            max_age: Maximum frames to keep inactive track before deletion
            similarity_threshold: Threshold for association score
            use_kalman_filter: Whether to use Kalman filter for prediction (default: True)
            use_hungarian_algorithm: Whether to use Hungarian algorithm for optimal assignment (default: True)
        """
        self.max_age = max_age
        self.similarity_threshold = similarity_threshold
        self.use_kalman_filter = use_kalman_filter
        self.use_hungarian_algorithm = use_hungarian_algorithm and SCIPY_AVAILABLE
        self.next_track_id = 1
        self.active_tracks: Dict[int, Track] = {}
        self.inactive_tracks: Dict[int, Track] = {}
        self.kalman_filters: Dict[int, KalmanFilter] = {}  # One Kalman filter per track

        if use_hungarian_algorithm and not SCIPY_AVAILABLE:
            print("[TrackingManager] Warning: scipy not available, Hungarian algorithm disabled")
            self.use_hungarian_algorithm = False

        print(f"[TrackingManager] Initialized with Kalman={self.use_kalman_filter}, "
              f"Hungarian={self.use_hungarian_algorithm}")

    def associate_detections(
        self,
        frame_id: int,
        detections: List[Detection],
        features: Optional[np.ndarray] = None
    ) -> List[Detection]:
        """
        Associate detections with existing tracks using Kalman filter and Hungarian algorithm.

        Args:
            frame_id: Current frame index
            detections: List of detections in current frame
            features: Optional feature vectors for better association

        Returns:
            Detections with assigned track IDs
        """
        if not self.active_tracks:
            # First frame: assign new IDs to all detections
            for det in detections:
                det.track_id = self.next_track_id
                # Initialize Kalman filter if enabled
                if self.use_kalman_filter:
                    self.kalman_filters[self.next_track_id] = KalmanFilter(det.bbox)
                self.next_track_id += 1
            return detections

        # Step 1: Predict next positions for active tracks
        track_predictions = {}
        if self.use_kalman_filter:
            for track_id in self.active_tracks.keys():
                if track_id in self.kalman_filters:
                    predicted_bbox = self.kalman_filters[track_id].predict()
                    track_predictions[track_id] = predicted_bbox
                else:
                    # Fallback to last detection if no Kalman filter
                    last_det = self.active_tracks[track_id].detections[-1]
                    track_predictions[track_id] = last_det.bbox

        # Step 2: Create cost matrix for assignment
        if self.use_hungarian_algorithm and len(detections) > 0 and len(self.active_tracks) > 0:
            cost_matrix = self._create_cost_matrix(detections, track_predictions, features)
            # Solve assignment problem using Hungarian algorithm
            assignments = optimal_assignment(cost_matrix)
        else:
            # Fallback to greedy assignment
            assignments = self._greedy_assignment(detections, track_predictions, features)

        # Step 3: Update tracks
        used_tracks = set()

        for det_idx, det in enumerate(detections):
            # Find matching track
            matching_track_id = None
            for track_id, det_assigned_idx in assignments.items():
                if det_assigned_idx == det_idx:
                    matching_track_id = track_id
                    break

            if matching_track_id is not None and matching_track_id in self.active_tracks:
                det.track_id = matching_track_id
                used_tracks.add(matching_track_id)

                # Update Kalman filter with measurement
                if self.use_kalman_filter and matching_track_id in self.kalman_filters:
                    self.kalman_filters[matching_track_id].update(det.bbox)
            else:
                # Create new track
                det.track_id = self.next_track_id
                if self.use_kalman_filter:
                    self.kalman_filters[self.next_track_id] = KalmanFilter(det.bbox)
                self.next_track_id += 1

        # Step 4: Update active tracks
        for det in detections:
            if det.track_id in self.active_tracks:
                self.active_tracks[det.track_id].detections.append(det)
            else:
                self.active_tracks[det.track_id] = Track(
                    track_id=det.track_id,
                    detections=[det],
                    start_frame=frame_id,
                    end_frame=frame_id
                )

        return detections

    def _create_cost_matrix(
        self,
        detections: List[Detection],
        track_predictions: Dict[int, np.ndarray],
        features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create cost matrix for Hungarian algorithm.

        Delegates to utils.hungarian_algorithm.create_cost_matrix for portability.

        Args:
            detections: List of detections
            track_predictions: Dict of track_id -> predicted bbox
            features: Optional feature vectors

        Returns:
            Cost matrix (N_tracks, N_detections)
        """
        # Extract bboxes from Detection objects
        detection_bboxes = [det.bbox for det in detections]

        # Use utils function
        return create_cost_matrix(
            detection_bboxes,
            track_predictions,
            similarity_threshold=self.similarity_threshold
        )

    def _greedy_assignment(
        self,
        detections: List[Detection],
        track_predictions: Dict[int, np.ndarray],
        features: Optional[np.ndarray] = None
    ) -> Dict[int, int]:
        """
        Greedy assignment (fallback when Hungarian algorithm unavailable).

        Delegates to utils.hungarian_algorithm.greedy_assignment for portability.

        Returns:
            Dict mapping track_idx -> detection_index
        """
        # Create cost matrix
        cost_matrix = self._create_cost_matrix(detections, track_predictions, features)

        # Use utils function for greedy assignment
        return greedy_assignment(cost_matrix)

    def _compute_similarity(
        self,
        det: Detection,
        last_det: Detection,
        features: Optional[np.ndarray] = None
    ) -> float:
        """Compute similarity between two detections (legacy method)"""
        # Simple IoU-based similarity
        iou = self._bbox_iou(det.bbox, last_det.bbox)
        return iou

    @staticmethod
    def _bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bboxes [x, y, w, h]"""
        x1_min, y1_min = bbox1[:2]
        x1_max, y1_max = x1_min + bbox1[2], y1_min + bbox1[3]

        x2_min, y2_min = bbox2[:2]
        x2_max, y2_max = x2_min + bbox2[2], y2_min + bbox2[3]

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-6)


class PersonSegmenter:
    """Handles per-person segmentation using SAM or similar"""

    def __init__(self, use_sam: bool = False):
        """
        Args:
            use_sam: Whether to use SAM for segmentation
        """
        self.use_sam = use_sam
        self.sam = None

        if use_sam:
            try:
                from segment_anything import sam_model_registry
                self.sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
            except ImportError:
                print("[Stage2] Warning: SAM not available, using simple segmentation")
                self.use_sam = False

    def segment_person(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        keypoints_2d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Segment a single person from image.

        Args:
            frame: Input image (H, W, 3) in RGB
            bbox: Bounding box [x, y, w, h]
            keypoints_2d: Optional 2D keypoints for refinement

        Returns:
            Binary mask (H, W)
        """
        if self.use_sam:
            return self._segment_with_sam(frame, bbox, keypoints_2d)
        else:
            return self._segment_with_bbox(frame, bbox)

    def _segment_with_bbox(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Simple bounding box-based segmentation"""
        H, W = frame.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)

        x, y, w, h = bbox.astype(int)
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(W, x + w)
        y_max = min(H, y + h)

        mask[y_min:y_max, x_min:x_max] = 1

        return mask

    def _segment_with_sam(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        keypoints_2d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Segmentation using Segment Anything Model"""
        # Use bbox as prompt for SAM
        x, y, w, h = bbox
        box = np.array([x, y, x + w, y + h])

        # Use keypoints as additional prompts if available
        points = None
        point_labels = None
        if keypoints_2d is not None:
            # Use torso/hip keypoints as prompts
            valid_points = keypoints_2d[keypoints_2d[:, 2] > 0.5][:, :2]
            if len(valid_points) > 0:
                points = valid_points
                point_labels = np.ones(len(points))

        # SAM inference would go here
        # For now, return bbox-based mask as fallback
        return self._segment_with_bbox(frame, bbox)


class PHALP_Tracker:
    """Wrapper around PHALP+ for multi-person tracking"""

    def __init__(
        self,
        device: str = 'cuda',
        model_path: Optional[str] = None,
        use_kalman_filter: bool = True,
        use_hungarian_algorithm: bool = True
    ):
        """
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            model_path: Path to pretrained PHALP model
            use_kalman_filter: Whether to use Kalman filter (default: True)
            use_hungarian_algorithm: Whether to use Hungarian algorithm (default: True)
        """
        self.device = device
        self.phalp_wrapper = PHALPWrapper(model_path=model_path, device=device)
        self.segmenter = PersonSegmenter(use_sam=False)
        self.tracking_manager = TrackingManager(
            use_kalman_filter=use_kalman_filter,
            use_hungarian_algorithm=use_hungarian_algorithm
        )

    def detect_and_track(
        self,
        frames: np.ndarray,
        camera_poses: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run multi-person detection and tracking on video.

        Args:
            frames: Video frames (T, H, W, 3) in RGB
            camera_poses: Optional camera poses for 3D association

        Returns:
            Dictionary with tracking results
        """
        T, H, W = frames.shape[:3]
        all_detections = []
        all_keypoints = []
        all_masks = []

        print(f"[Stage2] Running detection and tracking on {T} frames...")

        start_time = time.time()

        for frame_idx in range(T):
            if frame_idx % 10 == 0:
                print(f"  - Frame {frame_idx}/{T}")

            frame = frames[frame_idx]

            # Run detection (using PHALP or simple detector)
            detections = self._detect_frame(frame)

            # Track
            detections = self.tracking_manager.associate_detections(
                frame_idx, detections
            )

            # Segment
            for det in detections:
                mask = self.segmenter.segment_person(frame, det.bbox, det.keypoints_2d)
                det.mask = mask

            all_detections.append(detections)
            all_keypoints.append([d.keypoints_2d for d in detections])
            all_masks.append([d.mask for d in detections])

        elapsed = time.time() - start_time

        # Consolidate results
        results = {
            'detections': all_detections,
            'keypoints_2d': all_keypoints,
            'masks': all_masks,
            'num_people': len(self.tracking_manager.active_tracks),
            'num_frames': T,
            'processing_time': elapsed,
            'fps': T / elapsed
        }

        print(f"[Stage2] Detection/tracking completed in {elapsed:.2f}s ({T/elapsed:.2f} fps)")
        print(f"  - Detected {results['num_people']} people")

        return results

    def _detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: Input frame (H, W, 3) in RGB

        Returns:
            List of detections
        """
        if self.model is not None:
            return self._detect_with_phalp(frame)
        else:
            return self._detect_simple(frame)

    def _detect_with_phalp(self, frame: np.ndarray) -> List[Detection]:
        """Detection using PHALP+"""
        try:
            # Use PHALP wrapper
            phalp_detections = self.phalp_wrapper.detect_frame(frame)

            detections = []
            for phalp_det in phalp_detections:
                det = Detection(
                    frame_id=0,  # Set properly in parent function
                    track_id=phalp_det.track_id,
                    bbox=phalp_det.bbox,
                    mask=None,
                    keypoints_2d=phalp_det.keypoints_2d if phalp_det.keypoints_2d is not None else np.zeros((17, 3)),
                    confidence=phalp_det.confidence,
                    visibility=1.0 if phalp_det.confidence > 0.5 else 0.0
                )
                detections.append(det)

            return detections
        except Exception as e:
            print(f"[Stage2] PHALP detection failed: {e}")
            return []

    def _detect_simple(self, frame: np.ndarray) -> List[Detection]:
        """Simple detection fallback (returns empty list)"""
        # In a real implementation, would use YOLOv7 or similar
        return []


def run_tracking(
    video_path: str,
    camera_poses: Optional[np.ndarray],
    output_dir: Path,
    config: Dict,
    max_frames: Optional[int] = None
) -> Dict:
    """
    Run Stage 2: Multi-person tracking.

    Args:
        video_path: Path to input video
        camera_poses: Camera poses from Stage 1
        output_dir: Directory to save results
        config: Configuration dictionary with optional keys:
            - 'device': Device to use ('cuda' or 'cpu', default: 'cuda')
            - 'use_kalman_filter': Enable Kalman filter (default: True)
            - 'use_hungarian_algorithm': Enable Hungarian algorithm (default: True)
        max_frames: Maximum frames to process

    Returns:
        Dictionary with tracking results
    """
    print("\n" + "="*80)
    print("STAGE 2: MULTI-PERSON DETECTION, SEGMENTATION & TRACKING")
    print("="*80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video
    print(f"[Stage2] Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_idx += 1

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    frames = np.stack(frames)

    print(f"[Stage2] Loaded {len(frames)} frames")

    # Initialize tracker with configuration
    device = config.get('device', 'cuda')
    use_kalman = config.get('use_kalman_filter', True)
    use_hungarian = config.get('use_hungarian_algorithm', True)

    tracker = PHALP_Tracker(
        device=device,
        use_kalman_filter=use_kalman,
        use_hungarian_algorithm=use_hungarian
    )

    # Run tracking
    results = tracker.detect_and_track(frames, camera_poses)

    # Save results
    save_tracking_results(results, output_dir)

    # Print summary
    print("\n" + "-"*80)
    print("STAGE 2 SUMMARY")
    print("-"*80)
    print(f"Number of frames: {results['num_frames']}")
    print(f"Number of people detected: {results['num_people']}")
    print(f"Processing time: {results['processing_time']:.2f} seconds")
    print(f"Average speed: {results['fps']:.2f} fps")
    print("-"*80 + "\n")

    return results


def save_tracking_results(results: Dict, output_dir: Path):
    """Save tracking results to disk"""
    print(f"[Stage2] Saving results to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detections
    detections_dir = output_dir / 'detections'
    detections_dir.mkdir(exist_ok=True)

    for frame_idx, frame_detections in enumerate(results['detections']):
        frame_data = {
            'frame_id': frame_idx,
            'detections': []
        }

        for det in frame_detections:
            det_data = {
                'track_id': det.track_id,
                'bbox': det.bbox.tolist(),
                'confidence': float(det.confidence),
                'visibility': float(det.visibility),
                'keypoints_2d': det.keypoints_2d.tolist() if det.keypoints_2d is not None else None
            }
            frame_data['detections'].append(det_data)

        with open(detections_dir / f'frame_{frame_idx:06d}.pkl', 'wb') as f:
            pickle.dump(frame_data, f)

    # Save masks
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)

    for frame_idx, frame_masks in enumerate(results['masks']):
        for person_idx, mask in enumerate(frame_masks):
            if mask is not None:
                np.save(
                    masks_dir / f'frame_{frame_idx:06d}_person_{person_idx:03d}.npy',
                    mask
                )

    # Save metadata
    metadata = {
        'num_frames': results['num_frames'],
        'num_people': results['num_people'],
        'processing_time': results['processing_time'],
        'fps': results['fps']
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("[Stage2] Results saved successfully")


if __name__ == "__main__":
    """Test Stage 2 independently"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Test Stage 2: Tracking")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default="configs/vggt.yaml", help="Config file")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'device': 'cuda',
            'max_age': 30,
            'similarity_threshold': 0.5
        }

    # Run Stage 2
    results = run_tracking(
        video_path=args.video,
        camera_poses=None,
        output_dir=Path(args.output),
        config=config,
        max_frames=args.max_frames
    )

    print("[Stage2] Test complete!")
