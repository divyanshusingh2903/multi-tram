"""
Stage 3: Per-Person 3D Pose & Shape Estimation (VIMO)
Estimates SMPL parameters for each tracked person using video transformer
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import json
from dataclasses import dataclass, asdict
import pickle

import torch
import torch.nn as nn

from src.models.vimo_wrapper import VIMOWrapper, VIMOBatchPredictor


@dataclass
class SMPLOutput:
    """SMPL output for a single person at a single timestep"""
    frame_id: int
    poses: np.ndarray  # (23, 3) body pose in axis-angle
    betas: np.ndarray  # (10,) body shape
    global_orient: np.ndarray  # (3,) root orientation
    transl: np.ndarray  # (3,) root translation in camera frame
    joints: Optional[np.ndarray] = None  # (24, 3) 3D joint positions
    vertices: Optional[np.ndarray] = None  # (6890, 3) mesh vertices
    keypoints_2d: Optional[np.ndarray] = None  # (24, 3) 2D projections with conf
    confidence: float = 0.0


@dataclass
class PersonPose:
    """Complete pose sequence for one person"""
    track_id: int
    start_frame: int
    end_frame: int
    frames: List[int]
    poses: List[SMPLOutput]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'track_id': self.track_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'num_frames': len(self.frames),
            'frames': self.frames
        }


class VIMOPosePredictor:
    """Video Transformer for Human Motion (VIMO) pose estimator wrapper"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        backbone: str = 'vit_huge',
        num_frames: int = 16
    ):
        """
        Initialize VIMO predictor.

        Args:
            model_path: Path to pretrained VIMO weights
            device: Device to run on ('cuda' or 'cpu')
            backbone: ViT backbone size ('vit_huge', 'vit_large', etc.)
            num_frames: Number of frames to process at once
        """
        self.device = device
        self.backbone = backbone
        self.num_frames = num_frames

        # Use VIMO wrapper
        self.vimo_wrapper = VIMOWrapper(
            model_path=model_path,
            device=device,
            backbone=backbone,
            num_frames=num_frames
        )

        # Image normalization
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])

    def predict_person_pose(
        self,
        frames: np.ndarray,
        masks: Optional[np.ndarray] = None,
        bboxes: Optional[np.ndarray] = None,
        keypoints_2d: Optional[np.ndarray] = None
    ) -> List[SMPLOutput]:
        """
        Predict SMPL pose for a sequence of frames of one person.

        Args:
            frames: Person frames (T, H, W, 3) RGB, cropped and resized to 256x256
            masks: Binary masks for person (T, 256, 256)
            bboxes: Bounding boxes (T, 4) [x, y, w, h]
            keypoints_2d: 2D keypoints (T, K, 3)

        Returns:
            List of SMPL outputs for each frame
        """
        T = len(frames)
        print(f"[Stage3] Predicting pose for {T}-frame sequence")

        # Prepare input - normalize frames to [0, 1]
        frames_normalized = self._normalize_frames(frames)

        # Run inference using VIMO wrapper
        vimo_output = self.vimo_wrapper.predict_sequence(
            frames_normalized,
            masks=masks,
            keypoints_2d=keypoints_2d,
            use_temporal_smoothing=True
        )

        # Convert VIMO output to SMPLOutput format
        outputs = []
        for frame_idx in range(T):
            out = SMPLOutput(
                frame_id=frame_idx,
                poses=vimo_output.poses[frame_idx],
                betas=vimo_output.betas,
                global_orient=vimo_output.global_orient[frame_idx],
                transl=vimo_output.transl[frame_idx],
                joints=vimo_output.joints[frame_idx] if vimo_output.joints is not None else None,
                vertices=vimo_output.vertices[frame_idx] if vimo_output.vertices is not None else None,
                keypoints_2d=None,
                confidence=vimo_output.confidence[frame_idx] if vimo_output.confidence is not None else 0.0
            )
            outputs.append(out)

        return outputs

    def _normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Normalize frames for input to model"""
        # Frames should already be 256x256 from preprocessing
        # Just ensure they're in [0, 1] range
        normalized = []

        for frame in frames:
            frame = frame.astype(np.float32)

            # Normalize to [0, 1] if needed
            if frame.max() > 1.0:
                frame = frame / 255.0

            normalized.append(frame)

        return np.stack(normalized)


    def extract_person_frames(
        self,
        video_frames: np.ndarray,
        masks: List[np.ndarray],
        bboxes: List[Tuple[float, float, float, float]],
        valid_frame_ids: List[int]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract and preprocess frames for a single person.

        Args:
            video_frames: All video frames
            masks: Per-frame person masks
            bboxes: Per-frame bounding boxes
            valid_frame_ids: Frame indices where person is visible

        Returns:
            Tuple of (cropped_frames, cropped_masks, bbox_coords)
        """
        cropped_frames = []
        cropped_masks = []

        for frame_id in valid_frame_ids:
            frame = video_frames[frame_id]
            mask = masks[frame_id]
            bbox = bboxes[frame_id]

            # Crop to bbox
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            H, W = frame.shape[:2]
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(W, x + w)
            y_max = min(H, y + h)

            cropped = frame[y_min:y_max, x_min:x_max].copy()
            cropped_mask = mask[y_min:y_max, x_min:x_max].copy()

            # Resize to 256x256 maintaining aspect ratio
            cropped = cv2.resize(cropped, (256, 256))
            cropped_mask = cv2.resize(cropped_mask, (256, 256))

            cropped_frames.append(cropped)
            cropped_masks.append(cropped_mask)

        cropped_frames = np.stack(cropped_frames)
        cropped_masks = np.stack(cropped_masks) if cropped_masks else None

        return cropped_frames, cropped_masks, np.array(bboxes)


class PoseEstimator:
    """High-level pose estimation manager"""

    def __init__(self, config: Dict):
        """
        Initialize pose estimator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('device', 'cuda')
        self.vimo = VIMOPosePredictor(
            model_path=config.get('vimo_model_path'),
            device=self.device,
            num_frames=config.get('num_frames_per_batch', 16)
        )

    def estimate_poses_for_track(
        self,
        video_frames: np.ndarray,
        track_id: int,
        detections_per_frame: List,
        masks_per_frame: List[Optional[np.ndarray]]
    ) -> PersonPose:
        """
        Estimate SMPL pose for a single person track.

        Args:
            video_frames: All video frames
            track_id: Person track ID
            detections_per_frame: Detections per frame (from Stage 2)
            masks_per_frame: Segmentation masks per frame

        Returns:
            PersonPose object with complete sequence
        """
        # Find frames where this person is visible
        valid_frames = []
        bboxes = []
        masks = []
        keypoints = []

        for frame_id, detections in enumerate(detections_per_frame):
            for det in detections:
                if det.track_id == track_id:
                    valid_frames.append(frame_id)
                    bboxes.append(det.bbox)
                    masks.append(
                        masks_per_frame[frame_id] if masks_per_frame else None
                    )
                    keypoints.append(det.keypoints_2d)
                    break

        if not valid_frames:
            print(f"[Stage3] Warning: No frames found for track {track_id}")
            return PersonPose(
                track_id=track_id,
                start_frame=-1,
                end_frame=-1,
                frames=[],
                poses=[]
            )

        print(f"[Stage3] Estimating pose for track {track_id} "
              f"({len(valid_frames)} frames)")

        # Extract person frames
        person_frames, person_masks, _ = self.vimo.extract_person_frames(
            video_frames, masks, bboxes, valid_frames
        )

        # Predict pose
        smpl_outputs = self.vimo.predict_person_pose(
            person_frames, person_masks, np.array(bboxes), np.array(keypoints)
        )

        # Create PersonPose object
        return PersonPose(
            track_id=track_id,
            start_frame=min(valid_frames),
            end_frame=max(valid_frames),
            frames=valid_frames,
            poses=smpl_outputs
        )


def run_pose_estimation(
    video_frames: np.ndarray,
    tracking_results: Dict,
    output_dir: Path,
    config: Dict
) -> Dict:
    """
    Run Stage 3: Per-person pose estimation.

    Args:
        video_frames: Video frames (T, H, W, 3)
        tracking_results: Results from Stage 2
        output_dir: Directory to save results
        config: Configuration dictionary

    Returns:
        Dictionary with pose estimation results
    """
    print("\n" + "="*80)
    print("STAGE 3: PER-PERSON 3D POSE & SHAPE ESTIMATION (VIMO)")
    print("="*80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize estimator
    estimator = PoseEstimator(config)

    # Get detections and masks from Stage 2
    detections_per_frame = tracking_results.get('detections', [])
    masks_per_frame = tracking_results.get('masks', [])

    # Get all track IDs
    track_ids = set()
    for detections in detections_per_frame:
        for det in detections:
            track_ids.add(det.track_id)

    print(f"[Stage3] Estimating pose for {len(track_ids)} people")

    all_person_poses = []
    start_time = time.time()

    # Process each person
    for track_id in sorted(track_ids):
        person_pose = estimator.estimate_poses_for_track(
            video_frames,
            track_id,
            detections_per_frame,
            masks_per_frame
        )

        all_person_poses.append(person_pose)

    elapsed = time.time() - start_time

    # Save results
    save_pose_results(all_person_poses, output_dir)

    results = {
        'num_people': len(all_person_poses),
        'person_poses': all_person_poses,
        'processing_time': elapsed,
        'config': config
    }

    # Print summary
    print("\n" + "-"*80)
    print("STAGE 3 SUMMARY")
    print("-"*80)
    print(f"Number of people: {len(all_person_poses)}")
    for person_pose in all_person_poses:
        print(f"  - Person {person_pose.track_id}: "
              f"frames {person_pose.start_frame}-{person_pose.end_frame} "
              f"({len(person_pose.poses)} frames)")
    print(f"Processing time: {elapsed:.2f} seconds")
    print("-"*80 + "\n")

    return results


def save_pose_results(person_poses: List[PersonPose], output_dir: Path):
    """Save pose estimation results to disk"""
    print(f"[Stage3] Saving results to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-person results
    for person_pose in person_poses:
        person_dir = output_dir / f'person_{person_pose.track_id:03d}'
        person_dir.mkdir(exist_ok=True)

        # Save SMPL parameters
        poses_array = np.stack([p.poses for p in person_pose.poses])
        betas = person_pose.poses[0].betas  # Same for all frames
        global_orients = np.stack([p.global_orient for p in person_pose.poses])
        transl = np.stack([p.transl for p in person_pose.poses])

        np.savez(
            person_dir / 'smpl_params_camera.npz',
            poses=poses_array,
            betas=betas,
            global_orient=global_orients,
            transl=transl
        )

        # Save joints and vertices if available
        if person_pose.poses[0].joints is not None:
            joints = np.stack([p.joints for p in person_pose.poses])
            np.save(person_dir / 'joints_camera.npy', joints)

        if person_pose.poses[0].vertices is not None:
            vertices = np.stack([p.vertices for p in person_pose.poses])
            np.save(person_dir / 'vertices_camera.npy', vertices)

        # Save 2D keypoints if available
        if person_pose.poses[0].keypoints_2d is not None:
            keypoints_2d = np.stack([p.keypoints_2d for p in person_pose.poses])
            np.save(person_dir / 'keypoints_2d.npy', keypoints_2d)

        # Save metadata
        metadata = person_pose.to_dict()
        with open(person_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    # Save overall metadata
    metadata = {
        'num_people': len(person_poses),
        'people': [p.to_dict() for p in person_poses]
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("[Stage3] Results saved successfully")


if __name__ == "__main__":
    """Test Stage 3 independently"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Test Stage 3: Pose Estimation")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--tracking_results", type=str, required=True,
                        help="Path to Stage 2 results")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default="configs/vggt.yaml",
                        help="Config file")

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {'device': 'cuda', 'num_frames_per_batch': 16}

    # Load video
    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    frames = np.stack(frames)

    # Load tracking results
    with open(args.tracking_results, 'rb') as f:
        tracking_results = pickle.load(f)

    # Run Stage 3
    results = run_pose_estimation(frames, tracking_results, Path(args.output), config)

    print("[Stage3] Test complete!")
