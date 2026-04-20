"""
Stage 3: Per-Person 3D Pose & Shape Estimation (VIMO)
Estimates SMPL parameters for each tracked person using video transformer.

VIMO requires image file paths on disk, not tensors. This stage:
  1. Saves video frames to a temp directory
  2. For each tracked person, passes their frame file paths + raw bboxes
     to model.inference(imgfiles, boxes, img_focal, img_center)
  3. Converts rotation-matrix output to axis-angle SMPL parameters
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import json
import tempfile
import os
from dataclasses import dataclass, asdict
import pickle

import torch

from src.models.vimo_wrapper import VIMOWrapper, VIMOOutput


@dataclass
class _SimpleDetection:
    """Lightweight detection record reconstructed from tracks.npz."""
    track_id: int
    bbox: np.ndarray        # (4,) [x1, y1, x2, y2]
    mask: Optional[np.ndarray]  # (H, W) bool or None
    keypoints_2d: np.ndarray    # (24, 3)


@dataclass
class SMPLOutput:
    """SMPL output for a single person at a single timestep"""
    frame_id: int
    poses: np.ndarray          # (23, 3) body pose axis-angle
    betas: np.ndarray          # (10,) body shape
    global_orient: np.ndarray  # (3,) root orientation
    transl: np.ndarray         # (3,) root translation in camera frame
    joints: Optional[np.ndarray] = None      # (24, 3) 3D joint positions
    vertices: Optional[np.ndarray] = None    # (6890, 3) mesh vertices
    keypoints_2d: Optional[np.ndarray] = None
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
        return {
            'track_id': self.track_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'num_frames': len(self.frames),
            'frames': self.frames,
        }


class PoseEstimator:
    """Runs VIMO inference per person track."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        model_path = config.get('vimo_model_path')
        # Resolve relative path from project root
        if model_path and not Path(model_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            model_path = str(project_root / model_path)
        self.wrapper = VIMOWrapper(
            model_path=model_path,
            device=self.device,
            num_frames=config.get('num_frames_per_batch', 16),
        )

    def estimate_poses_for_track(
        self,
        frame_imgfiles: List[str],   # imgfiles[i] = path to video frame i
        track_id: int,
        detections_per_frame: List[List[_SimpleDetection]],
        img_focal: Optional[float] = None,
        img_center: Optional[np.ndarray] = None,
    ) -> PersonPose:
        """
        Estimate SMPL pose for one person track.

        Args:
            frame_imgfiles:        Paths to saved video frames (indexed by frame id).
            track_id:              Person track ID.
            detections_per_frame:  Per-frame list of _SimpleDetection objects.
            img_focal:             Focal length in pixels (None → VIMO estimates).
            img_center:            Principal point [cx, cy] (None → VIMO estimates).
        """
        valid_frames = []
        imgfiles = []
        boxes = []

        for frame_id, detections in enumerate(detections_per_frame):
            for det in detections:
                if det.track_id == track_id:
                    valid_frames.append(frame_id)
                    imgfiles.append(frame_imgfiles[frame_id])
                    boxes.append(det.bbox)  # [x1, y1, x2, y2]
                    break

        if not valid_frames:
            print(f"[Stage3] Warning: No frames found for track {track_id}")
            return PersonPose(track_id=track_id, start_frame=-1, end_frame=-1,
                              frames=[], poses=[])

        T = len(valid_frames)
        boxes_arr = np.array(boxes, dtype=np.float32)  # (T, 4)
        frame_indices = np.array(valid_frames, dtype=np.int64)

        print(f"[Stage3] Track {track_id}: {T} frames "
              f"({valid_frames[0]}–{valid_frames[-1]})")

        vimo_out: VIMOOutput = self.wrapper.predict_from_files(
            imgfiles=imgfiles,
            boxes=boxes_arr,
            img_focal=img_focal,
            img_center=img_center,
            frame_indices=np.arange(T),  # local indices 0..T-1
        )

        # Build SMPLOutput list (one per valid frame, local indexing)
        smpl_outputs = []
        for local_i, frame_id in enumerate(valid_frames):
            out = SMPLOutput(
                frame_id=frame_id,
                poses=vimo_out.poses[local_i],
                betas=vimo_out.betas,
                global_orient=vimo_out.global_orient[local_i],
                transl=vimo_out.transl[local_i],
                confidence=float(vimo_out.confidence[local_i])
                           if vimo_out.confidence is not None else 0.0,
            )
            smpl_outputs.append(out)

        return PersonPose(
            track_id=track_id,
            start_frame=min(valid_frames),
            end_frame=max(valid_frames),
            frames=valid_frames,
            poses=smpl_outputs,
        )


def _save_frames_to_disk(video_frames: np.ndarray, tmp_dir: Path) -> List[str]:
    """
    Save (T, H, W, 3) RGB frames as JPEG files; return list of paths.
    These file paths are what VIMO's TrackDataset reads with cv2.imread().
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(video_frames):
        fpath = tmp_dir / f"frame_{i:06d}.jpg"
        # cv2.imread expects BGR, so convert RGB→BGR before saving
        cv2.imwrite(str(fpath), frame[:, :, ::-1])
        paths.append(str(fpath))
    return paths


def run_pose_estimation(
    video_path: str,
    camera_data,
    tracking_data,
    output_dir: Path,
    config: Dict,
    max_frames: Optional[int] = None,
    person_ids: Optional[List[int]] = None,
) -> Dict:
    """
    Run Stage 3: per-person pose estimation.

    Args:
        video_path:    Path to input video.
        camera_data:   Loaded cameras.npz with keys 'poses' (T,4,4) and
                       optionally 'intrinsics' (T,3,3).
        tracking_data: Loaded tracks.npz from Stage 2.
        output_dir:    Directory for results.
        config:        Config dict.
        max_frames:    Cap on frames loaded.
        person_ids:    Restrict to these track IDs.
    """
    print("\n" + "="*80)
    print("STAGE 3: PER-PERSON 3D POSE & SHAPE ESTIMATION (VIMO)")
    print("="*80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load video frames ---
    cap = cv2.VideoCapture(str(video_path))
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames_list) >= max_frames:
            break
    cap.release()
    video_frames = np.stack(frames_list)
    T = len(video_frames)
    print(f"[Stage3] Loaded {T} video frames")

    # --- Extract camera intrinsics for VIMO (use first-frame values) ---
    img_focal = None
    img_center = None
    if 'intrinsics' in camera_data.files if hasattr(camera_data, 'files') else 'intrinsics' in camera_data:
        K = camera_data['intrinsics']   # (T, 3, 3) or (3, 3)
        K0 = K[0] if K.ndim == 3 else K
        img_focal  = float((K0[0, 0] + K0[1, 1]) / 2.0)
        img_center = np.array([K0[0, 2], K0[1, 2]], dtype=np.float32)
        print(f"[Stage3] Camera intrinsics: focal={img_focal:.1f}, "
              f"center=({img_center[0]:.1f}, {img_center[1]:.1f})")
    else:
        print("[Stage3] No intrinsics in camera_data; VIMO will estimate per frame")

    # --- Save frames to disk so VIMO can read them ---
    tmp_dir = output_dir / "_tmp_frames"
    print(f"[Stage3] Saving {T} frames to {tmp_dir} ...")
    frame_imgfiles = _save_frames_to_disk(video_frames, tmp_dir)

    # --- Reconstruct per-frame detections from tracks.npz ---
    detections_per_frame: List[List[_SimpleDetection]] = [[] for _ in range(T)]
    track_keys = {k.split("_")[1] for k in tracking_data.files
                  if k.startswith("track_") and k.endswith("_frames")}

    stage2_masks_dir = Path(output_dir).parent / "2_tracking" / "masks"

    for tid_str in sorted(track_keys, key=int):
        tid = int(tid_str)
        if person_ids and tid not in person_ids:
            continue
        frames_arr = tracking_data[f"track_{tid}_frames"]
        bboxes_arr = tracking_data[f"track_{tid}_bboxes"]

        for i, frame_id in enumerate(frames_arr):
            frame_id = int(frame_id)
            if frame_id >= T:
                continue

            mask_path = stage2_masks_dir / f"frame_{frame_id:06d}_person_{tid:03d}.npy"
            mask = np.load(mask_path) if mask_path.exists() else None

            detections_per_frame[frame_id].append(_SimpleDetection(
                track_id=tid,
                bbox=bboxes_arr[i].astype(np.float32),
                mask=mask,
                keypoints_2d=np.zeros((24, 3), dtype=np.float32),
            ))

    track_ids_found = {d.track_id for dets in detections_per_frame for d in dets}
    if person_ids:
        track_ids_found = track_ids_found & set(person_ids)

    print(f"[Stage3] Estimating pose for {len(track_ids_found)} people: "
          f"{sorted(track_ids_found)}")

    estimator = PoseEstimator(config)
    all_person_poses = []
    start_time = time.time()

    for track_id in sorted(track_ids_found):
        person_pose = estimator.estimate_poses_for_track(
            frame_imgfiles=frame_imgfiles,
            track_id=track_id,
            detections_per_frame=detections_per_frame,
            img_focal=img_focal,
            img_center=img_center,
        )
        all_person_poses.append(person_pose)

    elapsed = time.time() - start_time

    # --- Clean up temp frames ---
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    save_pose_results(all_person_poses, output_dir)

    results = {
        'num_people': len(all_person_poses),
        'person_poses': all_person_poses,
        'processing_time': elapsed,
        'config': config,
    }

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
    """Save pose estimation results to disk."""
    print(f"[Stage3] Saving results to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for person_pose in person_poses:
        if not person_pose.poses:
            continue
        person_dir = output_dir / f'person_{person_pose.track_id:03d}'
        person_dir.mkdir(exist_ok=True)

        poses_array    = np.stack([p.poses for p in person_pose.poses])
        betas          = person_pose.poses[0].betas
        global_orients = np.stack([p.global_orient for p in person_pose.poses])
        transl         = np.stack([p.transl for p in person_pose.poses])

        np.savez(
            person_dir / 'smpl_params_camera.npz',
            poses=poses_array,
            betas=betas,
            global_orient=global_orients,
            transl=transl,
        )

        if person_pose.poses[0].joints is not None:
            joints = np.stack([p.joints for p in person_pose.poses])
            np.save(person_dir / 'joints_camera.npy', joints)

        if person_pose.poses[0].vertices is not None:
            vertices = np.stack([p.vertices for p in person_pose.poses])
            np.save(person_dir / 'vertices_camera.npy', vertices)

        metadata = person_pose.to_dict()
        with open(person_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    metadata = {
        'num_people': len(person_poses),
        'people': [p.to_dict() for p in person_poses],
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("[Stage3] Results saved successfully")
