"""
Stage 2 — Multi-Person Detection & Tracking (spec §3).

Sub-steps:
  A. PHALP+ tracking (primary tracker) → per-frame detections with stable IDs.
  B. VGGT depth-guided world-frame ID correction → fix ID switches caused by
     camera motion, merge fragmented tracks, discard short tracks.

After this stage, each Track object has SAM 2 masks for every frame the person
is visible (used by Stage 3 for masked VIMO crops).

Output artifacts (relative to output_dir/):
  tracks.npz                               track metadata for all N persons
  detections/frame_{t:06d}.pkl             list of Detection per frame
  masks/frame_{t:06d}_person_{id:03d}.npy  bool (H, W) — SAM 2 mask per person
  id_corrections.json                      log of corrected ID switches
  metadata.json
"""
import json
import pickle
import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.models.phalp_wrapper import PHALPWrapper, PHALPDetection
from src.models.sam_wrapper import SAMWrapper
from src.utils.world_correction import PHALPTrack, correct_world_frame_ids


# ---------------------------------------------------------------------------
# Output data type (spec §3.2 "Output per track")
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """One tracked person across the video."""
    track_id: int
    start_frame: int
    end_frame: int
    frames: List[int]              # all frame indices where visible
    bboxes: np.ndarray             # float32 (T_i, 4) [cx,cy,w,h] — PHALP format
    masks: np.ndarray              # bool    (T_i, H, W) — SAM 2 per-frame mask
    keypoints_2d: np.ndarray       # float32 (T_i, 24, 3)
    smpl_init: Dict                # coarse SMPL from PHALP+ (input to VIMO)


# ---------------------------------------------------------------------------
# Stage 2 main entry point (spec §3.4)
# ---------------------------------------------------------------------------

def run_tracking(
    frames: np.ndarray,
    camera_results: Dict,
    output_dir: Path,
    config: Dict,
) -> List[Track]:
    """
    Run Stage 2: Multi-Person Detection & Tracking.

    Args:
        frames:         (T, H, W, 3) uint8 RGB.
        camera_results: Stage 1 output dict — keys: poses, intrinsics,
                        depths, track_features, confidence, method_used.
        output_dir:     Directory to write all artifacts.
        config:         Dict with keys (defaults shown):
                          phalp_cfg            "configs/phalp.yaml" (unused — PHALP
                                               uses its own Hydra config; pass
                                               model_path / device instead)
                          device               "cuda"
                          phalp_model_path     None
                          min_track_len        8
                          v_max_world          2.0
                          embed_sim_threshold  0.6
                          depth_roi_px         5
                          sam_model            "sam2_hiera_large.yaml"
                          sam_checkpoint       None
                          sam_dilate_px        5

    Returns:
        List of Track objects — one per unique person, ordered by track_id.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: MULTI-PERSON DETECTION & TRACKING (PHALP+)")
    print("=" * 80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames.shape

    device = config.get("device", "cuda")

    # ------------------------------------------------------------------
    # Sub-step A — PHALP+ tracking (spec §3.2)
    # ------------------------------------------------------------------
    print("[Stage2] Sub-step A: PHALP+ tracking...")
    phalp = PHALPWrapper(
        model_path=config.get("phalp_model_path", None),
        device=device,
    )
    t0 = time.time()
    dets_by_frame: Dict[int, List[PHALPDetection]] = phalp.track_frames(
        frames, tmp_dir=output_dir / "_tmp"
    )
    phalp_time = time.time() - t0
    print(f"[Stage2] PHALP+ done in {phalp_time:.1f}s")

    # ------------------------------------------------------------------
    # Build PHALPTrack objects for world-frame correction
    # ------------------------------------------------------------------
    phalp_tracks: Dict[int, PHALPTrack] = {}

    for frame_id in range(T):
        dets = dets_by_frame.get(frame_id, [])
        for det in dets:
            tid = det.track_id
            if tid not in phalp_tracks:
                phalp_tracks[tid] = PHALPTrack(
                    track_id=tid,
                    frames=[],
                    bboxes=np.zeros((0, 4), dtype=np.float32),
                    keypoints_2d=np.zeros((0, 24, 3), dtype=np.float32),
                    smpl_init={},
                    embeddings=np.zeros((0, 4096), dtype=np.float32),
                    world_positions=np.zeros((0, 3), dtype=np.float32),
                )
            trk = phalp_tracks[tid]
            trk.frames.append(frame_id)
            trk.bboxes = np.concatenate(
                [trk.bboxes, det.bbox[None]], axis=0
            )
            trk.keypoints_2d = np.concatenate(
                [trk.keypoints_2d, det.keypoints_2d[None]], axis=0
            )
            trk.embeddings = np.concatenate(
                [trk.embeddings, det.embedding[None]], axis=0
            )
            # Accumulate SMPL — keep last-frame value (VIMO overrides in Stage 3)
            trk.smpl_init = det.smpl

    tracks_list = list(phalp_tracks.values())
    print(f"[Stage2] PHALP+ found {len(tracks_list)} track(s) before correction.")

    # ------------------------------------------------------------------
    # Sub-step B — World-frame ID correction (spec §3.3)
    # ------------------------------------------------------------------
    print("[Stage2] Sub-step B: VGGT depth-guided world-frame ID correction...")
    corrected_tracks, correction_log = correct_world_frame_ids(
        tracks_list,
        camera_results,
        config={
            "v_max_world":          config.get("v_max_world",         2.0),
            "embed_sim_threshold":  config.get("embed_sim_threshold",  0.6),
            "depth_roi_px":         config.get("depth_roi_px",         5),
            "min_track_len":        config.get("min_track_len",        8),
        },
    )
    print(f"[Stage2] After correction: {len(corrected_tracks)} track(s).")

    # Save id_corrections.json
    with open(output_dir / "id_corrections.json", "w") as f:
        json.dump(correction_log, f, indent=2)

    # ------------------------------------------------------------------
    # SAM 2 person masks (spec §3.2 "Output per track" — masks field)
    # ------------------------------------------------------------------
    print("[Stage2] Computing SAM 2 person masks...")
    sam = SAMWrapper(
        model_cfg=config.get("sam_model", "sam2_hiera_large.yaml"),
        checkpoint=config.get("sam_checkpoint", None),
        device=device,
        dilate_px=config.get("sam_dilate_px", 5),
    )
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    # Build per-person mask arrays and write per-frame mask files
    person_masks: Dict[int, List[np.ndarray]] = {trk.track_id: [] for trk in corrected_tracks}
    tid_to_track = {trk.track_id: trk for trk in corrected_tracks}

    for t in range(T):
        active = [(trk, idx) for trk in corrected_tracks
                  for idx, f in enumerate(trk.frames) if f == t]

        if not active:
            continue

        frame = frames[t]
        # PHALP bboxes are [cx, cy, w, h]; SAM2 expects [x1, y1, x2, y2]
        cxywh = np.array([trk.bboxes[idx] for trk, idx in active], dtype=np.float32)
        boxes = np.stack([cxywh[:, 0] - cxywh[:, 2] / 2,
                          cxywh[:, 1] - cxywh[:, 3] / 2,
                          cxywh[:, 0] + cxywh[:, 2] / 2,
                          cxywh[:, 1] + cxywh[:, 3] / 2], axis=1)
        inst_masks = sam.segment_boxes(frame, boxes)

        for (trk, _), mask in zip(active, inst_masks):
            person_masks[trk.track_id].append(mask)
            np.save(
                masks_dir / f"frame_{t:06d}_person_{trk.track_id:03d}.npy",
                mask.astype(bool),
            )

    # ------------------------------------------------------------------
    # Assemble final Track objects (spec §3.2)
    # ------------------------------------------------------------------
    final_tracks: List[Track] = []
    for ptrk in corrected_tracks:
        tid = ptrk.track_id
        mlist = person_masks.get(tid, [])
        # Pad missing masks with zeros if needed
        while len(mlist) < len(ptrk.frames):
            mlist.append(np.zeros((H, W), dtype=bool))

        track = Track(
            track_id=tid,
            start_frame=ptrk.frames[0],
            end_frame=ptrk.frames[-1],
            frames=ptrk.frames,
            bboxes=ptrk.bboxes,
            masks=np.stack(mlist, axis=0),
            keypoints_2d=ptrk.keypoints_2d,
            smpl_init=ptrk.smpl_init,
        )
        final_tracks.append(track)

    # ------------------------------------------------------------------
    # Save artifacts (spec §3.4)
    # ------------------------------------------------------------------
    _save_tracking_results(final_tracks, dets_by_frame, T, output_dir)

    total_time = time.time() - t0
    print(f"\n[Stage2] Done in {total_time:.1f}s  —  {len(final_tracks)} person(s) tracked.")
    return final_tracks


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_tracking_results(
    tracks: List[Track],
    dets_by_frame: Dict[int, List[PHALPDetection]],
    T: int,
    output_dir: Path,
):
    # tracks.npz — track metadata (no large arrays)
    track_meta = {
        f"track_{trk.track_id}_frames": np.array(trk.frames, dtype=np.int32)
        for trk in tracks
    }
    track_meta.update({
        f"track_{trk.track_id}_bboxes": trk.bboxes for trk in tracks
    })
    np.savez(output_dir / "tracks.npz", **track_meta)

    # detections/frame_{t:06d}.pkl
    det_dir = output_dir / "detections"
    det_dir.mkdir(exist_ok=True)
    for frame_id in range(T):
        dets = dets_by_frame.get(frame_id, [])
        det_list = []
        for d in dets:
            det_list.append({
                "track_id":    d.track_id,
                "bbox":        d.bbox.tolist(),
                "confidence":  d.confidence,
                "keypoints_2d": d.keypoints_2d.tolist(),
            })
        with open(det_dir / f"frame_{frame_id:06d}.pkl", "wb") as f:
            pickle.dump(det_list, f)

    # metadata.json
    meta = {
        "num_tracks": len(tracks),
        "num_frames": T,
        "track_ids":  [trk.track_id for trk in tracks],
        "track_lengths": {
            trk.track_id: len(trk.frames) for trk in tracks
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Stage2] Saved tracks.npz, {T} detection frames, masks, metadata.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser(description="Stage 2: Multi-Person Tracking")
    parser.add_argument("--video",        required=True)
    parser.add_argument("--stage1_dir",   required=True, help="Stage 1 output directory")
    parser.add_argument("--output",       required=True)
    parser.add_argument("--config",       default="configs/tracking.yaml")
    parser.add_argument("--max_frames",   type=int, default=None)
    args = parser.parse_args()

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # Load Stage 1 results from disk
    stage1_dir = Path(args.stage1_dir)
    cam = np.load(stage1_dir / "cameras.npz")
    T_cam = len(cam["poses"])

    depths = np.stack([
        np.load(stage1_dir / "depth_maps" / f"depth_{t:06d}.npy")
        for t in range(T_cam)
    ])

    camera_results = {
        "poses":      cam["poses"],
        "intrinsics": cam["intrinsics"],
        "confidence": cam["confidence"],
        "depths":     depths,
        "method_used": "vggt",
    }

    # Load frames
    cap = cv2.VideoCapture(args.video)
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if args.max_frames and len(frames_list) >= args.max_frames:
            break
    cap.release()
    frames = np.stack(frames_list)

    tracks = run_tracking(frames, camera_results, Path(args.output), cfg)
    print(f"[Stage2] Done — {len(tracks)} tracks.")
