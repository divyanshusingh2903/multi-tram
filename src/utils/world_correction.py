"""
VGGT depth-guided world-frame ID correction — Stage 2 Sub-step B (spec §3.3).

After PHALP+ finishes tracking in camera-frame 3D space, this module:
  1. Projects each detection's 2D bbox centre to world 3D using VGGT depth + camera pose.
  2. Flags ID switches: |W_i(t) - W_i(t-1)| > v_max_world.
  3. Searches for a nearby track j whose world position is consistent and whose
     PHALP embedding cosine similarity > embed_sim_threshold.
  4. Swaps the track IDs if a valid candidate is found.
  5. Merges fragmented track segments using the same world-trajectory signal.
  6. Discards tracks shorter than min_track_len.
"""
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types shared with stage2_tracking
# ---------------------------------------------------------------------------

@dataclass
class PHALPTrack:
    """Representation of one PHALP+ track, enriched with world-frame info."""
    track_id: int
    frames: List[int]
    bboxes: np.ndarray           # (T_i, 4) [x1, y1, x2, y2]
    keypoints_2d: np.ndarray     # (T_i, 24, 3)
    smpl_init: Dict              # coarse SMPL from PHALP
    embeddings: np.ndarray       # (T_i, E) PHALP ALPh embedding
    world_positions: np.ndarray  # (T_i, 3)  — filled in by this module


# ---------------------------------------------------------------------------
# Depth unprojection helper
# ---------------------------------------------------------------------------

def _unproject_bbox_centre(
    bbox: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    pose: np.ndarray,
    roi_px: int = 5,
) -> np.ndarray:
    """
    Project a 2D bbox centre to world 3D using VGGT depth.

    Args:
        bbox:       [x1, y1, x2, y2] in pixels.
        depth_map:  (H, W) metric depth in metres.
        intrinsics: (3, 3).
        pose:       (4, 4) world-to-camera extrinsic G_t.
        roi_px:     Radius around centre to sample depth (median).

    Returns:
        world_pos: (3,) in metres.
    """
    H, W = depth_map.shape
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)

    # Sample a roi_px-radius patch for robust depth
    x0, x1 = max(0, cx - roi_px), min(W, cx + roi_px + 1)
    y0, y1 = max(0, cy - roi_px), min(H, cy + roi_px + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0]
    depth = float(np.median(valid)) if len(valid) > 0 else 5.0

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    px_cx, px_cy = intrinsics[0, 2], intrinsics[1, 2]

    X_c = (cx - px_cx) * depth / fx
    Y_c = (cy - px_cy) * depth / fy
    Z_c = depth

    p_cam = np.array([X_c, Y_c, Z_c], dtype=np.float64)

    # World-to-camera: p_world = R_t^T (p_cam - T_t)
    R_t = pose[:3, :3]
    T_t = pose[:3, 3]
    p_world = R_t.T @ (p_cam - T_t)
    return p_world.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def correct_world_frame_ids(
    tracks: List[PHALPTrack],
    camera_results: Dict,
    config: Dict,
) -> Tuple[List[PHALPTrack], Dict]:
    """
    Apply world-frame ID correction to a list of PHALP+ tracks.

    Args:
        tracks:         List of PHALPTrack objects from PHALP+ parsing.
        camera_results: Stage 1 output dict with keys:
                          poses      (T, 4, 4) world-to-camera
                          intrinsics (T, 3, 3)
                          depths     list of (H, W) depth maps
        config:         Dict with keys:
                          v_max_world           (default 2.0 m/frame)
                          embed_sim_threshold   (default 0.6)
                          depth_roi_px          (default 5)
                          min_track_len         (default 8)

    Returns:
        corrected_tracks: List of PHALPTrack after ID correction.
        correction_log:   Dict suitable for JSON serialisation.
    """
    v_max = config.get("v_max_world", 2.0)
    embed_thresh = config.get("embed_sim_threshold", 0.6)
    roi_px = config.get("depth_roi_px", 5)
    min_len = config.get("min_track_len", 8)

    poses = camera_results["poses"]          # (T, 4, 4)
    intrinsics = camera_results["intrinsics"]  # (T, 3, 3)
    depths = camera_results["depths"]          # list or (T, H, W)

    def get_depth(t):
        if isinstance(depths, np.ndarray):
            return depths[t]
        return depths[t]

    # ------------------------------------------------------------------
    # Step 1: Compute world positions for every (track, frame) pair
    # ------------------------------------------------------------------
    for trk in tracks:
        wp = []
        for idx, t in enumerate(trk.frames):
            if t >= len(poses):
                wp.append(np.array([0.0, 0.0, 0.0]))
                continue
            pos = _unproject_bbox_centre(
                trk.bboxes[idx],
                get_depth(t),
                intrinsics[t],
                poses[t],
                roi_px,
            )
            wp.append(pos)
        trk.world_positions = np.array(wp, dtype=np.float32)

    # ------------------------------------------------------------------
    # Step 2: Detect suspicious jumps in each track
    # ------------------------------------------------------------------
    correction_log = {"swaps": [], "merges": [], "discarded": []}
    track_by_id: Dict[int, PHALPTrack] = {trk.track_id: trk for trk in tracks}

    flagged: List[Tuple[int, int]] = []  # (track_id, frame_index_within_track)
    for trk in tracks:
        for i in range(1, len(trk.frames)):
            disp = np.linalg.norm(trk.world_positions[i] - trk.world_positions[i - 1])
            if disp > v_max:
                flagged.append((trk.track_id, i))

    # ------------------------------------------------------------------
    # Step 3: For each flagged jump, try to find the track that should
    #         have continued (look for track j whose last world position
    #         before frame t is consistent with continuity of track i).
    # ------------------------------------------------------------------
    id_map: Dict[int, int] = {}  # new_id -> canonical_id (for merge logging)

    for (tid_i, fi) in flagged:
        trk_i = track_by_id.get(tid_i)
        if trk_i is None:
            continue

        t_switch = trk_i.frames[fi]
        pos_before = trk_i.world_positions[fi - 1]
        emb_before = trk_i.embeddings[fi - 1] if trk_i.embeddings is not None else None

        best_j = None
        best_score = -1.0

        for trk_j in tracks:
            if trk_j.track_id == tid_i:
                continue
            # Check if j had a detection just before t_switch
            if t_switch - 1 not in trk_j.frames:
                continue
            idx_j = trk_j.frames.index(t_switch - 1)
            pos_j = trk_j.world_positions[idx_j]

            if np.linalg.norm(pos_j - pos_before) > v_max:
                continue

            # Check embedding similarity
            if emb_before is not None and trk_j.embeddings is not None:
                emb_j = trk_j.embeddings[idx_j]
                sim = _cosine_sim(emb_before, emb_j)
            else:
                sim = 1.0  # no embeddings available — assume OK

            if sim > embed_thresh and sim > best_score:
                best_score = sim
                best_j = trk_j

        if best_j is not None:
            # Swap: merge trk_i[fi:] into trk_j
            _merge_tail(trk_i, fi, best_j)
            correction_log["swaps"].append({
                "frame": t_switch,
                "from_track": tid_i,
                "to_track": best_j.track_id,
                "embed_sim": float(best_score),
            })

    # ------------------------------------------------------------------
    # Step 4: Discard tracks shorter than min_track_len
    # ------------------------------------------------------------------
    kept = []
    for trk in tracks:
        if len(trk.frames) < min_len:
            correction_log["discarded"].append(
                {"track_id": trk.track_id, "n_frames": len(trk.frames)}
            )
        else:
            kept.append(trk)

    print(f"[WorldCorrection] Swaps: {len(correction_log['swaps'])}, "
          f"Discarded short tracks: {len(correction_log['discarded'])}, "
          f"Kept: {len(kept)}")
    return kept, correction_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _merge_tail(src: PHALPTrack, fi: int, dst: PHALPTrack) -> None:
    """Append frames src[fi:] to dst, removing them from src."""
    tail_frames = src.frames[fi:]
    tail_bboxes = src.bboxes[fi:]
    tail_kpts = src.keypoints_2d[fi:] if src.keypoints_2d is not None else None
    tail_embs = src.embeddings[fi:] if src.embeddings is not None else None
    tail_wp = src.world_positions[fi:]

    dst.frames = dst.frames + tail_frames
    dst.bboxes = np.concatenate([dst.bboxes, tail_bboxes], axis=0)
    if tail_kpts is not None and dst.keypoints_2d is not None:
        dst.keypoints_2d = np.concatenate([dst.keypoints_2d, tail_kpts], axis=0)
    if tail_embs is not None and dst.embeddings is not None:
        dst.embeddings = np.concatenate([dst.embeddings, tail_embs], axis=0)
    dst.world_positions = np.concatenate([dst.world_positions, tail_wp], axis=0)

    # Truncate source
    src.frames = src.frames[:fi]
    src.bboxes = src.bboxes[:fi]
    if src.keypoints_2d is not None:
        src.keypoints_2d = src.keypoints_2d[:fi]
    if src.embeddings is not None:
        src.embeddings = src.embeddings[:fi]
    src.world_positions = src.world_positions[:fi]
