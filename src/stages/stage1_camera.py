"""
Stage 1 — Camera Estimation (spec §2).

Sub-steps:
  A. Generalized dynamic masking (YOLOv8-x + SAM 2) → M_dyn(t)
  B. VGGT feed-forward camera estimation on masked frames (primary)
  C. Masked DROID-SLAM fallback (only if VGGT fails)

Output artifacts (relative to output_dir/):
  cameras.npz                        poses (T,4,4), intrinsics (T,3,3), confidence (T,)
  depth_maps/depth_{t:06d}.npy       float32 (H, W)
  point_maps/pmap_{t:06d}.npy        float32 (H, W, 3)
  track_features/feat_{t:06d}.npy    float32 (128, H, W)   [TODO: non-zero once wired]
  depth_uncertainty/sigma_{t:06d}.npy float32 (H, W)
  masks/frame_{t:06d}_dynamic.npy    bool (H, W)
  masks/frame_{t:06d}_instances.pkl  list[dict]
  metadata.json
"""
import json
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional

from src.models.vggt_wrapper import VGGTWrapper
from src.models.droid_slam_wrapper import DROIDSLAMWrapper
from src.utils.masking import generate_dynamic_masks, apply_mask_to_frames


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video(video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Load video frames as (T, H, W, 3) uint8 RGB array.
    """
    print(f"[Stage1] Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.1f}  |  Total frames: {total}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    arr = np.stack(frames, axis=0)
    print(f"  Loaded {len(arr)} frames {arr.shape[1]}×{arr.shape[2]}")
    return arr


# ---------------------------------------------------------------------------
# Save helpers (spec §2.5 file interface)
# ---------------------------------------------------------------------------

def save_results(results: Dict, output_dir: Path):
    """
    Save Stage 1 results to disk in the format specified by §2.5.

      cameras.npz            — poses, intrinsics, confidence
      depth_maps/            — per-frame depth  (6-digit index)
      point_maps/            — per-frame point maps
      track_features/        — per-frame dense features
      depth_uncertainty/     — per-frame 1-depth_conf
      metadata.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Stage1] Saving results to {output_dir}")

    poses      = results["poses"]        # (T, 4, 4)
    intrinsics = results["intrinsics"]   # (T, 3, 3)
    confidence = results["confidence"]   # (T,)
    depths     = results["depths"]       # (T, H, W)
    depth_conf = results.get("depth_conf", 1.0 - results.get("depth_uncertainty",
                              np.zeros_like(results["depths"])))
    world_pts  = results.get("world_points")   # (T, H, W, 3) or None
    track_feat = results.get("track_features") # (T, 128, H, W) or None

    # cameras.npz
    np.savez(
        output_dir / "cameras.npz",
        poses=poses,
        intrinsics=intrinsics,
        confidence=confidence,
    )

    T = len(poses)

    # depth_maps/depth_{t:06d}.npy
    dm_dir = output_dir / "depth_maps"
    dm_dir.mkdir(exist_ok=True)
    for t in range(T):
        np.save(dm_dir / f"depth_{t:06d}.npy", depths[t].astype(np.float32))

    # point_maps/pmap_{t:06d}.npy
    if world_pts is not None:
        pm_dir = output_dir / "point_maps"
        pm_dir.mkdir(exist_ok=True)
        for t in range(T):
            np.save(pm_dir / f"pmap_{t:06d}.npy", world_pts[t].astype(np.float32))

    # track_features/feat_{t:06d}.npy
    tf_dir = output_dir / "track_features"
    tf_dir.mkdir(exist_ok=True)
    if track_feat is not None:
        for t in range(T):
            np.save(tf_dir / f"feat_{t:06d}.npy", track_feat[t].astype(np.float32))

    # depth_uncertainty/sigma_{t:06d}.npy  (1 - depth_conf)
    du_dir = output_dir / "depth_uncertainty"
    du_dir.mkdir(exist_ok=True)
    if isinstance(depth_conf, np.ndarray) and depth_conf.ndim == 3:
        sigma = (1.0 - depth_conf).astype(np.float32)
    else:
        sigma = np.zeros_like(depths, dtype=np.float32)
    for t in range(T):
        np.save(du_dir / f"sigma_{t:06d}.npy", sigma[t])

    # metadata.json
    meta = {
        "num_frames":   T,
        "image_size":   list(results.get("original_size", [0, 0])),
        "method_used":  results.get("method_used", "unknown"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Stage1] Saved cameras.npz, {T} depth maps, {T} point maps, "
          f"{T} uncertainty maps, {T} feature maps.")


# ---------------------------------------------------------------------------
# Main stage function (spec §2.5 interface)
# ---------------------------------------------------------------------------

def run_camera_estimation(
    frames: np.ndarray,
    output_dir: Path,
    config: Dict,
    max_frames: Optional[int] = None,
) -> Dict:
    """
    Run Stage 1: Camera Estimation.

    Args:
        frames:     (T, H, W, 3) uint8 RGB.
        output_dir: Directory to write all artifacts.
        config:     Dict with keys (all optional, defaults shown):
                      method             "vggt" | "droid"  (default "vggt")
                      device             "cuda"
                      max_frames         200
                      fallback_to_droid  True
                      mask_classes       [0,1,2,3,5,7]
                      yolo_conf          0.3
                      sam_dilate_px      5
                      sam_model          "sam2_hiera_large.yaml"
                      vggt_model_path    None  (downloads from HF)
        max_frames: Deprecated — use config["max_frames"] instead.

    Returns:
        dict with keys: poses, intrinsics, depths, depth_conf, world_points,
                        track_features, confidence, method_used, original_size.
    """
    print("\n" + "=" * 80)
    print("STAGE 1: CAMERA ESTIMATION")
    print("=" * 80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_frames is not None:
        frames = frames[:max_frames]

    device = config.get("device", "cuda")
    method = config.get("method", "vggt")

    # ------------------------------------------------------------------
    # Sub-step A — Dynamic masking (spec §2.2)
    # ------------------------------------------------------------------
    print("[Stage1] Sub-step A: Generating dynamic masks...")
    try:
        dynamic_masks = generate_dynamic_masks(frames, config, output_dir=output_dir)
        masked_frames = apply_mask_to_frames(frames, dynamic_masks)
        print(f"[Stage1] Masking complete. "
              f"Mean masked fraction: "
              f"{np.mean([m.mean() for m in dynamic_masks])*100:.1f}%")
    except Exception as e:
        print(f"[Stage1] Warning: dynamic masking failed ({e}) — proceeding without masks.")
        masked_frames = frames
        dynamic_masks = [np.zeros(frames.shape[1:3], dtype=bool)] * len(frames)

    # ------------------------------------------------------------------
    # Sub-step B — VGGT (primary) or DROID-SLAM
    # ------------------------------------------------------------------
    t0 = time.time()

    try:
        if method in ("vggt", "auto"):
            print("[Stage1] Sub-step B: VGGT camera estimation...")
            estimator = VGGTWrapper(
                model_path=config.get("vggt_model_path", None),
                device=device,
                max_frames=config.get("max_frames", 200),
            )
            results = estimator.estimate_cameras(masked_frames)

            # Check VGGT confidence — fall back if too low (spec §2.4)
            conf = results["confidence"]
            low_conf_frac = float((conf < 0.3).mean())
            if low_conf_frac > 0.20 and config.get("fallback_to_droid", True):
                print(f"[Stage1] VGGT low confidence on {low_conf_frac*100:.1f}% of frames "
                      f"— triggering DROID-SLAM fallback.")
                raise RuntimeError("VGGT confidence below threshold")

        elif method == "droid":
            raise RuntimeError("DROID-SLAM forced by config")

        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        # ------------------------------------------------------------------
        # Sub-step C — Masked DROID-SLAM fallback (spec §2.4)
        # ------------------------------------------------------------------
        if method != "droid" and not config.get("fallback_to_droid", True):
            raise

        print(f"[Stage1] Sub-step C: DROID-SLAM fallback ({e})...")
        estimator = DROIDSLAMWrapper(
            weights_path=config.get("droid_weights_path", None),
            device=device,
            buffer_size=config.get("buffer_size", 512),
        )
        results = estimator.estimate_cameras(masked_frames)
        results["method_used"] = "droid"

    elapsed = time.time() - t0
    T = len(results["poses"])
    print(f"\n[Stage1] Camera estimation done in {elapsed:.2f}s  "
          f"({T / elapsed:.1f} fps)  method={results['method_used']}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_results(results, output_dir)
    _print_summary(results, elapsed)

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: Dict, elapsed: float):
    from scipy.spatial.transform import Rotation as Rscipy
    poses = results["poses"]
    T = len(poses)

    centers, euler = [], []
    for pose in poses:
        pi = np.linalg.inv(pose)
        centers.append(pi[:3, 3])
        euler.append(Rscipy.from_matrix(pi[:3, :3]).as_euler("xyz", degrees=True))

    centers = np.array(centers)
    euler   = np.array(euler)
    traj_len = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    total_rot = np.sum(np.abs(np.diff(euler, axis=0)), axis=0)

    print("\n" + "-" * 80)
    print("STAGE 1 SUMMARY")
    print("-" * 80)
    print(f"Frames:              {T}")
    print(f"Method:              {results.get('method_used', '?')}")
    print(f"Trajectory length:   {traj_len:.2f} m")
    print(f"Mean camera height:  {np.mean(centers[:, 1]):.2f} m")
    print(f"Total rotation:      Roll {total_rot[0]:.1f}°  "
          f"Pitch {total_rot[1]:.1f}°  Yaw {total_rot[2]:.1f}°")
    print(f"Processing time:     {elapsed:.2f} s  ({T/elapsed:.1f} fps)")
    print("-" * 80 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser(description="Stage 1: Camera Estimation")
    parser.add_argument("--video",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--config",     default="configs/vggt.yaml")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--method",     default="vggt", choices=["vggt", "droid"])
    args = parser.parse_args()

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    cfg["method"] = args.method

    frames = load_video(args.video, args.max_frames)
    run_camera_estimation(frames, Path(args.output), cfg)
    print("[Stage1] Done.")
