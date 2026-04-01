"""
VGGT Wrapper for Camera Estimation — Stage 1 primary method (spec §2.3).

Correct VGGT API (confirmed from thirdparty/vggt demo scripts):
  - Input:  images (S, 3, 518, 518) float32, ImageNet-normalised, no batch dim.
            Model internally handles the batch dimension.
  - Output: dict with keys pose_enc (1,S,9), depth (1,S,H,W,1),
            depth_conf (1,S,H,W), world_points (1,S,H,W,3).

Preprocessing: resize to 518×518, normalise with ImageNet mean/std.
Chunking: simple concatenation for sequences > max_frames.
  TODO: replace with PnP-based chunk alignment using overlapping keyframes.
  Until then, keep max_frames=200 so chunking only triggers for long sequences.
"""
import sys
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ImageNet statistics (matches VGGT's load_and_preprocess_images)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_VGGT_IMG_SIZE = 518  # hardcoded in pretrained model architecture


class VGGTWrapper:
    """
    Wrapper for VGGT (Visual Geometry Grounded Transformer).

    Returns metric-scale camera poses, depth maps, point maps, and
    depth-confidence maps from a single feed-forward pass.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        max_frames: int = 200,
        image_size: int = None,  # kept for API compatibility, always 518
    ):
        """
        Args:
            model_path: Local .pt checkpoint. If None, downloads from HuggingFace.
            device:     "cuda" or "cpu".
            max_frames: Frames per chunk (spec default: 200).
            image_size: Deprecated — VGGT always uses 518×518.
        """
        self.device = device
        self.max_frames = max_frames
        self.image_size = _VGGT_IMG_SIZE

        if image_size is not None and image_size != _VGGT_IMG_SIZE:
            print(f"[VGGT] Warning: image_size={image_size} ignored; model requires 518×518.")

        # Add VGGT to import path
        vggt_root = Path(__file__).parent.parent.parent / "thirdparty" / "vggt"
        if str(vggt_root) not in sys.path:
            sys.path.insert(0, str(vggt_root))

        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        print(f"[VGGT] Initialising model (img_size=518)...")
        self.model = VGGT(img_size=518).to(device)
        self._pose_dec = pose_encoding_to_extri_intri

        if model_path is not None:
            p = Path(model_path)
            if not p.exists():
                raise FileNotFoundError(f"VGGT checkpoint not found: {model_path}")
            print(f"[VGGT] Loading from local file: {model_path}")
            state = torch.load(model_path, map_location=device)
        else:
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            print("[VGGT] Downloading checkpoint from HuggingFace Hub...")
            state = torch.hub.load_state_dict_from_url(_URL, map_location=device)

        self.model.load_state_dict(state)
        self.model.eval()
        print("[VGGT] Model ready.")

        # Choose dtype based on GPU capability
        if device == "cuda" and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()[0]
            self._dtype = torch.bfloat16 if cap >= 8 else torch.float16
        else:
            self._dtype = torch.float32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_cameras(self, frames: np.ndarray) -> Dict:
        """
        Estimate camera poses and auxiliary outputs from video frames.

        Args:
            frames: (T, H, W, 3) uint8 RGB.

        Returns:
            dict with keys:
              poses          (T, 4, 4) world-to-camera extrinsics
              intrinsics     (T, 3, 3) camera intrinsics (scaled to original res)
              depths         (T, H, W) metric depth in metres
              depth_conf     (T, H, W) depth confidence [0, 1]
              world_points   (T, H, W, 3) 3D point map in world frame
              track_features (T, C, H, W) dense tracking features  [TODO: stub]
              confidence     (T,)  per-frame confidence (mean depth_conf)
              original_size  (H, W) tuple
              method_used    "vggt"
        """
        T = len(frames)
        print(f"[VGGT] Processing {T} frames...")

        if T > self.max_frames:
            print(f"[VGGT] Chunking into segments of {self.max_frames} frames.")
            return self._process_chunks(frames)

        return self._run_single_chunk(frames)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """
        (T, H, W, 3) uint8 → (T, 3, 518, 518) float32 on device, ImageNet-normalised.
        """
        T = len(frames)
        out = np.empty((T, _VGGT_IMG_SIZE, _VGGT_IMG_SIZE, 3), dtype=np.float32)
        for i, frame in enumerate(frames):
            resized = cv2.resize(frame, (_VGGT_IMG_SIZE, _VGGT_IMG_SIZE))
            out[i] = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        # (T, H, W, 3) → (T, 3, H, W)
        tensor = torch.from_numpy(out).permute(0, 3, 1, 2).to(self.device)
        return tensor

    def _run_single_chunk(self, frames: np.ndarray) -> Dict:
        H_orig, W_orig = frames.shape[1], frames.shape[2]
        images = self._preprocess(frames)   # (T, 3, 518, 518)

        with torch.amp.autocast(device_type=self.device if self.device == "cuda" else "cpu",
                                dtype=self._dtype):
            preds = self.model(images)      # model handles batch dim internally

        # --- Pose decoding ---
        pose_enc = preds["pose_enc"]  # may be (S, 9) or (1, S, 9)
        if pose_enc.dim() == 2:
            pose_enc = pose_enc.unsqueeze(0)  # → (1, S, 9)

        extrinsics, intrinsics_vggt = self._pose_dec(
            pose_enc, image_size_hw=(_VGGT_IMG_SIZE, _VGGT_IMG_SIZE)
        )
        # extrinsics: (1, S, 3, 4) → (S, 3, 4)
        ext_np = extrinsics.squeeze(0).cpu().float().numpy()
        intr_np = intrinsics_vggt.squeeze(0).cpu().float().numpy()  # (S, 3, 3)

        S = ext_np.shape[0]
        poses = np.zeros((S, 4, 4), dtype=np.float32)
        poses[:, :3, :] = ext_np
        poses[:, 3, 3] = 1.0

        # Scale intrinsics from 518×518 back to original resolution
        sx, sy = W_orig / _VGGT_IMG_SIZE, H_orig / _VGGT_IMG_SIZE
        intr_scaled = intr_np.copy()
        intr_scaled[:, 0, 0] *= sx; intr_scaled[:, 0, 2] *= sx
        intr_scaled[:, 1, 1] *= sy; intr_scaled[:, 1, 2] *= sy

        # --- Depth ---
        # shape: (1, S, H, W, 1) or (S, H, W, 1)
        depth_raw = preds["depth"]
        if depth_raw.dim() == 5:
            depth_raw = depth_raw.squeeze(0)   # (S, H, W, 1)
        depth_np = depth_raw.squeeze(-1).cpu().float().numpy()   # (S, H518, W518)
        depth_np = self._resize_maps(depth_np, H_orig, W_orig)   # (S, H, W)

        # --- Depth confidence ---
        dconf_raw = preds["depth_conf"]
        if dconf_raw.dim() == 4:
            dconf_raw = dconf_raw.squeeze(0)   # (S, H, W)
        dconf_np = dconf_raw.cpu().float().numpy()
        dconf_np = self._resize_maps(dconf_np, H_orig, W_orig)   # (S, H, W)

        # --- World points (point maps) ---
        wp_raw = preds["world_points"]
        if wp_raw.dim() == 5:
            wp_raw = wp_raw.squeeze(0)   # (S, H, W, 3)
        wp_np = wp_raw.cpu().float().numpy()   # (S, H518, W518, 3)
        # resize along spatial dims
        wp_resized = np.stack([
            cv2.resize(wp_np[t], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            for t in range(S)
        ])  # (S, H, W, 3)

        # --- Track features (spec §2.3 T_i) ---
        # TODO: extract DPT intermediate features from TrackHead.feature_extractor.
        # For now, save a zero-filled placeholder so downstream code can run.
        # Shape should be (T, 128, H, W) once implemented.
        track_features = np.zeros((S, 128, H_orig, W_orig), dtype=np.float32)

        # --- Per-frame confidence (mean depth_conf) ---
        confidence = dconf_np.mean(axis=(1, 2))  # (S,)

        return {
            "poses": poses,
            "intrinsics": intr_scaled,
            "depths": depth_np,
            "depth_conf": dconf_np,
            "world_points": wp_resized,
            "track_features": track_features,
            "confidence": confidence,
            "original_size": (H_orig, W_orig),
            "method_used": "vggt",
        }

    def _process_chunks(self, frames: np.ndarray) -> Dict:
        """
        Process a long video in non-overlapping chunks of max_frames.

        TODO: Replace the simple concatenation below with PnP-based chunk alignment
        using overlapping keyframes to remove the coordinate discontinuity at chunk
        boundaries. Acceptable limitation for sequences < 200 frames where chunking
        is not triggered.
        """
        all_poses, all_intr, all_depths, all_dconf = [], [], [], []
        all_wp, all_tf, all_conf = [], [], []

        for start in range(0, len(frames), self.max_frames):
            chunk = frames[start:start + self.max_frames]
            print(f"[VGGT] Chunk {start}–{start + len(chunk) - 1}")
            res = self._run_single_chunk(chunk)

            all_poses.append(res["poses"])
            all_intr.append(res["intrinsics"])
            all_depths.append(res["depths"])
            all_dconf.append(res["depth_conf"])
            all_wp.append(res["world_points"])
            all_tf.append(res["track_features"])
            all_conf.append(res["confidence"])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        H, W = frames.shape[1], frames.shape[2]
        return {
            "poses": np.concatenate(all_poses, axis=0),
            "intrinsics": np.concatenate(all_intr, axis=0),
            "depths": np.concatenate(all_depths, axis=0),
            "depth_conf": np.concatenate(all_dconf, axis=0),
            "world_points": np.concatenate(all_wp, axis=0),
            "track_features": np.concatenate(all_tf, axis=0),
            "confidence": np.concatenate(all_conf, axis=0),
            "original_size": (H, W),
            "method_used": "vggt",
        }

    @staticmethod
    def _resize_maps(maps: np.ndarray, H: int, W: int) -> np.ndarray:
        """Resize (S, h, w) or (S, h, w, C) maps to (S, H, W[, C])."""
        S = maps.shape[0]
        if maps.shape[1] == H and maps.shape[2] == W:
            return maps
        out = []
        for t in range(S):
            out.append(cv2.resize(maps[t], (W, H), interpolation=cv2.INTER_LINEAR))
        return np.stack(out)
