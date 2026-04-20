"""
PHALP+ Wrapper — primary tracker for Stage 2 (spec §3.2).

Calling convention: file-based (PHALP's native API is tracker.track(video_path)).
  - track_video(video_path) → per-frame detection dicts
  - track_frames(frames, tmp_dir) → writes temp video, then calls track_video

PHALP+ gives per-frame:
  track_id, bbox [x1,y1,x2,y2], smpl {poses,betas,transl},
  keypoints (24,3), embedding (4096,)

Install:
  pip install "phalp[all]@git+https://github.com/brjathu/PHALP.git"
"""
import json
import tempfile
import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PHALPDetection:
    """One person detection from PHALP+ for a single frame."""
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    keypoints_2d: np.ndarray  # (24, 3) — 2D joints + confidence
    smpl: Dict                # {"poses": (24,3), "betas": (10,), "transl": (3,)}
    embedding: np.ndarray     # (4096,) — ALPh joint embedding for re-ID


class PHALPWrapper:
    """
    Wrapper for PHALP+ (Rajasegaran et al., CVPR 2022).

    The spec uses `tracker.track(video_path)` — this wrapper exposes that
    interface and a convenience `track_frames` for numpy arrays.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        detector_backbone: str = "vitdet",
    ):
        """
        Args:
            model_path:         Optional path to custom PHALP weights.
            device:             "cuda" or "cpu".
            detector_backbone:  "vitdet" (stronger ViT detector, recommended
                                per spec §8.3) or "maskrcnn".
        """
        self.device = device
        self.model_path = model_path
        self.tracker = None
        self._init_model(detector_backbone)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_model(self, detector_backbone: str):
        import torch
        # PHALP checkpoints contain omegaconf/typing internals not in PyTorch
        # 2.6+'s default safe-globals list. Patch torch.load for the init
        # scope so every internal load call defaults weights_only=False, then
        # restore immediately after PHALP(cfg) returns.
        _orig_load = torch.load
        def _load_compat(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)
        torch.load = _load_compat
        try:
            from phalp.trackers.PHALP import PHALP
            from omegaconf import OmegaConf

            # Build minimal config; PHALP merges with its own defaults
            cfg = OmegaConf.create({
                "device": self.device,
                "phalp": {
                    "detector": detector_backbone,
                    "max_age_track": 30,
                    "track_history": 7,
                },
                "render": {"enable": False},
                "video": {"source": "", "output_dir": "outputs/"},
            })

            try:
                from phalp.configs.base import FullConfig
                from dataclasses import asdict
                defaults = OmegaConf.create(asdict(FullConfig()))
                cfg = OmegaConf.merge(defaults, cfg)
            except Exception:
                pass  # Use partial config if FullConfig unavailable

            self.tracker = PHALP(cfg)

            if self.model_path:
                ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.tracker.load_state_dict(ckpt, strict=False)
                print(f"[PHALPWrapper] Loaded custom weights: {self.model_path}")
            else:
                print("[PHALPWrapper] Using default PHALP+ weights.")

        except ImportError as e:
            print(f"[PHALPWrapper] PHALP not installed — {e}")
            print("  Install: pip install 'phalp[all]@git+https://github.com/brjathu/PHALP.git'")
        except Exception as e:
            print(f"[PHALPWrapper] Init failed — {e}")
        finally:
            torch.load = _orig_load

    # ------------------------------------------------------------------
    # Public API (spec §3.2)
    # ------------------------------------------------------------------

    def track_video(self, video_path: str) -> Dict[int, List[PHALPDetection]]:
        """
        Run PHALP+ on a video file.

        Args:
            video_path: Path to an MP4 / AVI video.

        Returns:
            detections_by_frame: dict mapping frame_id → List[PHALPDetection].
        """
        if self.tracker is None:
            print("[PHALPWrapper] Model not available — returning empty results.")
            return {}

        # PHALP.track() takes no arguments; video path is set via cfg.video.source
        self.tracker.cfg.video.source = video_path
        results = self.tracker.track()
        return self._parse_results(results)

    def track_frames(
        self,
        frames: np.ndarray,
        tmp_dir: Optional[Path] = None,
        fps: float = 30.0,
    ) -> Dict[int, List[PHALPDetection]]:
        """
        Convenience wrapper: write frames to a temp video then call track_video.

        Args:
            frames:  (T, H, W, 3) uint8 RGB.
            tmp_dir: Directory for temp file. Uses system tmp if None.
            fps:     Frame rate for the temp video (default 30).

        Returns:
            detections_by_frame: dict mapping frame_id → List[PHALPDetection].
        """
        if self.tracker is None:
            return {}

        use_tmp = tmp_dir is None
        ctx = tempfile.TemporaryDirectory() if use_tmp else None

        try:
            base = Path(ctx.name if ctx else tmp_dir)
            base.mkdir(parents=True, exist_ok=True)
            video_path = str(base / "phalp_input.mp4")
            self._write_video(frames, video_path, fps)
            return self.track_video(video_path)
        finally:
            if ctx is not None:
                ctx.cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_results(self, raw) -> Dict[int, List[PHALPDetection]]:
        """
        Parse PHALP+ tracker output into per-frame PHALPDetection lists.

        PHALP.track() may return:
          - tuple (results, extras) — unpack first element
          - list  — flat list of per-person dicts, each with a 'time' key
          - dict  — frame_key → list[per-person dict]
        Each per-person dict has keys: time, tid, bbox, conf, smpl, 2d_joints, ...
        """
        # Unpack tuple wrapper if present
        if isinstance(raw, tuple):
            raw = raw[0]

        # Normalise to a flat iterable of per-person dicts
        if isinstance(raw, list):
            person_dicts = raw
        elif isinstance(raw, dict):
            # Values may be lists of per-person dicts (keyed by frame name/id)
            # or the dict itself may be one per-person record (has 'tid' key)
            if "tid" in raw or "time" in raw:
                person_dicts = [raw]
            else:
                person_dicts = []
                for v in raw.values():
                    if isinstance(v, list):
                        person_dicts.extend(v)
                    elif isinstance(v, dict):
                        person_dicts.append(v)
        else:
            print(f"[PHALPWrapper] Unexpected result type: {type(raw)}")
            return {}

        print(f"[PHALPWrapper] Parsing {len(person_dicts)} person-frame entries")
        if person_dicts:
            d0 = person_dicts[0]
            print(f"[PHALPWrapper] Sample keys: {list(d0.keys())}")
            smpl0 = d0.get("smpl", None)
            print(f"[PHALPWrapper] smpl type={type(smpl0)}, "
                  f"keys={list(smpl0.keys()) if isinstance(smpl0, dict) else 'N/A'}")
            tids = {d.get("tid", d.get("track_id", "?")) for d in person_dicts if isinstance(d, dict)}
            print(f"[PHALPWrapper] Unique track IDs in results: {sorted(tids)}")
        out: Dict[int, List[PHALPDetection]] = {}

        for d in person_dicts:
            if not isinstance(d, dict):
                continue

            frame_id = int(d.get("time", -1))
            if frame_id < 0:
                continue

            # --- confidence: PHALP stores detection scores as a list ---
            conf_raw = d.get("conf", 1.0)
            if isinstance(conf_raw, (list, np.ndarray)):
                arr = np.asarray(conf_raw, dtype=np.float32).flatten()
                conf_val = float(arr[0]) if arr.size > 0 else 1.0
            else:
                conf_val = float(conf_raw)

            # --- bbox: may be (5,) with confidence appended ---
            bbox_raw = np.asarray(d.get("bbox", [0, 0, 1, 1]), dtype=np.float32).flatten()
            bbox = bbox_raw[:4]

            # --- 2D joints: key is '2d_joints', shape (J, 3) or (J*3,) ---
            joints_raw = d.get("2d_joints", d.get("keypoints_2d", np.zeros((24, 3))))
            joints = np.asarray(joints_raw, dtype=np.float32).reshape(-1, 3)
            if joints.shape[0] < 24:
                pad = np.zeros((24 - joints.shape[0], 3), dtype=np.float32)
                joints = np.concatenate([joints, pad], axis=0)
            joints = joints[:24]

            # --- SMPL: dict with standard keys or flat array ---
            smpl_raw = d.get("smpl", {})
            if isinstance(smpl_raw, dict):
                # Standard SMPL dict: global_orient + body_pose or full pose
                go   = np.asarray(smpl_raw.get("global_orient", smpl_raw.get("global_pose", np.zeros(3))),  dtype=np.float32).flatten()[:3]
                bp   = np.asarray(smpl_raw.get("body_pose",     smpl_raw.get("pose",        np.zeros(69))), dtype=np.float32).flatten()[:69]
                poses = np.concatenate([go, bp]).reshape(24, 3)  # (24, 3) incl. root
                betas = np.asarray(smpl_raw.get("betas",  np.zeros(10)), dtype=np.float32).flatten()[:10]
                transl = np.asarray(smpl_raw.get("transl", np.zeros(3)), dtype=np.float32).flatten()[:3]
            else:
                smpl_arr = np.asarray(smpl_raw, dtype=np.float32).flatten()
                poses  = smpl_arr[:72].reshape(24, 3) if smpl_arr.size >= 72 else np.zeros((24, 3), dtype=np.float32)
                betas  = smpl_arr[72:82]              if smpl_arr.size >= 82 else np.zeros(10,       dtype=np.float32)
                transl = np.asarray(d.get("camera_bbox", np.zeros(3)), dtype=np.float32).flatten()[:3]

            # --- embedding: stored in extra_data or directly ---
            extra = d.get("extra_data", {}) or {}
            emb_raw = extra.get("embedding", d.get("embedding", np.zeros(4096)))
            embedding = np.asarray(emb_raw, dtype=np.float32).flatten()
            if embedding.size < 4096:
                embedding = np.zeros(4096, dtype=np.float32)

            det = PHALPDetection(
                track_id=int(d.get("tid", d.get("track_id", -1))),
                bbox=bbox,
                confidence=conf_val,
                keypoints_2d=joints,
                smpl={"poses": poses, "betas": betas, "transl": transl},
                embedding=embedding,
            )

            out.setdefault(frame_id, []).append(det)

        return out

    @staticmethod
    def _write_video(frames: np.ndarray, path: str, fps: float):
        T, H, W = frames.shape[:3]
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
