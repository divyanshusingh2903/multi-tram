"""
VIMO Wrapper for Per-Person Pose Estimation
Wraps HMR_VIMO for temporally coherent SMPL estimation.

The correct inference API is:
    results = model.inference(imgfiles, boxes, img_focal, img_center)
where:
    imgfiles  – list of image file paths (one per frame)
    boxes     – (N, 4) float array [x1, y1, x2, y2] in pixel coords
    img_focal – scalar focal length (pixels); if None, estimated from image size
    img_center – [cx, cy] principal point; if None, estimated from image size

Output keys: pred_rotmat (N,24,3,3), pred_shape (N,10), pred_trans (N,1,3), frame (N,)
"""
import torch
import numpy as np
from pathlib import Path
import sys
import types
from typing import Dict, List, Optional
from dataclasses import dataclass


def _inject_tram_stubs():
    """
    tools.py in thirdparty/tram imports segment_anything, deva, and detectron2
    utilities at module level, but hmr_vimo.py only uses parse_chunks from it.
    Inject a minimal stub so the import chain resolves without those packages.
    Must be called AFTER tram is on sys.path.
    """
    if 'lib.pipeline.tools' in sys.modules:
        return  # already loaded (real or stub)

    import numpy as _np

    def parse_chunks(frame, boxes, min_len=16):
        frame_chunks, boxes_chunks = [], []
        step = frame[1:] - frame[:-1]
        step = _np.concatenate([[0], step])
        breaks = _np.where(step != 1)[0]
        start = 0
        for bk in breaks:
            f_chunk, b_chunk = frame[start:bk], boxes[start:bk]
            start = bk
            if len(f_chunk) >= min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)
            if bk == breaks[-1]:
                f_chunk, b_chunk = frame[bk:], boxes[bk:]
                if len(f_chunk) >= min_len:
                    frame_chunks.append(f_chunk)
                    boxes_chunks.append(b_chunk)
        return frame_chunks, boxes_chunks

    stub = types.ModuleType('lib.pipeline.tools')
    stub.parse_chunks = parse_chunks
    sys.modules['lib.pipeline.tools'] = stub

    # Also stub the parent package if not present
    if 'lib.pipeline' not in sys.modules:
        pkg = types.ModuleType('lib.pipeline')
        pkg.tools = stub
        sys.modules['lib.pipeline'] = pkg


@dataclass
class VIMOOutput:
    """VIMO prediction output"""
    poses: np.ndarray          # (T, 23, 3) body pose axis-angle
    betas: np.ndarray          # (10,) body shape
    global_orient: np.ndarray  # (T, 3) root orientation axis-angle
    transl: np.ndarray         # (T, 3) root translation
    joints: Optional[np.ndarray] = None    # (T, 24, 3) 3D joints
    vertices: Optional[np.ndarray] = None  # (T, 6890, 3) mesh vertices
    confidence: Optional[np.ndarray] = None  # (T,) per-frame confidence


class VIMOWrapper:
    """
    Wrapper for HMR_VIMO (from thirdparty/tram).

    Primary entry point: predict_from_files()
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        backbone: str = 'vit_huge',
        num_frames: int = 16,
        temporal_model: str = 'transformer'
    ):
        self.device = device
        self.model_path = model_path
        self.backbone = backbone
        self.num_frames = num_frames
        self.temporal_model = temporal_model
        self.model = None

        # Add tram lib to path so `from lib.models import ...` works
        tram_path = Path(__file__).parent.parent.parent / 'thirdparty' / 'tram'
        if str(tram_path) not in sys.path:
            sys.path.insert(0, str(tram_path))

        self._init_model()

    def _init_model(self):
        """Initialize HMR_VIMO model."""
        try:
            _inject_tram_stubs()
            from lib.models import get_hmr_vimo

            # Resolve relative paths against project root
            project_root = Path(__file__).parent.parent.parent
            checkpoint = None
            if self.model_path:
                p = Path(self.model_path)
                if not p.is_absolute():
                    p = project_root / p
                checkpoint = str(p) if p.exists() else None
                if not p.exists():
                    print(f"[VIMOWrapper] Checkpoint not found: {p}")
            self.model = get_hmr_vimo(checkpoint=checkpoint, device=self.device)
            print(f"[VIMOWrapper] HMR-VIMO loaded"
                  + (f" from {checkpoint}" if checkpoint else " (no checkpoint)"))

        except ImportError as e:
            print(f"[VIMOWrapper] VIMO not importable ({e}), using dummy model")
            self.model = None
        except Exception as e:
            print(f"[VIMOWrapper] Error loading VIMO: {e}")
            self.model = None

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def predict_from_files(
        self,
        imgfiles: List[str],
        boxes: np.ndarray,
        img_focal: Optional[float] = None,
        img_center: Optional[np.ndarray] = None,
        frame_indices: Optional[np.ndarray] = None,
        valid: Optional[np.ndarray] = None,
    ) -> VIMOOutput:
        """
        Run VIMO inference given image file paths and bounding boxes.

        Args:
            imgfiles:      List of per-frame image file paths (length T).
            boxes:         (T, 4) float array [x1, y1, x2, y2].
            img_focal:     Scalar focal length in pixels (None → estimated per frame).
            img_center:    [cx, cy] principal point (None → estimated per frame).
            frame_indices: (T,) int array mapping list position → video frame id.
                           Defaults to np.arange(T).
            valid:         (T,) bool mask selecting frames to use (None → all).

        Returns:
            VIMOOutput with SMPL parameters.
        """
        T = len(imgfiles)
        if self.model is None:
            return self._predict_dummy(T)

        if frame_indices is None:
            frame_indices = np.arange(T)

        try:
            with torch.no_grad():
                results = self.model.inference(
                    imgfiles=imgfiles,
                    boxes=boxes,
                    img_focal=img_focal,
                    img_center=img_center,
                    valid=valid,
                    frame=frame_indices,
                    device=self.device,
                )

            if results is None:
                print("[VIMOWrapper] model.inference returned None "
                      "(fewer than 16 valid frames?)")
                return self._predict_dummy(T)

            return self._parse_inference_results(results, T)

        except Exception as e:
            print(f"[VIMOWrapper] inference failed: {e}")
            import traceback; traceback.print_exc()
            return self._predict_dummy(T)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_inference_results(self, results: Dict, T: int) -> VIMOOutput:
        """Convert raw model output to VIMOOutput."""
        pred_rotmat = results['pred_rotmat'].cpu().numpy()  # (N, 24, 3, 3)
        pred_shape  = results['pred_shape'].cpu().numpy()   # (N, 10)
        pred_trans  = results['pred_trans'].cpu().numpy()   # (N, 1, 3)
        frame_idx   = results['frame'].cpu().numpy()        # (N,) indices into imgfiles

        N = len(frame_idx)

        # Convert rotation matrices to axis-angle
        global_orient_all = self._rotmat_to_aa(pred_rotmat[:, 0:1, :, :])   # (N, 1, 3)
        body_poses_all    = self._rotmat_to_aa(pred_rotmat[:, 1:, :, :])     # (N, 23, 3)
        transl_all        = pred_trans[:, 0, :]                               # (N, 3)

        # Scatter back into T-length arrays (frames not in frame_idx stay zero)
        global_orient = np.zeros((T, 3), dtype=np.float32)
        body_poses    = np.zeros((T, 23, 3), dtype=np.float32)
        transl        = np.zeros((T, 3), dtype=np.float32)
        confidence    = np.zeros(T, dtype=np.float32)

        for out_i, t in enumerate(frame_idx):
            if t < T:
                global_orient[t] = global_orient_all[out_i, 0]
                body_poses[t]    = body_poses_all[out_i]
                transl[t]        = transl_all[out_i]
                confidence[t]    = 1.0

        # Mean shape across sequence
        betas = pred_shape.mean(axis=0)  # (10,)

        return VIMOOutput(
            poses=body_poses,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
            confidence=confidence,
        )

    @staticmethod
    def _rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrices to axis-angle representation.

        Args:
            rotmat: (..., 3, 3)

        Returns:
            axis-angle: (..., 3)
        """
        try:
            from scipy.spatial.transform import Rotation as ScipyR
            orig_shape = rotmat.shape[:-2]
            flat = rotmat.reshape(-1, 3, 3)
            aa = ScipyR.from_matrix(flat).as_rotvec()
            return aa.reshape(*orig_shape, 3).astype(np.float32)
        except ImportError:
            # Manual Rodrigues fallback
            orig_shape = rotmat.shape[:-2]
            flat = rotmat.reshape(-1, 3, 3)
            aa = np.zeros((len(flat), 3), dtype=np.float32)
            for i, R in enumerate(flat):
                angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
                if abs(angle) < 1e-6:
                    aa[i] = 0.0
                else:
                    vec = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
                    aa[i] = angle / (2 * np.sin(angle)) * vec
            return aa.reshape(*orig_shape, 3)

    def _predict_dummy(self, num_frames: int) -> VIMOOutput:
        """Return zero SMPL predictions (fallback when model unavailable)."""
        return VIMOOutput(
            poses=np.zeros((num_frames, 23, 3), dtype=np.float32),
            betas=np.zeros(10, dtype=np.float32),
            global_orient=np.zeros((num_frames, 3), dtype=np.float32),
            transl=np.zeros((num_frames, 3), dtype=np.float32),
            confidence=np.zeros(num_frames, dtype=np.float32),
        )


class VIMOBatchPredictor:
    """Batch predictor for multiple people (convenience wrapper)."""

    def __init__(self, vimo_model: VIMOWrapper):
        self.vimo = vimo_model

    def predict_batch(
        self,
        person_imgfiles_list: List[List[str]],
        person_boxes_list: List[np.ndarray],
        img_focal: Optional[float] = None,
        img_center: Optional[np.ndarray] = None,
    ) -> List[VIMOOutput]:
        results = []
        for imgfiles, boxes in zip(person_imgfiles_list, person_boxes_list):
            out = self.vimo.predict_from_files(
                imgfiles=imgfiles,
                boxes=boxes,
                img_focal=img_focal,
                img_center=img_center,
            )
            results.append(out)
        return results
