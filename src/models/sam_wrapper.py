"""
SAM 2 Wrapper for instance segmentation.

Used in:
- Stage 1: dynamic object masking (one mask per YOLO bbox, merged → M_dyn)
- Stage 2: per-person masks for VIMO crop preparation

Install SAM 2:
    pip install git+https://github.com/facebookresearch/sam2.git

Falls back to dilated-bbox masks when SAM 2 is not installed.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class SAMWrapper:
    """
    Thin wrapper around SAM 2's image predictor.

    Accepts bounding boxes in [x1, y1, x2, y2] format and returns binary
    instance masks of shape (H, W).
    """

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_large.yaml",
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        dilate_px: int = 5,
    ):
        """
        Args:
            model_cfg:   SAM 2 config name (e.g. "sam2_hiera_large.yaml").
            checkpoint:  Path to SAM 2 checkpoint (.pt). If None uses default.
            device:      "cuda" or "cpu".
            dilate_px:   Pixels to dilate each mask after prediction (spec: 5 px).
        """
        self.device = device
        self.dilate_px = dilate_px
        self.predictor = None

        if SAM2_AVAILABLE:
            try:
                if checkpoint is None:
                    # Try common download locations
                    default_ckpt = Path("checkpoints/sam2/sam2_hiera_large.pt")
                    if default_ckpt.exists():
                        checkpoint = str(default_ckpt)

                model = build_sam2(model_cfg, checkpoint, device=device)
                self.predictor = SAM2ImagePredictor(model)
                print("[SAMWrapper] SAM 2 loaded successfully")
            except Exception as e:
                print(f"[SAMWrapper] Warning: could not load SAM 2 — {e}")
                print("[SAMWrapper] Falling back to dilated-bbox masks")
        else:
            print("[SAMWrapper] SAM 2 not installed — using dilated-bbox fallback")
            print("  Install: pip install git+https://github.com/facebookresearch/sam2.git")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Segment one image given a set of bounding boxes.

        Args:
            image: (H, W, 3) uint8 RGB.
            boxes: (N, 4) float32 [x1, y1, x2, y2].

        Returns:
            masks: List of N binary (H, W) bool arrays, one per box.
                   Uses SAM 2 when available, dilated bbox otherwise.
        """
        if len(boxes) == 0:
            return []

        H, W = image.shape[:2]

        if self.predictor is not None:
            return self._sam2_segment(image, boxes)
        else:
            return self._bbox_fallback(H, W, boxes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sam2_segment(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[np.ndarray]:
        import torch

        self.predictor.set_image(image)
        masks_out = []

        boxes_tensor = torch.from_numpy(boxes).float().to(self.device)
        with torch.no_grad():
            # SAM 2 accepts batched boxes
            raw_masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_tensor,
                multimask_output=False,
            )
        # raw_masks: (N, 1, H, W) or (N, H, W)
        if raw_masks.ndim == 4:
            raw_masks = raw_masks[:, 0]

        for mask in raw_masks:
            binary = mask.astype(bool)
            if self.dilate_px > 0:
                binary = self._dilate(binary, self.dilate_px)
            masks_out.append(binary)

        return masks_out

    def _bbox_fallback(
        self,
        H: int,
        W: int,
        boxes: np.ndarray,
    ) -> List[np.ndarray]:
        """Create dilated bounding-box masks when SAM 2 is unavailable."""
        masks = []
        d = self.dilate_px
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            mask = np.zeros((H, W), dtype=bool)
            y1c = max(0, y1 - d)
            y2c = min(H, y2 + d)
            x1c = max(0, x1 - d)
            x2c = min(W, x2 + d)
            mask[y1c:y2c, x1c:x2c] = True
            masks.append(mask)
        return masks

    @staticmethod
    def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)
        )
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
