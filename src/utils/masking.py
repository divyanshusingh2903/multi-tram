"""
Dynamic object masking — Stage 1 Sub-step A (spec §2.2).

Pipeline per frame:
  1. YOLOv8-x detects all dynamic-class objects (persons, vehicles, cyclists…).
  2. SAM 2 produces pixel-accurate masks for each detection bbox.
     Falls back to dilated-bbox masks when SAM 2 is unavailable.
  3. All instance masks are unioned into a single binary M_dyn(t).

Output per frame:
  masks/frame_{t:06d}_dynamic.npy   bool (H, W)   — union mask
  masks/frame_{t:06d}_instances.pkl list[dict]     — per-instance info
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from src.models.sam_wrapper import SAMWrapper

# COCO class IDs that represent dynamic / moving objects (spec §2.2)
DEFAULT_DYNAMIC_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorbike, bus, truck


def generate_dynamic_masks(
    frames: np.ndarray,
    config: Dict,
    output_dir: Optional[Path] = None,
) -> List[np.ndarray]:
    """
    Generate per-frame binary union masks for all dynamic objects.

    Args:
        frames:     (T, H, W, 3) uint8 RGB.
        config:     Dict with keys:
                      yolo_conf       float  (default 0.3)
                      mask_classes    list   (default DEFAULT_DYNAMIC_CLASSES)
                      sam_dilate_px   int    (default 5)
                      sam_model       str    (default "sam2_hiera_large.yaml")
                      sam_checkpoint  str    (optional)
                      device          str    (default "cuda")
        output_dir: If given, save mask files to output_dir/masks/.

    Returns:
        dynamic_masks: List[np.ndarray] — T binary (H, W) bool arrays.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics required: pip install ultralytics")

    T, H, W, _ = frames.shape
    device = config.get("device", "cuda")
    yolo_conf = config.get("yolo_conf", 0.3)
    mask_classes = config.get("mask_classes", DEFAULT_DYNAMIC_CLASSES)
    dilate_px = config.get("sam_dilate_px", 5)

    # Load YOLOv8-x
    yolo_path = config.get("yolo_model_path", "yolov8x.pt")
    print(f"[Masking] Loading YOLOv8-x from {yolo_path}...")
    yolo = YOLO(yolo_path)

    # Load SAM 2 wrapper
    sam = SAMWrapper(
        model_cfg=config.get("sam_model", "sam2_hiera_large.yaml"),
        checkpoint=config.get("sam_checkpoint", None),
        device=device,
        dilate_px=dilate_px,
    )

    if output_dir is not None:
        mask_dir = Path(output_dir) / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

    dynamic_masks: List[np.ndarray] = []

    print(f"[Masking] Processing {T} frames...")
    for t, frame in enumerate(frames):
        union_mask = np.zeros((H, W), dtype=bool)
        instance_info = []

        # --- YOLO detection ---
        results = yolo(frame, conf=yolo_conf, iou=0.45, verbose=False)
        boxes_xyxy = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in mask_classes:
                    continue
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                boxes_xyxy.append(xyxy)
                instance_info.append({
                    "class_id": cls_id,
                    "bbox": xyxy,
                    "confidence": conf,
                })

        # --- SAM 2 segmentation ---
        if len(boxes_xyxy) > 0:
            boxes_arr = np.array(boxes_xyxy, dtype=np.float32)
            instance_masks = sam.segment_boxes(frame, boxes_arr)

            for i, inst_mask in enumerate(instance_masks):
                union_mask |= inst_mask
                instance_info[i]["mask"] = inst_mask

        dynamic_masks.append(union_mask)

        # --- Save to disk ---
        if output_dir is not None:
            np.save(mask_dir / f"frame_{t:06d}_dynamic.npy", union_mask)
            with open(mask_dir / f"frame_{t:06d}_instances.pkl", "wb") as fh:
                pickle.dump(instance_info, fh)

        if t % 20 == 0:
            n_dyn = int(union_mask.sum())
            frac = n_dyn / (H * W)
            print(f"  Frame {t:4d}/{T}: {len(instance_info):2d} objects, "
                  f"{frac*100:.1f}% pixels masked")

    return dynamic_masks


def apply_mask_to_frames(
    frames: np.ndarray,
    masks: List[np.ndarray],
) -> np.ndarray:
    """
    Zero out dynamic pixels in each frame.

    frames: (T, H, W, 3) uint8
    masks:  T × (H, W) bool — True where dynamic

    Returns masked_frames: (T, H, W, 3) uint8
    """
    masked = frames.copy()
    for t, mask in enumerate(masks):
        masked[t][mask] = 0
    return masked
