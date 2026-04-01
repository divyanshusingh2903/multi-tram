"""
YOLO Wrapper for Person Detection
Wraps YOLOv8/v11 for fast and accurate person detection

Usage Examples:
    # Load default YOLOv8s model
    yolo = YOLOWrapper()

    # Load YOLOv11s for faster inference
    yolo = YOLOWrapper(model_name='yolov11s')

    # Detect people in single frame
    detections = yolo.detect_frame(frame)

    # Detect people in batch of frames
    all_detections = yolo.detect_batch(frames)

    # Results contain:
    # - bbox: [x, y, w, h] in pixels
    # - confidence: detection confidence [0, 1]
    # - bbox_xyxy: [x1, y1, x2, y2] format
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
from dataclasses import dataclass


@dataclass
class YOLODetection:
    """Single person detection from YOLO"""
    bbox: np.ndarray  # [x, y, w, h] format
    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2] format
    confidence: float
    crop: Optional[np.ndarray] = None  # Cropped image for ReID


class YOLOWrapper:
    """
    Wrapper for YOLO (v8/v11) Person Detection

    Provides fast and accurate person detection using state-of-the-art
    YOLOv8 or YOLOv11 models from Ultralytics.

    Key features:
    - 2-3x faster than Mask R-CNN
    - Better detection accuracy (~48-49% mAP)
    - Better small person detection
    - Modular design for easy integration
    """

    def __init__(
        self,
        model_name: str = 'yolov8s',
        device: str = 'cuda',
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        return_crops: bool = True
    ):
        """
        Initialize YOLO wrapper

        Args:
            model_name: YOLO model variant
                - 'yolov8n': Fastest (23ms, 3.2M params)
                - 'yolov8s': Balanced (28ms, 11.2M params) [RECOMMENDED]
                - 'yolov8m': Accurate (35ms, 25.9M params)
                - 'yolov11n': Ultra-fast (13.5ms, 2.6M params)
                - 'yolov11s': Best balanced (19ms, 9.4M params)
            device: Device to run on ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            return_crops: Whether to return cropped person images
        """
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.return_crops = return_crops
        self.model = None

        self._init_model()

    def _init_model(self):
        """Initialize YOLO model"""
        try:
            from ultralytics import YOLO

            print(f"[YOLOWrapper] Loading {self.model_name} model...")

            # Load pretrained model
            # Ultralytics will auto-download if not present
            self.model = YOLO(f'{self.model_name}.pt')

            # Move to device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
                print(f"[YOLOWrapper] Model loaded on GPU")
            else:
                print(f"[YOLOWrapper] Model loaded on CPU")

            print(f"[YOLOWrapper] {self.model_name} initialized successfully")

        except ImportError as e:
            print(f"[YOLOWrapper] Error: Could not import ultralytics - {e}")
            print("[YOLOWrapper] Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"[YOLOWrapper] Error initializing YOLO: {e}")
            self.model = None

    def detect_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        return_crops: Optional[bool] = None
    ) -> List[YOLODetection]:
        """
        Detect people in a single frame

        Args:
            frame: Input frame (H, W, 3) RGB in [0, 255]
            conf_threshold: Override default confidence threshold
            return_crops: Override default crop return setting

        Returns:
            List of YOLODetection objects
        """
        if self.model is None:
            return []

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        crops = return_crops if return_crops is not None else self.return_crops

        try:
            # Run YOLO detection
            # classes=0 means only detect 'person' class
            results = self.model(
                frame,
                conf=conf,
                iou=self.iou_threshold,
                classes=[0],  # Person class only
                verbose=False
            )

            detections = []

            # Process results
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    # Get bounding box in xyxy format
                    xyxy = boxes.xyxy[i].cpu().numpy()

                    # Convert to xywh format
                    x1, y1, x2, y2 = xyxy
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1

                    # Get confidence
                    confidence = float(boxes.conf[i].cpu().numpy())

                    # Crop image if requested
                    crop = None
                    if crops:
                        x1_int, y1_int = int(max(0, x1)), int(max(0, y1))
                        x2_int, y2_int = int(min(frame.shape[1], x2)), int(min(frame.shape[0], y2))
                        crop = frame[y1_int:y2_int, x1_int:x2_int].copy()

                    det = YOLODetection(
                        bbox=np.array([x, y, w, h]),
                        bbox_xyxy=xyxy,
                        confidence=confidence,
                        crop=crop
                    )
                    detections.append(det)

            return detections

        except Exception as e:
            print(f"[YOLOWrapper] Detection failed: {e}")
            return []

    def detect_batch(
        self,
        frames: np.ndarray,
        conf_threshold: Optional[float] = None,
        batch_size: int = 8
    ) -> List[List[YOLODetection]]:
        """
        Detect people in batch of frames

        Args:
            frames: Video frames (T, H, W, 3) RGB in [0, 255]
            conf_threshold: Override default confidence threshold
            batch_size: Number of frames to process at once

        Returns:
            List of detection lists, one per frame
        """
        if self.model is None:
            return [[] for _ in range(len(frames))]

        T = len(frames)
        all_detections = []

        try:
            # Process in batches for efficiency
            for batch_start in range(0, T, batch_size):
                batch_end = min(batch_start + batch_size, T)
                batch_frames = frames[batch_start:batch_end]

                # Detect in each frame of batch
                for frame in batch_frames:
                    frame_dets = self.detect_frame(frame, conf_threshold)
                    all_detections.append(frame_dets)

            return all_detections

        except Exception as e:
            print(f"[YOLOWrapper] Batch detection failed: {e}")
            return [[] for _ in range(T)]

    def get_model_info(self) -> dict:
        """
        Get model information

        Returns:
            Dictionary with model specs
        """
        model_specs = {
            'yolov8n': {'params': '3.2M', 'inference_time': '23ms', 'mAP': '~45%'},
            'yolov8s': {'params': '11.2M', 'inference_time': '28ms', 'mAP': '~48%'},
            'yolov8m': {'params': '25.9M', 'inference_time': '35ms', 'mAP': '~51%'},
            'yolov11n': {'params': '2.6M', 'inference_time': '13.5ms', 'mAP': '~46%'},
            'yolov11s': {'params': '9.4M', 'inference_time': '19ms', 'mAP': '~49%'},
        }

        return {
            'model_name': self.model_name,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'specs': model_specs.get(self.model_name, {})
        }


if __name__ == "__main__":
    """Test YOLO wrapper"""
    import argparse

    parser = argparse.ArgumentParser(description="Test YOLO wrapper")
    parser.add_argument("--image", type=str, help="Test image path")
    parser.add_argument("--video", type=str, help="Test video path")
    parser.add_argument("--model", type=str, default='yolov8s',
                       help="YOLO model (yolov8s, yolov11s, etc.)")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")

    args = parser.parse_args()

    # Initialize wrapper
    yolo = YOLOWrapper(
        model_name=args.model,
        device=args.device,
        conf_threshold=args.conf
    )

    print(f"\nModel Info:")
    for k, v in yolo.get_model_info().items():
        print(f"  {k}: {v}")

    if args.image:
        # Test single frame
        img = cv2.imread(args.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = yolo.detect_frame(img)
        print(f"\nDetected {len(detections)} people in image")
        for i, det in enumerate(detections):
            print(f"  Person {i+1}: bbox={det.bbox}, conf={det.confidence:.3f}")

    if args.video:
        # Test video
        cap = cv2.VideoCapture(args.video)
        frames = []
        max_frames = 30

        print(f"\nReading video frames...")
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        print(f"Read {len(frames)} frames")

        # Test batch detection
        frames = np.stack(frames)
        all_detections = yolo.detect_batch(frames)

        total_people = sum(len(dets) for dets in all_detections)
        avg_people = total_people / len(frames)

        print(f"\nProcessed {len(frames)} frames")
        print(f"Total detections: {total_people}")
        print(f"Average people per frame: {avg_people:.1f}")
