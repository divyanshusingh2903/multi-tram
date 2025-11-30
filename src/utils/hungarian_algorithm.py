"""
Hungarian Algorithm for Optimal Track-Detection Assignment

Provides optimal matching between detections and track predictions using
scipy's linear_sum_assignment implementation. Falls back to greedy matching
if scipy is unavailable.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def optimal_assignment(
    cost_matrix: np.ndarray
) -> Dict[int, int]:
    """
    Solve optimal assignment problem using Hungarian algorithm.

    Args:
        cost_matrix: Cost matrix (N_tracks × M_detections)
                    Lower cost = better match

    Returns:
        Dictionary mapping track_idx -> detection_idx
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Hungarian algorithm. Install with: pip install scipy")

    if cost_matrix.size == 0:
        return {}

    # Solve assignment problem
    track_indices, det_indices = linear_sum_assignment(cost_matrix)

    # Convert to dictionary (track_idx -> det_idx)
    assignments = {}
    for track_idx, det_idx in zip(track_indices, det_indices):
        # Only include if cost is reasonable (not too high)
        if cost_matrix[track_idx, det_idx] < 1e5:
            assignments[track_idx] = det_idx

    return assignments


def greedy_assignment(
    cost_matrix: np.ndarray
) -> Dict[int, int]:
    """
    Fallback greedy assignment when Hungarian unavailable.

    Sequentially assigns each track to its best matching detection.

    Args:
        cost_matrix: Cost matrix (N_tracks × M_detections)

    Returns:
        Dictionary mapping track_idx -> detection_idx
    """
    assignments = {}
    used_dets = set()

    n_tracks, n_dets = cost_matrix.shape

    # For each track, find best detection
    for track_idx in range(n_tracks):
        best_det_idx = -1
        best_cost = 1e6

        for det_idx in range(n_dets):
            if det_idx in used_dets:
                continue

            cost = cost_matrix[track_idx, det_idx]

            if cost < best_cost:
                best_cost = cost
                best_det_idx = det_idx

        if best_det_idx >= 0 and best_cost < 1e5:
            assignments[track_idx] = best_det_idx
            used_dets.add(best_det_idx)

    return assignments


def create_cost_matrix(
    detections: List[np.ndarray],
    track_predictions: Dict[int, np.ndarray],
    similarity_threshold: float = 0.5,
    iou_fn=None
) -> np.ndarray:
    """
    Create cost matrix for assignment problem.

    Cost = 1 - IoU (higher IoU = lower cost = better match)

    Args:
        detections: List of detection bboxes [x, y, w, h]
        track_predictions: Dict mapping track_id -> predicted bbox [x, y, w, h]
        similarity_threshold: Minimum IoU to consider as valid match
        iou_fn: Custom IoU function (optional)

    Returns:
        Cost matrix (N_tracks × N_detections)
    """
    if iou_fn is None:
        iou_fn = compute_bbox_iou

    n_tracks = len(track_predictions)
    n_dets = len(detections)

    cost_matrix = np.ones((n_tracks, n_dets)) * 1e6  # High cost for no match

    track_ids = list(track_predictions.keys())

    for track_idx, track_id in enumerate(track_ids):
        predicted_bbox = track_predictions[track_id]

        for det_idx, det_bbox in enumerate(detections):
            # Compute IoU
            iou = iou_fn(predicted_bbox, det_bbox)

            # Cost is inverse of similarity
            similarity = iou
            cost = 1.0 - similarity

            # Penalize assignments below threshold
            if similarity < similarity_threshold:
                cost = 1e6

            cost_matrix[track_idx, det_idx] = cost

    return cost_matrix


def compute_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: Bounding box [x, y, w, h]
        bbox2: Bounding box [x, y, w, h]

    Returns:
        IoU value [0, 1]
    """
    x1_min, y1_min = bbox1[:2]
    x1_max, y1_max = x1_min + bbox1[2], y1_min + bbox1[3]

    x2_min, y2_min = bbox2[:2]
    x2_max, y2_max = x2_min + bbox2[2], y2_min + bbox2[3]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)
