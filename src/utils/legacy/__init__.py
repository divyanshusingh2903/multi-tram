"""
Legacy tracking utilities — not used in the main pipeline.
PHALP+ handles association internally; world-frame correction
uses world-frame position + PHALP embedding similarity.
Kept here in case they are useful for the world-correction
Hungarian-matching step or ablation experiments.
"""
from .kalman_filter import KalmanFilter
from .hungarian_algorithm import (
    optimal_assignment,
    greedy_assignment,
    create_cost_matrix,
    compute_bbox_iou,
    SCIPY_AVAILABLE,
)

__all__ = [
    "KalmanFilter",
    "optimal_assignment",
    "greedy_assignment",
    "create_cost_matrix",
    "compute_bbox_iou",
    "SCIPY_AVAILABLE",
]
