"""
Utility modules for multi-TRAM pipeline.

Active utilities (spec §7 repository structure):
  masking.py         — YOLOv8 + SAM 2 dynamic object mask generation (Stage 1)
  world_correction.py— VGGT depth-guided world-frame ID correction (Stage 2)
  geometry.py        — SE(3) ops, ground plane RANSAC, Y-up alignment (Stage 4)
  smpl_utils.py      — SMPL forward kinematics helpers (Stage 3 / 4)
  visualization.py   — Rendering utilities
  world_frame.py     — Depth unprojection and world-frame transform helpers

Legacy (not used in main pipeline):
  legacy/kalman_filter.py      — bbox Kalman filter (kept for ablations)
  legacy/hungarian_algorithm.py— Hungarian matching (kept for ablations)
"""
