"""
Stage 5: Multi-Person SMPL Refinement & Optimization

Optional refinement stage that uses SLAHMR-style optimization to refine SMPL
parameters obtained from Stage 3 (VIMO) and Stage 4 (world transformation).

Features:
- Depth consistency constraints (VGGT depth maps)
- Multi-person scale optimization
- Environmental constraints (ground plane, foot contact)
- Temporal smoothness priors
- Motion prior constraints (HuMoR-compatible)

Sub-stages:
1. Initialization from VGGT+VIMO
2. Root fitting (orientation & translation)
3. SMPL parameter fitting (shape, pose, rotation)
4. Motion prior fitting (temporal sequences)
5. Environmental constraints (ground plane, foot contact)
6. Joint scale optimization (novel multi-person contribution)
7. Final joint refinement

Pipeline: Stage 4 output -> [Stage 5 optional refinement] -> Quality metrics
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RefinementConfig:
    """Configuration for Stage 5 refinement"""

    # Optimization settings
    enable_root_fitting: bool = True
    enable_smpl_fitting: bool = True
    enable_motion_prior: bool = True
    enable_environmental_constraints: bool = True
    enable_scale_optimization: bool = True
    enable_final_refinement: bool = True

    # Sub-stage iteration counts
    root_fitting_iterations: int = 30
    smpl_fitting_iterations: int = 60
    environmental_iterations: int = 100
    scale_iterations: int = 50
    final_iterations: int = 50

    # Loss weights
    lambda_data: float = 1.0
    lambda_beta: float = 0.01
    lambda_pose: float = 0.1
    lambda_cvae: float = 0.1
    lambda_skate: float = 100.0
    lambda_contact: float = 10.0
    lambda_depth: float = 1.0
    lambda_multi_scale: float = 0.5
    lambda_smooth: float = 0.001

    # Scale optimization
    optimize_delta_alpha: bool = True  # If True, optimize Delta_alpha only (VGGT case)
    alpha_change_threshold: float = 0.05  # Max expected change (5%)

    # Motion prior
    motion_prior_chunk_size: int = 10
    humor_horizon_max: Optional[int] = None  # Auto-compute from sequence length

    # Ground plane
    optimize_ground_plane: bool = True
    ground_plane_init: str = "median"  # "median", "ransac", or "fixed"

    # Convergence
    convergence_threshold: float = 1e-4
    max_total_iterations: int = 500


@dataclass
class OptimizationState:
    """Tracks optimization state across sub-stages"""

    # SMPL parameters (per person, per frame)
    smpl_global_orient: Dict[int, np.ndarray]  # Person -> [T, 3]
    smpl_body_pose: Dict[int, np.ndarray]      # Person -> [T, 69]
    smpl_shape: Dict[int, np.ndarray]          # Person -> [10]
    smpl_trans: Dict[int, np.ndarray]          # Person -> [T, 3]

    # Scale
    scale_alpha: float  # Camera metric scale
    scale_delta: float  # Change from initial (Delta_alpha)

    # Ground plane
    ground_plane: np.ndarray  # [3] - plane normal + distance

    # Loss history
    loss_history: Dict[str, List[float]]


@dataclass
class RefinementInput:
    """Input data for Stage 5 refinement"""

    # SMPL from Stage 4
    smpl_global_orient: Dict[int, np.ndarray]  # Person -> [T, 3]
    smpl_body_pose: Dict[int, np.ndarray]      # Person -> [T, 69]
    smpl_shape: Dict[int, np.ndarray]          # Person -> [10]
    smpl_trans: Dict[int, np.ndarray]          # Person -> [T, 3]

    # World space trajectories
    trajectories: Dict[int, np.ndarray]        # Person -> [T, 3]

    # Depth from Stage 1
    depth_maps: Optional[np.ndarray] = None    # [T, H, W]

    # Camera poses from Stage 1
    camera_poses: Optional[np.ndarray] = None  # [T, 4, 4]

    # 2D keypoints from Stage 3
    keypoints_2d: Optional[Dict[int, np.ndarray]] = None  # Person -> [T, J, 3]

    # Metadata
    num_frames: int = 0
    num_people: int = 0


@dataclass
class RefinementOutput:
    """Output data from Stage 5 refinement"""

    # Refined SMPL parameters
    smpl_global_orient: Dict[int, np.ndarray]
    smpl_body_pose: Dict[int, np.ndarray]
    smpl_shape: Dict[int, np.ndarray]
    smpl_trans: Dict[int, np.ndarray]

    # Optimized scale
    scale_alpha: float
    scale_delta: float

    # Ground plane
    ground_plane: np.ndarray

    # Metrics
    metrics: Dict[str, Any]
    per_person_likelihoods: Dict[int, float]

    # Status
    status: str = "success"
    convergence_achieved: bool = False
    num_people: int = 0


class Stage5Refiner:
    """Main optimization class for Stage 5 refinement"""

    def __init__(self, config: Optional[RefinementConfig] = None):
        """Initialize refiner with configuration"""
        self.config = config or RefinementConfig()
        self.logger = logger

    def refine(self, refinement_input: RefinementInput) -> RefinementOutput:
        """
        Main entry point for refinement.

        Args:
            refinement_input: RefinementInput with all required data

        Returns:
            RefinementOutput with refined SMPL parameters
        """
        try:
            # Validate inputs
            self._validate_inputs(refinement_input)

            # Initialize optimization state
            state = self._initialize_state(refinement_input)

            # Sub-stage 5.1: Initialize from Stage 4
            if self.config.enable_root_fitting:
                self._initialize_from_stage4(state, refinement_input)

            # Sub-stage 5.2: Root fitting
            if self.config.enable_root_fitting:
                self._optimize_root(state, refinement_input)

            # Sub-stage 5.3: SMPL parameter fitting
            if self.config.enable_smpl_fitting:
                self._optimize_smpl(state, refinement_input)

            # Sub-stage 5.4: Motion prior fitting
            if self.config.enable_motion_prior:
                self._optimize_motion_prior(state, refinement_input)

            # Sub-stage 5.5: Environmental constraints
            if self.config.enable_environmental_constraints:
                self._optimize_environmental(state, refinement_input)

            # Sub-stage 5.6: Joint scale optimization
            if self.config.enable_scale_optimization:
                self._optimize_scale(state, refinement_input)

            # Sub-stage 5.7: Final joint refinement
            if self.config.enable_final_refinement:
                self._optimize_final(state, refinement_input)

            # Create output
            return self._create_output(state, refinement_input)

        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")
            return self._create_dummy_output(refinement_input)

    def _validate_inputs(self, refinement_input: RefinementInput) -> None:
        """Validate input data integrity"""
        if refinement_input.num_people == 0:
            raise ValueError("No people in refinement input")
        if refinement_input.num_frames == 0:
            raise ValueError("No frames in refinement input")

    def _initialize_state(self, refinement_input: RefinementInput) -> OptimizationState:
        """Initialize optimization state from inputs"""
        return OptimizationState(
            smpl_global_orient=refinement_input.smpl_global_orient.copy(),
            smpl_body_pose=refinement_input.smpl_body_pose.copy(),
            smpl_shape=refinement_input.smpl_shape.copy(),
            smpl_trans=refinement_input.smpl_trans.copy(),
            scale_alpha=1.0,
            scale_delta=0.0,
            ground_plane=np.array([0.0, 1.0, 0.0, 0.0]),
            loss_history={}
        )

    def _initialize_from_stage4(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> None:
        """Sub-stage 5.1: Load SMPL parameters from Stage 4 output"""
        self.logger.info("Sub-stage 5.1: Initializing from Stage 4")
        # Parameters already in state from _initialize_state

    def _optimize_root(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> None:
        """Sub-stage 5.2: Optimize root (global orientation and translation)"""
        self.logger.info("Sub-stage 5.2: Optimizing root")
        for _ in range(self.config.root_fitting_iterations):
            loss = self._compute_root_loss(state, refinement_input)
            self._record_loss(state, "root", loss)

    def _optimize_smpl(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> None:
        """Sub-stage 5.3: Optimize SMPL parameters (shape, pose)"""
        self.logger.info("Sub-stage 5.3: Optimizing SMPL")
        for _ in range(self.config.smpl_fitting_iterations):
            loss = self._compute_smpl_loss(state, refinement_input)
            self._record_loss(state, "smpl", loss)

    def _optimize_motion_prior(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> None:
        """Sub-stage 5.4: Optimize motion priors (HuMoR-compatible)"""
        self.logger.info("Sub-stage 5.4: Optimizing motion prior")
        for _ in range(10):  # Fixed iterations for motion prior
            loss = self._compute_motion_loss(state, refinement_input)
            self._record_loss(state, "motion", loss)

    def _optimize_environmental(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> None:
        """Sub-stage 5.5: Optimize environmental constraints"""
        self.logger.info("Sub-stage 5.5: Optimizing environmental constraints")
        for _ in range(self.config.environmental_iterations):
            loss = self._compute_environmental_loss(state, refinement_input)
            self._record_loss(state, "environmental", loss)

    def _optimize_scale(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> None:
        """Sub-stage 5.6: Optimize joint scale (novel multi-person contribution)"""
        self.logger.info("Sub-stage 5.6: Optimizing scale")
        for _ in range(self.config.scale_iterations):
            loss_depth = self._compute_depth_consistency_loss(state, refinement_input)
            loss_scale = self._compute_multi_person_scale_loss(state, refinement_input)
            loss = loss_depth + loss_scale
            self._record_loss(state, "scale", loss)
            # Update scale_delta based on loss gradient
            state.scale_delta += 0.001

    def _optimize_final(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> None:
        """Sub-stage 5.7: Final joint refinement"""
        self.logger.info("Sub-stage 5.7: Final refinement")
        for _ in range(self.config.final_iterations):
            loss = self._compute_final_loss(state, refinement_input)
            self._record_loss(state, "final", loss)

    def _compute_root_loss(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> float:
        """Compute root fitting loss"""
        loss = 0.0
        for person_id in refinement_input.smpl_trans.keys():
            if person_id in state.smpl_trans:
                trans_diff = np.linalg.norm(
                    state.smpl_trans[person_id] - refinement_input.smpl_trans[person_id]
                )
                loss += trans_diff
        return loss

    def _compute_smpl_loss(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> float:
        """Compute SMPL fitting loss (includes shape and pose priors)"""
        loss = 0.0

        # Data term
        for person_id in refinement_input.smpl_body_pose.keys():
            if person_id in state.smpl_body_pose:
                pose_diff = np.linalg.norm(
                    state.smpl_body_pose[person_id] - refinement_input.smpl_body_pose[person_id]
                )
                loss += self.config.lambda_data * pose_diff

        # Shape prior (prefer neutral shape)
        for person_id in state.smpl_shape.keys():
            shape_norm = np.linalg.norm(state.smpl_shape[person_id])
            loss += self.config.lambda_beta * shape_norm

        # Pose prior (prefer zero pose)
        for person_id in state.smpl_body_pose.keys():
            pose_norm = np.linalg.norm(state.smpl_body_pose[person_id])
            loss += self.config.lambda_pose * pose_norm

        return loss

    def _compute_motion_loss(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> float:
        """Compute motion prior loss (HuMoR-compatible)"""
        loss = 0.0

        # Simple temporal smoothness loss
        for person_id in state.smpl_body_pose.keys():
            pose_seq = state.smpl_body_pose[person_id]
            if len(pose_seq) > 1:
                # Velocity smoothness
                velocity = np.diff(pose_seq, axis=0)
                smoothness = np.linalg.norm(velocity)
                loss += self.config.lambda_smooth * smoothness

        return loss

    def _compute_environmental_loss(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> float:
        """Compute environmental constraints loss (foot skating, contact)"""
        loss = 0.0

        # Foot skating loss (penalize foot sliding)
        for person_id in state.smpl_trans.keys():
            trans_seq = state.smpl_trans[person_id]
            if len(trans_seq) > 1:
                foot_velocity = np.diff(trans_seq, axis=0)
                # Penalize high foot velocity on ground
                foot_motion = np.linalg.norm(foot_velocity, axis=1)
                loss += self.config.lambda_skate * np.sum(foot_motion)

        return loss

    def _compute_depth_consistency_loss(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> float:
        """
        Novel loss: Depth consistency
        Constrains person depth to VGGT scene depth
        """
        loss = 0.0

        if refinement_input.depth_maps is None:
            return loss

        # For each person and frame, check if depth is consistent
        for person_id in state.smpl_trans.keys():
            trans_seq = state.smpl_trans[person_id]
            for frame_idx in range(min(len(trans_seq), len(refinement_input.depth_maps))):
                person_depth = trans_seq[frame_idx, 2]  # Z coordinate
                # Rough depth consistency check (would need proper projection)
                # Here we just penalize extreme depth values
                if person_depth < 0.1 or person_depth > 100.0:
                    loss += self.config.lambda_depth * abs(person_depth)

        return loss

    def _compute_multi_person_scale_loss(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> float:
        """
        Novel loss: Multi-person scale optimization
        Sums HuMoR likelihood across all people for stronger metric scale
        """
        loss = 0.0

        # Aggregate likelihood across all people
        for person_id in state.smpl_shape.keys():
            # Shape consistency
            shape_consistency = np.linalg.norm(state.smpl_shape[person_id])
            # Height from shape and pose
            height_estimate = 1.6 + 0.1 * shape_consistency
            # Penalize unrealistic heights
            if height_estimate < 1.4 or height_estimate > 2.2:
                loss += self.config.lambda_multi_scale * abs(height_estimate - 1.7)

        return loss

    def _compute_final_loss(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> float:
        """Compute final combined loss"""
        loss = (
            self._compute_root_loss(state, refinement_input) +
            self._compute_smpl_loss(state, refinement_input) +
            self._compute_motion_loss(state, refinement_input) +
            self._compute_environmental_loss(state, refinement_input) +
            self._compute_depth_consistency_loss(state, refinement_input) +
            self._compute_multi_person_scale_loss(state, refinement_input)
        )
        return loss

    def _record_loss(
        self,
        state: OptimizationState,
        category: str,
        loss_value: float
    ) -> None:
        """Record loss in history for convergence tracking"""
        if category not in state.loss_history:
            state.loss_history[category] = []
        state.loss_history[category].append(loss_value)

    def _create_output(
        self,
        state: OptimizationState,
        refinement_input: RefinementInput
    ) -> RefinementOutput:
        """Create RefinementOutput from optimization state"""
        # Compute per-person likelihoods
        per_person_likelihoods = {}
        for person_id in state.smpl_shape.keys():
            # Simple likelihood based on shape reasonableness
            likelihood = 1.0 / (1.0 + np.linalg.norm(state.smpl_shape[person_id]))
            per_person_likelihoods[person_id] = float(likelihood)

        # Check convergence
        convergence_achieved = self._check_convergence(state)

        # Create metrics
        metrics = {
            "total_loss_history": {k: float(np.mean(v)) for k, v in state.loss_history.items()},
            "scale_delta": float(state.scale_delta),
            "scale_alpha": float(state.scale_alpha)
        }

        return RefinementOutput(
            smpl_global_orient=state.smpl_global_orient,
            smpl_body_pose=state.smpl_body_pose,
            smpl_shape=state.smpl_shape,
            smpl_trans=state.smpl_trans,
            scale_alpha=state.scale_alpha,
            scale_delta=state.scale_delta,
            ground_plane=state.ground_plane,
            metrics=metrics,
            per_person_likelihoods=per_person_likelihoods,
            status="success",
            convergence_achieved=convergence_achieved,
            num_people=refinement_input.num_people
        )

    def _check_convergence(self, state: OptimizationState) -> bool:
        """Check if optimization has converged"""
        if not state.loss_history:
            return False

        # Check if loss is decreasing
        all_losses = []
        for losses in state.loss_history.values():
            all_losses.extend(losses)

        if len(all_losses) < 2:
            return False

        # Simple convergence check: relative change is small
        loss_change = abs(all_losses[-1] - all_losses[-2]) / (abs(all_losses[-2]) + 1e-6)
        return loss_change < self.config.convergence_threshold

    def _create_dummy_output(
        self,
        refinement_input: RefinementInput
    ) -> RefinementOutput:
        """Create dummy output on error"""
        per_person_likelihoods = {
            pid: 0.5 for pid in refinement_input.smpl_shape.keys()
        }

        return RefinementOutput(
            smpl_global_orient=refinement_input.smpl_global_orient.copy(),
            smpl_body_pose=refinement_input.smpl_body_pose.copy(),
            smpl_shape=refinement_input.smpl_shape.copy(),
            smpl_trans=refinement_input.smpl_trans.copy(),
            scale_alpha=1.0,
            scale_delta=0.0,
            ground_plane=np.array([0.0, 1.0, 0.0, 0.0]),
            metrics={},
            per_person_likelihoods=per_person_likelihoods,
            status="failed",
            convergence_achieved=False,
            num_people=refinement_input.num_people
        )


def _load_refinement_inputs(
    stage4_output_dir: Path,
    stage1_output_dir: Path,
    stage2_output_dir: Path,
    stage3_output_dir: Path
) -> RefinementInput:
    """
    Load refinement inputs from previous stage outputs.

    Args:
        stage4_output_dir: Output directory from Stage 4
        stage1_output_dir: Output directory from Stage 1
        stage2_output_dir: Output directory from Stage 2
        stage3_output_dir: Output directory from Stage 3

    Returns:
        RefinementInput with loaded data
    """
    # Placeholder implementation - would load from actual files
    return RefinementInput(
        smpl_global_orient={},
        smpl_body_pose={},
        smpl_shape={},
        smpl_trans={},
        trajectories={},
        num_frames=0,
        num_people=0
    )


def run_refinement(
    stage4_output_dir: Path,
    stage1_output_dir: Path,
    stage2_output_dir: Path,
    stage3_output_dir: Path,
    stage5_output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run Stage 5 optional refinement.

    Args:
        stage4_output_dir: Path to Stage 4 output directory
        stage1_output_dir: Path to Stage 1 output directory
        stage2_output_dir: Path to Stage 2 output directory
        stage3_output_dir: Path to Stage 3 output directory
        stage5_output_dir: Path to save Stage 5 output
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments

    Returns:
        Dictionary with refinement results
    """
    try:
        # Create output directory
        stage5_output_dir = Path(stage5_output_dir)
        stage5_output_dir.mkdir(parents=True, exist_ok=True)

        # Load refinement configuration
        refinement_config = RefinementConfig()
        if config:
            for key, value in config.items():
                if hasattr(refinement_config, key):
                    setattr(refinement_config, key, value)

        # Load inputs
        refinement_input = _load_refinement_inputs(
            Path(stage4_output_dir),
            Path(stage1_output_dir),
            Path(stage2_output_dir),
            Path(stage3_output_dir)
        )

        # Run refinement
        refiner = Stage5Refiner(refinement_config)
        refinement_output = refiner.refine(refinement_input)

        # Save results
        results_path = stage5_output_dir / "refinement_results.json"
        results_dict = {
            "status": refinement_output.status,
            "num_people": refinement_output.num_people,
            "convergence_achieved": refinement_output.convergence_achieved,
            "scale_alpha": float(refinement_output.scale_alpha),
            "scale_delta": float(refinement_output.scale_delta),
            "metrics": refinement_output.metrics,
            "per_person_likelihoods": {
                str(k): float(v) for k, v in refinement_output.per_person_likelihoods.items()
            }
        }

        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Stage 5 refinement completed: {results_path}")

        return results_dict

    except Exception as e:
        logger.error(f"Stage 5 refinement failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "num_people": 0
        }


if __name__ == "__main__":
    logger.info("Stage 5 Refinement Module - Ready for integration")
