"""
Stage 4: World-Space Transformation
Transforms all people from camera frame to shared world frame using camera poses
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import json
from dataclasses import dataclass, asdict

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None


@dataclass
class SMPLParametersWorld:
    """SMPL parameters in world frame"""
    track_id: int
    frame_id: int
    global_orient_world: np.ndarray
    transl_world: np.ndarray
    poses: np.ndarray
    betas: np.ndarray
    joints_world: Optional[np.ndarray] = None
    vertices_world: Optional[np.ndarray] = None
    global_orient_camera: Optional[np.ndarray] = None
    transl_camera: Optional[np.ndarray] = None


class WorldTransformer:
    """Handles transformation from camera to world frame"""

    def __init__(self, world_origin: np.ndarray = None, up_axis: int = 1):
        """
        Initialize transformer.

        Args:
            world_origin: Origin of world frame
            up_axis: Which axis is up (0=X, 1=Y, 2=Z)
        """
        self.world_origin = world_origin
        self.up_axis = up_axis
        self.world_origin_set = False
        self.camera_poses_world = None

    def set_camera_poses(self, camera_poses: Dict):
        """
        Set camera poses from Stage 1.

        Args:
            camera_poses: Dictionary with 'R' and 'T' keys
        """
        R_poses = camera_poses.get('R')
        T_poses = camera_poses.get('T')

        if R_poses is None or T_poses is None:
            raise ValueError("Camera poses must contain 'R' and 'T' keys")

        T = len(R_poses)
        self.camera_poses_world = np.zeros((T, 4, 4))

        for t in range(T):
            T_WC = np.eye(4)
            T_WC[:3, :3] = R_poses[t].T
            T_WC[:3, 3] = -R_poses[t].T @ T_poses[t]
            self.camera_poses_world[t] = T_WC

        print(f"[Stage4] Camera poses set: {T} frames")

    def transform_person_to_world(
        self,
        smpl_params_camera: Dict,
        joints_camera: np.ndarray,
        frame_camera_pose: np.ndarray
    ) -> Dict:
        """
        Transform person from camera frame to world frame.

        Args:
            smpl_params_camera: SMPL params in camera frame
            joints_camera: 3D joints in camera frame (24, 3)
            frame_camera_pose: Camera pose matrix (4, 4)

        Returns:
            Dictionary with world-frame SMPL parameters
        """
        poses = smpl_params_camera.get('poses')
        betas = smpl_params_camera.get('betas')
        global_orient_camera = smpl_params_camera.get('global_orient')
        transl_camera = smpl_params_camera.get('transl')

        R_camera = frame_camera_pose[:3, :3]
        T_camera = frame_camera_pose[:3, 3]

        if global_orient_camera is not None:
            if R is not None:
                r_camera_mat = R.from_rotvec(global_orient_camera).as_matrix()
                r_world_mat = R_camera.T @ r_camera_mat
                global_orient_world = R.from_matrix(r_world_mat).as_rotvec()
            else:
                global_orient_world = global_orient_camera.copy()
        else:
            global_orient_world = np.zeros(3)

        transl_world = R_camera.T @ (transl_camera - T_camera)

        joints_world = None
        if joints_camera is not None:
            joints_centered = joints_camera - T_camera[np.newaxis, :]
            joints_world = joints_centered @ R_camera

        return {
            'poses': poses,
            'betas': betas,
            'global_orient_world': global_orient_world,
            'transl_world': transl_world,
            'joints_world': joints_world,
            'global_orient_camera': global_orient_camera,
            'transl_camera': transl_camera
        }

    def get_trajectory(
        self,
        person_smpl_sequence: List[Dict],
        person_joints_sequence: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract root position trajectory in world frame.

        Args:
            person_smpl_sequence: List of SMPL params
            person_joints_sequence: List of joint positions

        Returns:
            Root position trajectory (T, 3)
        """
        trajectory = []

        for smpl_params in person_smpl_sequence:
            transl = smpl_params.get('transl_world')
            if transl is not None:
                trajectory.append(transl)
            else:
                trajectory.append(np.array([0, 0, 0]))

        return np.stack(trajectory)

    def ensure_gravity_alignment(
        self,
        trajectory: np.ndarray,
        up_axis: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure trajectory is gravity-aligned.

        Args:
            trajectory: Root positions (T, 3)
            up_axis: Which axis is up

        Returns:
            Aligned trajectory and alignment transform
        """
        mean_height = np.mean(trajectory[:, up_axis])
        return trajectory, np.eye(3)


def compose_people_in_world(
    person_poses_list: List,
    camera_poses: Dict,
    output_dir: Path,
    config: Dict
) -> Dict:
    """
    Compose all people in shared world frame.

    Args:
        person_poses_list: List of PersonPose objects from Stage 3
        camera_poses: Camera poses from Stage 1
        output_dir: Output directory
        config: Configuration dictionary

    Returns:
        Dictionary with world-frame results
    """
    print("\n" + "="*80)
    print("STAGE 4: WORLD-SPACE TRANSFORMATION")
    print("="*80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_data = np.load(camera_poses) if isinstance(camera_poses, (str, Path)) else camera_poses

    transformer = WorldTransformer()
    transformer.set_camera_poses(camera_data)

    all_people_world = {}
    start_time = time.time()

    for person_idx, person_pose in enumerate(person_poses_list):
        print(f"[Stage4] Transforming person {person_pose.track_id}...")

        person_dir = output_dir / f'person_{person_pose.track_id:03d}'
        person_dir.mkdir(exist_ok=True)

        smpl_file = Path(config.get('camera_frame_smpl_dir', '.')) / \
                    f'person_{person_pose.track_id:03d}' / 'smpl_params_camera.npz'

        if not smpl_file.exists():
            print(f"[Stage4] Warning: SMPL file not found: {smpl_file}")
            continue

        camera_smpl = np.load(smpl_file)
        poses_array = camera_smpl.get('poses')
        betas = camera_smpl.get('betas')
        global_orients = camera_smpl.get('global_orient')
        transl = camera_smpl.get('transl')

        joints_file = Path(config.get('camera_frame_smpl_dir', '.')) / \
                      f'person_{person_pose.track_id:03d}' / 'joints_camera.npy'

        joints_all = None
        if joints_file.exists():
            joints_all = np.load(joints_file)

        poses_world = []
        joints_world_list = []
        trajectories = []

        for frame_idx, frame_id in enumerate(person_pose.frames):
            if frame_id >= len(transformer.camera_poses_world):
                print(f"[Stage4] Warning: Frame {frame_id} out of range")
                continue

            frame_camera_pose = transformer.camera_poses_world[frame_id]

            smpl_camera = {
                'poses': poses_array[frame_idx] if poses_array is not None else None,
                'betas': betas,
                'global_orient': global_orients[frame_idx] if global_orients is not None else None,
                'transl': transl[frame_idx] if transl is not None else None
            }

            joints_camera = joints_all[frame_idx] if joints_all is not None else None

            world_params = transformer.transform_person_to_world(
                smpl_camera,
                joints_camera,
                frame_camera_pose
            )

            poses_world.append(world_params)
            joints_world_list.append(world_params.get('joints_world'))
            trajectories.append(world_params['transl_world'])

        trajectory_world = np.stack(trajectories) if trajectories else None

        if poses_world:
            world_poses = np.stack([p['poses'] for p in poses_world if p['poses'] is not None])
            world_orients = np.stack([p['global_orient_world'] for p in poses_world])
            world_transl = np.stack([p['transl_world'] for p in poses_world])

            np.savez(
                person_dir / 'smpl_params_world.npz',
                poses=world_poses,
                betas=betas,
                global_orient=world_orients,
                transl=world_transl
            )

            np.save(person_dir / 'trajectory_world.npy', trajectory_world)

            if joints_world_list[0] is not None:
                joints_world = np.stack(joints_world_list)
                np.save(person_dir / 'joints_world.npy', joints_world)

        all_people_world[person_pose.track_id] = {
            'frames': person_pose.frames,
            'trajectory_world': trajectory_world,
            'num_frames': len(person_pose.frames)
        }

    elapsed = time.time() - start_time

    combined_file = output_dir / 'all_people_world.npz'
    np.savez(combined_file, **{
        f'person_{track_id}': data
        for track_id, data in all_people_world.items()
    })

    metadata = {
        'num_people': len(all_people_world),
        'people': list(all_people_world.keys()),
        'world_origin': [0.0, 0.0, 0.0],
        'up_axis': 1,
        'processing_time': elapsed
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "-"*80)
    print("STAGE 4 SUMMARY")
    print("-"*80)
    print(f"Number of people transformed: {len(all_people_world)}")

    for track_id, data in all_people_world.items():
        print(f"  - Person {track_id}: {data['num_frames']} frames")
        if data['trajectory_world'] is not None:
            traj = data['trajectory_world']
            traj_length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            print(f"    - Trajectory length: {traj_length:.2f}m")
            print(f"    - Mean position: ({traj.mean(0)[0]:.2f}, "
                  f"{traj.mean(0)[1]:.2f}, {traj.mean(0)[2]:.2f})")

    print(f"Processing time: {elapsed:.2f} seconds")
    print("-"*80 + "\n")

    return {
        'all_people_world': all_people_world,
        'processing_time': elapsed,
        'output_dir': str(output_dir)
    }


def run_world_transform(
    person_poses_list: List,
    camera_poses_path: Path,
    stage3_output_dir: Path,
    stage4_output_dir: Path,
    config: Dict
) -> Dict:
    """
    Run Stage 4: World-space transformation.

    Args:
        person_poses_list: List of PersonPose objects from Stage 3
        camera_poses_path: Path to cameras.npz from Stage 1
        stage3_output_dir: Output directory from Stage 3
        stage4_output_dir: Output directory for Stage 4
        config: Configuration dictionary

    Returns:
        Dictionary with world-frame results
    """
    config = dict(config)
    config['camera_frame_smpl_dir'] = stage3_output_dir

    camera_poses = np.load(camera_poses_path)

    results = compose_people_in_world(
        person_poses_list,
        camera_poses,
        stage4_output_dir,
        config
    )

    return results


if __name__ == "__main__":
    """Test Stage 4 independently"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Stage 4: World Transform")
    parser.add_argument("--camera_poses", type=str, required=True,
                        help="Path to cameras.npz from Stage 1")
    parser.add_argument("--stage3_dir", type=str, required=True,
                        help="Stage 3 output directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")

    args = parser.parse_args()

    config = {}

    results = run_world_transform(
        person_poses_list=[],
        camera_poses_path=Path(args.camera_poses),
        stage3_output_dir=Path(args.stage3_dir),
        stage4_output_dir=Path(args.output),
        config=config
    )

    print("[Stage4] Test complete!")
