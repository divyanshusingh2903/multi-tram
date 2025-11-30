"""
Stage 1: Camera Estimation
Uses VGGT for fast feed-forward estimation with DROID-SLAM fallback
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional
import time
import json

from src.models.vggt_wrapper import VGGTWrapper
from src.models.droid_slam_wrapper import DROIDSLAMWrapper


def load_video(video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Load video frames

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load

    Returns:
        frames: numpy array (T, H, W, 3) in RGB [0, 255]
    """
    print(f"[Stage1] Loading video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

    frames = np.stack(frames, axis=0)
    print(f"  - Loaded {len(frames)} frames with shape {frames.shape}")

    return frames


def save_results(results: Dict, output_dir: Path):
    """Save camera estimation results"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage1] Saving results to {output_dir}")

    # Save poses and intrinsics
    np.savez(
        output_dir / 'cameras.npz',
        poses=results['poses'],
        intrinsics=results['intrinsics'],
        original_size=results['original_size']
    )

    # Save depth maps
    depths_dir = output_dir / 'depth_maps'
    depths_dir.mkdir(exist_ok=True)
    for t, depth in enumerate(results['depths']):
        np.save(depths_dir / f'depth_{t:04d}.npy', depth)

    # Save point cloud
    if len(results['point_cloud']) > 0:
        save_point_cloud(
            results['point_cloud'],
            results['point_colors'],
            output_dir / 'point_cloud.ply'
        )

    # Save metadata
    metadata = {
        'num_frames': len(results['poses']),
        'image_size': results['original_size'],
        'num_points': len(results['point_cloud'])
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[Stage1] Results saved successfully")


def save_point_cloud(points: np.ndarray, colors: np.ndarray, output_path: Path):
    """Save point cloud in PLY format"""
    print(f"[Stage1] Saving point cloud with {len(points)} points")

    with open(output_path, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write points
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def visualize_trajectory(poses: np.ndarray, output_path: Path):
    """Create visualization of camera trajectory"""
    import matplotlib.pyplot as plt

    # Extract camera centers
    centers = []
    for pose in poses:
        pose_inv = np.linalg.inv(pose)
        center = pose_inv[:3, 3]
        centers.append(center)
    centers = np.array(centers)

    # Create figure with multiple views
    fig = plt.figure(figsize=(18, 6))

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(centers[:, 0], centers[:, 1], centers[:, 2], 'b-', linewidth=2)
    ax1.scatter(centers[0, 0], centers[0, 1], centers[0, 2],
                c='g', s=100, marker='o', label='Start')
    ax1.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2],
                c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Camera Trajectory')
    ax1.legend()

    # Top-down view (X-Z plane)
    ax2 = fig.add_subplot(132)
    ax2.plot(centers[:, 0], centers[:, 2], 'b-', linewidth=2)
    ax2.scatter(centers[0, 0], centers[0, 2], c='g', s=100, marker='o', label='Start')
    ax2.scatter(centers[-1, 0], centers[-1, 2], c='r', s=100, marker='x', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Top-Down View')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()

    # Side view (Y-Z plane)
    ax3 = fig.add_subplot(133)
    ax3.plot(centers[:, 2], centers[:, 1], 'b-', linewidth=2)
    ax3.scatter(centers[0, 2], centers[0, 1], c='g', s=100, marker='o', label='Start')
    ax3.scatter(centers[-1, 2], centers[-1, 1], c='r', s=100, marker='x', label='End')
    ax3.set_xlabel('Z (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Side View')
    ax3.axis('equal')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Stage1] Trajectory visualization saved to {output_path}")


def run_camera_estimation(video_path: str, output_dir: Path, config: Dict, max_frames: Optional[int] = None) -> Dict:
    """
    Run camera estimation stage

    Args:
        video_path: Path to input video
        output_dir: Directory to save results
        config: Configuration dictionary
        max_frames: Maximum number of frames to process

    Returns:
        Dictionary with camera estimation results
    """
    print("\n" + "="*80)
    print("STAGE 1: CAMERA ESTIMATION")
    print("="*80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video
    frames = load_video(video_path, max_frames=max_frames)

    # Initialize camera estimator
    method = config.get('method', 'vggt')  # 'vggt' or 'droid'
    device = config.get('device', 'cuda')

    start_time = time.time()

    try:
        if method == 'vggt':
            print("[Stage1] Using VGGT for camera estimation")
            model_path = config.get('vggt_model_path', 'data/pretrained_models/vggt/model.pth')

            estimator = VGGTWrapper(
                model_path=model_path,
                device=device,
                max_frames=config.get('max_frames', 100),
                image_size=config.get('image_size', 512)
            )

            results = estimator.estimate_cameras(frames)

        elif method == 'droid':
            print("[Stage1] Using DROID-SLAM for camera estimation")
            weights_path = config.get('droid_weights_path', None)

            estimator = DROIDSLAMWrapper(
                weights_path=weights_path,
                device=device,
                buffer_size=config.get('buffer_size', 512)
            )

            results = estimator.estimate_cameras(frames)

        else:
            raise ValueError(f"Unknown camera estimation method: {method}")

        elapsed_time = time.time() - start_time
        print(f"\n[Stage1] Camera estimation completed in {elapsed_time:.2f} seconds")
        print(f"  - Speed: {len(frames)/elapsed_time:.2f} fps")

    except Exception as e:
        print(f"\n[Stage1] Error with {method.upper()}: {e}")

        # Fallback to DROID-SLAM if VGGT fails
        if method == 'vggt' and config.get('fallback_to_droid', True):
            print("[Stage1] Falling back to DROID-SLAM...")

            estimator = DROIDSLAMWrapper(device=device)
            results = estimator.estimate_cameras(frames)

            elapsed_time = time.time() - start_time
            print(f"\n[Stage1] Camera estimation completed in {elapsed_time:.2f} seconds")
        else:
            raise

    # Save results
    save_results(results, output_dir)

    # Visualize trajectory
    visualize_trajectory(results['poses'], output_dir / 'trajectory.png')

    # Print summary statistics
    print("\n" + "-"*80)
    print("STAGE 1 SUMMARY")
    print("-"*80)
    print(f"Number of frames: {len(results['poses'])}")
    print(f"Image size: {results['original_size']}")
    print(f"Point cloud size: {len(results['point_cloud'])} points")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average speed: {len(frames)/elapsed_time:.2f} fps")

    # Compute trajectory statistics
    centers = []
    for pose in results['poses']:
        pose_inv = np.linalg.inv(pose)
        center = pose_inv[:3, 3]
        centers.append(center)
    centers = np.array(centers)

    trajectory_length = np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))
    print(f"Total trajectory length: {trajectory_length:.2f} meters")
    print(f"Average camera height: {np.mean(centers[:, 1]):.2f} meters")
    print("-"*80 + "\n")

    return results


if __name__ == "__main__":
    """Test Stage 1 independently"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Test Stage 1: Camera Estimation")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default="configs/vggt.yaml", help="Config file")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--method", type=str, default="vggt", choices=['vggt', 'droid'])

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'method': args.method,
            'device': 'cuda',
            'max_frames': 100,
            'image_size': 512,
            'fallback_to_droid': True
        }

    # Override method if specified
    config['method'] = args.method

    # Run Stage 1
    results = run_camera_estimation(
        video_path=args.video,
        output_dir=Path(args.output),
        config=config,
        max_frames=args.max_frames
    )

    print("\n[Stage1] Test complete!")