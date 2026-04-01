#!/usr/bin/env python3
"""
Stage 2: World-Frame Multi-Person Tracking Runner
Standalone script for SLURM execution
Uses YOLO detection + custom world-frame tracking

Depends on: Stage 1 outputs (camera_data.npz, depth_maps.npz)
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stages.stage2_tracking import run_tracking
import yaml


def main():
    parser = argparse.ArgumentParser(
        description='Run Stage 2: World-Frame Multi-Person Tracking (YOLO + Custom Tracker)'
    )
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--stage1_dir', type=str, required=True,
                        help='Directory containing Stage 1 VGGT outputs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for tracking results')
    parser.add_argument('--config', type=str, default='configs/tracking.yaml',
                        help='Path to tracking configuration file')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process')

    # YOLO/Tracking specific arguments (override config)
    parser.add_argument('--yolo_model', type=str, default=None,
                        help='YOLO model variant (yolov8s, yolov11s, etc.)')
    parser.add_argument('--reid_backbone', type=str, default=None,
                        help='ReID backbone (resnet50, osnet, simple)')
    parser.add_argument('--conf_threshold', type=float, default=None,
                        help='YOLO confidence threshold')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("STAGE 2: WORLD-FRAME MULTI-PERSON TRACKING")
    print("Using: YOLO Detection + Custom World-Frame Tracker")
    print("="*80)
    print(f"Video: {args.video}")
    print(f"Stage 1 input (VGGT): {args.stage1_dir}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    # Verify Stage 1 outputs exist
    stage1_dir = Path(args.stage1_dir)

    # Check for required VGGT outputs
    camera_file = stage1_dir / 'camera_data.npz'
    if not camera_file.exists():
        # Try alternative name
        camera_file = stage1_dir / 'cameras.npz'

    if not camera_file.exists():
        print(f"ERROR: Stage 1 camera data not found in {stage1_dir}")
        print("Expected: camera_data.npz or cameras.npz")
        print("Please run Stage 1 first!")
        return 1

    print(f"[Stage 2] Found camera data: {camera_file}")

    # Check for depth maps (optional but recommended)
    depth_file = stage1_dir / 'depth_maps.npz'
    if depth_file.exists():
        print(f"[Stage 2] Found depth maps: {depth_file}")
    else:
        print(f"[Stage 2] Warning: No depth maps found, will use default depths")

    # Load config
    config = {}
    config_path = project_root / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[Stage 2] Loaded config from: {config_path}")
    else:
        print(f"[Stage 2] Warning: Config file {config_path} not found, using defaults")
        config = {
            'yolo_model': 'yolov8s',
            'reid_backbone': 'resnet50',
            'device': 'cuda',
            'conf_threshold': 0.3,
            'max_age': 30,
            'min_hits': 3,
            'alpha': 0.4,
            'beta': 0.4,
            'gamma': 0.2
        }

    # Override config with command-line arguments
    if args.yolo_model:
        config['yolo_model'] = args.yolo_model
    if args.reid_backbone:
        config['reid_backbone'] = args.reid_backbone
    if args.conf_threshold:
        config['conf_threshold'] = args.conf_threshold
    if args.device:
        config['device'] = args.device

    print(f"\n[Stage 2] Configuration:")
    print(f"  - YOLO model: {config.get('yolo_model', 'yolov8s')}")
    print(f"  - ReID backbone: {config.get('reid_backbone', 'resnet50')}")
    print(f"  - Device: {config.get('device', 'cuda')}")
    print(f"  - Confidence threshold: {config.get('conf_threshold', 0.3)}")
    print(f"  - Cost weights: α={config.get('alpha', 0.4)}, β={config.get('beta', 0.4)}, γ={config.get('gamma', 0.2)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run world-frame tracking
    print("\n[Stage 2] Starting world-frame tracking...")

    results = run_tracking(
        video_path=args.video,
        vggt_output_dir=stage1_dir,
        output_dir=output_dir,
        config=config,
        max_frames=args.max_frames
    )

    print("\n" + "="*80)
    print("STAGE 2 COMPLETED")
    print("="*80)
    print(f"Total people tracked: {results.get('num_people', 'N/A')}")
    print(f"Total frames: {results.get('num_frames', 'N/A')}")
    print(f"Processing speed: {results.get('fps', 'N/A'):.2f} fps")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")

    # Print track summary
    if 'trajectories' in results:
        print("Track Summary:")
        for track_id, traj in results['trajectories'].items():
            print(f"  Track {track_id}: {len(traj)} frames")

    return 0


if __name__ == '__main__':
    sys.exit(main())
