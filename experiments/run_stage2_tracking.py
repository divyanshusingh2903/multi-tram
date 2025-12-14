#!/usr/bin/env python3
"""
Stage 2: Multi-Person Tracking Runner
Standalone script for SLURM execution
Depends on: Stage 1 outputs (cameras.npz, depth_maps/)
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
    parser = argparse.ArgumentParser(description='Run Stage 2: Multi-Person Tracking')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--stage1_dir', type=str, required=True,
                        help='Directory containing Stage 1 outputs (cameras.npz)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for tracking results')
    parser.add_argument('--config', type=str, default='configs/tracking.yaml',
                        help='Path to tracking configuration file')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Detection confidence threshold')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("STAGE 2: MULTI-PERSON TRACKING")
    print("="*80)
    print(f"Video: {args.video}")
    print(f"Stage 1 input: {args.stage1_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print("="*80 + "\n")

    # Verify Stage 1 outputs exist
    stage1_dir = Path(args.stage1_dir)
    cameras_file = stage1_dir / 'cameras.npz'

    if not cameras_file.exists():
        print(f"ERROR: Stage 1 output not found: {cameras_file}")
        print("Please run Stage 1 first!")
        return 1

    # Load camera data to check
    camera_data = np.load(cameras_file)
    print(f"[Stage 2] Loaded camera data: {camera_data['poses'].shape[0]} frames")

    # Load config if it exists
    config = {}
    config_path = project_root / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found, using defaults")

    # Override confidence threshold if specified
    if args.confidence_threshold:
        config['confidence_threshold'] = args.confidence_threshold

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run tracking
    results = run_tracking(
        video_path=args.video,
        camera_data=camera_data,
        output_dir=output_dir,
        config=config,
        max_frames=args.max_frames
    )

    print("\n" + "="*80)
    print("STAGE 2 COMPLETED")
    print("="*80)
    print(f"Total tracks: {results.get('num_tracks', 'N/A')}")
    print(f"Total frames: {results.get('num_frames', 'N/A')}")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
