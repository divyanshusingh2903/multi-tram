#!/usr/bin/env python3
"""
Stage 3: Per-Person Pose Estimation Runner
Standalone script for SLURM execution
Depends on: Stage 1 outputs (cameras.npz), Stage 2 outputs (tracks.npz)
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stages.stage3_pose import run_pose_estimation
import yaml


def main():
    parser = argparse.ArgumentParser(description='Run Stage 3: Per-Person Pose Estimation')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--stage1_dir', type=str, required=True,
                        help='Directory containing Stage 1 outputs (cameras.npz)')
    parser.add_argument('--stage2_dir', type=str, required=True,
                        help='Directory containing Stage 2 outputs (tracks.npz)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for pose estimation results')
    parser.add_argument('--config', type=str, default='configs/vimo.yaml',
                        help='Path to VIMO configuration file')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--person_ids', type=str, nargs='+', default=None,
                        help='Specific person IDs to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for VIMO processing')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("STAGE 3: PER-PERSON POSE ESTIMATION (VIMO)")
    print("="*80)
    print(f"Video: {args.video}")
    print(f"Stage 1 input: {args.stage1_dir}")
    print(f"Stage 2 input: {args.stage2_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    if args.person_ids:
        print(f"Processing specific persons: {args.person_ids}")
    print("="*80 + "\n")

    # Verify Stage 1 and Stage 2 outputs exist
    stage1_dir = Path(args.stage1_dir)
    stage2_dir = Path(args.stage2_dir)

    cameras_file = stage1_dir / 'cameras.npz'
    tracks_file = stage2_dir / 'tracks.npz'

    if not cameras_file.exists():
        print(f"ERROR: Stage 1 output not found: {cameras_file}")
        print("Please run Stage 1 first!")
        return 1

    if not tracks_file.exists():
        print(f"ERROR: Stage 2 output not found: {tracks_file}")
        print("Please run Stage 2 first!")
        return 1

    # Load camera and tracking data
    camera_data = np.load(cameras_file)
    tracking_data = np.load(tracks_file)

    print(f"[Stage 3] Loaded camera data: {camera_data['poses'].shape[0]} frames")
    print(f"[Stage 3] Loaded tracking data: {len(tracking_data.files)} tracks")

    # Load config if it exists
    config = {}
    config_path = project_root / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found, using defaults")

    # Override batch size if specified
    if args.batch_size:
        config['batch_size'] = args.batch_size

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pose estimation
    results = run_pose_estimation(
        video_path=args.video,
        camera_data=camera_data,
        tracking_data=tracking_data,
        output_dir=output_dir,
        config=config,
        max_frames=args.max_frames,
        person_ids=args.person_ids
    )

    print("\n" + "="*80)
    print("STAGE 3 COMPLETED")
    print("="*80)
    print(f"Processed {len(results.get('person_results', {}))} people")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
