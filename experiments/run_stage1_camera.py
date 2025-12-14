#!/usr/bin/env python3
"""
Stage 1: Camera Estimation Runner
Standalone script for SLURM execution
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stages.stage1_camera import run_camera_estimation
import yaml


def main():
    parser = argparse.ArgumentParser(description='Run Stage 1: Camera Estimation')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for camera results')
    parser.add_argument('--config', type=str, default='configs/vggt.yaml',
                        help='Path to configuration file')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--method', type=str, choices=['vggt', 'droid'], default='vggt',
                        help='Camera estimation method')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("STAGE 1: CAMERA ESTIMATION")
    print("="*80)
    print(f"Video: {args.video}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"Max frames: {args.max_frames}")
    print("="*80 + "\n")

    # Load config
    config_path = project_root / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override method if specified
    if args.method:
        config['method'] = args.method

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run camera estimation
    results = run_camera_estimation(
        video_path=args.video,
        output_dir=output_dir,
        config=config,
        max_frames=args.max_frames
    )

    print("\n" + "="*80)
    print("STAGE 1 COMPLETED")
    print("="*80)
    print(f"Processed {results.get('num_frames', 'N/A')} frames")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
