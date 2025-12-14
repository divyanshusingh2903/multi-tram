#!/usr/bin/env python3
"""
Stage 5: Optional Refinement (SLAHMR-based Optimization) Runner
Standalone script for SLURM execution
Depends on: All previous stage outputs
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stages.stage5_refinement import run_refinement
import yaml


def main():
    parser = argparse.ArgumentParser(description='Run Stage 5: Multi-Person Refinement')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--stage1_dir', type=str, required=True,
                        help='Directory containing Stage 1 outputs')
    parser.add_argument('--stage2_dir', type=str, required=True,
                        help='Directory containing Stage 2 outputs')
    parser.add_argument('--stage3_dir', type=str, required=True,
                        help='Directory containing Stage 3 outputs')
    parser.add_argument('--stage4_dir', type=str, required=True,
                        help='Directory containing Stage 4 outputs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for refined results')
    parser.add_argument('--config', type=str, default='configs/refinement.yaml',
                        help='Path to refinement configuration file')
    parser.add_argument('--person_ids', type=str, nargs='+', default=None,
                        help='Specific person IDs to process (default: all)')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of L-BFGS iterations')
    parser.add_argument('--lambda_depth', type=float, default=5.0,
                        help='Weight for depth consistency loss')
    parser.add_argument('--lambda_scale', type=float, default=2.0,
                        help='Weight for multi-person scale loss')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("STAGE 5: OPTIONAL REFINEMENT (SLAHMR-BASED)")
    print("="*80)
    print(f"Video: {args.video}")
    print(f"Stage 1 input: {args.stage1_dir}")
    print(f"Stage 2 input: {args.stage2_dir}")
    print(f"Stage 3 input: {args.stage3_dir}")
    print(f"Stage 4 input: {args.stage4_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Lambda depth: {args.lambda_depth}")
    print(f"Lambda scale: {args.lambda_scale}")
    if args.person_ids:
        print(f"Processing specific persons: {args.person_ids}")
    print("="*80 + "\n")

    # Verify all previous stage outputs exist
    stage1_dir = Path(args.stage1_dir)
    stage2_dir = Path(args.stage2_dir)
    stage3_dir = Path(args.stage3_dir)
    stage4_dir = Path(args.stage4_dir)

    cameras_file = stage1_dir / 'cameras.npz'
    tracks_file = stage2_dir / 'tracks.npz'

    required_files = [
        (cameras_file, "Stage 1 cameras.npz"),
        (tracks_file, "Stage 2 tracks.npz"),
    ]

    for file_path, name in required_files:
        if not file_path.exists():
            print(f"ERROR: {name} not found: {file_path}")
            print(f"Please run previous stages first!")
            return 1

    if not stage3_dir.exists():
        print(f"ERROR: Stage 3 output directory not found: {stage3_dir}")
        return 1

    if not stage4_dir.exists():
        print(f"ERROR: Stage 4 output directory not found: {stage4_dir}")
        return 1

    # Load camera and tracking data
    camera_data = np.load(cameras_file)
    tracking_data = np.load(tracks_file)

    print(f"[Stage 5] Loaded camera data: {camera_data['poses'].shape[0]} frames")
    print(f"[Stage 5] Loaded tracking data")

    # Load config if it exists
    config = {}
    config_path = project_root / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found, using defaults")

    # Override parameters if specified
    if args.num_iterations:
        config['num_iterations'] = args.num_iterations
    if args.lambda_depth:
        config['lambda_depth'] = args.lambda_depth
    if args.lambda_scale:
        config['lambda_scale'] = args.lambda_scale

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run refinement
    results = run_refinement(
        video_path=args.video,
        camera_data=camera_data,
        tracking_data=tracking_data,
        stage3_output_dir=stage3_dir,
        stage4_output_dir=stage4_dir,
        output_dir=output_dir,
        config=config,
        person_ids=args.person_ids
    )

    print("\n" + "="*80)
    print("STAGE 5 COMPLETED")
    print("="*80)
    print(f"Refined {len(results.get('person_results', {}))} people")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
