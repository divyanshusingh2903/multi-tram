#!/usr/bin/env python3
"""
Stage 4: World-Space Transformation Runner
Standalone script for SLURM execution
Depends on: Stage 1 outputs (cameras.npz), Stage 3 outputs (person_XXX/)
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stages.stage4_world_transform import run_world_transform
import yaml


def main():
    parser = argparse.ArgumentParser(description='Run Stage 4: World-Space Transformation')
    parser.add_argument('--stage1_dir', type=str, required=True,
                        help='Directory containing Stage 1 outputs (cameras.npz)')
    parser.add_argument('--stage3_dir', type=str, required=True,
                        help='Directory containing Stage 3 outputs (person_XXX/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for world-space results')
    parser.add_argument('--config', type=str, default='configs/world_transform.yaml',
                        help='Path to configuration file')
    parser.add_argument('--person_ids', type=str, nargs='+', default=None,
                        help='Specific person IDs to process (default: all)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("STAGE 4: WORLD-SPACE TRANSFORMATION")
    print("="*80)
    print(f"Stage 1 input: {args.stage1_dir}")
    print(f"Stage 3 input: {args.stage3_dir}")
    print(f"Output: {args.output_dir}")
    if args.person_ids:
        print(f"Processing specific persons: {args.person_ids}")
    print("="*80 + "\n")

    # Verify Stage 1 and Stage 3 outputs exist
    stage1_dir = Path(args.stage1_dir)
    stage3_dir = Path(args.stage3_dir)

    cameras_file = stage1_dir / 'cameras.npz'

    if not cameras_file.exists():
        print(f"ERROR: Stage 1 output not found: {cameras_file}")
        print("Please run Stage 1 first!")
        return 1

    if not stage3_dir.exists():
        print(f"ERROR: Stage 3 output directory not found: {stage3_dir}")
        print("Please run Stage 3 first!")
        return 1

    # Load camera data
    camera_data = np.load(cameras_file)
    print(f"[Stage 4] Loaded camera data: {camera_data['poses'].shape[0]} frames")

    # Find all person directories in Stage 3 output
    person_dirs = sorted(stage3_dir.glob('person_*'))
    if args.person_ids:
        person_dirs = [d for d in person_dirs if d.name.split('_')[1] in args.person_ids]

    print(f"[Stage 4] Found {len(person_dirs)} person directories")

    if len(person_dirs) == 0:
        print("ERROR: No person directories found in Stage 3 output!")
        return 1

    # Load config if it exists
    config = {}
    config_path = project_root / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found, using defaults")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run world transformation
    results = run_world_transform(
        camera_data=camera_data,
        stage3_output_dir=stage3_dir,
        output_dir=output_dir,
        config=config,
        person_ids=args.person_ids
    )

    print("\n" + "="*80)
    print("STAGE 4 COMPLETED")
    print("="*80)
    print(f"Transformed {len(results.get('person_results', {}))} people to world space")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
