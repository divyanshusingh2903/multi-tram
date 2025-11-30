"""
Multi-TRAM Pipeline: Complete multi-person 3D reconstruction from video
Orchestrates all stages: Camera → Tracking → Pose → World Transform → [Optional Refinement]
"""
import argparse
import sys
from pathlib import Path
import yaml
import json
import time
from typing import Dict, Optional

# Import stage modules
from src.stages.stage1_camera import run_camera_estimation
from src.stages.stage2_tracking import run_tracking
from src.stages.stage3_pose import run_pose_estimation, PersonPose
from src.stages.stage4_world_transform import run_world_transform
from src.stages.stage5_refinement import run_refinement


class MultiTRAMPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config_path: Path, output_base_dir: Path):
        """
        Initialize pipeline.

        Args:
            config_path: Path to configuration YAML file
            output_base_dir: Base directory for all outputs
        """
        self.config_path = Path(config_path)
        self.output_base_dir = Path(output_base_dir)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("\n" + "="*80)
        print("MULTI-TRAM PIPELINE INITIALIZATION")
        print("="*80)
        print(f"Configuration: {config_path}")
        print(f"Output directory: {output_base_dir}")
        print("="*80 + "\n")

        # Create output directories
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        skip_stages: Optional[list] = None
    ) -> Dict:
        """
        Run complete multi-person 3D reconstruction pipeline.

        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (for testing)
            skip_stages: List of stage numbers to skip (e.g., [5] to skip refinement)

        Returns:
            Dictionary with complete results
        """
        if skip_stages is None:
            skip_stages = []

        print("\n" + "="*80)
        print("MULTI-TRAM COMPLETE PIPELINE")
        print("="*80)
        print(f"Input video: {video_path}")
        if max_frames:
            print(f"Max frames: {max_frames}")
        print("="*80 + "\n")

        pipeline_start = time.time()
        results = {}

        # Stage 1: Camera Estimation
        print("[PIPELINE] Starting Stage 1: Camera Estimation...")
        stage1_start = time.time()

        stage1_output_dir = self.output_base_dir / '1_camera_estimation'
        camera_results = run_camera_estimation(
            video_path=video_path,
            output_dir=stage1_output_dir,
            config=self.config.get('stage1', {}),
            max_frames=max_frames
        )

        results['stage1'] = {
            'output_dir': str(stage1_output_dir),
            'num_frames': camera_results.get('num_frames'),
            'processing_time': time.time() - stage1_start
        }

        print(f"[PIPELINE] Stage 1 completed in {results['stage1']['processing_time']:.2f}s\n")

        # Load video for later stages
        import cv2
        import numpy as np

        print("[PIPELINE] Loading video frames...")
        cap = cv2.VideoCapture(video_path)

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        frames = np.stack(frames)
        print(f"[PIPELINE] Loaded {len(frames)} frames\n")

        # Stage 2: Multi-person Tracking
        if 2 not in skip_stages:
            print("[PIPELINE] Starting Stage 2: Multi-person Tracking...")
            stage2_start = time.time()

            stage2_output_dir = self.output_base_dir / '2_tracking'
            tracking_results = run_tracking(
                video_path=video_path,
                camera_poses=camera_results.get('poses'),
                output_dir=stage2_output_dir,
                config=self.config.get('stage2', {}),
                max_frames=max_frames
            )

            results['stage2'] = {
                'output_dir': str(stage2_output_dir),
                'num_people': tracking_results.get('num_people'),
                'num_frames': tracking_results.get('num_frames'),
                'processing_time': time.time() - stage2_start
            }

            print(f"[PIPELINE] Stage 2 completed in {results['stage2']['processing_time']:.2f}s\n")
        else:
            print("[PIPELINE] Skipping Stage 2\n")
            tracking_results = {'detections': [], 'masks': []}

        # Stage 3: Per-person Pose Estimation
        if 3 not in skip_stages:
            print("[PIPELINE] Starting Stage 3: Per-person Pose Estimation...")
            stage3_start = time.time()

            stage3_output_dir = self.output_base_dir / '3_pose_estimation'
            pose_results = run_pose_estimation(
                video_frames=frames,
                tracking_results=tracking_results,
                output_dir=stage3_output_dir,
                config=self.config.get('stage3', {})
            )

            results['stage3'] = {
                'output_dir': str(stage3_output_dir),
                'num_people': pose_results.get('num_people'),
                'processing_time': time.time() - stage3_start
            }

            person_poses = pose_results.get('person_poses', [])

            print(f"[PIPELINE] Stage 3 completed in {results['stage3']['processing_time']:.2f}s\n")
        else:
            print("[PIPELINE] Skipping Stage 3\n")
            person_poses = []

        # Stage 4: World-space Transformation
        if 4 not in skip_stages:
            print("[PIPELINE] Starting Stage 4: World-space Transformation...")
            stage4_start = time.time()

            stage4_output_dir = self.output_base_dir / '4_world_space'
            world_results = run_world_transform(
                person_poses_list=person_poses,
                camera_poses_path=stage1_output_dir / 'cameras.npz',
                stage3_output_dir=stage3_output_dir,
                stage4_output_dir=stage4_output_dir,
                config=self.config.get('stage4', {})
            )

            results['stage4'] = {
                'output_dir': str(stage4_output_dir),
                'num_people': len(person_poses),
                'processing_time': time.time() - stage4_start
            }

            print(f"[PIPELINE] Stage 4 completed in {results['stage4']['processing_time']:.2f}s\n")
        else:
            print("[PIPELINE] Skipping Stage 4\n")

        # Stage 5: Optional Multi-person Refinement
        if 5 not in skip_stages:
            print("[PIPELINE] Starting Stage 5: Optional Multi-person SMPL Refinement...")
            stage5_start = time.time()

            stage5_output_dir = self.output_base_dir / '5_refinement'
            refinement_results = run_refinement(
                stage4_output_dir=stage4_output_dir,
                stage1_output_dir=stage1_output_dir,
                stage2_output_dir=stage2_output_dir,
                stage3_output_dir=stage3_output_dir,
                stage5_output_dir=stage5_output_dir,
                config=self.config.get('stage5', {})
            )

            results['stage5'] = {
                'status': refinement_results.get('status', 'unknown'),
                'output_dir': str(stage5_output_dir),
                'num_people': refinement_results.get('num_people', 0),
                'processing_time': time.time() - stage5_start
            }

            print(f"[PIPELINE] Stage 5 completed in {results['stage5']['processing_time']:.2f}s\n")
        else:
            print("[PIPELINE] Skipping Stage 5 (optional refinement)\n")

        # Total timing
        total_time = time.time() - pipeline_start

        # Save pipeline results summary
        summary = {
            'video_path': video_path,
            'num_frames': len(frames),
            'output_directory': str(self.output_base_dir),
            'total_processing_time': total_time,
            'stages': results,
            'config': self.config
        }

        summary_path = self.output_base_dir / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print final summary
        print("\n" + "="*80)
        print("MULTI-TRAM PIPELINE COMPLETE")
        print("="*80)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average speed: {len(frames) / total_time:.2f} fps")
        print(f"\nResults saved to: {self.output_base_dir}")
        print("\nStage Summary:")
        print("-" * 80)

        for stage_num, stage_results in sorted(results.items()):
            if isinstance(stage_results, dict):
                stage_name = stage_num.replace('stage', 'Stage ')
                proc_time = stage_results.get('processing_time', 0)
                print(f"  {stage_name}: {proc_time:.2f}s")
                if 'num_people' in stage_results:
                    print(f"    - People: {stage_results['num_people']}")
                if 'num_frames' in stage_results:
                    print(f"    - Frames: {stage_results['num_frames']}")

        print("="*80 + "\n")

        return summary

    def run_stages(
        self,
        video_path: str,
        stages: list = [1, 2, 3, 4],
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        Run selected stages of the pipeline.

        Args:
            video_path: Path to input video
            stages: List of stage numbers to run (default: [1,2,3,4], excludes optional stage 5)
            max_frames: Maximum frames to process

        Returns:
            Results dictionary
        """
        skip_stages = [s for s in range(1, 7) if s not in stages]
        return self.run_full_pipeline(video_path, max_frames, skip_stages)


def main():
    """Command-line interface for pipeline"""
    parser = argparse.ArgumentParser(
        description="Multi-TRAM: Multi-person 3D reconstruction pipeline"
    )

    parser.add_argument(
        "video",
        type=str,
        help="Input video file path"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="results",
        help="Output directory (default: results)"
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/vggt.yaml",
        help="Configuration file (default: configs/vggt.yaml)"
    )

    parser.add_argument(
        "-m", "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to process (for testing)"
    )

    parser.add_argument(
        "-s", "--stages",
        type=int,
        nargs='+',
        default=[1, 2, 3, 4],
        help="Stages to run (default: 1 2 3 4)"
    )

    parser.add_argument(
        "--skip_stages",
        type=int,
        nargs='+',
        default=[],
        help="Stages to skip"
    )

    args = parser.parse_args()

    # Determine which stages to run
    if args.skip_stages:
        stages_to_run = [s for s in args.stages if s not in args.skip_stages]
    else:
        stages_to_run = args.stages

    # Create output directory with video name
    video_name = Path(args.video).stem
    output_dir = Path(args.output) / video_name

    # Initialize and run pipeline
    pipeline = MultiTRAMPipeline(
        config_path=args.config,
        output_base_dir=output_dir
    )

    try:
        results = pipeline.run_stages(
            video_path=args.video,
            stages=stages_to_run,
            max_frames=args.max_frames
        )

        print(f"\n[SUCCESS] Pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
