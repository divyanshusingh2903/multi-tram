# multi-tram

A research codebase that extends TRAM for multibody tracking and dense reconstruction. This repository contains experiments, datasets, and utilities to develop and evaluate methods that recover poses, shapes, and trajectories for multiple interacting rigid and articulated bodies.

## Goals
- Extend TRAM to handle multiple concurrently moving objects.
- Jointly estimate per-object tracking and dense 3D reconstruction.
- Provide a modular codebase for experiments and benchmarking.

## Key ideas
- Multi-instance data association and temporal consistency.
- Scene-aware bundle adjustment that enforces multi-body constraints.
- Integration of learned priors for object shape and motion.

## Repository layout
- `data/` - dataset download scripts and example inputs
- `configs/` - experiment and model configuration files
- `src/` - core implementation (tracking, reconstruction, optimization)
- `experiments/` - training and evaluation scripts
- `notebooks/` - analysis and visualization notebooks
- `docs/` - design notes and papers
- `benchmarks/` - evaluation code and metrics

## Getting started
1. Clone the repo:
    git clone <repo-url>
2. Create environment and install dependencies:
    python -m venv .venv
    source .venv/bin/activate   # or .venv\Scripts\activate on Windows
    pip install -r requirements.txt
3. Download example data:
    ./scripts/download_example_data.sh
4. Run a demo experiment:
    python experiments/run_demo.py --config configs/demo.yaml

(Adjust commands to match your environment and shells.)

## Usage
- Tracking module: `src/tracking/`
- Reconstruction module: `src/reconstruction/`
- Optimization / BA: `src/optimization/`
- Visualize results: `notebooks/visualize_results.ipynb`

## Configuration
Use YAML config files under `configs/` to specify:
- dataset and input paths
- model checkpoints and hyperparameters
- optimization schedule and constraints
- evaluation metrics

## Experiments & Evaluation
- Include ablations for data association, multi-body priors, and bundle adjustment choices.
- Report metrics for tracking accuracy (MOTA/MOTP or task-specific) and reconstruction error (Chamfer, reprojection).

## Development & Contributing
- Follow the Python style and add tests under `tests/`.
- Open issues for feature requests and bugs.
- Use branches for experiments: `feat/*`, `exp/*`, `fix/*`.

## Citation
If you use this work in publications, please cite TRAM and this repository (add citation details here).

## License
Add an appropriate open-source license (e.g., MIT, Apache 2.0) in `LICENSE`.

## TODO
- Integrate multi-object data association module.
- Add synthetic multi-body dataset generator.
- Implement joint multi-body bundle adjustment and evaluation scripts.

Contact: maintainers (add email or GitHub handles)