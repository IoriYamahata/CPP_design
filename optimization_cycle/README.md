# optimization_cycle

Pipeline for iterative optimization (preprocessing, training, optimization, and I/O).

## Repository Layout

```
.
├── bayesian_optimization/
├── diffdock/
├── md_simulation/
├── prediction/
├── input_data/
├── results/
└── directory_setting.sh
```

## Requirements

- Python packages listed in `environment.yml`
- DiffDock as an external dependency (clone into `diffdock/DiffDock` or set `DIFFDOCK_ROOT`)
- MD tools installed locally (GROMACS, etc.)
  - `gmx` and `gmx_MMPBSA` available on `PATH`
- `obabel` available on `PATH` for DiffDock postprocessing

## Environment Setup (conda)

```
conda env create -f environment.yml
conda activate optimization_cycle
```

## Usage (Example)

1. Prepare `input_data/` and `results/` under the project root.
2. `directory_setting.sh` uses `PROJECT_DIR` (defaults to repo root).
3. Run the scripts in order below.

### Prediction

```
# Build training data
bash prediction/scripts/prediction_preprocess.sh

# Train + predict
bash prediction/scripts/prediction_training.sh

# Merge prediction outputs
bash prediction/scripts/prediction_postprocess.sh
```

### Bayesian Optimization

```
bash bayesian_optimization/scripts/bayesian_optimization.sh
```

### DiffDock (external)

```
# Prepare DiffDock inputs, then run inference and postprocess
bash diffdock/scripts/docking_preprocess.sh . <input_csv_name>
bash diffdock/scripts/docking_simulation.sh
bash diffdock/scripts/docking_postprocess.sh
```

### MD Simulation

After setting required environment variables in `md_simulation/scripts/CONFIG` (e.g., GROMACS paths), run:

```
bash md_simulation/scripts/run_1.sh
bash md_simulation/scripts/run_2.sh
bash md_simulation/scripts/run_3.sh
```

## Notes

- Prepare input data according to your experimental setup.
