# TokenGate Calibration Module

Calibration pipeline for TokenGate policies, from grid generation to policy selection.

## Files
- `runner.py`: Orchestrates calibration runs and output artifacts.
- `grid.py`: Parameter grid generation and validity filtering.
- `metrics.py`: Per-case metrics and aggregation helpers.
- `selection.py`: Pareto/utopia selection and weighted fallbacks.
- `io.py`: Manifest/config loading and artifact writing.
- `models.py`: Dataclasses for policies and metrics.
