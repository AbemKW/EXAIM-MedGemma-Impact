# CLI Module

Command-line entry points for trace generation, replay, validation, and metrics.

## Commands
- `make_traces.py`: Generate timed traces from MAC.
- `run_variants.py`: Replay traces through evaluation variants.
- `compute_metrics.py`: Compute per-case and aggregate metrics.
- `error_analysis.py`: Extract trace excerpts and decisions for outlier cases.
- `validate_traces.py`: Validate trace integrity.
- `validate_logs.py`: Validate log files against schemas.
- `generate_stoplists.py`: Build stoplists from frozen traces.
- `replay_trace.py`: Inspect trace replay output.
- `calibrate_tokengate.py`: Run TokenGate calibration.

Each CLI is a thin argparse wrapper over a core module in `evals/src/`.

## Usage
Run via modules from the repo root, e.g.:

```bash
python -m evals.cli.make_traces --config configs/mas_generation.yaml --dry-run
```
