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
- `calibrate_v3.py`: Compute V3 chunk size from V0 TokenGate flush logs.

## V3 Calibration Usage

The `calibrate_v3.py` command accepts either a directory or individual files for `--v0-run-log`:

```bash
# Using a directory (recommended - automatically finds all .jsonl.gz files):
python -m evals.cli.calibrate_v3 \
    --case-list data/manifests/<case_list.jsonl> \
    --v0-run-log data/runs/V0 \
    --output data/calibration/v3_calibration_report.json

# Or specify individual files (can be repeated):
python -m evals.cli.calibrate_v3 \
    --case-list data/manifests/<case_list.jsonl> \
    --v0-run-log data/runs/V0/case-1.trace.jsonl.gz \
    --v0-run-log data/runs/V0/case-2.trace.jsonl.gz \
    --output data/calibration/v3_calibration_report.json
```

## V3 Calibration Provenance
`calibrate_v3.py` writes `data/calibration/v3_calibration_report.json` with:
- trace dataset hash
- V0 TokenGate config hash
- EXAID commit hash
- V0 run log hashes

V3 runtime loading validates these fields to ensure the calibrated
`chunk_size_ctu` matches the intended dataset and TokenGate policy.
Set `EXAID_ALLOW_COMMIT_MISMATCH=1` (legacy env var name) to allow commit mismatches while
preserving provenance in the report.

Each CLI is a thin argparse wrapper over a core module in `evals/src/`.

## Usage
Run via modules from the repo root, e.g.:

```bash
python -m evals.cli.make_traces --config configs/mas_generation.yaml --dry-run
```
