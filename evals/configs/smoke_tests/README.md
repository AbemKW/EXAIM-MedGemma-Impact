# TokenGate Calibration Smoke Tests

This directory contains minimal configuration files for smoke testing the TokenGate calibration pipeline. These configs use small parameter grids and relaxed constraints to enable quick validation that the calibration code works correctly.

## Available Smoke Tests

### `smoke_test_quick.yaml` - Absolute Minimum (4 policies)
- **Grid**: 2×2×1×1 = 4 policies
- **Purpose**: Fastest possible validation
- **Use when**: You want to verify the pipeline works with minimal time investment
- **Expected runtime**: ~1-2 minutes (depends on trace count)

### `smoke_test_minimal.yaml` - Minimal (16 policies)
- **Grid**: 2×2×2×2 = 16 policies
- **Purpose**: Minimal but meaningful validation
- **Use when**: You want basic coverage without waiting too long
- **Expected runtime**: ~5-10 minutes (depends on trace count)

### `smoke_test_small.yaml` - Small (36 policies)
- **Grid**: 3×3×2×2 = 36 policies
- **Purpose**: Better coverage while still running quickly
- **Use when**: You want more confidence but still need fast feedback
- **Expected runtime**: ~15-20 minutes (depends on trace count)

## Usage

**Important:** Paths are relative to the `evals/` directory. Run commands from the `evals/` directory, or adjust paths accordingly.

### Running from `evals/` directory (Recommended):

#### Bash/Linux/Mac:
```bash
cd evals

# Quick test (4 policies)
python -m evals.cli.calibrate_tokengate \
    --traces data/traces/ \
    --manifest data/manifests/exaid_traces_*.manifest.jsonl \
    --config configs/smoke_tests/smoke_test_quick.yaml \
    --output data/calibration_smoke/

# Minimal test (16 policies)
python -m evals.cli.calibrate_tokengate \
    --traces data/traces/ \
    --manifest data/manifests/exaid_traces_*.manifest.jsonl \
    --config configs/smoke_tests/smoke_test_minimal.yaml \
    --output data/calibration_smoke/

# Small test (36 policies)
python -m evals.cli.calibrate_tokengate \
    --traces data/traces/ \
    --manifest data/manifests/exaid_traces_*.manifest.jsonl \
    --config configs/smoke_tests/smoke_test_small.yaml \
    --output data/calibration_smoke/
```

#### PowerShell (Windows):
```powershell
cd evals

# Quick test (4 policies) - use backticks for line continuation
python -m evals.cli.calibrate_tokengate `
    --traces data/traces/ `
    --manifest data/manifests/exaid_traces_*.manifest.jsonl `
    --config configs/smoke_tests/smoke_test_quick.yaml `
    --output data/calibration_smoke/

# Or use a single line:
python -m evals.cli.calibrate_tokengate --traces data/traces/ --manifest data/manifests/exaid_traces_*.manifest.jsonl --config configs/smoke_tests/smoke_test_quick.yaml --output data/calibration_smoke/
```

### Running from repo root:

If running from the repo root, prefix paths with `evals/`:

#### PowerShell (Windows):
```powershell
# From repo root - prefix paths with evals/
python -m evals.cli.calibrate_tokengate --traces evals/data/traces/ --manifest evals/data/manifests/exaid_traces_*.manifest.jsonl --config evals/configs/smoke_tests/smoke_test_quick.yaml --output evals/data/calibration_smoke/
```

### Docker:
```bash
# Quick test (4 policies) - paths are relative to /app/evals in container
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.calibrate_tokengate --traces data/traces/ --manifest data/manifests/exaid_traces_*.manifest.jsonl --config configs/smoke_tests/smoke_test_quick.yaml --output data/calibration_smoke/
```

## Differences from Full Calibration

These smoke test configs differ from `calibration_sweep.yaml` in several ways:

1. **Smaller parameter grids**: Fewer combinations to evaluate (4-36 vs 625)
2. **Relaxed constraints**: Higher thresholds so more policies pass validation
3. **Disabled safety checks**: 
   - `verify_trace_hashes: false` - Skips hash verification for speed
   - `verify_determinism: false` - Skips determinism checks for speed
4. **Relaxed failure rates**: Higher `max_replay_failure_rate` tolerance

## What to Verify

After running a smoke test, verify:

1. **No import errors**: The calibration code imports correctly from the new package structure
2. **Output artifacts generated**: Check that all expected files are created in the output directory
3. **Valid results**: Review `calibration_summary.json` to ensure policies were evaluated
4. **Selected policy**: Verify that a policy was selected (or check why none were selected)

## When to Use

- **After code reorganization**: Verify imports and structure work correctly
- **Before full calibration**: Quick sanity check that everything is configured correctly
- **CI/CD pipelines**: Fast validation that calibration code hasn't broken
- **Development**: Quick feedback loop when modifying calibration logic

## Notes

- Smoke tests use relaxed constraints, so results may differ from full calibration
- These configs are **not** suitable for production calibration runs
- For production use, use `configs/calibration_sweep.yaml` with the full 625-policy grid

## Troubleshooting

### FileNotFoundError: Config file not found
**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'configs\\smoke_tests\\smoke_test_quick.yaml'`

**Solution:** Make sure you're running from the `evals/` directory, or adjust paths:
- **From repo root:** Use `evals/configs/smoke_tests/smoke_test_quick.yaml`
- **From evals/:** Use `configs/smoke_tests/smoke_test_quick.yaml`

### PowerShell Line Continuation Error
If you see `Missing expression after unary operator '--'` in PowerShell, use backticks (`) instead of backslashes (\) for line continuation, or use a single-line command.

### Docker Import Errors
If you see import errors referencing old paths (e.g., `tokengate_calibration_runner.py`), rebuild the Docker image:
```bash
docker compose -f docker-compose.evals.yml build --no-cache evals
```

### Python Cache Issues
If imports fail after reorganization, clear Python cache:
```bash
# Linux/Mac:
find . -type d -name __pycache__ -exec rm -r {} +
find . -name "*.pyc" -delete

# PowerShell (Windows):
Get-ChildItem -Path . -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -Filter *.pyc | Remove-Item -Force
```

