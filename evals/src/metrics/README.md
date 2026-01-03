# Metrics Module

Shared metric helpers and dataclasses used by evaluation pipelines.

## Files
- `computations.py`: Core per-event and aggregate metric computations.
- `aggregation.py`: Bootstrap confidence intervals and aggregation helpers.
- `extractor.py`: Cached extraction wrapper for metrics.
- `integrity.py`: Manifest and trace provenance helpers.
- `constants.py`: Metric constants and schema versions.
- `types.py`: Dataclasses for per-case and aggregate metrics.
- `runner.py`: Metrics computation workflow used by `cli/compute_metrics.py`.
