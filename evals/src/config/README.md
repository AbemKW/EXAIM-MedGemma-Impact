# Config Module

Centralized configuration loaders used throughout evaluation runs.

## Files
- `config_loader.py`: Loads extractor and variant configs, resolves paths, and computes hashes.

## Common entry points
- `load_extractor_config()` for extractor settings with provenance hashes.
- `load_variant_config()` for variant YAMLs in `evals/configs/variants/`.
- `get_stoplists_provenance()` for run metadata.
