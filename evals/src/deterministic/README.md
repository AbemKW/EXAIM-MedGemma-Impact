# Deterministic Module

Utilities that guarantee byte-stable outputs and deterministic replay behavior.

## Files
- `io.py`: Deterministic JSON/JSONL/gzip writing plus run-log helpers.
- `utils.py`: Deterministic timestamp derivation, CTU computation, and ID helpers.

## Notes
- Keep new output serialization paths inside `io.py` so determinism rules stay centralized.
