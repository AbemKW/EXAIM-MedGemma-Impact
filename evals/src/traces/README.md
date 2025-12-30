# Traces Module

Trace parsing, canonical text reconstruction, and deterministic replay utilities.

## Files
- `trace_text.py`: Canonical trace text and window reconstruction (single source of truth).
- `trace_replay_engine.py`: Deterministic replay engine with content/control-plane classification.

## Notes
- Import trace text helpers from `trace_text.py` rather than reimplementing them.
