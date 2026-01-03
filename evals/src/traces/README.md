# Traces Module

Trace parsing, canonical text reconstruction, and deterministic replay utilities.

## Files
- `trace_text.py`: Canonical trace text and window reconstruction (single source of truth).
- `trace_replay_engine.py`: Backward-compatible replay API (delegates to split modules).
- `models.py`: Replay dataclasses and shared exceptions.
- `parser.py`: Trace record parsing and agent-label derivation.
- `classifier.py`: Conservative turn classification rules and audit flag logic.
- `replay.py`: Deterministic replay engine implementation and iterators.
- `generation.py`: Trace generation workflow used by `cli/make_traces.py`.

## Notes
- Import trace text helpers from `trace_text.py` rather than reimplementing them.
