"""Trace parsing and replay utilities."""

from .trace_replay_engine import (
    AuditFlag,
    ReplayEvent,
    StubTraceError,
    TraceReplayEngine,
    TraceReplayError,
    TraceValidationError,
    classify_turn,
    derive_agent_labels,
    is_suspicious_label_like,
    replay_trace,
)
from .trace_text import (
    TraceParsingError,
    TraceParsingStats,
    build_canonical_trace_text,
    build_window_text,
    get_chunk_by_seq,
    iter_trace_chunks,
    iter_trace_records,
    load_trace_chunks_for_case,
)

__all__ = [
    "AuditFlag",
    "ReplayEvent",
    "StubTraceError",
    "TraceParsingError",
    "TraceParsingStats",
    "TraceReplayEngine",
    "TraceReplayError",
    "TraceValidationError",
    "build_canonical_trace_text",
    "build_window_text",
    "classify_turn",
    "derive_agent_labels",
    "get_chunk_by_seq",
    "is_suspicious_label_like",
    "iter_trace_chunks",
    "iter_trace_records",
    "load_trace_chunks_for_case",
    "replay_trace",
]
