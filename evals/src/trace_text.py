#!/usr/bin/env python3
"""
Canonical trace text construction - SINGLE SOURCE OF TRUTH.

Paper hook: "Trace text is defined as the concatenation of message-type
chunks only, excluding orchestrator summaries and system artifacts (Section 3.1)"

This module is the SINGLE SOURCE OF TRUTH for trace text construction.
It MUST be used by:
    - generate_stoplists.py (Phase 0)
    - compute_metrics.py for trace concept sets (M4/M5)
    - compute_metrics.py for window reconstruction (M6a/M6b)
    - run_variants.py for buffer hash computation
    
INVARIANT: No other module may implement trace/window text logic.
All must import and use build_canonical_trace_text() or build_window_text().

Dependencies:
    - Python 3.10+
    - gzip (stdlib)
    - json (stdlib)
"""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple


@dataclass
class TraceParsingStats:
    """Statistics from trace parsing for auditability."""
    total_records: int = 0
    chunk_records: int = 0
    included_message_chunks: int = 0
    excluded_orchestrator_summary: int = 0
    excluded_system_note: int = 0
    excluded_other_subtype: int = 0
    missing_event_subtype: int = 0
    non_chunk_records: int = 0


class TraceParsingError(Exception):
    """Raised when trace parsing fails validation."""
    pass


def iter_trace_records(trace_file: Path) -> Iterator[dict]:
    """
    Iterate over ALL records from a trace file.
    
    Handles both .jsonl and .jsonl.gz files.
    """
    open_fn = gzip.open if str(trace_file).endswith(".gz") else open
    with open_fn(trace_file, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise TraceParsingError(
                    f"Invalid JSON at {trace_file}:{line_num}: {e}"
                )


def is_chunk_record(record: dict) -> bool:
    """
    Determine if a record is a chunk (vs trace_meta, turn_boundary, etc).
    
    Schema-robust: checks multiple possible indicators.
    
    Paper hook: "Chunk records identified by record_type='chunk' or presence
    of seq+text+agent fields (Section 3.1)"
    """
    # Primary check: record_type field
    if record.get("record_type") == "chunk":
        return True
    
    # Secondary check: presence of chunk-specific fields
    # (for backwards compatibility with traces that may not have record_type)
    has_seq = "seq" in record or "sequence_num" in record
    has_text = "text_chunk" in record or "content" in record
    has_agent = "agent_id" in record
    
    # Must have seq + text + agent to be considered a chunk
    if has_seq and has_text and has_agent:
        # Exclude if it has record_type that's NOT chunk
        if "record_type" in record and record["record_type"] != "chunk":
            return False
        return True
    
    return False


def is_message_chunk(chunk: dict) -> Tuple[bool, str]:
    """
    Determine if a chunk is a message (vs orchestrator_summary/system_note).
    
    STRICT ALLOWLIST RULE:
        - Include ONLY if event_subtype == "message"
        - Include if event_subtype is missing BUT agent_id is a known specialist
          (NOT orchestrator, NOT _system)
    
    Paper hook: "Message chunks identified by event_subtype='message' allowlist
    with fallback to agent_id exclusion list (Section 3.1)"
    
    Returns:
        Tuple of (is_message: bool, reason: str)
    """
    subtype = chunk.get("event_subtype")
    agent_id = chunk.get("agent_id", "").lower()
    
    # Explicit subtype check (preferred)
    if subtype is not None:
        if subtype == "message":
            return True, "explicit_message"
        elif subtype == "orchestrator_summary":
            return False, "orchestrator_summary"
        elif subtype == "system_note":
            return False, "system_note"
        else:
            return False, f"other_subtype:{subtype}"
    
    # Missing subtype: use agent_id as secondary discriminator
    # STRICT: Only include known specialist agents
    excluded_agents = {"orchestrator", "_system", "system", "meta"}
    if agent_id in excluded_agents:
        return False, "excluded_agent_no_subtype"
    
    # If agent looks like a specialist and no subtype, cautiously include
    # but track as missing_subtype for audit
    return True, "missing_subtype_included"


def iter_trace_chunks(
    trace_file: Path,
    stats: Optional[TraceParsingStats] = None
) -> Iterator[dict]:
    """
    Iterate over chunk records from a trace file.
    
    Args:
        trace_file: Path to trace JSONL file
        stats: Optional stats object to populate
        
    Yields:
        Chunk records only (filters out trace_meta, etc)
    """
    if stats is None:
        stats = TraceParsingStats()
    
    for record in iter_trace_records(trace_file):
        stats.total_records += 1
        
        if is_chunk_record(record):
            stats.chunk_records += 1
            yield record
        else:
            stats.non_chunk_records += 1


def build_canonical_trace_text(
    trace_file: Path,
    fail_on_empty: bool = True
) -> Tuple[str, TraceParsingStats]:
    """
    Build canonical trace text from a trace file.
    
    Paper hook: "Full trace text is the concatenation of message-type chunks
    in seq order, excluding orchestrator summaries (Section 3.1)"
    
    Args:
        trace_file: Path to trace JSONL file
        fail_on_empty: If True, raise error if no message chunks found
        
    Returns:
        Tuple of (canonical_text, parsing_stats)
        
    Raises:
        TraceParsingError: If fail_on_empty and no message chunks found
    """
    stats = TraceParsingStats()
    chunks_with_text = []
    
    for chunk in iter_trace_chunks(trace_file, stats):
        is_msg, reason = is_message_chunk(chunk)
        
        # Update stats based on reason
        if reason == "explicit_message":
            stats.included_message_chunks += 1
        elif reason == "missing_subtype_included":
            stats.included_message_chunks += 1
            stats.missing_event_subtype += 1
        elif reason == "orchestrator_summary":
            stats.excluded_orchestrator_summary += 1
            continue
        elif reason == "system_note":
            stats.excluded_system_note += 1
            continue
        elif reason == "excluded_agent_no_subtype":
            stats.excluded_other_subtype += 1
            continue
        else:
            stats.excluded_other_subtype += 1
            continue
        
        if not is_msg:
            continue
        
        # Extract text and seq
        text = chunk.get("text_chunk") or chunk.get("content") or ""
        seq = chunk.get("seq", chunk.get("sequence_num", 0))
        
        if text.strip():
            chunks_with_text.append((seq, text))
    
    # FAIL-FAST: Raise if no message chunks found
    if fail_on_empty and stats.included_message_chunks == 0:
        raise TraceParsingError(
            f"No message chunks found in {trace_file}. "
            f"Stats: total_records={stats.total_records}, "
            f"chunk_records={stats.chunk_records}, "
            f"excluded_orchestrator={stats.excluded_orchestrator_summary}, "
            f"excluded_system_note={stats.excluded_system_note}, "
            f"excluded_other={stats.excluded_other_subtype}"
        )
    
    # Sort by seq and concatenate
    chunks_with_text.sort(key=lambda x: x[0])
    canonical_text = " ".join(text for _, text in chunks_with_text)
    
    return canonical_text, stats


def build_window_text(
    trace_chunks: list[dict],
    start_seq: int,
    end_seq: int
) -> str:
    """
    Build window text for a seq range.
    
    Paper hook: "Window text for M6a/M6b is the canonical trace text
    within [start_seq, end_seq] inclusive (Section 8.2)"
    
    Args:
        trace_chunks: List of chunk records (pre-loaded)
        start_seq: Start sequence number (inclusive)
        end_seq: End sequence number (inclusive)
    
    Returns:
        Concatenated text of message chunks in the window.
    """
    window_texts = []
    
    for chunk in trace_chunks:
        seq = chunk.get("seq", chunk.get("sequence_num", 0))
        
        if start_seq <= seq <= end_seq:
            is_msg, _ = is_message_chunk(chunk)
            if is_msg:
                text = chunk.get("text_chunk") or chunk.get("content") or ""
                if text.strip():
                    window_texts.append((seq, text))
    
    window_texts.sort(key=lambda x: x[0])
    return " ".join(text for _, text in window_texts)


def load_trace_chunks_for_case(trace_file: Path) -> list[dict]:
    """
    Load all chunk records for a case (for window reconstruction).
    
    Paper hook: "Trace chunks loaded once per case and cached for
    window reconstruction efficiency (Section 8.2)"
    
    Returns:
        List of chunk dicts, sorted by seq.
    """
    chunks = list(iter_trace_chunks(trace_file))
    chunks.sort(key=lambda c: c.get("seq", c.get("sequence_num", 0)))
    return chunks


def get_chunk_by_seq(trace_chunks: list[dict], seq: int) -> Optional[dict]:
    """
    Find a chunk by sequence number.
    
    Args:
        trace_chunks: List of chunk records
        seq: Sequence number to find
        
    Returns:
        Chunk dict or None if not found
    """
    for chunk in trace_chunks:
        chunk_seq = chunk.get("seq", chunk.get("sequence_num", -1))
        if chunk_seq == seq:
            return chunk
    return None


if __name__ == "__main__":
    # Simple test when run directly
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python trace_text.py <trace_file.jsonl.gz>")
        sys.exit(1)
    
    trace_path = Path(sys.argv[1])
    try:
        text, stats = build_canonical_trace_text(trace_path)
        print(f"Parsed {trace_path.name}:")
        print(f"  Total records: {stats.total_records}")
        print(f"  Chunk records: {stats.chunk_records}")
        print(f"  Included message chunks: {stats.included_message_chunks}")
        print(f"  Excluded orchestrator: {stats.excluded_orchestrator_summary}")
        print(f"  Excluded system_note: {stats.excluded_system_note}")
        print(f"  Excluded other: {stats.excluded_other_subtype}")
        print(f"  Missing event_subtype: {stats.missing_event_subtype}")
        print(f"  Canonical text length: {len(text)} chars")
    except TraceParsingError as e:
        print(f"ERROR: {e}")
        sys.exit(1)





