#!/usr/bin/env python3
"""
EXAID Trace Replay Engine - Deterministic replay with conservative classification.

Paper hook: "Traces are replayed deterministically with virtual time, classifying
turns as content_plane or control_plane using exact-match rules (Section 3.2)"

Two-pass architecture:
    Pass 1: Derive agent_labels from turn_boundary records only (authoritative source)
    Pass 2: Classify turns and yield replay events

Design Principles:
    - Conservative: Only filter when 100% certain (exact match)
    - Auditable: Every turn has classification_reason + suspicious flags
    - Deterministic: Two-pass architecture, no runtime randomness
    - Self-contained: Labels derived from trace, no external config

Dependencies:
    - Python 3.10+
    - gzip (stdlib)
    - json (stdlib)
"""

import gzip
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional


# =============================================================================
# Exceptions
# =============================================================================

class TraceReplayError(Exception):
    """Base exception for trace replay errors."""
    pass


class TraceValidationError(TraceReplayError):
    """Raised when trace validation fails."""
    pass


class StubTraceError(TraceReplayError):
    """Raised when attempting to replay a stub trace without override."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class ReplayEvent:
    """
    Single replay event yielded by the engine.
    
    Represents either a delta (stream content) or a boundary (turn start/end).
    """
    event_type: Literal["delta", "turn_start", "turn_end"]
    virtual_time_ms: int
    stream_mode: Literal["full", "content_plane"]
    seq: int
    turn_id: int
    agent_id: str
    
    # Delta-specific (None for boundaries)
    delta_text: Optional[str] = None
    
    # Boundary-specific (None for deltas)
    boundary: Optional[Literal["start", "end"]] = None
    content_hash: Optional[str] = None  # Only on turn_end


@dataclass
class TurnClassification:
    """
    Classification result for a single turn.
    
    Every turn is classified as either content_plane or control_plane,
    with an auditable classification_reason.
    """
    turn_id: int
    turn_type: Literal["content_plane", "control_plane"]
    agent_id: str
    turn_text: str
    classification_reason: str
    delta_count: int
    start_seq: int
    end_seq: int


@dataclass
class AuditFlag:
    """
    Flags for reviewer visibility on suspicious but unfiltered turns.
    
    When a turn looks label-like but doesn't match derived labels,
    it's kept as content_plane but flagged for audit.
    """
    turn_id: int
    flag_type: str
    details: str


@dataclass
class TraceMeta:
    """
    Parsed trace metadata from trace_meta record.
    """
    schema_version: str
    case_id: str
    mas_run_id: str
    mac_commit: str
    model: str
    created_at: str
    t0_emitted_ms: int
    stub_mode: bool
    total_turns: Optional[int]
    total_deltas: Optional[int]
    decoding: Optional[dict] = None


@dataclass
class _TurnData:
    """Internal: Accumulated data for a turn during reconstruction."""
    turn_id: int
    agent_id: str
    deltas: list = field(default_factory=list)  # List of (seq, delta_text, t_rel_ms)
    start_seq: Optional[int] = None
    end_seq: Optional[int] = None
    start_t_rel_ms: Optional[int] = None
    end_t_rel_ms: Optional[int] = None
    content_hash: Optional[str] = None


# =============================================================================
# Constants
# =============================================================================

# Sentinel values that trigger control_plane classification
TERMINATE_SENTINELS = frozenset({"TERMINATE"})

# Agent IDs to exclude from derived label set
EXCLUDED_AGENT_IDS = frozenset({"unknown", "null", "none", ""})

# Boundary timestamp tolerance (milliseconds)
# Due to millisecond resolution and async timing, boundary timestamps may be
# off by 1-2ms from delta timestamps. This is cosmetic and does not affect
# evaluation semantics.
BOUNDARY_TIME_EPSILON_MS = 2


# =============================================================================
# Pass 1: Agent Label Derivation
# =============================================================================

def iter_trace_records(trace_path: Path) -> Iterator[dict]:
    """
    Iterate over ALL records from a trace file.
    
    Handles both .jsonl and .jsonl.gz files.
    """
    open_fn = gzip.open if str(trace_path).endswith(".gz") else open
    with open_fn(trace_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise TraceValidationError(
                    f"Invalid JSON at {trace_path}:{line_num}: {e}"
                )


def derive_agent_labels(trace_path: Path) -> frozenset[str]:
    """
    Pass 1: Scan turn_boundary records to collect agent labels.
    
    Deterministic:
        - Processes records in file order
        - Collects agent_id from turn_boundary records ONLY (authoritative)
        - Normalizes to lowercase
        - Excludes null/empty/"unknown"
    
    Args:
        trace_path: Path to trace JSONL file
        
    Returns:
        Frozen set of normalized labels (e.g., {"doctor0", "supervisor"})
    """
    labels = set()
    
    for record in iter_trace_records(trace_path):
        # Only consider turn_boundary records (authoritative source)
        if record.get("record_type") != "turn_boundary":
            continue
        
        agent_id = record.get("agent_id")
        if agent_id:
            normalized = agent_id.strip().lower()
            if normalized and normalized not in EXCLUDED_AGENT_IDS:
                labels.add(normalized)
    
    return frozenset(labels)


# =============================================================================
# Classification Logic
# =============================================================================

def is_suspicious_label_like(turn_text: str, derived_labels: frozenset[str]) -> bool:
    """
    Detect turns that LOOK like speaker-selection but don't match derived labels.
    
    Suspicious if:
        - Single word (after strip)
        - Looks identifier-ish (alphanumeric, possibly with digits)
        - NOT in derived_agent_labels
    
    Args:
        turn_text: Reconstructed turn text
        derived_labels: Set of known agent labels
        
    Returns:
        True if suspicious (should be flagged for audit)
    """
    text = turn_text.strip()
    words = text.split()
    
    if len(words) != 1:
        return False
    
    word = words[0].lower()
    
    # Already matched -> not suspicious (it was filtered)
    if word in derived_labels:
        return False
    
    # Looks identifier-ish? (contains letters, possibly digits, no spaces)
    if word.replace("_", "").isalnum() and any(c.isalpha() for c in word):
        return True
    
    return False


def classify_turn(
    turn_id: int,
    turn_text: str,
    delta_count: int,
    agent_id: str,
    start_seq: int,
    end_seq: int,
    derived_agent_labels: frozenset[str],
    audit_flags: list[AuditFlag],
) -> TurnClassification:
    """
    Conservative classification: exact match only.
    
    Classification rules (in order):
    1. Exact agent label match → control_plane
    2. TERMINATE sentinel → control_plane
    3. Empty/whitespace-only turn → control_plane
    4. Default → content_plane
    
    Suspicious turns are kept as content_plane and flagged.
    
    Args:
        turn_id: Turn identifier
        turn_text: Reconstructed text from deltas
        delta_count: Number of deltas in this turn
        agent_id: Agent that produced this turn
        start_seq: First seq in turn
        end_seq: Last seq in turn
        derived_agent_labels: Set of known agent labels
        audit_flags: List to append audit flags to (mutated)
        
    Returns:
        TurnClassification with type and reason
    """
    normalized = turn_text.strip().lower()
    
    # Rule 1: Exact agent label match
    if normalized in derived_agent_labels:
        return TurnClassification(
            turn_id=turn_id,
            turn_type="control_plane",
            agent_id=agent_id,
            turn_text=turn_text,
            classification_reason=f"exact_label_match:{normalized}",
            delta_count=delta_count,
            start_seq=start_seq,
            end_seq=end_seq,
        )
    
    # Rule 2: TERMINATE sentinel (exact, case-insensitive)
    if turn_text.strip().upper() in TERMINATE_SENTINELS:
        return TurnClassification(
            turn_id=turn_id,
            turn_type="control_plane",
            agent_id=agent_id,
            turn_text=turn_text,
            classification_reason="terminate_sentinel",
            delta_count=delta_count,
            start_seq=start_seq,
            end_seq=end_seq,
        )
    
    # Rule 3: Empty turn (whitespace-only)
    if not turn_text.strip():
        return TurnClassification(
            turn_id=turn_id,
            turn_type="control_plane",
            agent_id=agent_id,
            turn_text=turn_text,
            classification_reason="empty_turn",
            delta_count=delta_count,
            start_seq=start_seq,
            end_seq=end_seq,
        )
    
    # Default: content_plane
    classification = TurnClassification(
        turn_id=turn_id,
        turn_type="content_plane",
        agent_id=agent_id,
        turn_text=turn_text,
        classification_reason="default_content",
        delta_count=delta_count,
        start_seq=start_seq,
        end_seq=end_seq,
    )
    
    # Check for suspicious label-like (add audit flag, don't filter)
    if is_suspicious_label_like(turn_text, derived_agent_labels):
        audit_flags.append(AuditFlag(
            turn_id=turn_id,
            flag_type="suspicious_label_like_unmatched",
            details=f"turn_text='{turn_text.strip()}' not in derived_agent_labels={derived_agent_labels}"
        ))
    
    return classification


# =============================================================================
# Timing Utilities
# =============================================================================

def compute_time_shift(records: list[dict]) -> int:
    """
    Find minimum t_rel_ms across ALL records (deltas AND boundaries).
    
    Important: turn_boundary.t_rel_ms may be negative!
    
    Args:
        records: List of all trace records
        
    Returns:
        Minimum t_rel_ms value (will be negative if any boundary is before t0)
    """
    min_t_rel = 0  # Default if no records with t_rel_ms
    
    for record in records:
        t_rel = record.get("t_rel_ms")
        if t_rel is not None:
            min_t_rel = min(min_t_rel, t_rel)
    
    return min_t_rel


# =============================================================================
# Main Replay Engine
# =============================================================================

class TraceReplayEngine:
    """
    Deterministic trace replay with conservative classification.
    
    Two-pass architecture:
        Pass 1: Derive agent_labels from turn_boundary records only (authoritative source)
        Pass 2: Classify turns and yield replay events
    
    Example usage:
        from pathlib import Path
        
        # Import from evals package (if running from repo root)
        from evals.src.trace_replay_engine import TraceReplayEngine
        
        # Alternative: If running from evals/ directory, add src to path first:
        # import sys
        # sys.path.insert(0, "src")
        # from trace_replay_engine import TraceReplayEngine
        
        engine = TraceReplayEngine(Path("trace.jsonl.gz"))
        
        # Get metadata
        meta = engine.get_metadata()
        labels = engine.get_derived_agent_labels()
        
        # Replay full stream
        for event in engine.replay_full():
            print(f"t={event.virtual_time_ms}ms: {event.event_type}")
        
        # Replay content_plane stream only
        for event in engine.replay_content_plane():
            process_content(event)
    """
    
    def __init__(
        self,
        trace_path: Path,
        *,
        strict_stub_guard: bool = True,
        strict_validation: bool = True,
        shift_to_zero: bool = False,
    ):
        """
        Initialize the replay engine.
        
        Args:
            trace_path: Path to trace JSONL(.gz) file
            strict_stub_guard: If True, raise error on stub_mode traces
            strict_validation: If True, raise error on validation failures
            shift_to_zero: If True, shift timeline so minimum virtual_time = 0
        """
        self._trace_path = Path(trace_path)
        self._strict_stub_guard = strict_stub_guard
        self._strict_validation = strict_validation
        self._shift_to_zero = shift_to_zero
        
        # Lazy-loaded state
        self._records: Optional[list[dict]] = None
        self._metadata: Optional[TraceMeta] = None
        self._derived_labels: Optional[frozenset[str]] = None
        self._classifications: Optional[dict[int, TurnClassification]] = None
        self._audit_flags: Optional[list[AuditFlag]] = None
        self._turn_data: Optional[dict[int, _TurnData]] = None
        self._time_shift: int = 0
        self._initialized: bool = False
    
    def _ensure_initialized(self) -> None:
        """Load and process trace if not already done."""
        if self._initialized:
            return
        
        # Pass 1: Derive agent labels from boundaries only
        self._derived_labels = derive_agent_labels(self._trace_path)
        
        if not self._derived_labels:
            warnings.warn(
                f"No agent labels derived from {self._trace_path}. "
                "This may indicate a data quality issue."
            )
        
        # Load all records for Pass 2
        self._records = list(iter_trace_records(self._trace_path))
        
        if not self._records:
            raise TraceValidationError(f"Trace file is empty: {self._trace_path}")
        
        # Parse metadata
        self._parse_metadata()
        
        # Check stub guard
        if self._strict_stub_guard and self._metadata.stub_mode:
            raise StubTraceError(
                f"Trace {self._trace_path} is a stub trace (stub_mode=true). "
                "Stub traces must not be used for evaluation. "
                "Pass strict_stub_guard=False to override."
            )
        
        # Validate trace
        self._validate_trace()
        
        # Compute time shift if needed
        if self._shift_to_zero:
            self._time_shift = compute_time_shift(self._records)
        
        # Reconstruct turns and classify
        self._reconstruct_turns()
        self._classify_all_turns()
        
        self._initialized = True
    
    def _parse_metadata(self) -> None:
        """Parse trace_meta record."""
        first_record = self._records[0]
        
        if first_record.get("record_type") != "trace_meta":
            raise TraceValidationError(
                f"First record must be trace_meta, got: {first_record.get('record_type')}"
            )
        
        self._metadata = TraceMeta(
            schema_version=first_record.get("schema_version", "unknown"),
            case_id=first_record.get("case_id", "unknown"),
            mas_run_id=first_record.get("mas_run_id", "unknown"),
            mac_commit=first_record.get("mac_commit", "unknown"),
            model=first_record.get("model", "unknown"),
            created_at=first_record.get("created_at", "unknown"),
            t0_emitted_ms=first_record.get("t0_emitted_ms", 0),
            stub_mode=first_record.get("stub_mode", False),
            total_turns=first_record.get("total_turns"),
            total_deltas=first_record.get("total_deltas"),
            decoding=first_record.get("decoding"),
        )
    
    def _validate_trace(self) -> None:
        """
        Validate trace integrity.
        
        Checks:
            - seq strictly increasing
            - t_emitted_ms non-decreasing for deltas
            - Boundary start/end pairs match
            - Per-record t_rel_ms consistency (critical for virtual time)
                * stream_delta: t_rel_ms == t_emitted_ms - t0
                * turn_boundary: t_rel_ms == t_ms - t0
            - Per-turn boundary-time containment
                * turn_start.t_ms <= first_delta.t_emitted_ms + epsilon
                * turn_end.t_ms >= last_delta.t_emitted_ms - epsilon
                * Skips turns with boundaries but zero deltas
            - Turns with deltas but no boundaries (schema violation)
            - content_hash matches (warning only)
        
        Behavior:
            - strict_validation=True: Violations → errors → raises TraceValidationError
            - strict_validation=False: Violations → warnings → continues execution
        """
        errors = []
        warnings_list = []
        
        t0 = self._metadata.t0_emitted_ms
        prev_seq = -1
        prev_t_emitted = -1
        open_turns: dict[int, dict] = {}  # turn_id -> start boundary
        turn_deltas: dict[int, list[str]] = {}  # turn_id -> list of delta_text
        turn_delta_times: dict[int, list[int]] = {}  # turn_id -> list of t_emitted_ms
        turn_boundaries: dict[int, dict] = {}  # turn_id -> {start: record, end: record}
        turns_with_deltas: set[int] = set()  # Track which turns have deltas
        turns_with_boundaries: set[int] = set()  # Track which turns have boundaries
        
        for idx, record in enumerate(self._records):
            if idx == 0 and record.get("record_type") == "trace_meta":
                continue
            
            record_type = record.get("record_type")
            seq = record.get("seq", -1)
            
            # Check seq strictly increasing
            if seq <= prev_seq:
                errors.append(f"seq {seq} not > prev {prev_seq} at index {idx}")
            prev_seq = seq
            
            if record_type == "stream_delta":
                t_emitted = record.get("t_emitted_ms", 0)
                t_rel = record.get("t_rel_ms")
                turn_id = record.get("turn_id", 0)
                delta_text = record.get("delta_text", "")
                
                # Check t_emitted non-decreasing
                if t_emitted < prev_t_emitted:
                    errors.append(
                        f"t_emitted {t_emitted} < prev {prev_t_emitted} at index {idx}"
                    )
                prev_t_emitted = t_emitted
                
                # Check t_rel_ms consistency for deltas
                if t_rel is not None:
                    expected_t_rel = t_emitted - t0
                    if t_rel != expected_t_rel:
                        msg = (
                            f"stream_delta at index {idx}: t_rel_ms {t_rel} != expected {expected_t_rel} "
                            f"(t_emitted_ms {t_emitted} - t0 {t0})"
                        )
                        if self._strict_validation:
                            errors.append(msg)
                        else:
                            warnings_list.append(msg)
                
                # Accumulate deltas for content_hash verification
                if turn_id not in turn_deltas:
                    turn_deltas[turn_id] = []
                    turn_delta_times[turn_id] = []
                turn_deltas[turn_id].append(delta_text)
                turn_delta_times[turn_id].append(t_emitted)
                turns_with_deltas.add(turn_id)
                
            elif record_type == "turn_boundary":
                turn_id = record.get("turn_id", 0)
                boundary = record.get("boundary")
                t_ms = record.get("t_ms", 0)
                t_rel = record.get("t_rel_ms")
                
                # Check t_rel_ms consistency for boundaries
                if t_rel is not None:
                    expected_t_rel = t_ms - t0
                    if t_rel != expected_t_rel:
                        msg = (
                            f"turn_boundary at index {idx} (turn {turn_id}, {boundary}): "
                            f"t_rel_ms {t_rel} != expected {expected_t_rel} "
                            f"(t_ms {t_ms} - t0 {t0})"
                        )
                        if self._strict_validation:
                            errors.append(msg)
                        else:
                            warnings_list.append(msg)
                
                if boundary == "start":
                    open_turns[turn_id] = record
                    if turn_id not in turn_boundaries:
                        turn_boundaries[turn_id] = {}
                    turn_boundaries[turn_id]["start"] = record
                    turns_with_boundaries.add(turn_id)
                elif boundary == "end":
                    if turn_id not in open_turns:
                        errors.append(f"turn {turn_id} end without start at index {idx}")
                    else:
                        del open_turns[turn_id]
                    
                    if turn_id not in turn_boundaries:
                        turn_boundaries[turn_id] = {}
                    turn_boundaries[turn_id]["end"] = record
                    
                    # Verify content_hash
                    content_hash = record.get("content_hash")
                    if content_hash and turn_id in turn_deltas:
                        deltas = turn_deltas[turn_id]
                        content = "".join(deltas)
                        expected_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
                        if content_hash != expected_hash:
                            warnings_list.append(
                                f"content_hash mismatch for turn {turn_id}"
                            )
        
        # Check for unclosed turns
        for turn_id in open_turns:
            errors.append(f"turn {turn_id} has start but no end")
        
        # Check for turns with deltas but no boundaries (schema violation)
        turns_with_deltas_no_boundaries = turns_with_deltas - turns_with_boundaries
        if turns_with_deltas_no_boundaries:
            msg = (
                f"Turns with deltas but no boundaries (schema violation): "
                f"{sorted(turns_with_deltas_no_boundaries)}"
            )
            if self._strict_validation:
                errors.append(msg)
            else:
                warnings_list.append(msg)
        
        # Per-turn boundary-time containment checks
        for turn_id, boundaries in turn_boundaries.items():
            if turn_id not in turn_delta_times or len(turn_delta_times[turn_id]) == 0:
                # Turn has boundaries but zero deltas - skip boundary-time check
                continue
            
            first_delta_t = min(turn_delta_times[turn_id])
            last_delta_t = max(turn_delta_times[turn_id])
            
            start_boundary = boundaries.get("start")
            end_boundary = boundaries.get("end")
            
            if start_boundary:
                start_t_ms = start_boundary.get("t_ms", 0)
                if start_t_ms > first_delta_t:
                    diff_ms = start_t_ms - first_delta_t
                    msg = (
                        f"turn {turn_id}: turn_start.t_ms {start_t_ms} > first_delta.t_emitted_ms {first_delta_t} "
                        f"(diff={diff_ms}ms, exceeds epsilon={BOUNDARY_TIME_EPSILON_MS}ms)"
                    )
                    if diff_ms > BOUNDARY_TIME_EPSILON_MS:
                        if self._strict_validation:
                            errors.append(msg)
                        else:
                            warnings_list.append(msg)
                    else:
                        warnings_list.append(
                            f"turn {turn_id}: turn_start.t_ms {start_t_ms} > first_delta.t_emitted_ms {first_delta_t} "
                            f"(diff={diff_ms}ms, within epsilon={BOUNDARY_TIME_EPSILON_MS}ms)"
                        )
            
            if end_boundary:
                end_t_ms = end_boundary.get("t_ms", 0)
                if end_t_ms < last_delta_t:
                    diff_ms = last_delta_t - end_t_ms
                    msg = (
                        f"turn {turn_id}: turn_end.t_ms {end_t_ms} < last_delta.t_emitted_ms {last_delta_t} "
                        f"(diff={diff_ms}ms, exceeds epsilon={BOUNDARY_TIME_EPSILON_MS}ms)"
                    )
                    if diff_ms > BOUNDARY_TIME_EPSILON_MS:
                        if self._strict_validation:
                            errors.append(msg)
                        else:
                            warnings_list.append(msg)
                    else:
                        warnings_list.append(
                            f"turn {turn_id}: turn_end.t_ms {end_t_ms} < last_delta.t_emitted_ms {last_delta_t} "
                            f"(diff={diff_ms}ms, within epsilon={BOUNDARY_TIME_EPSILON_MS}ms)"
                        )
        
        # Report warnings
        for warning in warnings_list:
            warnings.warn(warning)
        
        # Handle errors
        if errors and self._strict_validation:
            raise TraceValidationError(
                f"Trace validation failed with {len(errors)} error(s):\n" +
                "\n".join(f"  - {e}" for e in errors[:10])
            )
    
    def _reconstruct_turns(self) -> None:
        """
        Reconstruct turn data by accumulating deltas between boundaries.
        """
        self._turn_data = {}
        
        for record in self._records:
            record_type = record.get("record_type")
            
            if record_type == "turn_boundary":
                turn_id = record.get("turn_id", 0)
                agent_id = record.get("agent_id", "unknown")
                boundary = record.get("boundary")
                seq = record.get("seq", 0)
                t_rel_ms = record.get("t_rel_ms", 0)
                
                if turn_id not in self._turn_data:
                    self._turn_data[turn_id] = _TurnData(
                        turn_id=turn_id,
                        agent_id=agent_id,
                    )
                
                if boundary == "start":
                    self._turn_data[turn_id].start_seq = seq
                    self._turn_data[turn_id].start_t_rel_ms = t_rel_ms
                elif boundary == "end":
                    self._turn_data[turn_id].end_seq = seq
                    self._turn_data[turn_id].end_t_rel_ms = t_rel_ms
                    self._turn_data[turn_id].content_hash = record.get("content_hash")
                    
            elif record_type == "stream_delta":
                turn_id = record.get("turn_id", 0)
                seq = record.get("seq", 0)
                delta_text = record.get("delta_text", "")
                t_rel_ms = record.get("t_rel_ms", 0)
                
                if turn_id not in self._turn_data:
                    # Delta without boundary start - create turn data
                    self._turn_data[turn_id] = _TurnData(
                        turn_id=turn_id,
                        agent_id=record.get("agent_id", "unknown"),
                    )
                
                self._turn_data[turn_id].deltas.append((seq, delta_text, t_rel_ms))
    
    def _classify_all_turns(self) -> None:
        """Classify all turns using conservative rules."""
        self._classifications = {}
        self._audit_flags = []
        
        for turn_id, turn_data in self._turn_data.items():
            # Reconstruct turn text from deltas (sorted by seq)
            sorted_deltas = sorted(turn_data.deltas, key=lambda x: x[0])
            turn_text = "".join(d[1] for d in sorted_deltas)
            
            classification = classify_turn(
                turn_id=turn_id,
                turn_text=turn_text,
                delta_count=len(sorted_deltas),
                agent_id=turn_data.agent_id,
                start_seq=turn_data.start_seq or (sorted_deltas[0][0] if sorted_deltas else 0),
                end_seq=turn_data.end_seq or (sorted_deltas[-1][0] if sorted_deltas else 0),
                derived_agent_labels=self._derived_labels,
                audit_flags=self._audit_flags,
            )
            
            self._classifications[turn_id] = classification
    
    def _compute_virtual_time(self, t_rel_ms: int) -> int:
        """Compute virtual time from t_rel_ms, applying shift if configured."""
        return t_rel_ms - self._time_shift
    
    def _iter_events(self, stream_mode: Literal["full", "content_plane"]) -> Iterator[ReplayEvent]:
        """
        Internal iterator that yields replay events.
        
        Args:
            stream_mode: "full" for all events, "content_plane" for content only
        """
        self._ensure_initialized()
        
        # Get set of control_plane turn IDs for filtering
        control_plane_turns = {
            turn_id for turn_id, cls in self._classifications.items()
            if cls.turn_type == "control_plane"
        }
        
        for record in self._records:
            record_type = record.get("record_type")
            
            if record_type == "trace_meta":
                continue
            
            turn_id = record.get("turn_id", 0)
            
            # Skip control_plane turns in content_plane mode
            if stream_mode == "content_plane" and turn_id in control_plane_turns:
                continue
            
            if record_type == "stream_delta":
                yield ReplayEvent(
                    event_type="delta",
                    virtual_time_ms=self._compute_virtual_time(record.get("t_rel_ms", 0)),
                    stream_mode=stream_mode,
                    seq=record.get("seq", 0),
                    turn_id=turn_id,
                    agent_id=record.get("agent_id", "unknown"),
                    delta_text=record.get("delta_text", ""),
                )
                
            elif record_type == "turn_boundary":
                boundary = record.get("boundary")
                yield ReplayEvent(
                    event_type="turn_start" if boundary == "start" else "turn_end",
                    virtual_time_ms=self._compute_virtual_time(record.get("t_rel_ms", 0)),
                    stream_mode=stream_mode,
                    seq=record.get("seq", 0),
                    turn_id=turn_id,
                    agent_id=record.get("agent_id", "unknown"),
                    boundary=boundary,
                    content_hash=record.get("content_hash") if boundary == "end" else None,
                )
    
    # =========================================================================
    # Public API: Streams
    # =========================================================================
    
    def replay_full(self) -> Iterator[ReplayEvent]:
        """
        Yield ALL events (deltas + boundaries) in seq order.
        
        Includes both content_plane and control_plane turns.
        """
        return self._iter_events("full")
    
    def replay_content_plane(self) -> Iterator[ReplayEvent]:
        """
        Yield content_plane events only (excludes control_plane turns).
        
        Timing gaps from excluded turns are preserved - virtual time is
        NOT compressed. This ensures TokenGate sees realistic delays.
        """
        return self._iter_events("content_plane")
    
    # =========================================================================
    # Public API: Metadata
    # =========================================================================
    
    def get_metadata(self) -> TraceMeta:
        """Return parsed trace_meta record."""
        self._ensure_initialized()
        return self._metadata
    
    def get_derived_agent_labels(self) -> frozenset[str]:
        """
        Return the agent label set derived from this trace.
        
        Labels are derived from turn_boundary records only (authoritative source).
        """
        self._ensure_initialized()
        return self._derived_labels
    
    def get_turn_classifications(self) -> dict[int, TurnClassification]:
        """
        Return turn_id -> classification mapping.
        
        Every turn has a classification_reason for auditability.
        """
        self._ensure_initialized()
        return self._classifications.copy()
    
    # =========================================================================
    # Public API: Audit
    # =========================================================================
    
    def get_audit_flags(self) -> list[AuditFlag]:
        """
        Return list of audit flags for suspicious turns.
        
        Suspicious turns are classified as content_plane (conservative)
        but flagged for reviewer visibility.
        """
        self._ensure_initialized()
        return self._audit_flags.copy()


# =============================================================================
# Module-level convenience functions
# =============================================================================

def replay_trace(
    trace_path: Path,
    stream: Literal["full", "content_plane"] = "full",
    **kwargs
) -> Iterator[ReplayEvent]:
    """
    Convenience function to replay a trace.
    
    Args:
        trace_path: Path to trace file
        stream: Which stream to replay ("full" or "content_plane")
        **kwargs: Passed to TraceReplayEngine
        
    Yields:
        ReplayEvent objects
    """
    engine = TraceReplayEngine(trace_path, **kwargs)
    
    if stream == "full":
        yield from engine.replay_full()
    else:
        yield from engine.replay_content_plane()


if __name__ == "__main__":
    # Simple test when run directly
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cli/replay_trace.py <trace_file.jsonl.gz>")
        sys.exit(1)
    
    trace_path = Path(sys.argv[1])
    
    try:
        engine = TraceReplayEngine(trace_path, strict_stub_guard=False)
        meta = engine.get_metadata()
        labels = engine.get_derived_agent_labels()
        classifications = engine.get_turn_classifications()
        
        print(f"=== Trace Metadata ===")
        print(f"case_id: {meta.case_id}")
        print(f"schema_version: {meta.schema_version}")
        print(f"stub_mode: {meta.stub_mode}")
        print(f"derived_agent_labels: {labels}")
        print()
        
        print(f"=== Turn Classifications ===")
        content_count = sum(1 for c in classifications.values() if c.turn_type == "content_plane")
        control_count = sum(1 for c in classifications.values() if c.turn_type == "control_plane")
        print(f"Total: {len(classifications)} turns ({content_count} content_plane, {control_count} control_plane)")
        
        for turn_id in sorted(classifications.keys()):
            cls = classifications[turn_id]
            text_preview = cls.turn_text[:50].replace("\n", " ") + "..." if len(cls.turn_text) > 50 else cls.turn_text.replace("\n", " ")
            print(f"  turn={turn_id}: {cls.turn_type:14} | {cls.classification_reason:30} | \"{text_preview}\"")
        
        flags = engine.get_audit_flags()
        if flags:
            print()
            print(f"=== Audit Flags ({len(flags)}) ===")
            for flag in flags:
                print(f"  turn={flag.turn_id}: {flag.flag_type} - {flag.details[:80]}")
        
    except TraceReplayError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

