#!/usr/bin/env python3
"""
Deterministic utilities for replay engine.

Paper hook: "Event timestamps are derived deterministically from frozen
trace emission times (Section 3.1)"

STRICT RULES:
    1. No current time usage in production paths
    2. No silent 1970 epoch fallback
    3. Track and report all missing timestamps
    4. Fail validation if missing exceeds threshold

Dependencies:
    - Python 3.10+
    - datetime (stdlib)
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class TimestampDerivationStats:
    """
    Track timestamp derivation for auditability.
    
    Paper hook: "Timestamp derivation statistics logged in run_meta
    for reviewer verification (Section 3.2)"
    """
    derived_from_t_emitted: int = 0
    derived_from_fallback: int = 0
    missing_t_emitted_ms_count: int = 0


class DeterministicTimestamps:
    """
    Derive deterministic timestamps from trace data.
    
    Paper hook: "Timestamps derived from trace t_emitted_ms ensure
    byte-stable replay across runs (Section 3.1)"
    
    STRICT: No silent fallbacks. All missing timestamps are tracked
    and reported. Fallbacks use deterministic derivation from run_start_time.
    """
    
    def __init__(
        self,
        trace_chunks: list[dict],
        max_missing_threshold: int = 10
    ):
        """
        Initialize with trace chunks.
        
        Args:
            trace_chunks: List of stream_delta records (record_type == "stream_delta")
            max_missing_threshold: Max allowed missing timestamps before warning
            
        Raises:
            ValueError: If no chunks available
        """
        if not trace_chunks:
            raise ValueError("Cannot derive timestamps: no chunks available")
        
        self.chunks_by_seq = {
            c.get("seq", c.get("sequence_num", 0)): c
            for c in trace_chunks
        }
        self.max_missing_threshold = max_missing_threshold
        self.stats = TimestampDerivationStats()
        
        # Derive run_start_time deterministically from first chunk
        self.run_start_time = self._derive_run_start_time()
    
    def _derive_run_start_time(self) -> int:
        """
        Derive run start time from first available chunk timestamp.
        
        NO arbitrary epoch fallback. Uses first available timestamp
        from trace data.
        
        Returns:
            Unix timestamp in milliseconds
        """
        # Find first chunk with t_emitted_ms
        for seq in sorted(self.chunks_by_seq.keys()):
            chunk = self.chunks_by_seq[seq]
            if "t_emitted_ms" in chunk and chunk["t_emitted_ms"] is not None:
                return int(chunk["t_emitted_ms"])
        
        # Try ISO timestamp field as secondary source
        for seq in sorted(self.chunks_by_seq.keys()):
            chunk = self.chunks_by_seq[seq]
            if "timestamp" in chunk and chunk["timestamp"]:
                try:
                    # Parse ISO timestamp
                    ts_str = chunk["timestamp"]
                    if ts_str.endswith("Z"):
                        ts_str = ts_str[:-1] + "+00:00"
                    dt = datetime.fromisoformat(ts_str)
                    return int(dt.timestamp() * 1000)
                except (ValueError, TypeError):
                    continue
        
        # Last resort: ALL chunks missing timestamps
        # This is a data quality issue that MUST be visible in output
        # Use 0 so it's obvious in timestamps (not hidden 1970)
        self.stats.missing_t_emitted_ms_count = len(self.chunks_by_seq)
        warnings.warn(
            f"ALL {len(self.chunks_by_seq)} chunks missing timestamps. "
            "Using 0 as run_start_time. Trace data quality degraded."
        )
        return 0
    
    def get_chunk_timestamp(self, seq: int) -> str:
        """
        Get deterministic timestamp for a chunk-level event.
        
        Args:
            seq: Sequence number of the chunk
            
        Returns:
            ISO-8601 UTC timestamp string
        """
        chunk = self.chunks_by_seq.get(seq)
        
        if chunk and "t_emitted_ms" in chunk and chunk["t_emitted_ms"] is not None:
            self.stats.derived_from_t_emitted += 1
            return self._ms_to_iso(chunk["t_emitted_ms"])
        
        # Fallback: deterministic offset from run_start_time
        # NOT silent - tracked in stats
        self.stats.derived_from_fallback += 1
        self.stats.missing_t_emitted_ms_count += 1
        fallback_ms = self.run_start_time + (seq * 1000)
        return self._ms_to_iso(fallback_ms)
    
    def get_window_timestamp(self, end_seq: int) -> str:
        """
        Get timestamp for a window event (summary_event, buffer_decision).
        
        Paper hook: "Window event timestamps use end_seq chunk's t_emitted_ms
        for deterministic ordering (Section 3.1)"
        
        Args:
            end_seq: End sequence number of the window
            
        Returns:
            ISO-8601 UTC timestamp string
        """
        return self.get_chunk_timestamp(end_seq)
    
    def get_stats(self) -> dict:
        """
        Get derivation stats for run_meta logging.
        
        Returns:
            Dict with timestamp derivation statistics
        """
        return {
            "method": "trace_t_emitted_ms",
            "fallback_method": "run_start_time_plus_seq",
            "derived_from_t_emitted": self.stats.derived_from_t_emitted,
            "derived_from_fallback": self.stats.derived_from_fallback,
            "missing_t_emitted_ms_count": self.stats.missing_t_emitted_ms_count
        }
    
    def validate(self) -> bool:
        """
        Check if missing timestamps exceed threshold.
        
        Returns:
            True if within threshold, False otherwise
        """
        if self.stats.missing_t_emitted_ms_count > self.max_missing_threshold:
            warnings.warn(
                f"Missing t_emitted_ms count ({self.stats.missing_t_emitted_ms_count}) "
                f"exceeds threshold ({self.max_missing_threshold}). "
                "Trace data quality may be degraded."
            )
            return False
        return True
    
    @staticmethod
    def _ms_to_iso(ms: int) -> str:
        """
        Convert milliseconds to ISO-8601 UTC timestamp.
        
        Args:
            ms: Unix timestamp in milliseconds
            
        Returns:
            ISO-8601 formatted string with millisecond precision
        """
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        # Format: 2025-12-21T10:30:45.123Z
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ms % 1000):03d}Z"


def generate_event_id(case_id: str, variant_id: str, event_index: int) -> str:
    """
    Generate deterministic, collision-free event ID.
    
    Paper hook: "Event IDs are deterministic: {case_id}-{variant_id}-se-{index:03d}
    for reproducible reference (Section 3.2)"
    
    Args:
        case_id: Case identifier (e.g., "case-33651373")
        variant_id: Variant identifier (e.g., "V0")
        event_index: Zero-based event index
        
    Returns:
        Event ID string
    """
    return f"{case_id}-{variant_id}-se-{event_index:03d}"


def generate_decision_id(case_id: str, variant_id: str, decision_index: int) -> str:
    """
    Generate deterministic, collision-free decision ID.
    
    Args:
        case_id: Case identifier
        variant_id: Variant identifier
        decision_index: Zero-based decision index
        
    Returns:
        Decision ID string
    """
    return f"{case_id}-{variant_id}-bd-{decision_index:03d}"


def compute_ctu(text: str) -> int:
    """
    Compute Character-normalized Token Units.
    
    Paper hook: "CTU = ceil(len(text) / 4) provides vendor-agnostic,
    deterministic text unit measurement (Section 6.1)"
    
    Args:
        text: Input text string
        
    Returns:
        CTU count
    """
    import math
    return math.ceil(len(text) / 4)


def compute_text_hash(text: str) -> str:
    """
    Compute SHA-256 hash of text for verification.
    
    Args:
        text: Input text
        
    Returns:
        Hash string prefixed with "sha256:"
    """
    import hashlib
    h = hashlib.sha256(text.encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


class DeterministicRNG:
    """
    Deterministic random number generator for any randomized operations.
    
    Ensures reproducibility across runs.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize with fixed seed.
        
        Args:
            seed: Random seed (default: 42)
        """
        import random
        self.rng = random.Random(seed)
        self.seed = seed
    
    def choice(self, seq):
        """Deterministic choice from sequence."""
        return self.rng.choice(seq)
    
    def shuffle(self, seq):
        """Deterministic in-place shuffle."""
        self.rng.shuffle(seq)
    
    def sample(self, population, k):
        """Deterministic sample without replacement."""
        return self.rng.sample(population, k)





