"""Data models for TokenGate calibration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Policy:
    """A single parameter combination (policy)."""

    policy_id: str
    min_words: int
    max_words: int
    silence_timer_ms: int
    max_wait_timeout_ms: int
    boundary_cues: str = ".?!\n"


@dataclass
class FlushEvent:
    """Recorded flush event during replay."""

    agent_id: str
    flush_time_ms: int
    chunk_words: int
    flush_reason: str
    is_end_of_trace: bool = False


@dataclass
class CaseMetrics:
    """Per-case metrics computed from flush events."""

    case_id: str
    policy_id: str
    ttff_content_ms: Optional[int] = None
    ttff_trace_ms: Optional[int] = None
    flush_count: int = 0
    chunk_size_p50: Optional[float] = None
    chunk_size_p90: Optional[float] = None
    chunk_size_p95: Optional[float] = None
    chunk_size_max: Optional[int] = None
    worst_wait_ms: Optional[float] = None
    worst_wait_max_ms: Optional[int] = None
    spam_pct: Optional[float] = None
    timer_flush_pct: Optional[float] = None
    timer_under_min_pct: Optional[float] = None
    end_of_trace_flush_count: int = 0
    flush_events: List[FlushEvent] = field(default_factory=list)


@dataclass
class PolicyMetrics:
    """Aggregated metrics across all cases for a policy."""

    policy_id: str
    min_words: int
    max_words: int
    silence_timer_ms: int
    max_wait_timeout_ms: int

    # Aggregated metrics (mean/median/p95)
    ttff_content_p50_ms: Optional[float] = None
    ttff_content_p95_ms: Optional[float] = None
    ttff_trace_p50_ms: Optional[float] = None
    flush_count_mean: Optional[float] = None
    chunk_size_p50: Optional[float] = None
    chunk_size_p90: Optional[float] = None
    chunk_size_p95: Optional[float] = None
    chunk_size_max: Optional[int] = None
    worst_wait_p95_ms: Optional[float] = None
    worst_wait_max_ms: Optional[int] = None
    spam_pct_mean: Optional[float] = None
    timer_flush_pct_mean: Optional[float] = None
    timer_under_min_pct_mean: Optional[float] = None

    # Constraint violation flags
    constraint_violations: List[str] = field(default_factory=list)

    # Selection scores
    weighted_score: Optional[float] = None
