#!/usr/bin/env python3
"""
EXAID Evaluation - TokenGate Calibration Script
Phase 5: TokenGate Trigger Calibration

Calibrates TokenGate trigger parameters (min_words, max_words, silence_timer,
max_wait_timeout) by systematically evaluating literature-informed parameter
combinations across frozen v2.0.0 traces.

Selection Method:
    Uses 3-objective Pareto frontier + utopia-distance selection:
    - Objectives: Minimize TTFF, minimize flush_count, maximize chunk_size
    - Normalization: Data-driven percentile-based bounds (P05/P95) computed from survivors
    - Pareto frontier: k-dimensional non-dominated points (excludes dropped metrics)
    - Utopia distance: Dimension-normalized Euclidean distance to (1,1,1) in goodness space
    - Fallbacks: Weighted score (renormalized weights) or lexicographic tie-breaking

Usage:
    python src/calibrate_tokengate.py \
        --traces data/traces/ \
        --manifest data/manifests/exaid_traces_*.manifest.jsonl \
        --config configs/calibration_sweep.yaml \
        --output data/calibration/

Dependencies:
    - trace_replay_engine.py (trace replay)
    - token_gate.py (TokenGate with ManualClock)
"""

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# exaid_core should be available via PYTHONPATH=/app (set in Dockerfile)
# If not in Docker, add repo root to path
repo_root = Path(__file__).parent.parent.parent.resolve()
if (repo_root / "exaid_core").exists():
    sys.path.insert(0, str(repo_root))

from trace_replay_engine import TraceReplayEngine, StubTraceError, TraceValidationError
from exaid_core.token_gate.token_gate import TokenGate, ManualClock


# =============================================================================
# Data Structures
# =============================================================================

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


# =============================================================================
# Parameter Sweep Generation
# =============================================================================

def generate_policy_grid(config: dict) -> List[Policy]:
    """Generate all parameter combinations from grid."""
    grid = config["parameter_grid"]
    policies = []
    
    policy_idx = 0
    for min_words in grid["min_words"]:
        for max_words in grid["max_words"]:
            for silence_timer_ms in grid["silence_timer_ms"]:
                for max_wait_timeout_ms in grid["max_wait_timeout_ms"]:
                    policy_id = f"policy_{policy_idx:04d}"
                    policies.append(Policy(
                        policy_id=policy_id,
                        min_words=min_words,
                        max_words=max_words,
                        silence_timer_ms=silence_timer_ms,
                        max_wait_timeout_ms=max_wait_timeout_ms,
                        boundary_cues=grid.get("boundary_cues", ".?!\n")
                    ))
                    policy_idx += 1
    
    return policies


def filter_valid_policies(policies: List[Policy], config: dict) -> Tuple[List[Policy], List[Tuple[Policy, str]]]:
    """Filter out invalid parameter combinations."""
    constraints = config.get("validity_constraints", {})
    valid_policies = []
    invalid_policies = []
    
    for policy in policies:
        reasons = []
        
        # Constraint 1: min_words < max_words (strictly)
        if constraints.get("min_words_lt_max_words", True):
            if policy.min_words >= policy.max_words:
                reasons.append(f"min_words ({policy.min_words}) >= max_words ({policy.max_words})")
        
        # Constraint 2: max_wait_timeout_ms >= silence_timer_ms
        if constraints.get("max_wait_gte_silence", True):
            if policy.max_wait_timeout_ms < policy.silence_timer_ms:
                reasons.append(f"max_wait_timeout_ms ({policy.max_wait_timeout_ms}) < silence_timer_ms ({policy.silence_timer_ms})")
        
        # Constraint 3: Optional gap between min and max
        min_gap = constraints.get("min_gap_between_min_max")
        if min_gap is not None:
            if policy.max_words < policy.min_words + min_gap:
                reasons.append(f"max_words ({policy.max_words}) < min_words ({policy.min_words}) + {min_gap}")
        
        if reasons:
            invalid_policies.append((policy, "; ".join(reasons)))
        else:
            valid_policies.append(policy)
    
    return valid_policies, invalid_policies


# =============================================================================
# Reproducibility: Hash Computation
# =============================================================================

def compute_trace_dataset_hash(manifest_path: Path) -> str:
    """Compute SHA256 hash of canonical manifest fields."""
    # Load manifest and extract canonical fields
    records = []
    with open(manifest_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    
    # Extract canonical fields from manifest_meta and provenance
    canonical_data = {}
    for record in records:
        if record.get("record_type") == "manifest_meta":
            canonical_data["dataset_id"] = record.get("dataset_id", "")
            canonical_data["mas_run_id"] = record.get("mas_run_id", "")
        elif record.get("record_type") == "provenance":
            canonical_data["mac_commit"] = record.get("mac_commit", "")
            canonical_data["case_list_hash"] = record.get("case_list_hash", "")
            # Include trace entries (case_id, sha256 pairs)
            trace_entries = []
            for r in records:
                if r.get("record_type") == "trace_entry":
                    trace_entries.append({
                        "case_id": r.get("case_id", ""),
                        "sha256": r.get("sha256", "")
                    })
            canonical_data["trace_entries"] = sorted(trace_entries, key=lambda x: x["case_id"])
    
    # Canonicalize JSON
    canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_config_hash(config: dict) -> str:
    """Compute SHA256 hash of canonicalized sweep configuration."""
    # Create canonical copy (exclude non-deterministic fields)
    canonical_config = {
        "parameter_grid": config["parameter_grid"],
        "validity_constraints": config.get("validity_constraints", {}),
        "constraints": config.get("constraints", {}),
        "selection": config.get("selection", {}),
        "spam": config.get("spam", {})
    }
    
    canonical_json = json.dumps(canonical_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_exaid_commit() -> str:
    """Get current EXAID git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate_calibration_run_id(trace_dataset_hash: str, config_hash: str, exaid_commit: str) -> str:
    """Generate deterministic calibration run ID."""
    hash8 = lambda h: h[:8]
    return f"calib_{hash8(trace_dataset_hash)}_{hash8(config_hash)}_{hash8(exaid_commit)}"


# =============================================================================
# Event-Driven Replay with Timer Checks
# =============================================================================

async def replay_case_with_policy(
    trace_path: Path,
    policy: Policy,
    case_id: str,
    trace_meta,
    engine: Optional[TraceReplayEngine] = None,
    strict_stub_guard: bool = True,
    strict_validation: bool = True
) -> CaseMetrics:
    """
    Replay a single case with a policy, recording flush events.
    
    Production-faithful replay: Matches production behavior exactly:
    - Advance clock to event time
    - Call add_token() (which internally checks silence timer before adding)
    - Call check_timers() after add_token() (matching production line 144)
    - On turn_end, call flush() with reason="turn_end" (matching production flush_agent())
    
    We do NOT simulate timer expiries during gaps between events, as production
    has no background/tick loop. Timers are only checked synchronously after
    each token addition or at turn boundaries.
    
    Args:
        trace_path: Path to trace file
        policy: Policy to evaluate
        case_id: Case identifier
        trace_meta: Trace metadata (from engine.get_metadata())
        engine: Optional pre-created TraceReplayEngine (if None, creates one)
        strict_stub_guard: Whether to enforce strict stub guard (default: True)
        strict_validation: Whether to enforce strict validation (default: True)
    """
    
    # Initialize TokenGate with ManualClock
    # trace_meta.t0_emitted_ms is milliseconds since epoch
    t0_datetime = datetime.fromtimestamp(trace_meta.t0_emitted_ms / 1000.0, tz=timezone.utc)
    clock = ManualClock(t0_datetime)
    token_gate = TokenGate(
        min_words=policy.min_words,
        max_words=policy.max_words,
        boundary_cues=policy.boundary_cues,
        silence_timer=policy.silence_timer_ms / 1000.0,  # Convert ms to seconds
        max_wait_timeout=policy.max_wait_timeout_ms / 1000.0,  # Convert ms to seconds
        clock=clock
    )
    
    # Use provided engine or create one with specified strictness
    if engine is None:
        engine = TraceReplayEngine(
            trace_path,
            strict_stub_guard=strict_stub_guard,
            strict_validation=strict_validation
        )
    
    # Track flush events
    flush_events: List[FlushEvent] = []
    first_content_delta_time_ms: Optional[int] = None
    trace_t0_ms = trace_meta.t0_emitted_ms
    
    # Replay content_plane stream
    events = list(engine.replay_content_plane())
    
    if not events:
        # No content events - return empty metrics
        return CaseMetrics(case_id=case_id, policy_id=policy.policy_id)
    
    # Process events matching production behavior exactly
    for event in events:
        current_time_ms = event.virtual_time_ms
        
        # Advance clock to event time
        # trace_t0_ms is milliseconds since epoch, current_time_ms is relative ms
        event_datetime = datetime.fromtimestamp((trace_t0_ms + current_time_ms) / 1000.0, tz=timezone.utc)
        clock.set_time(event_datetime)
        
        # Process delta event
        if event.event_type == "delta" and event.delta_text:
            # Track first content delta for TTFF_content
            if first_content_delta_time_ms is None:
                first_content_delta_time_ms = current_time_ms
            
            agent_id = event.agent_id
            
            # Add token (internally checks silence timer before adding)
            flushed_text = await token_gate.add_token(agent_id, event.delta_text)
            
            if flushed_text:
                # Flush occurred from add_token()
                flush_reason = token_gate.get_last_flush_reason(agent_id) or "unknown"
                flush_time = token_gate.get_last_flush_time(agent_id)
                # Convert datetime to milliseconds since trace t0
                if flush_time:
                    flush_time_ms = int((flush_time.timestamp() * 1000 - trace_t0_ms))
                else:
                    flush_time_ms = current_time_ms
                
                chunk_words = len(flushed_text.split())
                
                flush_events.append(FlushEvent(
                    agent_id=agent_id,
                    flush_time_ms=flush_time_ms,
                    chunk_words=chunk_words,
                    flush_reason=flush_reason,
                    is_end_of_trace=False
                ))
            
            # Check timers after add_token() (matching production line 144)
            timer_chunk = await token_gate.check_timers(agent_id)
            if timer_chunk:
                flush_reason = token_gate.get_last_flush_reason(agent_id) or "timer"
                flush_time = token_gate.get_last_flush_time(agent_id)
                if flush_time:
                    flush_time_ms = int((flush_time.timestamp() * 1000 - trace_t0_ms))
                else:
                    flush_time_ms = current_time_ms
                
                chunk_words = len(timer_chunk.split())
                
                flush_events.append(FlushEvent(
                    agent_id=agent_id,
                    flush_time_ms=flush_time_ms,
                    chunk_words=chunk_words,
                    flush_reason=flush_reason,
                    is_end_of_trace=False
                ))
        
        # Process turn_end event (explicit flush, matching production flush_agent())
        elif event.event_type == "turn_end":
            agent_id = event.agent_id
            
            # Explicitly flush buffer at turn end (matching production flush_agent() behavior)
            flushed_text = await token_gate.flush(agent_id, reason="turn_end")
            
            if flushed_text:
                flush_time = token_gate.get_last_flush_time(agent_id)
                if flush_time:
                    flush_time_ms = int((flush_time.timestamp() * 1000 - trace_t0_ms))
                else:
                    flush_time_ms = current_time_ms
                
                chunk_words = len(flushed_text.split())
                
                flush_events.append(FlushEvent(
                    agent_id=agent_id,
                    flush_time_ms=flush_time_ms,
                    chunk_words=chunk_words,
                    flush_reason="turn_end",  # Explicit reason, not timer-based
                    is_end_of_trace=False  # Mid-trace turn end, not end of entire trace
                ))
    
    # End-of-trace handling: deterministic cleanup flush without timer simulation
    # Production-faithful: we do NOT advance time or simulate timer expiries.
    # Instead, we perform a deterministic cleanup flush for any remaining buffers.
    if events:
        last_event = events[-1]
        last_time_ms = last_event.virtual_time_ms
        
        # Flush any remaining buffers without advancing time (production-faithful)
        for agent_id in list(token_gate.buffers.keys()):
            if agent_id in token_gate.buffers and token_gate.buffers[agent_id]:
                # Flush with deterministic reason (not timer-based)
                flushed_text = await token_gate.flush(agent_id, reason="end_of_trace")
                if flushed_text:
                    chunk_words = len(flushed_text.split())
                    
                    flush_events.append(FlushEvent(
                        agent_id=agent_id,
                        flush_time_ms=last_time_ms,  # Use last event time, don't advance
                        chunk_words=chunk_words,
                        flush_reason="end_of_trace",
                        is_end_of_trace=True
                    ))
    
    # Compute metrics from flush events
    return compute_case_metrics(case_id, policy.policy_id, flush_events, first_content_delta_time_ms, trace_t0_ms, policy)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_case_metrics(
    case_id: str,
    policy_id: str,
    flush_events: List[FlushEvent],
    first_content_delta_time_ms: Optional[int],
    trace_t0_ms: int,
    policy: Policy,
    alpha: float = 0.7
) -> CaseMetrics:
    """Compute per-case metrics from flush events."""
    
    if not flush_events:
        return CaseMetrics(case_id=case_id, policy_id=policy_id, flush_events=flush_events)
    
    # Separate end-of-trace flushes
    regular_flushes = [e for e in flush_events if not e.is_end_of_trace]
    end_of_trace_flushes = [e for e in flush_events if e.is_end_of_trace]
    
    # Use regular_flush_count for cost constraints (excludes end-of-trace flushes)
    regular_flush_count = len(regular_flushes)
    
    metrics = CaseMetrics(
        case_id=case_id,
        policy_id=policy_id,
        flush_count=regular_flush_count,  # Cost metric: only regular flushes count
        end_of_trace_flush_count=len(end_of_trace_flushes),
        flush_events=flush_events
    )
    
    # TTFF metrics
    if regular_flushes:
        first_flush = regular_flushes[0]
        metrics.ttff_trace_ms = first_flush.flush_time_ms
        
        if first_content_delta_time_ms is not None:
            metrics.ttff_content_ms = first_flush.flush_time_ms - first_content_delta_time_ms
    
    # Chunk size statistics
    chunk_sizes = [e.chunk_words for e in regular_flushes]
    if chunk_sizes:
        sorted_sizes = sorted(chunk_sizes)
        n = len(sorted_sizes)
        metrics.chunk_size_p50 = sorted_sizes[n // 2] if n > 0 else None
        metrics.chunk_size_p90 = sorted_sizes[int(n * 0.9)] if n > 0 else None
        metrics.chunk_size_p95 = sorted_sizes[int(n * 0.95)] if n > 0 else None
        metrics.chunk_size_max = max(chunk_sizes)
    
    # Worst wait times (time between flushes)
    if len(regular_flushes) > 1:
        wait_times = []
        for i in range(1, len(regular_flushes)):
            wait_ms = regular_flushes[i].flush_time_ms - regular_flushes[i-1].flush_time_ms
            wait_times.append(wait_ms)
        
        if wait_times:
            sorted_waits = sorted(wait_times)
            n = len(sorted_waits)
            metrics.worst_wait_ms = sorted_waits[int(n * 0.95)] if n > 0 else None
            metrics.worst_wait_max_ms = max(wait_times)
    
    # Spam percentage (policy-relative)
    if regular_flushes:
        spam_threshold = alpha * policy.min_words
        spam_count = sum(1 for e in regular_flushes if e.chunk_words < spam_threshold)
        metrics.spam_pct = (spam_count / len(regular_flushes)) * 100.0
    
    # Timer flush percentage
    timer_flush_reasons = {"silence_timer", "max_wait_timeout"}
    timer_flushes = [e for e in regular_flushes if e.flush_reason in timer_flush_reasons]
    if regular_flushes:
        metrics.timer_flush_pct = (len(timer_flushes) / len(regular_flushes)) * 100.0
    
    # Timer under-minimum percentage
    if timer_flushes:
        under_min_count = sum(1 for e in timer_flushes if e.chunk_words < policy.min_words)
        metrics.timer_under_min_pct = (under_min_count / len(timer_flushes)) * 100.0
    
    return metrics


def aggregate_policy_metrics(
    policy: Policy,
    case_metrics_list: List[CaseMetrics]
) -> PolicyMetrics:
    """Aggregate metrics across all cases for a policy."""
    
    policy_metrics = PolicyMetrics(
        policy_id=policy.policy_id,
        min_words=policy.min_words,
        max_words=policy.max_words,
        silence_timer_ms=policy.silence_timer_ms,
        max_wait_timeout_ms=policy.max_wait_timeout_ms
    )
    
    if not case_metrics_list:
        return policy_metrics
    
    # Aggregate TTFF
    ttff_content_values = [m.ttff_content_ms for m in case_metrics_list if m.ttff_content_ms is not None]
    if ttff_content_values:
        sorted_ttff = sorted(ttff_content_values)
        n = len(sorted_ttff)
        policy_metrics.ttff_content_p50_ms = sorted_ttff[n // 2]
        policy_metrics.ttff_content_p95_ms = sorted_ttff[int(n * 0.95)]
    
    ttff_trace_values = [m.ttff_trace_ms for m in case_metrics_list if m.ttff_trace_ms is not None]
    if ttff_trace_values:
        sorted_ttff = sorted(ttff_trace_values)
        n = len(sorted_ttff)
        policy_metrics.ttff_trace_p50_ms = sorted_ttff[n // 2]
    
    # Aggregate flush count
    flush_counts = [m.flush_count for m in case_metrics_list]
    if flush_counts:
        policy_metrics.flush_count_mean = sum(flush_counts) / len(flush_counts)
    
    # Aggregate chunk sizes
    # p50: median of per-case medians
    chunk_size_p50s = [m.chunk_size_p50 for m in case_metrics_list if m.chunk_size_p50 is not None]
    if chunk_size_p50s:
        sorted_p50s = sorted(chunk_size_p50s)
        n = len(sorted_p50s)
        policy_metrics.chunk_size_p50 = sorted_p50s[n // 2]
    
    # p90/p95: aggregate from per-case p90/p95 values (not from p50s)
    # For hard constraint enforcement, use max() to guarantee no case exceeds threshold
    chunk_size_p90s = [m.chunk_size_p90 for m in case_metrics_list if m.chunk_size_p90 is not None]
    if chunk_size_p90s:
        sorted_p90s = sorted(chunk_size_p90s)
        n = len(sorted_p90s)
        policy_metrics.chunk_size_p90 = sorted_p90s[int(n * 0.9)] if n > 0 else None
    
    chunk_size_p95s = [m.chunk_size_p95 for m in case_metrics_list if m.chunk_size_p95 is not None]
    if chunk_size_p95s:
        # Use max() for strongest guarantee (reviewer-proof hard cap)
        policy_metrics.chunk_size_p95 = max(chunk_size_p95s)
    
    chunk_size_maxs = [m.chunk_size_max for m in case_metrics_list if m.chunk_size_max is not None]
    if chunk_size_maxs:
        policy_metrics.chunk_size_max = max(chunk_size_maxs)
    
    # Aggregate worst wait
    worst_waits = [m.worst_wait_ms for m in case_metrics_list if m.worst_wait_ms is not None]
    if worst_waits:
        sorted_waits = sorted(worst_waits)
        n = len(sorted_waits)
        policy_metrics.worst_wait_p95_ms = sorted_waits[int(n * 0.95)] if n > 0 else None
        policy_metrics.worst_wait_max_ms = max(worst_waits)
    
    # Aggregate spam percentage
    spam_pcts = [m.spam_pct for m in case_metrics_list if m.spam_pct is not None]
    if spam_pcts:
        policy_metrics.spam_pct_mean = sum(spam_pcts) / len(spam_pcts)
    
    # Aggregate timer percentages
    timer_flush_pcts = [m.timer_flush_pct for m in case_metrics_list if m.timer_flush_pct is not None]
    if timer_flush_pcts:
        policy_metrics.timer_flush_pct_mean = sum(timer_flush_pcts) / len(timer_flush_pcts)
    
    timer_under_min_pcts = [m.timer_under_min_pct for m in case_metrics_list if m.timer_under_min_pct is not None]
    if timer_under_min_pcts:
        policy_metrics.timer_under_min_pct_mean = sum(timer_under_min_pcts) / len(timer_under_min_pcts)
    
    return policy_metrics


# =============================================================================
# Constraint Filters
# =============================================================================

def check_constraints(policy_metrics: PolicyMetrics, config: dict) -> List[str]:
    """Check if policy violates any constraints. Returns list of violation reasons."""
    constraints = config.get("constraints", {})
    violations = []
    
    # ttff_content_p95_ms <= threshold
    threshold = constraints.get("ttff_content_p95_ms")
    if threshold is not None and policy_metrics.ttff_content_p95_ms is not None:
        if policy_metrics.ttff_content_p95_ms > threshold:
            violations.append(f"ttff_content_p95_ms ({policy_metrics.ttff_content_p95_ms:.1f}) > {threshold}")
    
    # spam_pct_mean <= threshold
    threshold = constraints.get("spam_pct_mean")
    if threshold is not None and policy_metrics.spam_pct_mean is not None:
        if policy_metrics.spam_pct_mean > threshold:
            violations.append(f"spam_pct_mean ({policy_metrics.spam_pct_mean:.1f}%) > {threshold}%")
    
    # timer_under_min_pct_mean <= threshold
    threshold = constraints.get("timer_under_min_pct_mean")
    if threshold is not None and policy_metrics.timer_under_min_pct_mean is not None:
        if policy_metrics.timer_under_min_pct_mean > threshold:
            violations.append(f"timer_under_min_pct_mean ({policy_metrics.timer_under_min_pct_mean:.1f}%) > {threshold}%")
    
    # chunk_size_p50 >= min_ratio * min_words
    min_ratio = constraints.get("chunk_size_p50_min_ratio")
    if min_ratio is not None and policy_metrics.chunk_size_p50 is not None:
        min_chunk_size = min_ratio * policy_metrics.min_words
        if policy_metrics.chunk_size_p50 < min_chunk_size:
            violations.append(f"chunk_size_p50 ({policy_metrics.chunk_size_p50:.1f}) < {min_ratio} * min_words ({min_chunk_size:.1f})")
    
    # chunk_size_p95 <= max
    threshold = constraints.get("chunk_size_p95_max")
    if threshold is not None and policy_metrics.chunk_size_p95 is not None:
        if policy_metrics.chunk_size_p95 > threshold:
            violations.append(f"chunk_size_p95 ({policy_metrics.chunk_size_p95:.1f}) > {threshold}")
    
    # worst_wait_p95_ms <= threshold
    threshold = constraints.get("worst_wait_p95_ms")
    if threshold is not None and policy_metrics.worst_wait_p95_ms is not None:
        if policy_metrics.worst_wait_p95_ms > threshold:
            violations.append(f"worst_wait_p95_ms ({policy_metrics.worst_wait_p95_ms:.1f}) > {threshold}")
    
    # flush_count_mean <= max (cost constraint: BufferAgent calls per case)
    threshold = constraints.get("flush_count_mean")
    if threshold is not None and policy_metrics.flush_count_mean is not None:
        if policy_metrics.flush_count_mean > threshold:
            violations.append(f"flush_count_mean ({policy_metrics.flush_count_mean:.1f}) > {threshold}")
    
    # chunk_size_p50 >= min (absolute cost constraint: minimum median chunk size)
    threshold = constraints.get("chunk_size_p50_min")
    if threshold is not None and policy_metrics.chunk_size_p50 is not None:
        if policy_metrics.chunk_size_p50 < threshold:
            violations.append(f"chunk_size_p50 ({policy_metrics.chunk_size_p50:.1f}) < {threshold}")
    
    return violations


# =============================================================================
# Selection Rules
# =============================================================================

def normalize_value(value: float, lower: float, upper: float, invert: bool = False) -> float:
    """Normalize value to [0, 1] range."""
    if value < lower:
        normalized = 0.0
    elif value > upper:
        normalized = 1.0
    else:
        normalized = (value - lower) / (upper - lower) if upper > lower else 0.0
    
    if invert:
        return 1.0 - normalized
    return normalized


# Epsilon constants for degenerate bounds detection
# Relaxed thresholds to avoid false positives from floating-point rounding
EPS_MS = 5.0  # For TTFF (milliseconds)
EPS_COUNT = 2.0  # For flush_count (counts)
EPS_WORDS = 2.0  # For chunk_size (words)


def compute_percentile_bounds(
    survivor_metrics: List[PolicyMetrics],
    metric_name: str,
    p05: float = 0.05,
    p95: float = 0.95,
    eps: float = 2.0  # Default epsilon (relaxed to avoid floating-point rounding issues)
) -> dict:
    """
    Compute percentile-based normalization bounds from survivor metrics.
    
    Handles small-N cases (len < 5) by using min/max instead of percentiles.
    Marks bounds as dropped if hi - lo < eps (degenerate case).
    
    Epsilon thresholds are relaxed (EPS_MS=5.0, EPS_COUNT=2.0, EPS_WORDS=2.0) to avoid
    false positives from floating-point rounding while still detecting truly degenerate metrics.
    
    Args:
        survivor_metrics: List of PolicyMetrics that passed constraints
        metric_name: Name of metric attribute (e.g., "ttff_content_p50_ms")
        p05: Lower percentile (default 0.05)
        p95: Upper percentile (default 0.95)
        eps: Epsilon threshold for degenerate bounds detection (relaxed to avoid floating-point issues)
        
    Returns:
        Dict with keys: lo, hi, method, computed_over, dropped (bool)
    """
    # Extract metric values
    values = []
    for pm in survivor_metrics:
        val = getattr(pm, metric_name, None)
        if val is not None:
            values.append(val)
    
    if not values:
        # No valid values - return dropped bounds
        return {
            "lo": None,
            "hi": None,
            "method": "none",
            "computed_over": "survivor_policies",
            "dropped": True
        }
    
    # Small-N handling: use min/max if len < 5
    if len(values) < 5:
        lo = min(values)
        hi = max(values)
        method = "min_max_small_n"
    else:
        # Compute percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)
        lo_idx = int(n * p05)
        hi_idx = int(n * p95)
        # Clamp indices
        lo_idx = max(0, min(lo_idx, n - 1))
        hi_idx = max(0, min(hi_idx, n - 1))
        lo = sorted_values[lo_idx]
        hi = sorted_values[hi_idx]
        method = "p05_p95"
    
    # Degeneracy check
    dropped = (hi - lo) < eps
    
    return {
        "lo": lo,
        "hi": hi,
        "method": method,
        "computed_over": "survivor_policies",
        "dropped": dropped
    }


def compute_all_normalization_bounds(
    survivor_metrics: List[PolicyMetrics]
) -> Tuple[dict, List[str]]:
    """
    Compute normalization bounds for all three metrics used in selection.
    
    Args:
        survivor_metrics: List of PolicyMetrics that passed constraints
        
    Returns:
        Tuple of (bounds_dict, dropped_metrics_list)
        bounds_dict: Dict mapping metric names to bounds metadata
        dropped_metrics_list: List of metric names that were dropped
    """
    bounds = {}
    dropped_metrics = []
    
    # TTFF bounds
    ttff_bounds = compute_percentile_bounds(
        survivor_metrics, "ttff_content_p50_ms", eps=EPS_MS
    )
    bounds["ttff_content_p50_ms"] = {
        "lo_ms": ttff_bounds["lo"],
        "hi_ms": ttff_bounds["hi"],
        "method": ttff_bounds["method"],
        "computed_over": ttff_bounds["computed_over"]
    }
    if ttff_bounds["dropped"]:
        dropped_metrics.append("ttff_content_p50_ms")
    
    # Flush count bounds
    flush_bounds = compute_percentile_bounds(
        survivor_metrics, "flush_count_mean", eps=EPS_COUNT
    )
    bounds["flush_count_mean"] = {
        "lo": flush_bounds["lo"],
        "hi": flush_bounds["hi"],
        "method": flush_bounds["method"],
        "computed_over": flush_bounds["computed_over"]
    }
    if flush_bounds["dropped"]:
        dropped_metrics.append("flush_count_mean")
    
    # Chunk size bounds
    chunk_bounds = compute_percentile_bounds(
        survivor_metrics, "chunk_size_p50", eps=EPS_WORDS
    )
    bounds["chunk_size_p50"] = {
        "lo": chunk_bounds["lo"],
        "hi": chunk_bounds["hi"],
        "method": chunk_bounds["method"],
        "computed_over": chunk_bounds["computed_over"]
    }
    if chunk_bounds["dropped"]:
        dropped_metrics.append("chunk_size_p50")
    
    return bounds, dropped_metrics


def normalize_to_goodness(
    value: float,
    lo: float,
    hi: float,
    lower_is_better: bool
) -> float:
    """
    Convert raw metric value to [0, 1] goodness space.
    
    Args:
        value: Raw metric value
        lo: Lower bound for normalization
        hi: Upper bound for normalization
        lower_is_better: True if lower values are better (e.g., TTFF, flush_count)
        
    Returns:
        Goodness value in [0, 1] where 1.0 is best
    """
    if hi == lo:
        # Degenerate bounds - return neutral value (shouldn't happen if bounds computed correctly)
        return 0.5
    
    # Normalize to [0, 1]
    norm = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    
    # Convert to goodness: if lower is better, invert
    if lower_is_better:
        return 1.0 - norm
    else:
        return norm


def compute_weighted_score(
    policy_metrics: PolicyMetrics,
    config: dict,
    computed_bounds: Optional[dict] = None,
    dropped_metrics: Optional[List[str]] = None
) -> Optional[float]:
    """
    Compute weighted objective function score.
    
    If computed_bounds and dropped_metrics are provided, uses data-driven bounds
    and renormalizes weights when metrics are dropped.
    
    Args:
        policy_metrics: PolicyMetrics to score
        config: Configuration dict
        computed_bounds: Optional computed bounds dict (if None, uses config bounds)
        dropped_metrics: Optional list of dropped metric names (if None, uses all metrics)
        
    Returns:
        Weighted score, or None if all weights were dropped
    """
    selection_config = config.get("selection", {}).get("weighted_score", {})
    weights = selection_config.get("weights", {})
    
    if dropped_metrics is None:
        dropped_metrics = []
    
    # Determine active metrics (not dropped)
    all_metrics = ["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50", "spam_pct_mean", "worst_wait_p95_ms"]
    active_metrics = [m for m in all_metrics if m not in dropped_metrics]
    
    if not active_metrics:
        # All metrics dropped - return None to signal lexicographic fallback
        return None
    
    # Compute active weights sum for renormalization
    active_weights_sum = sum(weights.get(m, 0.0) for m in active_metrics)
    
    if active_weights_sum == 0:
        # All active weights are zero - return None
        return None
    
    # Use computed bounds if provided, otherwise fallback to config bounds
    if computed_bounds is not None:
        bounds = computed_bounds
    else:
        bounds = selection_config.get("normalization_bounds", {})
    
    score = 0.0
    
    # TTFF (lower is better)
    if "ttff_content_p50_ms" in active_metrics and policy_metrics.ttff_content_p50_ms is not None:
        w = weights.get("ttff_content_p50_ms", 0.0)
        if w > 0:
            # Renormalize weight
            normalized_w = w / active_weights_sum
            if computed_bounds and "ttff_content_p50_ms" in computed_bounds:
                b = computed_bounds["ttff_content_p50_ms"]
                lo = b.get("lo_ms", b.get("lo", 5000))
                hi = b.get("hi_ms", b.get("hi", 30000))
            else:
                b = bounds.get("ttff_content_p50_ms", {"lower": 5000, "upper": 30000})
                lo = b.get("lower", 5000)
                hi = b.get("upper", 30000)
            score += normalized_w * normalize_value(policy_metrics.ttff_content_p50_ms, lo, hi, invert=True)
    
    # Flush count (lower is better)
    if "flush_count_mean" in active_metrics and policy_metrics.flush_count_mean is not None:
        w = weights.get("flush_count_mean", 0.0)
        if w > 0:
            normalized_w = w / active_weights_sum
            if computed_bounds and "flush_count_mean" in computed_bounds:
                b = computed_bounds["flush_count_mean"]
                lo = b.get("lo", 10)
                hi = b.get("hi", 150)
            else:
                b = bounds.get("flush_count_mean", {"lower": 10, "upper": 150})
                lo = b.get("lower", 10)
                hi = b.get("upper", 150)
            score += normalized_w * normalize_value(policy_metrics.flush_count_mean, lo, hi, invert=True)
    
    # Chunk size (higher is better)
    if "chunk_size_p50" in active_metrics and policy_metrics.chunk_size_p50 is not None:
        w = weights.get("chunk_size_p50", 0.0)
        if w > 0:
            normalized_w = w / active_weights_sum
            if computed_bounds and "chunk_size_p50" in computed_bounds:
                b = computed_bounds["chunk_size_p50"]
                lo = b.get("lo", 30)
                hi = b.get("hi", 160)
            else:
                b = bounds.get("chunk_size_p50", {"lower": 30, "upper": 160})
                lo = b.get("lower", 30)
                hi = b.get("upper", 160)
            score += normalized_w * normalize_value(policy_metrics.chunk_size_p50, lo, hi, invert=False)
    
    # Spam (lower is better)
    if "spam_pct_mean" in active_metrics and policy_metrics.spam_pct_mean is not None:
        w = weights.get("spam_pct_mean", 0.0)
        if w > 0:
            normalized_w = w / active_weights_sum
            if computed_bounds and "spam_pct_mean" in computed_bounds:
                b = computed_bounds["spam_pct_mean"]
                lo = b.get("lo", 0)
                hi = b.get("hi", 10)
            else:
                b = bounds.get("spam_pct_mean", {"lower": 0, "upper": 10})
                lo = b.get("lower", 0)
                hi = b.get("upper", 10)
            score += normalized_w * normalize_value(policy_metrics.spam_pct_mean, lo, hi, invert=True)
    
    # Worst wait (lower is better)
    if "worst_wait_p95_ms" in active_metrics and policy_metrics.worst_wait_p95_ms is not None:
        w = weights.get("worst_wait_p95_ms", 0.0)
        if w > 0:
            normalized_w = w / active_weights_sum
            if computed_bounds and "worst_wait_p95_ms" in computed_bounds:
                b = computed_bounds["worst_wait_p95_ms"]
                lo = b.get("lo", 10000)
                hi = b.get("hi", 60000)
            else:
                b = bounds.get("worst_wait_p95_ms", {"lower": 10000, "upper": 60000})
                lo = b.get("lower", 10000)
                hi = b.get("upper", 60000)
            score += normalized_w * normalize_value(policy_metrics.worst_wait_p95_ms, lo, hi, invert=True)
    
    return score


def build_pareto_frontier_3d(
    points: List[Tuple[PolicyMetrics, List[float]]],
    active_dimensions: List[int]
) -> List[Tuple[PolicyMetrics, List[float]]]:
    """
    Build k-dimensional Pareto frontier (non-dominated points).
    
    Excludes dropped metrics from dominance test by using only active dimensions.
    
    A point a dominates point b if:
    - For all active dimensions i: a[i] >= b[i] (all objectives better or equal)
    - AND exists dimension j where a[j] > b[j] (strictly better in at least one)
    
    Args:
        points: List of (policy_metrics, goodness_vector) tuples
        active_dimensions: List of dimension indices to use (e.g., [0, 1, 2] if all active, [0, 2] if dimension 1 dropped)
        
    Returns:
        List of non-dominated points (Pareto frontier)
    """
    if not points:
        return []
    
    if not active_dimensions:
        # No active dimensions - return all points (degenerate case)
        return points
    
    frontier = []
    for i, (pm_i, vec_i) in enumerate(points):
        is_dominated = False
        
        # Extract active dimensions for point i
        active_i = [vec_i[d] for d in active_dimensions]
        
        for j, (pm_j, vec_j) in enumerate(points):
            if i == j:
                continue
            
            # Extract active dimensions for point j
            active_j = [vec_j[d] for d in active_dimensions]
            
            # Check if point j dominates point i
            # j dominates i if: all active_j[k] >= active_i[k] AND exists k where active_j[k] > active_i[k]
            all_better_or_equal = all(active_j[k] >= active_i[k] for k in range(len(active_dimensions)))
            strictly_better = any(active_j[k] > active_i[k] for k in range(len(active_dimensions)))
            
            if all_better_or_equal and strictly_better:
                is_dominated = True
                break
        
        if not is_dominated:
            frontier.append((pm_i, vec_i))
    
    return frontier


def select_pareto_utopia(
    survivor_metrics: List[PolicyMetrics],
    computed_bounds: dict,
    dropped_metrics: List[str],
    config: dict
) -> Tuple[PolicyMetrics, str, dict]:
    """
    Select best policy using k-dimensional Pareto frontier + utopia-distance selection.
    
    Process:
    1. Extract 3D objective values (TTFF, flush_count, chunk_size) from survivors
    2. Normalize to goodness space using computed bounds
    3. Exclude dropped metrics from both Pareto dominance AND distance computation
    4. Build k-dimensional Pareto frontier (k = number of active dimensions)
    5. Compute dimension-normalized Euclidean distance to utopia point (1, 1, ..., 1)
    6. Select point with minimum distance
    7. Use deterministic tie-breaking if distances are equal
    
    Args:
        survivor_metrics: List of PolicyMetrics that passed constraints
        computed_bounds: Dict mapping metric names to bounds metadata
        dropped_metrics: List of metric names that were dropped (degenerate bounds)
        config: Configuration dict (for weighted score fallback)
        
    Returns:
        Tuple of (selected_policy, selection_method, metadata_dict)
    """
    import math
    
    if not survivor_metrics:
        raise ValueError("No survivor policies to select from")
    
    # Define metric order: [TTFF, flush_count, chunk_size]
    metric_order = ["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]
    metric_lower_is_better = [True, True, False]  # TTFF and flush_count: lower is better; chunk_size: higher is better
    
    # Determine active dimensions (exclude dropped metrics)
    active_dimensions = []
    dimension_to_metric = {}
    for i, metric_name in enumerate(metric_order):
        if metric_name not in dropped_metrics:
            active_dimensions.append(i)
            dimension_to_metric[i] = metric_name
    
    if not active_dimensions:
        # All metrics dropped - use lexicographic fallback
        print("DEBUG: All metrics dropped, using lexicographic fallback")
        # Deterministic tie-breaking order: lower flush_count, higher chunk_size, lower TTFF, smallest policy_id
        sorted_policies = sorted(
            survivor_metrics,
            key=lambda pm: (
                pm.flush_count_mean if pm.flush_count_mean is not None else float('inf'),
                -(pm.chunk_size_p50 if pm.chunk_size_p50 is not None else -float('inf')),
                pm.ttff_content_p50_ms if pm.ttff_content_p50_ms is not None else float('inf'),
                pm.policy_id
            )
        )
        return sorted_policies[0], "lexicographic_fallback", {"dropped_metrics": dropped_metrics}
    
    # Extract and normalize to goodness space
    points = []
    for pm in survivor_metrics:
        goodness_vector = []
        for i, metric_name in enumerate(metric_order):
            if i in active_dimensions:
                value = getattr(pm, metric_name, None)
                if value is None:
                    # Skip policies with missing values
                    break
                
                bounds = computed_bounds[metric_name]
                if metric_name == "ttff_content_p50_ms":
                    lo = bounds["lo_ms"]
                    hi = bounds["hi_ms"]
                else:
                    lo = bounds["lo"]
                    hi = bounds["hi"]
                
                goodness = normalize_to_goodness(value, lo, hi, metric_lower_is_better[i])
                goodness_vector.append(goodness)
            else:
                # Dropped metric - don't include in vector
                pass
        
        if len(goodness_vector) == len(active_dimensions):
            points.append((pm, goodness_vector))
    
    if not points:
        # No valid points - fallback to weighted score
        print("DEBUG: No valid points with all required metrics, falling back to weighted_score")
        for pm in survivor_metrics:
            pm.weighted_score = compute_weighted_score(pm, config, computed_bounds, dropped_metrics)
        best_policy = max(survivor_metrics, key=lambda pm: pm.weighted_score if pm.weighted_score is not None else -1)
        return best_policy, "weighted_fallback", {"dropped_metrics": dropped_metrics}
    
    # After compressing vectors to active dimensions, remap active_dimensions to [0, 1, ..., k-1]
    # This fixes the index mismatch bug when metrics are dropped
    k = len(points[0][1]) if points else 0
    compressed_active_dimensions = list(range(k))
    
    # Build k-dimensional Pareto frontier
    frontier = build_pareto_frontier_3d(points, compressed_active_dimensions)
    
    print(f"DEBUG: Pareto frontier: {len(frontier)} non-dominated points (filtered from {len(points)} total)")
    print(f"DEBUG: Active dimensions: {len(active_dimensions)} (excluded dropped metrics: {dropped_metrics})")
    
    if not frontier:
        # Empty frontier - fallback to weighted score
        print("DEBUG: Empty Pareto frontier, falling back to weighted_score")
        for pm in survivor_metrics:
            pm.weighted_score = compute_weighted_score(pm, config, computed_bounds, dropped_metrics)
        best_policy = max(survivor_metrics, key=lambda pm: pm.weighted_score if pm.weighted_score is not None else -1)
        return best_policy, "weighted_fallback", {"dropped_metrics": dropped_metrics}
    
    # Compute dimension-normalized Euclidean distance to utopia
    k = len(active_dimensions)
    utopia = [1.0] * k  # Utopia point: all goodness values = 1.0
    
    distances = []
    for pm, goodness_vec in frontier:
        # Dimension-normalized distance: sqrt(mean((1 - goodness_i)^2))
        squared_diffs = [(1.0 - goodness_vec[i]) ** 2 for i in range(k)]
        distance = math.sqrt(sum(squared_diffs) / k)
        distances.append((pm, goodness_vec, distance))
    
    # Find minimum distance
    min_distance = min(dist for _, _, dist in distances)
    
    # Collect candidates with minimum distance (within tolerance)
    tolerance = 1e-10
    candidates = [(pm, vec, dist) for pm, vec, dist in distances if abs(dist - min_distance) < tolerance]
    
    if len(candidates) == 1:
        selected_policy = candidates[0][0]
        print(f"DEBUG: Selected policy {selected_policy.policy_id} with utopia distance {min_distance:.6f}")
        return selected_policy, "pareto3_utopia", {"dropped_metrics": dropped_metrics, "utopia_distance": min_distance}
    
    # Tie-breaking: deterministic order
    print(f"DEBUG: {len(candidates)} candidates tied at distance {min_distance:.6f}, applying tie-breaking")
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (
            x[0].flush_count_mean if x[0].flush_count_mean is not None else float('inf'),
            -(x[0].chunk_size_p50 if x[0].chunk_size_p50 is not None else -float('inf')),
            x[0].ttff_content_p50_ms if x[0].ttff_content_p50_ms is not None else float('inf'),
            x[0].policy_id
        )
    )
    
    selected_policy = sorted_candidates[0][0]
    print(f"DEBUG: Selected policy {selected_policy.policy_id} after tie-breaking")
    return selected_policy, "pareto3_utopia", {"dropped_metrics": dropped_metrics, "utopia_distance": min_distance}


def compute_utopia_distances_for_all(
    survivor_metrics: List[PolicyMetrics],
    computed_bounds: dict,
    dropped_metrics: List[str]
) -> List[Tuple[PolicyMetrics, float]]:
    """
    Compute utopia distances for all survivor policies (not just frontier).
    
    This is used for ranking and reporting top policies by utopia distance.
    
    Args:
        survivor_metrics: List of all survivor policies
        computed_bounds: Computed normalization bounds
        dropped_metrics: List of dropped metric names
        
    Returns:
        List of (policy_metrics, utopia_distance) tuples, sorted by distance (ascending)
    """
    import math
    
    # Define metric order: [TTFF, flush_count, chunk_size]
    metric_order = ["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]
    metric_lower_is_better = [True, True, False]
    
    # Determine active dimensions
    active_dimensions = []
    for i, metric_name in enumerate(metric_order):
        if metric_name not in dropped_metrics:
            active_dimensions.append(i)
    
    if not active_dimensions:
        # All metrics dropped - return empty list
        return []
    
    k = len(active_dimensions)
    utopia = [1.0] * k
    
    distances = []
    for pm in survivor_metrics:
        goodness_vector = []
        for i, metric_name in enumerate(metric_order):
            if i in active_dimensions:
                value = getattr(pm, metric_name, None)
                if value is None:
                    break
                
                bounds = computed_bounds[metric_name]
                if metric_name == "ttff_content_p50_ms":
                    lo = bounds["lo_ms"]
                    hi = bounds["hi_ms"]
                else:
                    lo = bounds["lo"]
                    hi = bounds["hi"]
                
                goodness = normalize_to_goodness(value, lo, hi, metric_lower_is_better[i])
                goodness_vector.append(goodness)
        
        if len(goodness_vector) == k:
            # Compute dimension-normalized distance
            squared_diffs = [(1.0 - goodness_vector[i]) ** 2 for i in range(k)]
            distance = math.sqrt(sum(squared_diffs) / k)
            distances.append((pm, distance))
    
    # Sort by distance (ascending - lower is better)
    distances.sort(key=lambda x: x[1])
    return distances


# =============================================================================
# α Sensitivity Analysis
# =============================================================================

def compute_spam_sensitivity(
    case_metrics_list: List[CaseMetrics],
    policy: Policy,
    alpha_values: List[float]
) -> Dict[float, float]:
    """Recompute spam_pct for different α values."""
    results = {}
    
    for alpha in alpha_values:
        spam_pcts = []
        for case_metrics in case_metrics_list:
            # Recompute spam for this alpha
            regular_flushes = [e for e in case_metrics.flush_events if not e.is_end_of_trace]
            if regular_flushes:
                spam_threshold = alpha * policy.min_words
                spam_count = sum(1 for e in regular_flushes if e.chunk_words < spam_threshold)
                spam_pct = (spam_count / len(regular_flushes)) * 100.0
                spam_pcts.append(spam_pct)
        
        if spam_pcts:
            results[alpha] = sum(spam_pcts) / len(spam_pcts)
        else:
            results[alpha] = 0.0
    
    return results


# =============================================================================
# Output Generation
# =============================================================================

def write_calibration_results_csv(
    output_path: Path,
    policy_metrics_list: List[PolicyMetrics]
):
    """Write calibration results CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "policy_id", "min_words", "max_words", "silence_timer_ms", "max_wait_timeout_ms",
            "ttff_content_p50_ms", "ttff_content_p95_ms", "ttff_trace_p50_ms",
            "flush_count_mean", "chunk_size_p50", "chunk_size_p90", "chunk_size_p95", "chunk_size_max",
            "worst_wait_p95_ms", "worst_wait_max_ms",
            "spam_pct_mean", "timer_flush_pct_mean", "timer_under_min_pct_mean",
            "constraint_violations", "weighted_score"
        ])
        
        # Rows
        for pm in policy_metrics_list:
            violations_str = "; ".join(pm.constraint_violations) if pm.constraint_violations else ""
            writer.writerow([
                pm.policy_id, pm.min_words, pm.max_words, pm.silence_timer_ms, pm.max_wait_timeout_ms,
                pm.ttff_content_p50_ms, pm.ttff_content_p95_ms, pm.ttff_trace_p50_ms,
                pm.flush_count_mean, pm.chunk_size_p50, pm.chunk_size_p90, pm.chunk_size_p95, pm.chunk_size_max,
                pm.worst_wait_p95_ms, pm.worst_wait_max_ms,
                pm.spam_pct_mean, pm.timer_flush_pct_mean, pm.timer_under_min_pct_mean,
                violations_str, pm.weighted_score
            ])


def write_per_case_jsonl(
    output_path: Path,
    case_metrics_list: List[CaseMetrics]
):
    """Write per-case detailed results JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for cm in case_metrics_list:
            record = {
                "case_id": cm.case_id,
                "policy_id": cm.policy_id,
                "ttff_content_ms": cm.ttff_content_ms,
                "ttff_trace_ms": cm.ttff_trace_ms,
                "flush_count": cm.flush_count,
                "chunk_size_p50": cm.chunk_size_p50,
                "chunk_size_p90": cm.chunk_size_p90,
                "chunk_size_p95": cm.chunk_size_p95,
                "chunk_size_max": cm.chunk_size_max,
                "worst_wait_ms": cm.worst_wait_ms,
                "worst_wait_max_ms": cm.worst_wait_max_ms,
                "spam_pct": cm.spam_pct,
                "timer_flush_pct": cm.timer_flush_pct,
                "timer_under_min_pct": cm.timer_under_min_pct,
                "end_of_trace_flush_count": cm.end_of_trace_flush_count,
                "flush_events": [
                    {
                        "agent_id": e.agent_id,
                        "flush_time_ms": e.flush_time_ms,
                        "chunk_words": e.chunk_words,
                        "flush_reason": e.flush_reason,
                        "is_end_of_trace": e.is_end_of_trace
                    }
                    for e in cm.flush_events
                ]
            }
            f.write(json.dumps(record) + "\n")


def write_summary_json(
    output_path: Path,
    calibration_run_id: str,
    trace_dataset_hash: str,
    config_hash: str,
    exaid_commit: str,
    mas_run_id: str,
    valid_policies_count: int,
    invalid_policies_count: int,
    invalid_reasons: List[Tuple[Policy, str]],
    policy_metrics_list: List[PolicyMetrics],
    survivor_metrics: List[PolicyMetrics],
    selected_policy: PolicyMetrics,
    selection_method: str,
    config: dict,
    computed_bounds: Optional[dict] = None,
    dropped_metrics: Optional[List[str]] = None,
    selection_metadata: Optional[dict] = None
):
    """Write calibration summary JSON."""
    summary = {
        "calibration_run_id": calibration_run_id,
        "reproducibility": {
            "trace_dataset_hash": trace_dataset_hash,
            "trace_dataset_hash8": trace_dataset_hash[:8],
            "mas_run_id": mas_run_id,
            "exaid_commit": exaid_commit,
            "exaid_commit8": exaid_commit[:8],
            "calibration_config_hash": config_hash,
            "config_hash8": config_hash[:8]
        },
        "policy_validity": {
            "valid_policies_count": valid_policies_count,
            "invalid_policies_count": invalid_policies_count,
            "invalid_reasons": [
                {
                    "policy_id": p.policy_id,
                    "min_words": p.min_words,
                    "max_words": p.max_words,
                    "silence_timer_ms": p.silence_timer_ms,
                    "max_wait_timeout_ms": p.max_wait_timeout_ms,
                    "reason": reason
                }
                for p, reason in invalid_reasons[:10]  # Limit to first 10
            ]
        },
        "constraint_filtering": {
            "total_policies": len(policy_metrics_list),
            "survivor_count": len(survivor_metrics),
            "rejected_count": len(policy_metrics_list) - len(survivor_metrics)
        },
        "selection": {
            "method": selection_method,
            "selection_mode": selection_method,  # Alias for compatibility
            "selected_policy_id": selected_policy.policy_id if selected_policy else None,
            "selected_parameters": {
                "min_words": selected_policy.min_words if selected_policy else None,
                "max_words": selected_policy.max_words if selected_policy else None,
                "silence_timer_ms": selected_policy.silence_timer_ms if selected_policy else None,
                "max_wait_timeout_ms": selected_policy.max_wait_timeout_ms if selected_policy else None
            } if selected_policy else None
        },
        "weighted_scores": {
            pm.policy_id: pm.weighted_score
            for pm in survivor_metrics
            if pm.weighted_score is not None
        }
    }
    
    # Add normalization bounds (computed bounds if available, otherwise config bounds)
    if computed_bounds is not None:
        summary["normalization_bounds"] = computed_bounds
        summary["dropped_metrics"] = dropped_metrics if dropped_metrics else []
    else:
        summary["normalization_bounds"] = config.get("selection", {}).get("weighted_score", {}).get("normalization_bounds", {})
        summary["dropped_metrics"] = []
    
    # Add selection metadata if available
    if selection_metadata:
        summary["selection"]["metadata"] = selection_metadata
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_chosen_params_yaml(
    output_path: Path,
    selected_policy: PolicyMetrics
):
    """Write chosen TokenGate parameters YAML."""
    params = {
        "token_gate": {
            "min_words": selected_policy.min_words,
            "max_words": selected_policy.max_words,
            "boundary_cues": ".?!\n",  # Fixed
            "silence_timer": selected_policy.silence_timer_ms / 1000.0,  # Convert to seconds
            "max_wait_timeout": selected_policy.max_wait_timeout_ms / 1000.0  # Convert to seconds
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)


def write_calibration_report_md(
    output_path: Path,
    calibration_run_id: str,
    summary: dict,
    selected_policy: PolicyMetrics,
    selection_method: str,
    top_policies: List[PolicyMetrics],
    spam_sensitivity: Dict[str, Dict[float, float]],
    config: dict,
    top_5_by_utopia: Optional[List[PolicyMetrics]] = None,
    utopia_rankings: Optional[Dict[str, Tuple[int, float]]] = None
):
    """Write calibration report Markdown."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# TokenGate Calibration Report\n\n")
        f.write(f"**Calibration Run ID:** `{calibration_run_id}`\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"**Selected Policy:** `{selected_policy.policy_id}`\n\n")
        f.write(f"**Selection Method:** {selection_method}\n\n")
        f.write("**Selected Parameters:**\n")
        f.write(f"- `min_words`: {selected_policy.min_words}\n")
        f.write(f"- `max_words`: {selected_policy.max_words}\n")
        f.write(f"- `silence_timer_ms`: {selected_policy.silence_timer_ms}\n")
        f.write(f"- `max_wait_timeout_ms`: {selected_policy.max_wait_timeout_ms}\n\n")
        
        f.write("## Literature-Informed Grid Justification\n\n")
        lit_sources = config.get("literature_sources", {})
        for param, info in lit_sources.items():
            f.write(f"**{param}:** {info.get('note', 'N/A')}\n\n")
        
        f.write("## Policy Validity Filter Results\n\n")
        validity = summary.get("policy_validity", {})
        f.write(f"- **Valid policies:** {validity.get('valid_policies_count', 0)}\n")
        f.write(f"- **Invalid policies:** {validity.get('invalid_policies_count', 0)}\n\n")
        
        f.write("## Constraint Filter Results\n\n")
        constraint_filter = summary.get("constraint_filtering", {})
        f.write(f"- **Total policies evaluated:** {constraint_filter.get('total_policies', 0)}\n")
        f.write(f"- **Survivor policies:** {constraint_filter.get('survivor_count', 0)}\n")
        f.write(f"- **Rejected policies:** {constraint_filter.get('rejected_count', 0)}\n\n")
        
        f.write("## Selection Rule Explanation\n\n")
        f.write(f"**Method Used:** {selection_method}\n\n")
        if selection_method == "pareto3_utopia":
            f.write("3-objective Pareto frontier analysis with utopia-distance selection was used. ")
            f.write("Objectives: minimize TTFF (Time To First Flush), minimize flush count (BufferAgent calls), maximize chunk size. ")
            f.write("First, non-dominated points (Pareto frontier) were identified in the 3D objective space. ")
            f.write("Then, the policy with minimum dimension-normalized Euclidean distance to the utopia point (1, 1, 1) in goodness space was selected. ")
            f.write("Normalization bounds were computed from survivor policies using percentile-based methods (P05/P95, or min/max for small-N cases).\n\n")
        elif selection_method == "weighted_fallback":
            f.write("Weighted objective function was used as fallback (Pareto frontier was empty or invalid). ")
            f.write("Weights were renormalized to account for any dropped metrics.\n\n")
        elif selection_method == "lexicographic_fallback":
            f.write("Lexicographic tie-breaking was used as fallback (all metrics were dropped due to insufficient variance). ")
            f.write("Selection order: lower flush_count_mean, higher chunk_size_p50, lower ttff_content_p50_ms, smallest policy_id.\n\n")
        else:
            f.write("Weighted objective function was used for ranking policies.\n\n")
        
        # Add normalization bounds section
        f.write("## Normalization Bounds\n\n")
        bounds = summary.get("normalization_bounds", {})
        dropped = summary.get("dropped_metrics", [])
        
        if bounds:
            f.write("Normalization bounds were computed from survivor policies:\n\n")
            for metric_name, metric_bounds in bounds.items():
                if metric_name == "ttff_content_p50_ms":
                    lo = metric_bounds.get("lo_ms", "N/A")
                    hi = metric_bounds.get("hi_ms", "N/A")
                    unit = "ms"
                else:
                    lo = metric_bounds.get("lo", "N/A")
                    hi = metric_bounds.get("hi", "N/A")
                    unit = ""
                method = metric_bounds.get("method", "unknown")
                is_dropped = metric_name in dropped
                status = "⚠️ DROPPED (insufficient variance)" if is_dropped else "✅ Active"
                f.write(f"- **{metric_name}**: [{lo}{unit}, {hi}{unit}] (method: {method}) {status}\n")
            f.write("\n")
            
            if dropped:
                f.write(f"**Dropped Metrics:** {', '.join(dropped)} - These metrics had insufficient variance (hi - lo < epsilon) and were excluded from Pareto dominance and distance computation.\n\n")
        else:
            f.write("Normalization bounds were not computed (using config defaults).\n\n")
        
        f.write("## Top 5 Policies\n\n")
        f.write("| Policy ID | min_words | max_words | silence_timer_ms | max_wait_timeout_ms | TTFF (p50) | Chunk Size (p50) | Weighted Score |\n")
        f.write("|-----------|-----------|-----------|-------------------|---------------------|------------|------------------|----------------|\n")
        for pm in top_policies[:5]:
            ttff = f"{pm.ttff_content_p50_ms:.1f}" if pm.ttff_content_p50_ms else "N/A"
            chunk = f"{pm.chunk_size_p50:.1f}" if pm.chunk_size_p50 else "N/A"
            score = f"{pm.weighted_score:.4f}" if pm.weighted_score else "N/A"
            f.write(f"| {pm.policy_id} | {pm.min_words} | {pm.max_words} | {pm.silence_timer_ms} | {pm.max_wait_timeout_ms} | {ttff} | {chunk} | {score} |\n")
        
        f.write("\n## Spam Sensitivity Analysis\n\n")
        f.write("Spam metrics recomputed for different α values. Includes:\n")
        f.write("- Selected policy\n")
        if top_5_by_utopia:
            f.write("- Top 5 policies by utopia distance\n")
        f.write("- Top 5 policies by weighted score\n\n")
        
        f.write("| Policy ID | Rank (Utopia) | Utopia Dist | Rank (Weighted) | α=0.5 | α=0.6 | α=0.7 | α=0.8 |\n")
        f.write("|-----------|---------------|-------------|-----------------|-------|-------|-------|-------|\n")
        
        # Sort policies: selected first, then by utopia rank, then by weighted rank
        def get_sort_key(policy_id: str) -> Tuple[int, int, int, str]:
            utopia_rank = utopia_rankings.get(policy_id, (999, float('inf')))[0] if utopia_rankings else 999
            weighted_rank = next((i + 1 for i, pm in enumerate(top_policies) if pm.policy_id == policy_id), 999)
            is_selected = 0 if policy_id == selected_policy.policy_id else 1
            return (is_selected, utopia_rank, weighted_rank, policy_id)
        
        sorted_policy_ids = sorted(spam_sensitivity.keys(), key=get_sort_key)
        
        for policy_id in sorted_policy_ids:
            alpha_results = spam_sensitivity[policy_id]
            
            # Get rankings
            utopia_info = utopia_rankings.get(policy_id, (None, None)) if utopia_rankings else (None, None)
            utopia_rank = f"{utopia_info[0]}" if utopia_info[0] is not None else "-"
            utopia_dist = f"{utopia_info[1]:.4f}" if utopia_info[1] is not None else "-"
            
            weighted_rank = next((i + 1 for i, pm in enumerate(top_policies) if pm.policy_id == policy_id), None)
            weighted_rank_str = f"{weighted_rank}" if weighted_rank is not None else "-"
            
            # Mark selected policy
            policy_display = policy_id
            if policy_id == selected_policy.policy_id:
                policy_display = f"**{policy_id}** (selected)"
            
            row = f"| {policy_display} | {utopia_rank} | {utopia_dist} | {weighted_rank_str} |"
            for alpha in [0.5, 0.6, 0.7, 0.8]:
                spam_pct = alpha_results.get(alpha, 0.0)
                row += f" {spam_pct:.2f}% |"
            f.write(row + "\n")
        
        f.write("\n## Reproducibility\n\n")
        repro = summary.get("reproducibility", {})
        f.write(f"- **Trace Dataset Hash:** `{repro.get('trace_dataset_hash', 'N/A')}`\n")
        f.write(f"- **MAS Run ID:** `{repro.get('mas_run_id', 'N/A')}`\n")
        f.write(f"- **EXAID Commit:** `{repro.get('exaid_commit', 'N/A')}`\n")
        f.write(f"- **Config Hash:** `{repro.get('calibration_config_hash', 'N/A')}`\n\n")


# =============================================================================
# Main Execution
# =============================================================================

async def main_async():
    parser = argparse.ArgumentParser(
        description="EXAID TokenGate Calibration"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Input traces directory"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Manifest file pattern (e.g., data/manifests/exaid_traces_*.manifest.jsonl)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Calibration sweep configuration file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="Allow stub traces (for testing only)"
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Verify determinism by running same policy twice"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Find manifest file
    manifest_pattern = Path(args.manifest)
    if "*" in str(manifest_pattern):
        # Glob pattern
        import glob
        manifest_files = glob.glob(str(manifest_pattern))
        if not manifest_files:
            raise FileNotFoundError(f"No manifest files found matching: {args.manifest}")
        manifest_path = Path(manifest_files[0])
    else:
        manifest_path = manifest_pattern
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    # Load manifest and extract case list
    trace_entries = []
    mas_run_id = "unknown"
    with open(manifest_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_type = record.get("record_type")
            # Check both manifest_meta and provenance records for mas_run_id
            if record_type in ("manifest_meta", "provenance"):
                if mas_run_id == "unknown" and "mas_run_id" in record:
                    mas_run_id = record.get("mas_run_id", "unknown")
            elif record_type == "trace_entry":
                trace_entries.append(record)
    
    if not trace_entries:
        raise ValueError(f"No trace entries found in manifest: {manifest_path}")
    
    # Compute reproducibility hashes
    trace_dataset_hash = compute_trace_dataset_hash(manifest_path)
    config_hash = compute_config_hash(config)
    exaid_commit = get_exaid_commit()
    calibration_run_id = generate_calibration_run_id(trace_dataset_hash, config_hash, exaid_commit)
    
    print(f"Calibration Run ID: {calibration_run_id}")
    print(f"Trace Dataset Hash: {trace_dataset_hash[:8]}")
    print(f"Config Hash: {config_hash[:8]}")
    print(f"EXAID Commit: {exaid_commit[:8]}")
    print()
    
    # Generate policy grid
    print("Generating policy grid...")
    all_policies = generate_policy_grid(config)
    print(f"Total combinations: {len(all_policies)}")
    
    # Filter valid policies
    print("Filtering valid policies...")
    valid_policies, invalid_policies = filter_valid_policies(all_policies, config)
    print(f"Valid policies: {len(valid_policies)}")
    print(f"Invalid policies: {len(invalid_policies)}")
    
    # Create output directory
    output_dir = args.output / calibration_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration copy
    with open(output_dir / "calibration_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Verify trace hashes once before calibration (matching make_traces.py: hash uncompressed content)
    if config.get("safety", {}).get("verify_trace_hashes", True):
        print("\nVerifying trace file integrity...")
        for trace_entry in trace_entries:
            case_id = trace_entry["case_id"]
            trace_file = args.traces / trace_entry["file"]
            
            if not trace_file.exists():
                print(f"WARNING: Trace file not found: {trace_file}")
                continue
            
            # Hash uncompressed content (matching make_traces.py behavior)
            try:
                with gzip.open(trace_file, "rt", encoding="utf-8") as tf:
                    content = tf.read()
                content_bytes = content.encode("utf-8")
                actual_hash = hashlib.sha256(content_bytes).hexdigest()
                expected_hash = trace_entry.get("sha256", "").replace("sha256:", "")
                if expected_hash and actual_hash != expected_hash:
                    print(f"WARNING: Trace hash mismatch for {case_id}: expected {expected_hash[:8]}, got {actual_hash[:8]}")
                    print(f"  This may indicate the trace file was modified after manifest creation.")
                    print(f"  Continuing calibration, but results may not be reproducible.")
            except Exception as e:
                print(f"WARNING: Failed to verify hash for {case_id}: {e}")
    
    # Run calibration for each valid policy
    print(f"\nRunning calibration for {len(valid_policies)} policies across {len(trace_entries)} cases...")
    
    all_case_metrics: List[CaseMetrics] = []
    policy_metrics_list: List[PolicyMetrics] = []
    
    for policy_idx, policy in enumerate(valid_policies):
        if (policy_idx + 1) % 10 == 0:
            print(f"  Progress: {policy_idx + 1}/{len(valid_policies)} policies")
        
        case_metrics_for_policy: List[CaseMetrics] = []
        
        for trace_entry in trace_entries:
            case_id = trace_entry["case_id"]
            trace_file = args.traces / trace_entry["file"]
            
            if not trace_file.exists():
                print(f"WARNING: Trace file not found: {trace_file}")
                continue
            
            # Load trace and replay
            try:
                engine = TraceReplayEngine(
                    trace_file,
                    strict_stub_guard=not args.allow_stub,
                    strict_validation=config.get("safety", {}).get("strict_validation", True)
                )
                trace_meta = engine.get_metadata()
                
                case_metrics = await replay_case_with_policy(
                    trace_file,
                    policy,
                    case_id,
                    trace_meta,
                    engine=engine,  # Pass the already-created engine to avoid duplicate I/O
                    strict_stub_guard=not args.allow_stub,
                    strict_validation=config.get("safety", {}).get("strict_validation", True)
                )
                case_metrics_for_policy.append(case_metrics)
                all_case_metrics.append(case_metrics)
                
            except (StubTraceError, TraceValidationError) as e:
                print(f"ERROR: Failed to replay {case_id}: {e}")
                if config.get("safety", {}).get("strict_validation", True):
                    raise
        
        # Aggregate metrics for this policy
        policy_metrics = aggregate_policy_metrics(policy, case_metrics_for_policy)
        
        # Check constraints
        violations = check_constraints(policy_metrics, config)
        policy_metrics.constraint_violations = violations
        
        policy_metrics_list.append(policy_metrics)
    
    print(f"\nCalibration complete. Evaluated {len(policy_metrics_list)} policies.")
    
    # Filter survivors (policies that pass constraints)
    survivor_metrics = [pm for pm in policy_metrics_list if not pm.constraint_violations]
    print(f"Survivor policies: {len(survivor_metrics)}")
    
    # Diagnostic: Show constraint violation statistics
    violation_counts = defaultdict(int)
    if len(survivor_metrics) == 0:
        print("\n⚠️  WARNING: No policies passed all constraints!")
        print("\nConstraint violation statistics:")
        for pm in policy_metrics_list:
            for violation in pm.constraint_violations:
                # Extract constraint name from violation message
                constraint_name = violation.split("(")[0].strip()
                violation_counts[constraint_name] += 1
        
        for constraint, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
            pct = (count / len(policy_metrics_list)) * 100
            print(f"  {constraint}: {count}/{len(policy_metrics_list)} policies ({pct:.1f}%)")
        
        print("\nConsider:")
        print("  1. Relaxing constraint thresholds in config")
        print("  2. Checking if metrics are computed correctly")
        print("  3. Reviewing a sample of failed policies in the CSV output")
    
    # Write CSV output early (even if no survivors, for analysis)
    print("\nWriting output artifacts...")
    write_calibration_results_csv(output_dir / "calibration_results.csv", policy_metrics_list)
    
    if len(survivor_metrics) == 0:
        print("\n❌ Cannot select best policy: no survivors.")
        print("CSV output written for analysis. Review constraint violations and adjust thresholds.")
        # Write summary JSON with constraint stats (using dummy selected policy for structure)
        dummy_policy = policy_metrics_list[0] if policy_metrics_list else None
        write_summary_json(
            output_dir / "calibration_summary.json",
            calibration_run_id,
            trace_dataset_hash,
            config_hash,
            exaid_commit,
            mas_run_id,
            len(valid_policies),
            len(invalid_policies),
            invalid_policies[:10],
            policy_metrics_list,
            [],  # No survivors
            dummy_policy,  # Dummy for structure (will be ignored if None)
            "none",  # No selection method
            config,
            None,  # No computed bounds
            None,  # No dropped metrics
            None   # No selection metadata
        )
        # Write per-case JSONL even if no survivors
        write_per_case_jsonl(output_dir / "calibration_per_case.jsonl", all_case_metrics)
        return
    
    # Pass B: Compute normalization bounds from survivors
    print("\nComputing normalization bounds from survivor policies...")
    computed_bounds, dropped_metrics = compute_all_normalization_bounds(survivor_metrics)
    
    if dropped_metrics:
        print(f"⚠️  Dropped metrics (insufficient variance): {dropped_metrics}")
    else:
        print("✅ All metrics have sufficient variance")
    
    # Compute weighted scores for survivors (using computed bounds)
    for pm in survivor_metrics:
        pm.weighted_score = compute_weighted_score(pm, config, computed_bounds, dropped_metrics)
    
    # Select best policy using 3-objective Pareto + utopia distance
    selected_policy, selection_method, selection_metadata = select_pareto_utopia(
        survivor_metrics, computed_bounds, dropped_metrics, config
    )
    print(f"\nSelected policy: {selected_policy.policy_id} (method: {selection_method})")
    
    # Compute utopia distances for all survivors (for ranking)
    print("\nComputing utopia distances for all survivors...")
    utopia_distances = compute_utopia_distances_for_all(
        survivor_metrics, computed_bounds, dropped_metrics
    )
    
    # Get top 5 by utopia distance
    top_5_by_utopia = [pm for pm, _ in utopia_distances[:5]] if utopia_distances else []
    
    # Get top 5 by weighted score
    top_5_by_weighted = sorted(
        survivor_metrics,
        key=lambda pm: pm.weighted_score if pm.weighted_score is not None else -1,
        reverse=True
    )[:5]
    
    # Collect policies for spam sensitivity analysis
    policies_for_spam_analysis = set()
    policies_for_spam_analysis.add(selected_policy.policy_id)  # Selected policy
    for pm in top_5_by_utopia:
        policies_for_spam_analysis.add(pm.policy_id)
    for pm in top_5_by_weighted:
        policies_for_spam_analysis.add(pm.policy_id)
    
    # α Sensitivity Analysis
    print("\nComputing α sensitivity analysis...")
    spam_sensitivity = {}
    alpha_values = config.get("spam", {}).get("alpha_sensitivity", [0.5, 0.6, 0.7, 0.8])
    
    # Compute spam sensitivity for selected policy, top 5 by utopia, and top 5 by weighted score
    for policy_metrics in survivor_metrics:
        if policy_metrics.policy_id in policies_for_spam_analysis:
            policy = Policy(
                policy_id=policy_metrics.policy_id,
                min_words=policy_metrics.min_words,
                max_words=policy_metrics.max_words,
                silence_timer_ms=policy_metrics.silence_timer_ms,
                max_wait_timeout_ms=policy_metrics.max_wait_timeout_ms
            )
            case_metrics_for_policy = [cm for cm in all_case_metrics if cm.policy_id == policy.policy_id]
            spam_sensitivity[policy.policy_id] = compute_spam_sensitivity(
                case_metrics_for_policy,
                policy,
                alpha_values
            )
    
    # Write remaining outputs (CSV already written earlier)
    print("\nWriting remaining output artifacts...")
    
    write_per_case_jsonl(output_dir / "calibration_per_case.jsonl", all_case_metrics)
    
    summary_data = {
        "reproducibility": {
            "trace_dataset_hash": trace_dataset_hash,
            "mas_run_id": mas_run_id,
            "exaid_commit": exaid_commit,
            "calibration_config_hash": config_hash
        },
        "policy_validity": {
            "valid_policies_count": len(valid_policies),
            "invalid_policies_count": len(invalid_policies),
            "invalid_reasons": invalid_policies[:10]
        },
        "constraint_filtering": {
            "total_policies": len(policy_metrics_list),
            "survivor_count": len(survivor_metrics),
            "rejected_count": len(policy_metrics_list) - len(survivor_metrics)
        },
        "normalization_bounds": computed_bounds if computed_bounds else {},
        "dropped_metrics": dropped_metrics if dropped_metrics else []
    }
    
    write_summary_json(
        output_dir / "calibration_summary.json",
        calibration_run_id,
        trace_dataset_hash,
        config_hash,
        exaid_commit,
        mas_run_id,
        len(valid_policies),
        len(invalid_policies),
        invalid_policies,
        policy_metrics_list,
        survivor_metrics,
        selected_policy,
        selection_method,
        config,
        computed_bounds,
        dropped_metrics,
        selection_metadata
    )
    
    write_chosen_params_yaml(output_dir / "chosen_tokengate_params.yaml", selected_policy)
    
    # Sort survivors by weighted score for report
    top_policies = sorted(survivor_metrics, key=lambda pm: pm.weighted_score or -1, reverse=True)
    
    # Prepare utopia distance rankings for report
    utopia_rankings = {pm.policy_id: (idx + 1, dist) for idx, (pm, dist) in enumerate(utopia_distances)}
    
    write_calibration_report_md(
        output_dir / "calibration_report.md",
        calibration_run_id,
        summary_data,
        selected_policy,
        selection_method,
        top_policies,
        spam_sensitivity,
        config,
        top_5_by_utopia,
        utopia_rankings
    )
    
    # Write spam sensitivity JSON
    with open(output_dir / "spam_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(spam_sensitivity, f, indent=2)
    
    print(f"\nCalibration complete. Output directory: {output_dir}")
    print(f"Selected policy: {selected_policy.policy_id}")
    print(f"  min_words: {selected_policy.min_words}")
    print(f"  max_words: {selected_policy.max_words}")
    print(f"  silence_timer_ms: {selected_policy.silence_timer_ms}")
    print(f"  max_wait_timeout_ms: {selected_policy.max_wait_timeout_ms}")


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

