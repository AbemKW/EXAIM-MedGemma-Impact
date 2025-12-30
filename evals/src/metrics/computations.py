"""Core metric computations shared across evaluators."""

import math
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np

from trace_text import build_window_text

from .constants import BUFFER_COVERAGE_BUDGETS, REDUNDANCY_THRESHOLDS
from .extractor import ConceptExtractorWrapper
from .types import FaithfulnessResult


# ============================================================================
# Faithfulness Metrics (M6a, M6b)
# ============================================================================

def compute_faithfulness_for_event(
    event: dict,
    run_events: Dict[str, dict],
    trace_chunks: List[dict],
    extractor: ConceptExtractorWrapper,
) -> FaithfulnessResult:
    """
    Compute M6a/M6b for a single event with correct failure handling.

    Paper hook: "Schema failures excluded from denominators;
    empty-but-valid events return 0.0 (Section 5.1)"

    POLICY:
        - schema_ok=false → EXCLUDE (return None, not 0.0!)
        - schema_ok=true, empty CUIs → INCLUDE as 0.0
    """
    if not event.get("schema_ok", True):
        return FaithfulnessResult(
            m6a=None,
            m6b=None,
            excluded=True,
            exclusion_reason="schema_failure",
        )

    summary_text = event.get("summary_semantics_text", "")
    output_cuis = extractor.extract(summary_text)

    start_seq = event.get("start_seq", 0)
    end_seq = event.get("end_seq", 0)
    window_text = build_window_text(trace_chunks, start_seq, end_seq)
    window_cuis = extractor.extract(window_text)

    support_text = window_text
    if event.get("latest_summary_event_id"):
        latest_event = run_events.get(event["latest_summary_event_id"])
        if latest_event:
            support_text += " " + latest_event.get("summary_semantics_text", "")
    support_cuis = extractor.extract(support_text)

    if len(output_cuis) == 0:
        return FaithfulnessResult(
            m6a=0.0,
            m6b=0.0,
            excluded=False,
            exclusion_reason=None,
            output_cui_count=0,
            window_cui_count=len(window_cuis),
            support_cui_count=len(support_cuis),
        )

    unsupported_window = output_cuis - window_cuis
    m6a = len(unsupported_window) / len(output_cuis)

    unsupported_contract = output_cuis - support_cuis
    m6b = len(unsupported_contract) / len(output_cuis)

    return FaithfulnessResult(
        m6a=m6a,
        m6b=m6b,
        excluded=False,
        exclusion_reason=None,
        output_cui_count=len(output_cuis),
        window_cui_count=len(window_cuis),
        support_cui_count=len(support_cuis),
    )


# ============================================================================
# Redundancy Metric (M3)
# ============================================================================

def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union


def compute_redundancy(
    summary_events: List[dict],
    extractor: ConceptExtractorWrapper,
) -> List[float]:
    """Compute redundancy (Jaccard) between consecutive summaries."""
    if len(summary_events) < 2:
        return []

    cui_sets = []
    for event in summary_events:
        if not event.get("schema_ok", True):
            cuis = set()
        else:
            text = event.get("summary_semantics_text", "")
            cuis = extractor.extract(text)
        cui_sets.append(cuis)

    jaccards = []
    for i in range(1, len(cui_sets)):
        j = compute_jaccard_similarity(cui_sets[i - 1], cui_sets[i])
        jaccards.append(j)

    return jaccards


def compute_redundancy_threshold_rates(
    jaccards: List[float],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Optional[float]]:
    """Compute redundancy rates at each Jaccard threshold."""
    resolved_thresholds = thresholds if thresholds is not None else REDUNDANCY_THRESHOLDS
    if not jaccards:
        return {f"tau_{threshold:.2f}": None for threshold in resolved_thresholds}
    total = len(jaccards)
    return {
        f"tau_{threshold:.2f}": float(sum(1 for j in jaccards if j >= threshold) / total)
        for threshold in resolved_thresholds
    }


# ============================================================================
# Coverage Metrics (M4, M5)
# ============================================================================

def compute_coverage_metrics(
    summary_events: List[dict],
    trace_cuis: Set[str],
    extractor: ConceptExtractorWrapper,
) -> dict:
    """
    Compute coverage metrics with schema failure penalty.

    Paper hook: "Schema-failed events contribute empty concept sets,
    penalizing recall (Section 5.2)"
    """
    all_summary_cuis = set()
    per_summary_unsupported = []

    for event in summary_events:
        if not event.get("schema_ok", True):
            continue

        summary_text = event.get("summary_semantics_text", "")
        event_cuis = extractor.extract(summary_text)
        all_summary_cuis.update(event_cuis)

        if len(event_cuis) > 0:
            unsupported = event_cuis - trace_cuis
            unsupported_rate = len(unsupported) / len(event_cuis)
            per_summary_unsupported.append(unsupported_rate)

    if len(trace_cuis) == 0:
        coverage = 0.0
    else:
        covered = all_summary_cuis & trace_cuis
        coverage = len(covered) / len(trace_cuis)

    if len(all_summary_cuis) == 0:
        unsupported_global = 0.0
    else:
        unsupported = all_summary_cuis - trace_cuis
        unsupported_global = len(unsupported) / len(all_summary_cuis)

    unsupported_per_summary_mean = (
        float(np.mean(per_summary_unsupported)) if per_summary_unsupported else None
    )

    return {
        "trace_coverage": coverage,
        "unsupported_global_rate": unsupported_global,
        "unsupported_per_summary_mean": unsupported_per_summary_mean,
        "trace_cui_count": len(trace_cuis),
        "summary_cui_count": len(all_summary_cuis),
    }


def compute_coverage_by_budget(
    summary_events: List[dict],
    trace_cuis: Set[str],
    extractor: ConceptExtractorWrapper,
    budgets: List[int],
) -> Dict[str, float]:
    """Compute coverage at increasing summary CTU budgets (M7b)."""
    if not budgets:
        return {f"ctu_{budget}": 0.0 for budget in BUFFER_COVERAGE_BUDGETS}
    sorted_budgets = sorted(budgets)
    coverage_by_budget: Dict[str, float] = {}
    cumulative_ctu = 0
    cumulative_cuis: Set[str] = set()
    event_index = 0

    for budget in sorted_budgets:
        while event_index < len(summary_events):
            event = summary_events[event_index]
            event_ctu = event.get(
                "summary_ctu",
                compute_ctu(event.get("summary_semantics_text", "")),
            )
            if cumulative_ctu + event_ctu > budget:
                break
            cumulative_ctu += event_ctu
            if event.get("schema_ok", True):
                summary_text = event.get("summary_semantics_text", "")
                cumulative_cuis.update(extractor.extract(summary_text))
            event_index += 1

        if len(trace_cuis) == 0:
            coverage = 0.0
        else:
            coverage = len(cumulative_cuis & trace_cuis) / len(trace_cuis)
        coverage_by_budget[f"ctu_{budget}"] = coverage

    return coverage_by_budget


# ============================================================================
# CTU Computation
# ============================================================================

def compute_ctu(text: str) -> int:
    """Compute Character-normalized Token Units."""
    return math.ceil(len(text) / 4) if text else 0


def compute_distribution(values: List[float]) -> Optional[Dict[str, float]]:
    """Compute distribution statistics for a list of values."""
    if not values:
        return None
    arr = np.array(values, dtype=float)
    if arr.size < 2:
        return {
            "count": int(arr.size),
            "min": float(arr[0]),
            "max": float(arr[0]),
            "mean": float(arr[0]),
            "p50": float(arr[0]),
            "p90": float(arr[0]),
            "p95": float(arr[0]),
            "p99": float(arr[0]),
        }
    percentile_kwargs = {}
    try:
        np.percentile(arr, 50, method="linear")
        percentile_kwargs["method"] = "linear"
    except TypeError:
        percentile_kwargs["interpolation"] = "linear"
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50, **percentile_kwargs)),
        "p90": float(np.percentile(arr, 90, **percentile_kwargs)),
        "p95": float(np.percentile(arr, 95, **percentile_kwargs)),
        "p99": float(np.percentile(arr, 99, **percentile_kwargs)),
    }


def parse_timestamp_ms(timestamp: Optional[str]) -> Optional[int]:
    """Parse ISO-8601 timestamp to milliseconds since epoch."""
    if not timestamp:
        return None
    ts = timestamp
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def compute_flush_statistics(flushes: List[dict]) -> dict:
    """
    Compute flush statistics from tokengate_flush records.

    Expected flush record fields:
        - flush_index: monotonic index for ordering
        - timestamp: ISO-8601 timestamp for interval computation
        - accumulated_ctu: CTU since last flush
        - trigger_reason: reason for flush (e.g. timer, length)
    """
    stats: Dict[str, Any] = {
        "flush_count": len(flushes),
        "flush_reason_counts": {},
        "flush_accumulated_ctu_distribution": None,
        "flush_interval_ms_distribution": None,
        "flush_accumulated_ctu_values": [],
        "flush_interval_ms_values": [],
    }
    if not flushes:
        return stats

    for flush in flushes:
        reason = flush.get("trigger_reason", "unknown")
        stats["flush_reason_counts"][reason] = (
            stats["flush_reason_counts"].get(reason, 0) + 1
        )

    accumulated_ctu = [
        f.get("accumulated_ctu")
        for f in flushes
        if f.get("accumulated_ctu") is not None
    ]
    stats["flush_accumulated_ctu_distribution"] = compute_distribution(accumulated_ctu)
    stats["flush_accumulated_ctu_values"] = [float(value) for value in accumulated_ctu]

    intervals_ms = []
    prev_ts_ms = None
    sorted_flushes = sorted(flushes, key=lambda f: f.get("flush_index", 0))
    for flush in sorted_flushes:
        ts_ms = parse_timestamp_ms(flush.get("timestamp"))
        if ts_ms is None:
            continue
        if prev_ts_ms is not None:
            interval = ts_ms - prev_ts_ms
            if interval < 0:
                warnings.warn(
                    "Flush timestamps are not monotonic; skipping negative interval."
                )
                continue
            intervals_ms.append(interval)
        prev_ts_ms = ts_ms
    stats["flush_interval_ms_distribution"] = compute_distribution(intervals_ms)
    stats["flush_interval_ms_values"] = [float(value) for value in intervals_ms]
    return stats


def compute_virtual_time_throughput(
    trace_chunks: List[dict],
    trace_ctu: int,
    summary_ctu: int,
) -> dict:
    """Compute virtual-time throughput metrics from trace chunks."""
    t_rel_values = [
        c.get("t_rel_ms")
        for c in trace_chunks
        if c.get("t_rel_ms") is not None
    ]
    if not t_rel_values:
        return {
            "virtual_time_duration_ms": None,
            "trace_ctu_per_s": None,
            "summary_ctu_per_s": None,
        }
    duration_ms = max(t_rel_values) - min(t_rel_values)
    duration_s = duration_ms / 1000
    trace_ctu_per_s = None
    summary_ctu_per_s = None
    if duration_s > 0:
        trace_ctu_per_s = trace_ctu / duration_s
        summary_ctu_per_s = summary_ctu / duration_s
    return {
        "virtual_time_duration_ms": int(duration_ms),
        "trace_ctu_per_s": float(trace_ctu_per_s) if trace_ctu_per_s is not None else None,
        "summary_ctu_per_s": (
            float(summary_ctu_per_s) if summary_ctu_per_s is not None else None
        ),
    }


def compute_overhead_attribution(
    summary_latencies: List[float],
    buffer_latencies: List[float],
) -> dict:
    """
    Compute control-plane vs content-plane overhead attribution.

    summary_latencies -> content-plane (summarizer)
    buffer_latencies -> control-plane (buffer/decision)
    """
    content_plane = float(sum(summary_latencies))
    control_plane = float(sum(buffer_latencies))
    total = content_plane + control_plane
    return {
        "control_plane_latency_ms": control_plane,
        "content_plane_latency_ms": content_plane,
        "total_plane_latency_ms": total,
        "control_plane_latency_pct": (control_plane / total) if total > 0 else None,
        "content_plane_latency_pct": (content_plane / total) if total > 0 else None,
    }
