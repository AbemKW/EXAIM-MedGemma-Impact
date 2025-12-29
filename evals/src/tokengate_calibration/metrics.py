"""Metrics computation helpers for TokenGate calibration."""

from typing import Dict, List, Optional, Tuple

from .models import CaseMetrics, FlushEvent, Policy, PolicyMetrics


def nearest_rank_percentile(sorted_values: List[float], percentile: float) -> Optional[float]:
    """
    Return the nearest-rank percentile for a sorted list.

    This uses a discrete index (nearest-rank) rather than interpolation to match
    existing calibration semantics and avoid external dependencies.
    """
    if not sorted_values:
        return None
    if percentile <= 0.0:
        return sorted_values[0]
    if percentile >= 1.0:
        return sorted_values[-1]
    n = len(sorted_values)
    idx = int(n * percentile)
    idx = max(0, min(idx, n - 1))
    return sorted_values[idx]


def compute_case_metrics(
    case_id: str,
    policy_id: str,
    flush_events: List[FlushEvent],
    first_content_delta_time_ms: Optional[int],
    trace_t0_ms: int,
    policy: Policy,
    alpha: float = 0.7,
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
        flush_events=flush_events,
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
        metrics.chunk_size_p90 = nearest_rank_percentile(sorted_sizes, 0.9)
        metrics.chunk_size_p95 = nearest_rank_percentile(sorted_sizes, 0.95)
        metrics.chunk_size_max = max(chunk_sizes)

    # Worst wait times (time between flushes)
    if len(regular_flushes) > 1:
        wait_times = []
        for i in range(1, len(regular_flushes)):
            wait_ms = regular_flushes[i].flush_time_ms - regular_flushes[i - 1].flush_time_ms
            wait_times.append(wait_ms)

        if wait_times:
            sorted_waits = sorted(wait_times)
            metrics.worst_wait_ms = nearest_rank_percentile(sorted_waits, 0.95)
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


def aggregate_policy_metrics(policy: Policy, case_metrics_list: List[CaseMetrics]) -> PolicyMetrics:
    """Aggregate metrics across all cases for a policy."""

    policy_metrics = PolicyMetrics(
        policy_id=policy.policy_id,
        min_words=policy.min_words,
        max_words=policy.max_words,
        silence_timer_ms=policy.silence_timer_ms,
        max_wait_timeout_ms=policy.max_wait_timeout_ms,
    )

    if not case_metrics_list:
        return policy_metrics

    # Aggregate TTFF
    ttff_content_values = [
        m.ttff_content_ms for m in case_metrics_list if m.ttff_content_ms is not None
    ]
    if ttff_content_values:
        sorted_ttff = sorted(ttff_content_values)
        n = len(sorted_ttff)
        policy_metrics.ttff_content_p50_ms = sorted_ttff[n // 2]
        policy_metrics.ttff_content_p95_ms = nearest_rank_percentile(sorted_ttff, 0.95)

    ttff_trace_values = [
        m.ttff_trace_ms for m in case_metrics_list if m.ttff_trace_ms is not None
    ]
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
        policy_metrics.chunk_size_p90 = nearest_rank_percentile(sorted_p90s, 0.9)

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
        policy_metrics.worst_wait_p95_ms = nearest_rank_percentile(sorted_waits, 0.95)
        policy_metrics.worst_wait_max_ms = max(worst_waits)

    # Aggregate spam percentage
    spam_pcts = [m.spam_pct for m in case_metrics_list if m.spam_pct is not None]
    if spam_pcts:
        policy_metrics.spam_pct_mean = sum(spam_pcts) / len(spam_pcts)

    # Aggregate timer percentages
    timer_flush_pcts = [m.timer_flush_pct for m in case_metrics_list if m.timer_flush_pct is not None]
    if timer_flush_pcts:
        policy_metrics.timer_flush_pct_mean = sum(timer_flush_pcts) / len(timer_flush_pcts)

    timer_under_min_pcts = [
        m.timer_under_min_pct for m in case_metrics_list if m.timer_under_min_pct is not None
    ]
    if timer_under_min_pcts:
        policy_metrics.timer_under_min_pct_mean = sum(timer_under_min_pcts) / len(timer_under_min_pcts)

    return policy_metrics


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
    eps: float = 2.0,  # Default epsilon (relaxed to avoid floating-point rounding issues)
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
            "dropped": True,
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
        "dropped": dropped,
    }


def compute_all_normalization_bounds(
    survivor_metrics: List[PolicyMetrics],
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
    ttff_bounds = compute_percentile_bounds(survivor_metrics, "ttff_content_p50_ms", eps=EPS_MS)
    bounds["ttff_content_p50_ms"] = {
        "lo_ms": ttff_bounds["lo"],
        "hi_ms": ttff_bounds["hi"],
        "method": ttff_bounds["method"],
        "computed_over": ttff_bounds["computed_over"],
    }
    if ttff_bounds["dropped"]:
        dropped_metrics.append("ttff_content_p50_ms")

    # Flush count bounds
    flush_bounds = compute_percentile_bounds(survivor_metrics, "flush_count_mean", eps=EPS_COUNT)
    bounds["flush_count_mean"] = {
        "lo": flush_bounds["lo"],
        "hi": flush_bounds["hi"],
        "method": flush_bounds["method"],
        "computed_over": flush_bounds["computed_over"],
    }
    if flush_bounds["dropped"]:
        dropped_metrics.append("flush_count_mean")

    # Chunk size bounds
    chunk_bounds = compute_percentile_bounds(survivor_metrics, "chunk_size_p50", eps=EPS_WORDS)
    bounds["chunk_size_p50"] = {
        "lo": chunk_bounds["lo"],
        "hi": chunk_bounds["hi"],
        "method": chunk_bounds["method"],
        "computed_over": chunk_bounds["computed_over"],
    }
    if chunk_bounds["dropped"]:
        dropped_metrics.append("chunk_size_p50")

    return bounds, dropped_metrics


def check_constraints(policy_metrics: PolicyMetrics, config: dict) -> List[str]:
    """Check if policy violates any constraints. Returns list of violation reasons."""
    constraints = config.get("constraints", {})
    violations = []

    # ttff_content_p95_ms <= threshold
    threshold = constraints.get("ttff_content_p95_ms")
    if threshold is not None and policy_metrics.ttff_content_p95_ms is not None:
        if policy_metrics.ttff_content_p95_ms > threshold:
            violations.append(
                f"ttff_content_p95_ms ({policy_metrics.ttff_content_p95_ms:.1f}) > {threshold}"
            )

    # spam_pct_mean <= threshold
    threshold = constraints.get("spam_pct_mean")
    if threshold is not None and policy_metrics.spam_pct_mean is not None:
        if policy_metrics.spam_pct_mean > threshold:
            violations.append(f"spam_pct_mean ({policy_metrics.spam_pct_mean:.1f}%) > {threshold}%")

    # timer_under_min_pct_mean <= threshold
    threshold = constraints.get("timer_under_min_pct_mean")
    if threshold is not None and policy_metrics.timer_under_min_pct_mean is not None:
        if policy_metrics.timer_under_min_pct_mean > threshold:
            violations.append(
                f"timer_under_min_pct_mean ({policy_metrics.timer_under_min_pct_mean:.1f}%) > {threshold}%"
            )

    # chunk_size_p50 >= min_ratio * min_words
    min_ratio = constraints.get("chunk_size_p50_min_ratio")
    if min_ratio is not None and policy_metrics.chunk_size_p50 is not None:
        min_chunk_size = min_ratio * policy_metrics.min_words
        if policy_metrics.chunk_size_p50 < min_chunk_size:
            violations.append(
                f"chunk_size_p50 ({policy_metrics.chunk_size_p50:.1f}) < "
                f"{min_ratio} * min_words ({min_chunk_size:.1f})"
            )

    # chunk_size_p95 <= max
    threshold = constraints.get("chunk_size_p95_max")
    if threshold is not None and policy_metrics.chunk_size_p95 is not None:
        if policy_metrics.chunk_size_p95 > threshold:
            violations.append(
                f"chunk_size_p95 ({policy_metrics.chunk_size_p95:.1f}) > {threshold}"
            )

    # worst_wait_p95_ms <= threshold
    threshold = constraints.get("worst_wait_p95_ms")
    if threshold is not None and policy_metrics.worst_wait_p95_ms is not None:
        if policy_metrics.worst_wait_p95_ms > threshold:
            violations.append(
                f"worst_wait_p95_ms ({policy_metrics.worst_wait_p95_ms:.1f}) > {threshold}"
            )

    # flush_count_mean <= max (cost constraint: BufferAgent calls per case)
    threshold = constraints.get("flush_count_mean")
    if threshold is not None and policy_metrics.flush_count_mean is not None:
        if policy_metrics.flush_count_mean > threshold:
            violations.append(
                f"flush_count_mean ({policy_metrics.flush_count_mean:.1f}) > {threshold}"
            )

    # chunk_size_p50 >= min (absolute cost constraint: minimum median chunk size)
    threshold = constraints.get("chunk_size_p50_min")
    if threshold is not None and policy_metrics.chunk_size_p50 is not None:
        if policy_metrics.chunk_size_p50 < threshold:
            violations.append(
                f"chunk_size_p50 ({policy_metrics.chunk_size_p50:.1f}) < {threshold}"
            )

    return violations


def compute_spam_sensitivity(
    case_metrics_list: List[CaseMetrics],
    policy: Policy,
    alpha_values: List[float],
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

