"""Aggregation helpers for per-case metrics."""

from typing import List

import numpy as np

from .computations import compute_distribution, compute_redundancy_threshold_rates
from .constants import REDUNDANCY_THRESHOLDS
from .types import AggregateMetrics, PerCaseMetrics


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_ci(
    values: List[float],
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple:
    """
    Compute bootstrap confidence interval.

    Paper hook: "95% CIs computed via 10k bootstrap resamples
    with fixed seed for reproducibility (Section 5.3)"
    """
    if not values:
        return (None, None, None)

    rng = np.random.RandomState(seed)
    values_arr = np.array(values)
    n = len(values_arr)

    bootstrap_means = []
    for _ in range(n_resamples):
        resample = rng.choice(values_arr, size=n, replace=True)
        bootstrap_means.append(np.mean(resample))

    bootstrap_means = np.array(bootstrap_means)

    alpha = 1 - ci_level
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean = np.mean(values_arr)

    return (float(mean), float(ci_low), float(ci_high))


# ============================================================================
# Aggregate Metrics
# ============================================================================

def compute_aggregate_metrics(
    per_case_metrics: List[PerCaseMetrics],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> AggregateMetrics:
    """
    Aggregate per-case metrics with bootstrap CIs.
    """
    if not per_case_metrics:
        return AggregateMetrics(variant_id="unknown", n_cases=0)

    variant_id = per_case_metrics[0].variant_id
    agg = AggregateMetrics(variant_id=variant_id, n_cases=len(per_case_metrics))

    agg.update_count_mean = float(np.mean([m.update_count for m in per_case_metrics]))
    agg.output_ctu_mean = float(np.mean([m.output_ctu for m in per_case_metrics]))

    all_jaccards = []
    for m in per_case_metrics:
        all_jaccards.extend(m.redundancy_jaccard_values)
    if all_jaccards:
        agg.redundancy_jaccard_mean = float(np.mean(all_jaccards))
    agg.redundancy_threshold_rates = compute_redundancy_threshold_rates(
        all_jaccards, REDUNDANCY_THRESHOLDS
    )

    coverage_values = [m.trace_coverage for m in per_case_metrics]
    mean, ci_low, ci_high = bootstrap_ci(coverage_values, n_bootstrap, seed=seed)
    agg.trace_coverage_mean = mean or 0.0
    agg.trace_coverage_ci_low = ci_low or 0.0
    agg.trace_coverage_ci_high = ci_high or 0.0

    unsupported_global_values = [m.unsupported_global_rate for m in per_case_metrics]
    agg.unsupported_global_rate_mean = float(np.mean(unsupported_global_values))
    unsupported_per_summary_values = [
        m.unsupported_per_summary_mean
        for m in per_case_metrics
        if m.unsupported_per_summary_mean is not None
    ]
    if unsupported_per_summary_values:
        agg.unsupported_per_summary_mean = float(np.mean(unsupported_per_summary_values))

    m6a_values = [m.m6a_mean for m in per_case_metrics if m.m6a_mean is not None]
    if m6a_values:
        mean, ci_low, ci_high = bootstrap_ci(m6a_values, n_bootstrap, seed=seed)
        agg.m6a_mean = mean
        agg.m6a_ci_low = ci_low
        agg.m6a_ci_high = ci_high

    m6b_values = [m.m6b_mean for m in per_case_metrics if m.m6b_mean is not None]
    if m6b_values:
        mean, ci_low, ci_high = bootstrap_ci(m6b_values, n_bootstrap, seed=seed)
        agg.m6b_mean = mean
        agg.m6b_ci_low = ci_low
        agg.m6b_ci_high = ci_high

    budget_keys = {key for m in per_case_metrics for key in m.coverage_by_budget.keys()}
    for budget_key in sorted(budget_keys):
        values = [
            m.coverage_by_budget.get(budget_key)
            for m in per_case_metrics
            if budget_key in m.coverage_by_budget
        ]
        agg.coverage_by_budget_mean[budget_key] = float(np.mean(values)) if values else 0.0

    summary_latency_values = [
        value for m in per_case_metrics for value in m.summary_latency_values
    ]
    if summary_latency_values:
        agg.summary_latency_mean_ms = float(np.mean(summary_latency_values))
        agg.summary_latency_distribution_ms = compute_distribution(summary_latency_values)
    buffer_latency_values = [
        value for m in per_case_metrics for value in m.buffer_decision_latency_values
    ]
    if buffer_latency_values:
        agg.buffer_decision_latency_mean_ms = float(np.mean(buffer_latency_values))
        agg.buffer_decision_latency_distribution_ms = compute_distribution(buffer_latency_values)

    agg.prompt_ctu_mean = float(np.mean([m.total_prompt_ctu for m in per_case_metrics]))
    agg.completion_ctu_mean = float(
        np.mean([m.total_completion_ctu for m in per_case_metrics])
    )
    agg.total_llm_ctu_mean = float(np.mean([m.total_llm_ctu for m in per_case_metrics]))
    agg.buffer_decision_count_mean = float(
        np.mean([m.buffer_decision_count for m in per_case_metrics])
    )

    agg.flush_count_mean = float(np.mean([m.flush_count for m in per_case_metrics]))
    all_reasons = {
        reason for m in per_case_metrics for reason in m.flush_reason_counts.keys()
    }
    for reason in sorted(all_reasons):
        reason_values = [m.flush_reason_counts.get(reason, 0) for m in per_case_metrics]
        agg.flush_reason_counts_mean[reason] = float(np.mean(reason_values))

    flush_accum_values = [
        value for m in per_case_metrics for value in m.flush_accumulated_ctu_values
    ]
    if flush_accum_values:
        agg.flush_accumulated_ctu_distribution = compute_distribution(flush_accum_values)

    flush_interval_values = [
        value for m in per_case_metrics for value in m.flush_interval_ms_values
    ]
    if flush_interval_values:
        agg.flush_interval_ms_distribution = compute_distribution(flush_interval_values)

    duration_values = [
        m.virtual_time_duration_ms
        for m in per_case_metrics
        if m.virtual_time_duration_ms is not None
    ]
    if duration_values:
        agg.virtual_time_duration_ms_mean = float(np.mean(duration_values))
    agg.trace_ctu_mean = float(np.mean([m.trace_ctu for m in per_case_metrics]))
    trace_ctu_per_s_values = [
        m.trace_ctu_per_s for m in per_case_metrics if m.trace_ctu_per_s is not None
    ]
    if trace_ctu_per_s_values:
        agg.trace_ctu_per_s_mean = float(np.mean(trace_ctu_per_s_values))
    summary_ctu_per_s_values = [
        m.summary_ctu_per_s for m in per_case_metrics if m.summary_ctu_per_s is not None
    ]
    if summary_ctu_per_s_values:
        agg.summary_ctu_per_s_mean = float(np.mean(summary_ctu_per_s_values))

    agg.control_plane_latency_ms_total = sum(
        m.control_plane_latency_ms for m in per_case_metrics
    )
    agg.content_plane_latency_ms_total = sum(
        m.content_plane_latency_ms for m in per_case_metrics
    )
    agg.total_plane_latency_ms_total = sum(
        m.total_plane_latency_ms for m in per_case_metrics
    )
    if agg.total_plane_latency_ms_total > 0:
        agg.control_plane_latency_pct = (
            agg.control_plane_latency_ms_total / agg.total_plane_latency_ms_total
        )
        agg.content_plane_latency_pct = (
            agg.content_plane_latency_ms_total / agg.total_plane_latency_ms_total
        )

    agg.schema_failure_count = sum(m.schema_failure_count for m in per_case_metrics)
    agg.summary_event_count = sum(m.update_count for m in per_case_metrics)
    if agg.summary_event_count > 0:
        agg.schema_failure_rate_mean = agg.schema_failure_count / agg.summary_event_count
        agg.compliance_rate_mean = float(1 - agg.schema_failure_rate_mean)

    agg.faithfulness_valid_event_count = sum(
        m.faithfulness_valid_event_count for m in per_case_metrics
    )
    agg.excluded_from_faithfulness_count = sum(
        m.excluded_from_faithfulness_count for m in per_case_metrics
    )
    agg.schema_failures = sum(m.schema_failure_count for m in per_case_metrics)
    agg.stub_trace_count = sum(1 for m in per_case_metrics if m.stub_trace_detected)

    return agg
