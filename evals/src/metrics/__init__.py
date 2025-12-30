"""Metrics helpers for evaluation computations."""

from .aggregation import bootstrap_ci, compute_aggregate_metrics
from .computations import (
    compute_coverage_by_budget,
    compute_coverage_metrics,
    compute_ctu,
    compute_distribution,
    compute_faithfulness_for_event,
    compute_flush_statistics,
    compute_jaccard_similarity,
    compute_overhead_attribution,
    compute_redundancy,
    compute_redundancy_threshold_rates,
    compute_virtual_time_throughput,
    parse_timestamp_ms,
)
from .constants import (
    BUFFER_COVERAGE_BUDGETS,
    METRICS_SCHEMA_VERSION,
    REDUNDANCY_THRESHOLDS,
)
from .extractor import ConceptExtractorWrapper
from .integrity import (
    compute_manifest_hash,
    get_git_commit,
    load_manifest_provenance,
    load_trace_meta,
    safe_file_hash,
)
from .types import AggregateMetrics, FaithfulnessResult, PerCaseMetrics

__all__ = [
    "AggregateMetrics",
    "BUFFER_COVERAGE_BUDGETS",
    "ConceptExtractorWrapper",
    "FaithfulnessResult",
    "METRICS_SCHEMA_VERSION",
    "PerCaseMetrics",
    "REDUNDANCY_THRESHOLDS",
    "bootstrap_ci",
    "compute_aggregate_metrics",
    "compute_coverage_by_budget",
    "compute_coverage_metrics",
    "compute_ctu",
    "compute_distribution",
    "compute_faithfulness_for_event",
    "compute_flush_statistics",
    "compute_jaccard_similarity",
    "compute_manifest_hash",
    "compute_overhead_attribution",
    "compute_redundancy",
    "compute_redundancy_threshold_rates",
    "compute_virtual_time_throughput",
    "get_git_commit",
    "load_manifest_provenance",
    "load_trace_meta",
    "parse_timestamp_ms",
    "safe_file_hash",
]
