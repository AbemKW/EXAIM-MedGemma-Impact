"""Dataclasses used by evaluation metrics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FaithfulnessResult:
    """
    Result of faithfulness computation for one event.

    Paper hook: "Schema failures excluded from denominators;
    empty-but-valid events return 0.0 (Section 5.1)"
    """

    m6a: Optional[float]  # None if excluded (schema failure)
    m6b: Optional[float]  # None if excluded (schema failure)
    excluded: bool
    exclusion_reason: Optional[str]
    output_cui_count: int = 0
    window_cui_count: int = 0
    support_cui_count: int = 0


@dataclass
class PerCaseMetrics:
    """Metrics computed for a single case/variant combination."""

    case_id: str
    variant_id: str

    # M1: Update counts
    update_count: int = 0

    # M2: Output volume
    output_ctu: int = 0

    # M3: Redundancy (Jaccard thresholds)
    redundancy_jaccard_mean: Optional[float] = None
    redundancy_jaccard_values: List[float] = field(default_factory=list)
    redundancy_threshold_rates: Dict[str, Optional[float]] = field(default_factory=dict)

    # M4: Trace coverage
    trace_coverage: float = 0.0
    trace_cui_count: int = 0
    summary_cui_count: int = 0

    # M5: Unsupported
    unsupported_global_rate: float = 0.0
    unsupported_per_summary_mean: Optional[float] = None

    # M6a/M6b: Faithfulness
    m6a_mean: Optional[float] = None
    m6a_std: Optional[float] = None
    m6a_values: List[float] = field(default_factory=list)
    m6b_mean: Optional[float] = None
    m6b_std: Optional[float] = None
    m6b_values: List[float] = field(default_factory=list)
    faithfulness_valid_event_count: int = 0
    excluded_from_faithfulness_count: int = 0

    # M7b: Coverage vs budget
    coverage_by_budget: Dict[str, float] = field(default_factory=dict)

    # M8: Latency
    mean_summary_latency_ms: Optional[float] = None
    mean_buffer_decision_latency_ms: Optional[float] = None
    summary_latency_distribution_ms: Optional[Dict[str, float]] = None
    buffer_decision_latency_distribution_ms: Optional[Dict[str, float]] = None
    summary_latency_values: List[float] = field(default_factory=list)
    buffer_decision_latency_values: List[float] = field(default_factory=list)

    # M9: LLM usage
    total_prompt_ctu: int = 0
    total_completion_ctu: int = 0
    total_llm_ctu: int = 0
    buffer_decision_count: int = 0

    # Flush statistics
    flush_count: int = 0
    flush_reason_counts: Dict[str, int] = field(default_factory=dict)
    flush_accumulated_ctu_distribution: Optional[Dict[str, float]] = None
    flush_interval_ms_distribution: Optional[Dict[str, float]] = None
    flush_accumulated_ctu_values: List[float] = field(default_factory=list)
    flush_interval_ms_values: List[float] = field(default_factory=list)

    # Virtual-time throughput
    virtual_time_duration_ms: Optional[int] = None
    trace_ctu: int = 0
    trace_ctu_per_s: Optional[float] = None
    summary_ctu_per_s: Optional[float] = None

    # Overhead attribution
    control_plane_latency_ms: float = 0.0
    content_plane_latency_ms: float = 0.0
    total_plane_latency_ms: float = 0.0
    control_plane_latency_pct: Optional[float] = None
    content_plane_latency_pct: Optional[float] = None

    # M10: Compliance
    schema_failure_count: int = 0
    schema_failure_rate: float = 0.0
    compliance_rate: float = 0.0

    # Integrity checks
    tokengate_config_hash_match: Optional[bool] = None
    dataset_manifest_hash_valid: Optional[bool] = None
    stub_trace_detected: bool = False


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all cases for a variant."""

    variant_id: str
    n_cases: int = 0

    # M1: Update counts
    update_count_mean: float = 0.0

    # M2: Output volume
    output_ctu_mean: float = 0.0

    # M3: Redundancy
    redundancy_jaccard_mean: Optional[float] = None
    redundancy_threshold_rates: Dict[str, Optional[float]] = field(default_factory=dict)

    # M4: Coverage
    trace_coverage_mean: float = 0.0
    trace_coverage_ci_low: float = 0.0
    trace_coverage_ci_high: float = 0.0

    # M5: Unsupported
    unsupported_global_rate_mean: float = 0.0
    unsupported_per_summary_mean: Optional[float] = None

    # M6a/M6b: Faithfulness
    m6a_mean: Optional[float] = None
    m6a_ci_low: Optional[float] = None
    m6a_ci_high: Optional[float] = None

    m6b_mean: Optional[float] = None
    m6b_ci_low: Optional[float] = None
    m6b_ci_high: Optional[float] = None

    # M7b: Coverage vs budget
    coverage_by_budget_mean: Dict[str, float] = field(default_factory=dict)

    # M8: Latency
    summary_latency_mean_ms: Optional[float] = None
    buffer_decision_latency_mean_ms: Optional[float] = None
    summary_latency_distribution_ms: Optional[Dict[str, float]] = None
    buffer_decision_latency_distribution_ms: Optional[Dict[str, float]] = None

    # M9: Usage
    prompt_ctu_mean: float = 0.0
    completion_ctu_mean: float = 0.0
    total_llm_ctu_mean: float = 0.0
    buffer_decision_count_mean: float = 0.0

    # Flush statistics
    flush_count_mean: float = 0.0
    flush_reason_counts_mean: Dict[str, float] = field(default_factory=dict)
    flush_accumulated_ctu_distribution: Optional[Dict[str, float]] = None
    flush_interval_ms_distribution: Optional[Dict[str, float]] = None

    # Virtual-time throughput
    virtual_time_duration_ms_mean: Optional[float] = None
    trace_ctu_mean: float = 0.0
    trace_ctu_per_s_mean: Optional[float] = None
    summary_ctu_per_s_mean: Optional[float] = None

    # Overhead attribution
    control_plane_latency_ms_total: float = 0.0
    content_plane_latency_ms_total: float = 0.0
    total_plane_latency_ms_total: float = 0.0
    control_plane_latency_pct: Optional[float] = None
    content_plane_latency_pct: Optional[float] = None

    # M10: Compliance
    schema_failure_rate_mean: Optional[float] = None
    compliance_rate_mean: Optional[float] = None
    schema_failure_count: int = 0
    summary_event_count: int = 0

    faithfulness_valid_event_count: int = 0
    excluded_from_faithfulness_count: int = 0
    schema_failures: int = 0
    stub_trace_count: int = 0
