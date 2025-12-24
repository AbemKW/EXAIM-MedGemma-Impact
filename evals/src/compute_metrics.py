#!/usr/bin/env python3
"""
EXAID Evaluation - Metrics Computation

Paper hook: "All metrics computed from run logs and frozen traces
with schema failure handling per Section 5.1"

Implements M1-M10 metrics:
    M1: Compression ratio (CTU saved)
    M2: Summary count
    M3: Redundancy (Jaccard on CUI sets between consecutive summaries)
    M4: Trace coverage (fraction of trace CUIs in summaries)
    M5a: Unsupported global rate
    M5b: Unsupported per-summary rate
    M6a: Window-groundedness (unsupported vs window)
    M6b: Contract-groundedness (unsupported vs window+latest_summary)
    M7: Mean summary latency (ms)
    M8: LLM usage (CTU)
    M9: BufferAgent overhead (decisions, CTU)
    M10: Schema failure rate

FAITHFULNESS PARADOX RESOLUTION:
    - schema_ok=false → EXCLUDE from M6a/M6b means (not 0.0!)
    - schema_ok=true, empty CUIs → INCLUDE as 0.0 (grounded by absence)
    - Track excluded_from_faithfulness_count

Dependencies:
    - trace_text.py (canonical text, window reconstruction)
    - concept_extractor.py (CUI extraction)
    - deterministic_io.py (run log reading)
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Set, Any

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trace_text import build_canonical_trace_text, build_window_text, load_trace_chunks_for_case
from deterministic_io import read_run_log, write_json_deterministic, write_jsonl_deterministic
from config_loader import load_extractor_config, get_stoplists_provenance, get_configs_dir


# ============================================================================
# Data Classes
# ============================================================================

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
    
    # M1: Compression
    trace_ctu: int = 0
    summary_ctu: int = 0
    compression_ratio: float = 0.0
    
    # M2: Summary count
    summary_count: int = 0
    
    # M3: Redundancy (Jaccard)
    redundancy_jaccard_mean: Optional[float] = None
    redundancy_jaccard_values: List[float] = field(default_factory=list)
    
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
    
    # M7: Latency
    mean_summary_latency_ms: Optional[float] = None
    
    # M8: LLM usage
    total_prompt_ctu: int = 0
    total_completion_ctu: int = 0
    
    # M9: BufferAgent overhead
    buffer_decision_count: int = 0
    buffer_decision_ctu: int = 0
    
    # M10: Schema failures
    schema_failure_count: int = 0
    schema_failure_rate: float = 0.0


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all cases for a variant."""
    variant_id: str
    n_cases: int = 0
    
    # Means and CIs for each metric
    compression_ratio_mean: float = 0.0
    compression_ratio_ci_low: float = 0.0
    compression_ratio_ci_high: float = 0.0
    
    summary_count_mean: float = 0.0
    
    redundancy_jaccard_mean: Optional[float] = None
    
    trace_coverage_mean: float = 0.0
    trace_coverage_ci_low: float = 0.0
    trace_coverage_ci_high: float = 0.0
    
    m6a_mean: Optional[float] = None
    m6a_ci_low: Optional[float] = None
    m6a_ci_high: Optional[float] = None
    
    m6b_mean: Optional[float] = None
    m6b_ci_low: Optional[float] = None
    m6b_ci_high: Optional[float] = None
    
    faithfulness_valid_event_count: int = 0
    excluded_from_faithfulness_count: int = 0
    schema_failures: int = 0


# ============================================================================
# Concept Extractor Wrapper
# ============================================================================

class ConceptExtractorWrapper:
    """
    Wrapper for concept extraction with caching.
    
    Handles cases where scispaCy/UMLS linker may not be available.
    """
    
    def __init__(self, config: dict, no_linking: bool = False):
        self.config = config
        self.cache: Dict[str, Set[str]] = {}
        self._extractor = None
        self.no_linking = no_linking
        
        try:
            from concept_extractor import ConceptExtractor
            self._extractor = ConceptExtractor(config, no_linking=no_linking)
        except Exception as e:
            print(f"WARNING: Concept extractor unavailable: {e}")
            print("Using stub extraction (empty sets)")
    
    def extract(self, text: str) -> Set[str]:
        """Extract concepts with caching."""
        if not text:
            return set()
        
        # Check cache
        cache_key = hash(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Extract
        if self._extractor:
            concepts = self._extractor.extract(text)
        else:
            # Stub: return empty set
            concepts = set()
        
        self.cache[cache_key] = concepts
        return concepts
    
    def get_version_info(self) -> dict:
        """Get version info for logging."""
        if self._extractor:
            return self._extractor.get_version_info()
        return {"mode": "stub", "reason": "extractor_unavailable"}


# ============================================================================
# Faithfulness Metrics (M6a, M6b)
# ============================================================================

def compute_faithfulness_for_event(
    event: dict,
    run_events: Dict[str, dict],
    trace_chunks: List[dict],
    extractor: ConceptExtractorWrapper
) -> FaithfulnessResult:
    """
    Compute M6a/M6b for a single event with correct failure handling.
    
    Paper hook: "Schema failures excluded from denominators;
    empty-but-valid events return 0.0 (Section 5.1)"
    
    POLICY:
        - schema_ok=false → EXCLUDE (return None, not 0.0!)
        - schema_ok=true, empty CUIs → INCLUDE as 0.0
    
    Args:
        event: Summary event record
        run_events: Dict mapping event_id to event record
        trace_chunks: List of trace chunk records
        extractor: Concept extractor
        
    Returns:
        FaithfulnessResult with values or None if excluded
    """
    # CHECK: Schema failure → EXCLUDE (not 0.0!)
    if not event.get("schema_ok", True):
        return FaithfulnessResult(
            m6a=None,
            m6b=None,
            excluded=True,
            exclusion_reason="schema_failure"
        )
    
    # Extract output CUIs from summary
    summary_text = event.get("summary_semantics_text", "")
    output_cuis = extractor.extract(summary_text)
    
    # Build window text using canonical function
    start_seq = event.get("start_seq", 0)
    end_seq = event.get("end_seq", 0)
    window_text = build_window_text(trace_chunks, start_seq, end_seq)
    window_cuis = extractor.extract(window_text)
    
    # Build support text for M6b (window + latest_summary via ID lookup)
    support_text = window_text
    if event.get("latest_summary_event_id"):
        latest_event = run_events.get(event["latest_summary_event_id"])
        if latest_event:
            support_text += " " + latest_event.get("summary_semantics_text", "")
    support_cuis = extractor.extract(support_text)
    
    # Compute metrics
    # Empty output CUIs → 0.0 (grounded by absence), NOT excluded
    if len(output_cuis) == 0:
        return FaithfulnessResult(
            m6a=0.0,
            m6b=0.0,
            excluded=False,
            exclusion_reason=None,
            output_cui_count=0,
            window_cui_count=len(window_cuis),
            support_cui_count=len(support_cuis)
        )
    
    # M6a: window-groundedness (unsupported fraction)
    unsupported_window = output_cuis - window_cuis
    m6a = len(unsupported_window) / len(output_cuis)
    
    # M6b: contract-groundedness (support = window + latest_summary)
    unsupported_contract = output_cuis - support_cuis
    m6b = len(unsupported_contract) / len(output_cuis)
    
    return FaithfulnessResult(
        m6a=m6a,
        m6b=m6b,
        excluded=False,
        exclusion_reason=None,
        output_cui_count=len(output_cuis),
        window_cui_count=len(window_cuis),
        support_cui_count=len(support_cuis)
    )


# ============================================================================
# Redundancy Metric (M3)
# ============================================================================

def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Paper hook: "Redundancy measured via Jaccard similarity on CUI sets
    between consecutive summaries (Section 5.1)"
    """
    if not set_a and not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_redundancy(
    summary_events: List[dict],
    extractor: ConceptExtractorWrapper
) -> List[float]:
    """
    Compute redundancy (Jaccard) between consecutive summaries.
    
    Returns:
        List of Jaccard values for consecutive pairs
    """
    if len(summary_events) < 2:
        return []
    
    # Extract CUI sets for each summary
    cui_sets = []
    for event in summary_events:
        text = event.get("summary_semantics_text", "")
        cuis = extractor.extract(text)
        cui_sets.append(cuis)
    
    # Compute Jaccard for consecutive pairs
    jaccards = []
    for i in range(1, len(cui_sets)):
        j = compute_jaccard_similarity(cui_sets[i-1], cui_sets[i])
        jaccards.append(j)
    
    return jaccards


# ============================================================================
# Coverage Metrics (M4, M5)
# ============================================================================

def compute_coverage_metrics(
    summary_events: List[dict],
    trace_cuis: Set[str],
    extractor: ConceptExtractorWrapper
) -> dict:
    """
    Compute coverage metrics with schema failure penalty.
    
    Paper hook: "Schema-failed events contribute empty concept sets,
    penalizing recall (Section 5.2)"
    
    Returns:
        Dict with coverage metrics
    """
    all_summary_cuis = set()
    per_summary_unsupported = []
    
    for event in summary_events:
        if not event.get("schema_ok", True):
            # Schema failure → empty set (penalizes recall)
            continue
        
        summary_text = event.get("summary_semantics_text", "")
        event_cuis = extractor.extract(summary_text)
        all_summary_cuis.update(event_cuis)
        
        # Per-summary unsupported rate
        if len(event_cuis) > 0:
            unsupported = event_cuis - trace_cuis
            unsupported_rate = len(unsupported) / len(event_cuis)
            per_summary_unsupported.append(unsupported_rate)
    
    # M4: Coverage
    if len(trace_cuis) == 0:
        coverage = 0.0
    else:
        covered = all_summary_cuis & trace_cuis
        coverage = len(covered) / len(trace_cuis)
    
    # M5a: Unsupported global rate
    if len(all_summary_cuis) == 0:
        unsupported_global = 0.0
    else:
        unsupported = all_summary_cuis - trace_cuis
        unsupported_global = len(unsupported) / len(all_summary_cuis)
    
    # M5b: Mean per-summary unsupported
    unsupported_per_summary_mean = (
        float(np.mean(per_summary_unsupported)) 
        if per_summary_unsupported else None
    )
    
    return {
        "trace_coverage": coverage,
        "unsupported_global_rate": unsupported_global,
        "unsupported_per_summary_mean": unsupported_per_summary_mean,
        "trace_cui_count": len(trace_cuis),
        "summary_cui_count": len(all_summary_cuis)
    }


# ============================================================================
# CTU Computation
# ============================================================================

def compute_ctu(text: str) -> int:
    """Compute Character-normalized Token Units."""
    return math.ceil(len(text) / 4) if text else 0


# ============================================================================
# Per-Case Metrics Computation
# ============================================================================

def compute_per_case_metrics(
    case_id: str,
    variant_id: str,
    run_log_path: Path,
    trace_file: Path,
    extractor: ConceptExtractorWrapper
) -> PerCaseMetrics:
    """
    Compute all metrics for a single case/variant.
    
    Args:
        case_id: Case identifier
        variant_id: Variant identifier
        run_log_path: Path to run log file
        trace_file: Path to trace file
        extractor: Concept extractor
        
    Returns:
        PerCaseMetrics with all computed values
    """
    metrics = PerCaseMetrics(case_id=case_id, variant_id=variant_id)
    
    # Load run log
    records = read_run_log(run_log_path)
    
    # Parse records by type
    run_meta = None
    summary_events = []
    buffer_decisions = []
    
    run_events: Dict[str, dict] = {}  # event_id -> event
    
    for record in records:
        record_type = record.get("record_type")
        if record_type == "run_meta":
            run_meta = record
        elif record_type == "summary_event":
            summary_events.append(record)
            event_id = record.get("event_id")
            if event_id:
                run_events[event_id] = record
        elif record_type == "buffer_decision":
            buffer_decisions.append(record)
    
    # Sort events by index
    summary_events.sort(key=lambda e: e.get("event_index", 0))
    
    # Load trace chunks
    trace_chunks = load_trace_chunks_for_case(trace_file)
    
    # Build canonical trace text
    trace_text, _ = build_canonical_trace_text(trace_file, fail_on_empty=False)
    trace_cuis = extractor.extract(trace_text)
    
    # M1: Compression
    metrics.trace_ctu = compute_ctu(trace_text)
    metrics.summary_ctu = sum(
        e.get("summary_ctu", compute_ctu(e.get("summary_semantics_text", "")))
        for e in summary_events
    )
    if metrics.trace_ctu > 0:
        metrics.compression_ratio = 1 - (metrics.summary_ctu / metrics.trace_ctu)
    
    # M2: Summary count
    metrics.summary_count = len(summary_events)
    
    # M3: Redundancy
    jaccards = compute_redundancy(summary_events, extractor)
    metrics.redundancy_jaccard_values = jaccards
    if jaccards:
        metrics.redundancy_jaccard_mean = float(np.mean(jaccards))
    
    # M4, M5: Coverage
    coverage_results = compute_coverage_metrics(summary_events, trace_cuis, extractor)
    metrics.trace_coverage = coverage_results["trace_coverage"]
    metrics.unsupported_global_rate = coverage_results["unsupported_global_rate"]
    metrics.unsupported_per_summary_mean = coverage_results["unsupported_per_summary_mean"]
    metrics.trace_cui_count = coverage_results["trace_cui_count"]
    metrics.summary_cui_count = coverage_results["summary_cui_count"]
    
    # M6a, M6b: Faithfulness
    m6a_values = []
    m6b_values = []
    excluded_count = 0
    schema_failures = 0
    
    for event in summary_events:
        result = compute_faithfulness_for_event(
            event, run_events, trace_chunks, extractor
        )
        
        if result.excluded:
            excluded_count += 1
            if result.exclusion_reason == "schema_failure":
                schema_failures += 1
            continue  # Do NOT append anything
        
        # Valid event (including empty-but-valid as 0.0)
        m6a_values.append(result.m6a)
        m6b_values.append(result.m6b)
    
    metrics.m6a_values = m6a_values
    metrics.m6b_values = m6b_values
    
    if m6a_values:
        metrics.m6a_mean = float(np.mean(m6a_values))
        metrics.m6a_std = float(np.std(m6a_values))
    
    if m6b_values:
        metrics.m6b_mean = float(np.mean(m6b_values))
        metrics.m6b_std = float(np.std(m6b_values))
    
    metrics.faithfulness_valid_event_count = len(m6a_values)
    metrics.excluded_from_faithfulness_count = excluded_count
    
    # M7: Latency
    latencies = [e.get("latency_ms", 0) for e in summary_events]
    if latencies:
        metrics.mean_summary_latency_ms = float(np.mean(latencies))
    
    # M8: LLM usage
    for event in summary_events:
        llm_usage = event.get("llm_usage", {})
        metrics.total_prompt_ctu += llm_usage.get("prompt_ctu", 0)
        metrics.total_completion_ctu += llm_usage.get("completion_ctu", 0)
    
    # M9: BufferAgent overhead
    metrics.buffer_decision_count = len(buffer_decisions)
    for decision in buffer_decisions:
        metrics.buffer_decision_ctu += decision.get("input_ctu", 0)
    
    # M10: Schema failures
    metrics.schema_failure_count = schema_failures
    if len(summary_events) > 0:
        metrics.schema_failure_rate = schema_failures / len(summary_events)
    
    return metrics


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_ci(
    values: List[float],
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> tuple:
    """
    Compute bootstrap confidence interval.
    
    Paper hook: "95% CIs computed via 10k bootstrap resamples
    with fixed seed for reproducibility (Section 5.3)"
    
    Args:
        values: Sample values
        n_resamples: Number of bootstrap resamples
        ci_level: Confidence level (default 0.95)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (mean, ci_low, ci_high)
    """
    if not values:
        return (None, None, None)
    
    rng = np.random.RandomState(seed)
    values_arr = np.array(values)
    n = len(values_arr)
    
    # Bootstrap means
    bootstrap_means = []
    for _ in range(n_resamples):
        resample = rng.choice(values_arr, size=n, replace=True)
        bootstrap_means.append(np.mean(resample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute percentiles
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
    seed: int = 42
) -> AggregateMetrics:
    """
    Aggregate per-case metrics with bootstrap CIs.
    
    Args:
        per_case_metrics: List of per-case metrics
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed
        
    Returns:
        AggregateMetrics with means and CIs
    """
    if not per_case_metrics:
        return AggregateMetrics(variant_id="unknown", n_cases=0)
    
    variant_id = per_case_metrics[0].variant_id
    agg = AggregateMetrics(variant_id=variant_id, n_cases=len(per_case_metrics))
    
    # Compression ratio
    compression_values = [m.compression_ratio for m in per_case_metrics]
    mean, ci_low, ci_high = bootstrap_ci(compression_values, n_bootstrap, seed=seed)
    agg.compression_ratio_mean = mean or 0.0
    agg.compression_ratio_ci_low = ci_low or 0.0
    agg.compression_ratio_ci_high = ci_high or 0.0
    
    # Summary count
    agg.summary_count_mean = np.mean([m.summary_count for m in per_case_metrics])
    
    # Redundancy
    all_jaccards = []
    for m in per_case_metrics:
        all_jaccards.extend(m.redundancy_jaccard_values)
    if all_jaccards:
        agg.redundancy_jaccard_mean = float(np.mean(all_jaccards))
    
    # Trace coverage
    coverage_values = [m.trace_coverage for m in per_case_metrics]
    mean, ci_low, ci_high = bootstrap_ci(coverage_values, n_bootstrap, seed=seed)
    agg.trace_coverage_mean = mean or 0.0
    agg.trace_coverage_ci_low = ci_low or 0.0
    agg.trace_coverage_ci_high = ci_high or 0.0
    
    # Faithfulness M6a
    m6a_values = [m.m6a_mean for m in per_case_metrics if m.m6a_mean is not None]
    if m6a_values:
        mean, ci_low, ci_high = bootstrap_ci(m6a_values, n_bootstrap, seed=seed)
        agg.m6a_mean = mean
        agg.m6a_ci_low = ci_low
        agg.m6a_ci_high = ci_high
    
    # Faithfulness M6b
    m6b_values = [m.m6b_mean for m in per_case_metrics if m.m6b_mean is not None]
    if m6b_values:
        mean, ci_low, ci_high = bootstrap_ci(m6b_values, n_bootstrap, seed=seed)
        agg.m6b_mean = mean
        agg.m6b_ci_low = ci_low
        agg.m6b_ci_high = ci_high
    
    # Totals
    agg.faithfulness_valid_event_count = sum(
        m.faithfulness_valid_event_count for m in per_case_metrics
    )
    agg.excluded_from_faithfulness_count = sum(
        m.excluded_from_faithfulness_count for m in per_case_metrics
    )
    agg.schema_failures = sum(m.schema_failure_count for m in per_case_metrics)
    
    return agg


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXAID Metrics Computation"
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/runs"),
        help="Run logs directory"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("data/traces"),
        help="Traces directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metrics"),
        help="Output metrics directory"
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs"),
        help="Configs directory"
    )
    parser.add_argument(
        "--variant",
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Compute for specific variant only"
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of bootstrap samples for CI"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXAID Metrics Computation")
    print("=" * 60)
    print()
    
    # Initialize extractor via centralized config loader
    # Paper hook: "Extractor config loaded via centralized loader ensuring
    # drift-proof evaluation with resolved stoplist paths (Section 6.1)"
    extractor_config = load_extractor_config(args.configs, resolve_paths=True, include_hashes=True)
    
    # Get stoplists provenance for logging
    stoplists_provenance = get_stoplists_provenance(args.configs)
    print(f"Stoplists provenance:")
    for key, value in stoplists_provenance.items():
        print(f"  {key}: {value}")
    print()
    
    extractor = ConceptExtractorWrapper(extractor_config, no_linking=True)
    print(f"Extractor: {extractor.get_version_info()}")
    print()
    
    # Determine variants
    variants = [args.variant] if args.variant else ["V0", "V1", "V2", "V3", "V4"]
    
    # Output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    all_per_case = []
    all_aggregate = []
    
    for variant_id in variants:
        variant_dir = args.runs / variant_id
        if not variant_dir.exists():
            print(f"Skipping {variant_id}: no run logs found")
            continue
        
        print(f"Computing metrics for {variant_id}...")
        
        # Find run logs
        run_logs = sorted(variant_dir.glob("*.jsonl.gz"))
        if not run_logs:
            run_logs = sorted(variant_dir.glob("*.jsonl"))
        
        per_case_metrics = []
        
        for run_log_path in run_logs:
            case_id = run_log_path.stem.replace(".jsonl", "")
            trace_file = args.traces / f"{case_id}.jsonl.gz"
            
            if not trace_file.exists():
                trace_file = args.traces / f"{case_id}.jsonl"
            
            if not trace_file.exists():
                print(f"  WARNING: No trace file for {case_id}")
                continue
            
            if args.verbose:
                print(f"  {case_id}...", end=" ")
            
            try:
                metrics = compute_per_case_metrics(
                    case_id, variant_id, run_log_path, trace_file, extractor
                )
                per_case_metrics.append(metrics)
                all_per_case.append(metrics)
                
                if args.verbose:
                    print(f"coverage={metrics.trace_coverage:.2f}")
            except Exception as e:
                print(f"  ERROR: {case_id}: {e}")
        
        # Aggregate
        agg = compute_aggregate_metrics(
            per_case_metrics, 
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed
        )
        all_aggregate.append(agg)
        
        print(f"  {variant_id}: {agg.n_cases} cases, "
              f"coverage={agg.trace_coverage_mean:.3f} "
              f"[{agg.trace_coverage_ci_low:.3f}, {agg.trace_coverage_ci_high:.3f}]")
        print(f"    M6a={agg.m6a_mean:.3f if agg.m6a_mean else 'N/A'}, "
              f"M6b={agg.m6b_mean:.3f if agg.m6b_mean else 'N/A'}")
        print(f"    Schema failures: {agg.schema_failures}, "
              f"Excluded from faithfulness: {agg.excluded_from_faithfulness_count}")
    
    # Write outputs
    print()
    print("Writing outputs...")
    
    # Per-case JSONL
    per_case_output = args.output / "per_case.metrics.jsonl"
    per_case_records = []
    for m in all_per_case:
        record = {
            "case_id": m.case_id,
            "variant_id": m.variant_id,
            "compression_ratio": m.compression_ratio,
            "summary_count": m.summary_count,
            "redundancy_jaccard_mean": m.redundancy_jaccard_mean,
            "trace_coverage": m.trace_coverage,
            "unsupported_global_rate": m.unsupported_global_rate,
            "m6a_mean": m.m6a_mean,
            "m6b_mean": m.m6b_mean,
            "faithfulness_valid_event_count": m.faithfulness_valid_event_count,
            "excluded_from_faithfulness_count": m.excluded_from_faithfulness_count,
            "schema_failure_count": m.schema_failure_count,
            "schema_failure_rate": m.schema_failure_rate,
            "mean_summary_latency_ms": m.mean_summary_latency_ms,
            "total_prompt_ctu": m.total_prompt_ctu,
            "total_completion_ctu": m.total_completion_ctu,
            "buffer_decision_count": m.buffer_decision_count
        }
        per_case_records.append(record)
    
    write_jsonl_deterministic(per_case_records, per_case_output)
    print(f"  Per-case: {per_case_output}")
    
    # Aggregate JSON
    aggregate_output = args.output / "aggregate.metrics.json"
    aggregate_data = {
        "computed_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "variants": {}
    }
    
    for agg in all_aggregate:
        aggregate_data["variants"][agg.variant_id] = {
            "n_cases": agg.n_cases,
            "compression_ratio": {
                "mean": agg.compression_ratio_mean,
                "ci_low": agg.compression_ratio_ci_low,
                "ci_high": agg.compression_ratio_ci_high
            },
            "summary_count_mean": agg.summary_count_mean,
            "redundancy_jaccard_mean": agg.redundancy_jaccard_mean,
            "trace_coverage": {
                "mean": agg.trace_coverage_mean,
                "ci_low": agg.trace_coverage_ci_low,
                "ci_high": agg.trace_coverage_ci_high
            },
            "m6a": {
                "mean": agg.m6a_mean,
                "ci_low": agg.m6a_ci_low,
                "ci_high": agg.m6a_ci_high
            },
            "m6b": {
                "mean": agg.m6b_mean,
                "ci_low": agg.m6b_ci_low,
                "ci_high": agg.m6b_ci_high
            },
            "faithfulness_valid_event_count": agg.faithfulness_valid_event_count,
            "excluded_from_faithfulness_count": agg.excluded_from_faithfulness_count,
            "schema_failures": agg.schema_failures
        }
    
    write_json_deterministic(aggregate_data, aggregate_output)
    print(f"  Aggregate: {aggregate_output}")
    
    print()
    print("=" * 60)
    print("METRICS COMPUTATION COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
