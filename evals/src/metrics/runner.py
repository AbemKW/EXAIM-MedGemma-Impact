#!/usr/bin/env python3
"""
EXAIM Evaluation - Metrics Computation

Paper hook: "All metrics computed from run logs and frozen traces
with schema failure handling per Section 5.1"

Implements M1-M10 metrics:
    M1: Update counts
    M2: Output volume (summary CTU)
    M3: Redundancy (Jaccard thresholds between consecutive summaries)
    M4: Trace coverage (fraction of trace CUIs in summaries)
    M5: Unsupported content rates
    M6a: Window-groundedness (unsupported vs window)
    M6b: Contract-groundedness (unsupported vs window+latest_summary)
    M7b: Coverage-vs-budget curve
    M8: Latency (summary + BufferAgent)
    M9: LLM usage (CTU)
    M10: Compliance (schema_ok rate)

FAITHFULNESS PARADOX RESOLUTION:
    - schema_ok=false → EXCLUDE from M6a/M6b means (not 0.0!)
    - schema_ok=true, empty CUIs → INCLUDE as 0.0 (grounded by absence)
    - Track excluded_from_faithfulness_count

Dependencies:
    - traces/trace_text.py (canonical text, window reconstruction)
    - deterministic/io.py (run log reading)
    - metrics/ (shared metric helpers and dataclasses)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..traces.trace_text import (
    build_canonical_trace_text,
    load_trace_chunks_for_case,
)
from ..deterministic.io import (
    read_run_log,
    write_json_deterministic,
    write_jsonl_deterministic,
)
from ..config.config_loader import (
    load_extractor_config,
    get_stoplists_provenance,
    load_variant_config,
)
from ..utils.hashing import compute_tokengate_config_hash
from ..metrics import (
    BUFFER_COVERAGE_BUDGETS,
    ConceptExtractorWrapper,
    METRICS_SCHEMA_VERSION,
    PerCaseMetrics,
    REDUNDANCY_THRESHOLDS,
    compute_aggregate_metrics,
    compute_coverage_by_budget,
    compute_coverage_metrics,
    compute_ctu,
    compute_distribution,
    compute_faithfulness_for_event,
    compute_flush_statistics,
    compute_overhead_attribution,
    compute_redundancy,
    compute_redundancy_threshold_rates,
    compute_virtual_time_throughput,
    get_git_commit,
    load_manifest_provenance,
    load_trace_meta,
    safe_file_hash,
)


# ============================================================================
# Per-Case Metrics Computation
# ============================================================================

def _percentile_threshold(
    values: Sequence[Union[float, int]],
    percentile: float,
) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), percentile))


def apply_outlier_flags(per_case_metrics: List[PerCaseMetrics]) -> None:
    """
    Flag per-case outliers for latency spikes, excessive flushes, and low coverage.
    """
    latency_values = [
        metrics.mean_summary_latency_ms
        for metrics in per_case_metrics
        if metrics.mean_summary_latency_ms is not None
    ]
    flush_counts = [metrics.flush_count for metrics in per_case_metrics]
    coverage_values = [metrics.trace_coverage for metrics in per_case_metrics]

    latency_threshold = _percentile_threshold(latency_values, 95)
    flush_threshold = _percentile_threshold(flush_counts, 95)
    coverage_threshold = _percentile_threshold(coverage_values, 5)

    for metrics in per_case_metrics:
        metrics.outlier_latency_spike = False
        metrics.outlier_excessive_flushes = False
        metrics.outlier_low_coverage = False
        flags = []

        if (
            latency_threshold is not None
            and metrics.mean_summary_latency_ms is not None
            and metrics.mean_summary_latency_ms > latency_threshold
        ):
            metrics.outlier_latency_spike = True
            flags.append("latency_spike")

        if flush_threshold is not None and metrics.flush_count > flush_threshold:
            metrics.outlier_excessive_flushes = True
            flags.append("excessive_flushes")

        if (
            coverage_threshold is not None
            and metrics.trace_coverage < coverage_threshold
        ):
            metrics.outlier_low_coverage = True
            flags.append("low_coverage")

        metrics.outlier_flags = flags


def compute_per_case_metrics(
    case_id: str,
    variant_id: str,
    run_log_path: Path,
    trace_file: Path,
    extractor: ConceptExtractorWrapper,
    expected_tokengate_config_hash: Optional[str] = None,
    manifest_hash_valid: Optional[bool] = None,
    manifest_trace_dataset_hash: Optional[str] = None
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
    tokengate_flushes = []
    
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
        elif record_type == "tokengate_flush":
            tokengate_flushes.append(record)
    
    # Sort events by index
    summary_events.sort(key=lambda e: e.get("event_index", 0))

    if expected_tokengate_config_hash and not run_meta:
        raise ValueError(
            f"Missing run_meta for {case_id} ({variant_id}) in {run_log_path}; "
            "cannot validate TokenGate config hash."
        )
    if run_meta and expected_tokengate_config_hash:
        run_tokengate_hash = run_meta.get("tokengate_config_hash")
        metrics.tokengate_config_hash_match = (
            run_tokengate_hash == expected_tokengate_config_hash
        )
        if not metrics.tokengate_config_hash_match:
            raise ValueError(
                f"TokenGate config hash mismatch for {case_id} ({variant_id}): "
                f"{run_tokengate_hash} != {expected_tokengate_config_hash}"
            )
    dataset_manifest_valid = manifest_hash_valid
    if run_meta and manifest_trace_dataset_hash:
        run_trace_dataset_hash = run_meta.get("trace_dataset_hash")
        if run_trace_dataset_hash:
            trace_dataset_match = run_trace_dataset_hash == manifest_trace_dataset_hash
            dataset_manifest_valid = (
                trace_dataset_match
                if dataset_manifest_valid is None
                else dataset_manifest_valid and trace_dataset_match
            )
            if not trace_dataset_match:
                raise ValueError(
                    f"Trace dataset hash mismatch for {case_id}: "
                    f"{run_trace_dataset_hash} != {manifest_trace_dataset_hash}"
                )
    metrics.dataset_manifest_hash_valid = dataset_manifest_valid
    
    # Load trace chunks
    trace_chunks = load_trace_chunks_for_case(trace_file)

    trace_meta = load_trace_meta(trace_file)
    if trace_meta and trace_meta.get("stub_mode"):
        metrics.stub_trace_detected = True
    
    # Build canonical trace text
    trace_text, _ = build_canonical_trace_text(trace_file, fail_on_empty=False)
    metrics.trace_ctu = compute_ctu(trace_text)
    trace_cuis = extractor.extract(trace_text)
    
    # M1: Update counts
    metrics.update_count = len(summary_events)
    
    # M2: Output volume
    metrics.output_ctu = sum(
        e.get("summary_ctu", compute_ctu(e.get("summary_semantics_text", "")))
        for e in summary_events
    )
    
    # M3: Redundancy
    jaccards = compute_redundancy(summary_events, extractor)
    metrics.redundancy_jaccard_values = jaccards
    if jaccards:
        metrics.redundancy_jaccard_mean = float(np.mean(jaccards))
    metrics.redundancy_threshold_rates = compute_redundancy_threshold_rates(
        jaccards, REDUNDANCY_THRESHOLDS
    )
    
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
    
    # M7b: Coverage vs budget
    metrics.coverage_by_budget = compute_coverage_by_budget(
        summary_events, trace_cuis, extractor, BUFFER_COVERAGE_BUDGETS
    )
    
    # M8: Latency
    summary_latencies = [
        e.get("latency_ms")
        for e in summary_events
        if e.get("latency_ms") is not None
    ]
    if summary_latencies:
        metrics.mean_summary_latency_ms = float(np.mean(summary_latencies))
        metrics.summary_latency_distribution_ms = compute_distribution(summary_latencies)
        metrics.summary_latency_values = [float(value) for value in summary_latencies]
    decision_latencies = [
        d.get("latency_ms")
        for d in buffer_decisions
        if d.get("latency_ms") is not None
    ]
    if decision_latencies:
        metrics.mean_buffer_decision_latency_ms = float(np.mean(decision_latencies))
        metrics.buffer_decision_latency_distribution_ms = compute_distribution(decision_latencies)
        metrics.buffer_decision_latency_values = [float(value) for value in decision_latencies]

    overhead = compute_overhead_attribution(summary_latencies, decision_latencies)
    metrics.control_plane_latency_ms = overhead["control_plane_latency_ms"]
    metrics.content_plane_latency_ms = overhead["content_plane_latency_ms"]
    metrics.total_plane_latency_ms = overhead["total_plane_latency_ms"]
    metrics.control_plane_latency_pct = overhead["control_plane_latency_pct"]
    metrics.content_plane_latency_pct = overhead["content_plane_latency_pct"]
    
    # M9: LLM usage
    metrics.buffer_decision_count = len(buffer_decisions)
    for event in summary_events:
        llm_usage = event.get("llm_usage", {})
        metrics.total_prompt_ctu += llm_usage.get("prompt_ctu", 0)
        metrics.total_completion_ctu += llm_usage.get("completion_ctu", 0)
    for decision in buffer_decisions:
        llm_usage = decision.get("llm_usage", {})
        metrics.total_prompt_ctu += llm_usage.get("prompt_ctu", 0)
        metrics.total_completion_ctu += llm_usage.get("completion_ctu", 0)
    metrics.total_llm_ctu = metrics.total_prompt_ctu + metrics.total_completion_ctu

    # Flush statistics
    flush_stats = compute_flush_statistics(tokengate_flushes)
    metrics.flush_count = flush_stats["flush_count"]
    metrics.flush_reason_counts = flush_stats["flush_reason_counts"]
    metrics.flush_accumulated_ctu_distribution = flush_stats["flush_accumulated_ctu_distribution"]
    metrics.flush_interval_ms_distribution = flush_stats["flush_interval_ms_distribution"]
    metrics.flush_accumulated_ctu_values = flush_stats["flush_accumulated_ctu_values"]
    metrics.flush_interval_ms_values = flush_stats["flush_interval_ms_values"]

    # Virtual-time throughput
    throughput = compute_virtual_time_throughput(
        trace_chunks,
        metrics.trace_ctu,
        metrics.output_ctu,
    )
    metrics.virtual_time_duration_ms = throughput["virtual_time_duration_ms"]
    metrics.trace_ctu_per_s = throughput["trace_ctu_per_s"]
    metrics.summary_ctu_per_s = throughput["summary_ctu_per_s"]
    
    # M10: Compliance
    metrics.schema_failure_count = schema_failures
    if len(summary_events) > 0:
        metrics.schema_failure_rate = schema_failures / len(summary_events)
        metrics.compliance_rate = 1 - metrics.schema_failure_rate
    else:
        metrics.compliance_rate = 1.0
    
    return metrics


# ============================================================================
# Main Entry Point
# ============================================================================

def run_metrics(args) -> int:
    # Resolve relative paths relative to evals root
    evals_root = Path(__file__).resolve().parents[2]  # metrics -> src -> evals

    def resolve_path(path: Path) -> Path:
        """Resolve relative paths relative to evals root."""
        if path.is_absolute():
            return path
        # Try current directory first, then evals root
        if path.exists():
            return path
        evals_path = evals_root / path
        if evals_path.exists():
            return evals_path
        return path  # Return as-is if neither exists (let downstream handle error)

    args.runs = resolve_path(args.runs)
    args.traces = resolve_path(args.traces)
    args.output = resolve_path(args.output)
    args.configs = resolve_path(args.configs)
    args.manifests = resolve_path(args.manifests)
    if args.manifest:
        args.manifest = resolve_path(args.manifest)
    print("=" * 60)
    print("EXAIM Metrics Computation")
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
    
    extractor = ConceptExtractorWrapper(extractor_config, no_linking=False)
    print(f"Extractor: {extractor.get_version_info()}")
    print()

    manifest_path = args.manifest
    if manifest_path is None:
        if args.manifests.exists():
            manifest_candidates = sorted(args.manifests.glob("*.manifest.jsonl"))
            if len(manifest_candidates) == 1:
                manifest_path = manifest_candidates[0]
            elif len(manifest_candidates) > 1:
                raise ValueError(
                    "Multiple manifests found. Use --manifest to select one."
                )
    manifest_info = None
    if manifest_path and manifest_path.exists():
        manifest_info = load_manifest_provenance(manifest_path)
        if not manifest_info["manifest_hash_valid"]:
            raise ValueError(
                "Dataset manifest hash mismatch: "
                f"{manifest_info['manifest_hash']} != {manifest_info['computed_hash']}"
            )
    elif manifest_path:
        print(f"WARNING: Manifest path not found: {manifest_path}")
    
    # Determine variants
    variants = [args.variant] if args.variant else ["V0", "V1", "V2", "V3", "V4"]
    
    # Output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    all_per_case = []
    all_aggregate = []
    
    variant_provenance: Dict[str, dict] = {}

    for variant_id in variants:
        variant_dir = args.runs / variant_id
        if not variant_dir.exists():
            print(f"Skipping {variant_id}: no run logs found")
            continue
        
        print(f"Computing metrics for {variant_id}...")
        
        variant_config = load_variant_config(variant_id, args.configs)
        expected_tokengate_hash = compute_tokengate_config_hash(variant_config)
        variant_config_path = args.configs / "variants" / f"{variant_id}.yaml"
        variant_provenance[variant_id] = {
            "variant_config_hash": safe_file_hash(variant_config_path),
            "tokengate_config_hash": expected_tokengate_hash,
        }
        
        # Find run logs
        run_logs = sorted(variant_dir.glob("*.jsonl.gz"))
        if not run_logs:
            run_logs = sorted(variant_dir.glob("*.jsonl"))
        
        per_case_metrics = []
        
        manifest_hash_valid = manifest_info["manifest_hash_valid"] if manifest_info else None
        manifest_trace_dataset_hash = manifest_info["computed_hash"] if manifest_info else None

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
                    case_id,
                    variant_id,
                    run_log_path,
                    trace_file,
                    extractor,
                    expected_tokengate_config_hash=expected_tokengate_hash,
                    manifest_hash_valid=manifest_hash_valid,
                    manifest_trace_dataset_hash=manifest_trace_dataset_hash,
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
        
        m6a_display = f"{agg.m6a_mean:.3f}" if agg.m6a_mean is not None else "N/A"
        m6b_display = f"{agg.m6b_mean:.3f}" if agg.m6b_mean is not None else "N/A"
        print(f"  {variant_id}: {agg.n_cases} cases, "
              f"coverage={agg.trace_coverage_mean:.3f} "
              f"[{agg.trace_coverage_ci_low:.3f}, {agg.trace_coverage_ci_high:.3f}]")
        print(f"    M6a={m6a_display}, M6b={m6b_display}")
        print(f"    Schema failures: {agg.schema_failures}, "
              f"Excluded from faithfulness: {agg.excluded_from_faithfulness_count}")
    
    # Write outputs
    print()
    print("Writing outputs...")

    apply_outlier_flags(all_per_case)
    
    # Per-case JSONL
    per_case_output = args.output / "per_case.metrics.jsonl"
    per_case_records = []
    for m in all_per_case:
        record = {
            "schema_name": "exaid.metrics",
            "schema_version": METRICS_SCHEMA_VERSION,
            "metrics_type": "per_case",
            "case_id": m.case_id,
            "variant_id": m.variant_id,
            "m1_update_count": m.update_count,
            "m2_output_ctu": m.output_ctu,
            "m3_redundancy_jaccard_mean": m.redundancy_jaccard_mean,
            "m3_redundancy_threshold_rates": m.redundancy_threshold_rates,
            "m4_trace_coverage": m.trace_coverage,
            "m4_trace_cui_count": m.trace_cui_count,
            "m4_summary_cui_count": m.summary_cui_count,
            "m5_unsupported_global_rate": m.unsupported_global_rate,
            "m5_unsupported_per_summary_mean": m.unsupported_per_summary_mean,
            "m6a_mean": m.m6a_mean,
            "m6b_mean": m.m6b_mean,
            "faithfulness_valid_event_count": m.faithfulness_valid_event_count,
            "excluded_from_faithfulness_count": m.excluded_from_faithfulness_count,
            "m7b_coverage_by_budget": m.coverage_by_budget,
            "m8_summary_latency_ms_mean": m.mean_summary_latency_ms,
            "m8_buffer_decision_latency_ms_mean": m.mean_buffer_decision_latency_ms,
            "m8_summary_latency_distribution_ms": m.summary_latency_distribution_ms,
            "m8_buffer_decision_latency_distribution_ms": m.buffer_decision_latency_distribution_ms,
            "m9_prompt_ctu_total": m.total_prompt_ctu,
            "m9_completion_ctu_total": m.total_completion_ctu,
            "m9_total_llm_ctu": m.total_llm_ctu,
            "m9_buffer_decision_count": m.buffer_decision_count,
            "flush_count": m.flush_count,
            "flush_reason_counts": m.flush_reason_counts,
            "flush_accumulated_ctu_distribution": m.flush_accumulated_ctu_distribution,
            "flush_interval_ms_distribution": m.flush_interval_ms_distribution,
            "virtual_time_duration_ms": m.virtual_time_duration_ms,
            "trace_ctu": m.trace_ctu,
            "trace_ctu_per_s": m.trace_ctu_per_s,
            "summary_ctu_per_s": m.summary_ctu_per_s,
            "control_plane_latency_ms": m.control_plane_latency_ms,
            "content_plane_latency_ms": m.content_plane_latency_ms,
            "total_plane_latency_ms": m.total_plane_latency_ms,
            "control_plane_latency_pct": m.control_plane_latency_pct,
            "content_plane_latency_pct": m.content_plane_latency_pct,
            "m10_schema_failure_count": m.schema_failure_count,
            "m10_schema_failure_rate": m.schema_failure_rate,
            "m10_compliance_rate": m.compliance_rate,
            "outlier_flags": m.outlier_flags,
            "outlier_latency_spike": m.outlier_latency_spike,
            "outlier_excessive_flushes": m.outlier_excessive_flushes,
            "outlier_low_coverage": m.outlier_low_coverage,
            "tokengate_config_hash_match": m.tokengate_config_hash_match,
            "dataset_manifest_hash_valid": m.dataset_manifest_hash_valid,
            "stub_trace_detected": m.stub_trace_detected
        }
        per_case_records.append(record)
    
    write_jsonl_deterministic(per_case_records, per_case_output)
    print(f"  Per-case: {per_case_output}")
    
    # Aggregate JSON
    aggregate_output = args.output / "aggregate.metrics.json"
    aggregate_data = {
        "schema_name": "exaid.metrics",
        "schema_version": METRICS_SCHEMA_VERSION,
        "metrics_type": "aggregate",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "variants": {}
    }
    
    for agg in all_aggregate:
        aggregate_data["variants"][agg.variant_id] = {
            "n_cases": agg.n_cases,
            "m1_update_count_mean": agg.update_count_mean,
            "m2_output_ctu_mean": agg.output_ctu_mean,
            "m3_redundancy": {
                "jaccard_mean": agg.redundancy_jaccard_mean,
                "threshold_rates": agg.redundancy_threshold_rates
            },
            "m4_trace_coverage": {
                "mean": agg.trace_coverage_mean,
                "ci_low": agg.trace_coverage_ci_low,
                "ci_high": agg.trace_coverage_ci_high
            },
            "m5_unsupported": {
                "global_rate_mean": agg.unsupported_global_rate_mean,
                "per_summary_mean": agg.unsupported_per_summary_mean
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
            "m7b_coverage_by_budget_mean": agg.coverage_by_budget_mean,
            "m8_latency_ms_mean": {
                "summary": agg.summary_latency_mean_ms,
                "buffer_decision": agg.buffer_decision_latency_mean_ms
            },
            "m8_latency_ms_distribution": {
                "summary": agg.summary_latency_distribution_ms,
                "buffer_decision": agg.buffer_decision_latency_distribution_ms
            },
            "m9_usage_ctu_mean": {
                "prompt": agg.prompt_ctu_mean,
                "completion": agg.completion_ctu_mean,
                "total": agg.total_llm_ctu_mean,
                "buffer_decision_count": agg.buffer_decision_count_mean
            },
            "flush_statistics": {
                "flush_count_mean": agg.flush_count_mean,
                "flush_reason_counts_mean": agg.flush_reason_counts_mean,
                "flush_accumulated_ctu_distribution": agg.flush_accumulated_ctu_distribution,
                "flush_interval_ms_distribution": agg.flush_interval_ms_distribution
            },
            "virtual_time_throughput": {
                "duration_ms_mean": agg.virtual_time_duration_ms_mean,
                "trace_ctu_mean": agg.trace_ctu_mean,
                "trace_ctu_per_s_mean": agg.trace_ctu_per_s_mean,
                "summary_ctu_per_s_mean": agg.summary_ctu_per_s_mean
            },
            "overhead_attribution": {
                "control_plane_latency_ms_total": agg.control_plane_latency_ms_total,
                "content_plane_latency_ms_total": agg.content_plane_latency_ms_total,
                "total_plane_latency_ms_total": agg.total_plane_latency_ms_total,
                "control_plane_latency_pct": agg.control_plane_latency_pct,
                "content_plane_latency_pct": agg.content_plane_latency_pct
            },
            "m10_compliance": {
                "schema_failure_rate_mean": agg.schema_failure_rate_mean,
                "compliance_rate_mean": agg.compliance_rate_mean,
                "schema_failure_count": agg.schema_failure_count,
                "summary_event_count": agg.summary_event_count
            },
            "faithfulness_valid_event_count": agg.faithfulness_valid_event_count,
            "excluded_from_faithfulness_count": agg.excluded_from_faithfulness_count,
            "schema_failures": agg.schema_failures,
            "stub_trace_count": agg.stub_trace_count
        }
    
    write_json_deterministic(aggregate_data, aggregate_output)
    print(f"  Aggregate: {aggregate_output}")

    provenance_output = args.output / "metrics.provenance.json"
    provenance_data = {
        "schema_name": "exaid.metrics.provenance",
        "schema_version": "1.0.0",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "code_version": get_git_commit(Path(__file__).resolve().parents[2]),
        "configs": {
            "extractor_config_hash": safe_file_hash(args.configs / "extractor.yaml"),
            "metrics_config_hash": safe_file_hash(args.configs / "metrics.yaml"),
            "variant_configs": variant_provenance,
        },
        "dataset": manifest_info,
    }
    write_json_deterministic(provenance_data, provenance_output)
    print(f"  Provenance: {provenance_output}")
    
    print()
    print("=" * 60)
    print("METRICS COMPUTATION COMPLETE")
    print("=" * 60)
    
    return 0
