#!/usr/bin/env python3
"""
EXAID Evaluation - Metrics Computation

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
    - trace_text.py (canonical text, window reconstruction)
    - concept_extractor.py (CUI extraction)
    - deterministic_io.py (run log reading)
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Set, Any

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trace_text import (
    build_canonical_trace_text,
    build_window_text,
    load_trace_chunks_for_case,
    iter_trace_records,
)
from deterministic_io import read_run_log, write_json_deterministic, write_jsonl_deterministic
from config_loader import (
    load_extractor_config,
    get_stoplists_provenance,
    load_variant_config,
    compute_file_hash,
)
from run_variants import compute_tokengate_config_hash


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


# ============================================================================
# Metric Configuration
# ============================================================================

METRICS_SCHEMA_VERSION = "2.1.0"
REDUNDANCY_THRESHOLDS = [0.85, 0.90, 0.95]
BUFFER_COVERAGE_BUDGETS = [250, 500, 1000, 2000]


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
        trace_chunks: List of stream_delta records (record_type == "stream_delta")
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
        if not event.get("schema_ok", True):
            cuis = set()
        else:
            text = event.get("summary_semantics_text", "")
            cuis = extractor.extract(text)
        cui_sets.append(cuis)
    
    # Compute Jaccard for consecutive pairs
    jaccards = []
    for i in range(1, len(cui_sets)):
        j = compute_jaccard_similarity(cui_sets[i-1], cui_sets[i])
        jaccards.append(j)
    
    return jaccards


def compute_redundancy_threshold_rates(
    jaccards: List[float],
    thresholds: Optional[List[float]] = None
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


def compute_coverage_by_budget(
    summary_events: List[dict],
    trace_cuis: Set[str],
    extractor: ConceptExtractorWrapper,
    budgets: List[int]
) -> Dict[str, float]:
    """
    Compute coverage at increasing summary CTU budgets (M7b).
    """
    if not budgets:
        return {
            f"ctu_{budget}": 0.0 for budget in BUFFER_COVERAGE_BUDGETS
        }
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
                compute_ctu(event.get("summary_semantics_text", ""))
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
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
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
    """Compute flush statistics from tokengate_flush records."""
    stats: Dict[str, Any] = {
        "flush_count": len(flushes),
        "flush_reason_counts": {},
        "flush_accumulated_ctu_distribution": None,
        "flush_interval_ms_distribution": None,
    }
    if not flushes:
        return stats

    for flush in flushes:
        reason = flush.get("trigger_reason", "unknown")
        stats["flush_reason_counts"][reason] = stats["flush_reason_counts"].get(reason, 0) + 1

    accumulated_ctu = [
        f.get("accumulated_ctu")
        for f in flushes
        if f.get("accumulated_ctu") is not None
    ]
    stats["flush_accumulated_ctu_distribution"] = compute_distribution(accumulated_ctu)

    sorted_flushes = sorted(flushes, key=lambda f: f.get("flush_index", 0))
    intervals_ms = []
    prev_ts_ms = None
    for flush in sorted_flushes:
        ts_ms = parse_timestamp_ms(flush.get("timestamp"))
        if ts_ms is None:
            continue
        if prev_ts_ms is not None:
            intervals_ms.append(ts_ms - prev_ts_ms)
        prev_ts_ms = ts_ms
    stats["flush_interval_ms_distribution"] = compute_distribution(intervals_ms)
    return stats


def compute_virtual_time_throughput(
    trace_chunks: List[dict],
    trace_ctu: int,
    summary_ctu: int
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
    duration_s = duration_ms / 1000 if duration_ms > 0 else None
    trace_ctu_per_s = trace_ctu / duration_s if duration_s else None
    summary_ctu_per_s = summary_ctu / duration_s if duration_s else None
    return {
        "virtual_time_duration_ms": int(duration_ms),
        "trace_ctu_per_s": float(trace_ctu_per_s) if trace_ctu_per_s is not None else None,
        "summary_ctu_per_s": float(summary_ctu_per_s) if summary_ctu_per_s is not None else None,
    }


def compute_overhead_attribution(
    summary_latencies: List[float],
    buffer_latencies: List[float]
) -> dict:
    """Compute control-plane vs content-plane overhead attribution."""
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


def load_trace_meta(trace_file: Path) -> Optional[dict]:
    """Load trace_meta record from a trace file."""
    for record in iter_trace_records(trace_file):
        if record.get("record_type") == "trace_meta":
            return record
    return None


def compute_manifest_hash(
    mas_run_id: str,
    case_list_hash: str,
    trace_entries: List[tuple]
) -> str:
    """Compute manifest trace_dataset_hash per schema."""
    import hashlib

    canonical = {
        "mas_run_id": mas_run_id,
        "case_list_hash": case_list_hash,
        "traces": sorted(trace_entries),
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical_json.encode()).hexdigest()}"


def load_manifest_provenance(manifest_path: Path) -> dict:
    """Load manifest provenance and validate its hash."""
    manifest_meta = {}
    provenance = {}
    trace_entries = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_type = record.get("record_type")
            if record_type == "manifest_meta":
                manifest_meta = record
            elif record_type == "provenance":
                provenance = record
            elif record_type == "trace_entry":
                trace_entries.append((record.get("case_id", ""), record.get("sha256", "")))

    mas_run_id = manifest_meta.get("mas_run_id", "")
    case_list_hash = provenance.get("case_list_hash", "")
    computed_hash = compute_manifest_hash(mas_run_id, case_list_hash, trace_entries)
    manifest_hash = provenance.get("trace_dataset_hash")

    return {
        "dataset_id": manifest_meta.get("dataset_id"),
        "mas_run_id": mas_run_id,
        "case_list_hash": case_list_hash,
        "config_hash": provenance.get("config_hash"),
        "manifest_hash": manifest_hash,
        "computed_hash": computed_hash,
        "manifest_hash_valid": manifest_hash == computed_hash,
        "stub_mode": manifest_meta.get("stub_mode", False),
        "manifest_path": str(manifest_path),
    }


def get_git_commit(repo_root: Path) -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ============================================================================
# Per-Case Metrics Computation
# ============================================================================

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

    if run_meta and expected_tokengate_config_hash:
        run_tokengate_hash = run_meta.get("tokengate_config_hash")
        metrics.tokengate_config_hash_match = (
            run_tokengate_hash == expected_tokengate_config_hash
        )
        if not metrics.tokengate_config_hash_match:
            raise ValueError(
                f"TokenGate config hash mismatch for {case_id}: "
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
    decision_latencies = [
        d.get("latency_ms")
        for d in buffer_decisions
        if d.get("latency_ms") is not None
    ]
    if decision_latencies:
        metrics.mean_buffer_decision_latency_ms = float(np.mean(decision_latencies))
        metrics.buffer_decision_latency_distribution_ms = compute_distribution(decision_latencies)

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
    
    # M1: Update counts
    agg.update_count_mean = float(np.mean([m.update_count for m in per_case_metrics]))
    
    # M2: Output volume
    agg.output_ctu_mean = float(np.mean([m.output_ctu for m in per_case_metrics]))
    
    # M3: Redundancy
    all_jaccards = []
    for m in per_case_metrics:
        all_jaccards.extend(m.redundancy_jaccard_values)
    if all_jaccards:
        agg.redundancy_jaccard_mean = float(np.mean(all_jaccards))
    agg.redundancy_threshold_rates = compute_redundancy_threshold_rates(
        all_jaccards, REDUNDANCY_THRESHOLDS
    )
    
    # M4: Coverage
    coverage_values = [m.trace_coverage for m in per_case_metrics]
    mean, ci_low, ci_high = bootstrap_ci(coverage_values, n_bootstrap, seed=seed)
    agg.trace_coverage_mean = mean or 0.0
    agg.trace_coverage_ci_low = ci_low or 0.0
    agg.trace_coverage_ci_high = ci_high or 0.0

    # M5: Unsupported
    unsupported_global_values = [
        m.unsupported_global_rate for m in per_case_metrics
    ]
    agg.unsupported_global_rate_mean = float(np.mean(unsupported_global_values))
    unsupported_per_summary_values = [
        m.unsupported_per_summary_mean
        for m in per_case_metrics
        if m.unsupported_per_summary_mean is not None
    ]
    if unsupported_per_summary_values:
        agg.unsupported_per_summary_mean = float(
            np.mean(unsupported_per_summary_values)
        )
    
    # M6a: Faithfulness
    m6a_values = [m.m6a_mean for m in per_case_metrics if m.m6a_mean is not None]
    if m6a_values:
        mean, ci_low, ci_high = bootstrap_ci(m6a_values, n_bootstrap, seed=seed)
        agg.m6a_mean = mean
        agg.m6a_ci_low = ci_low
        agg.m6a_ci_high = ci_high
    
    # M6b: Faithfulness
    m6b_values = [m.m6b_mean for m in per_case_metrics if m.m6b_mean is not None]
    if m6b_values:
        mean, ci_low, ci_high = bootstrap_ci(m6b_values, n_bootstrap, seed=seed)
        agg.m6b_mean = mean
        agg.m6b_ci_low = ci_low
        agg.m6b_ci_high = ci_high

    # M7b: Coverage vs budget
    budget_keys = {key for m in per_case_metrics for key in m.coverage_by_budget.keys()}
    for budget_key in sorted(budget_keys):
        values = [
            m.coverage_by_budget.get(budget_key)
            for m in per_case_metrics
            if budget_key in m.coverage_by_budget
        ]
        agg.coverage_by_budget_mean[budget_key] = (
            float(np.mean(values)) if values else 0.0
        )

    # M8: Latency
    summary_latency_values = [
        m.mean_summary_latency_ms
        for m in per_case_metrics
        if m.mean_summary_latency_ms is not None
    ]
    if summary_latency_values:
        agg.summary_latency_mean_ms = float(np.mean(summary_latency_values))
        agg.summary_latency_distribution_ms = compute_distribution(summary_latency_values)
    buffer_latency_values = [
        m.mean_buffer_decision_latency_ms
        for m in per_case_metrics
        if m.mean_buffer_decision_latency_ms is not None
    ]
    if buffer_latency_values:
        agg.buffer_decision_latency_mean_ms = float(np.mean(buffer_latency_values))
        agg.buffer_decision_latency_distribution_ms = compute_distribution(buffer_latency_values)

    # M9: Usage
    agg.prompt_ctu_mean = float(np.mean([m.total_prompt_ctu for m in per_case_metrics]))
    agg.completion_ctu_mean = float(
        np.mean([m.total_completion_ctu for m in per_case_metrics])
    )
    agg.total_llm_ctu_mean = float(np.mean([m.total_llm_ctu for m in per_case_metrics]))
    agg.buffer_decision_count_mean = float(
        np.mean([m.buffer_decision_count for m in per_case_metrics])
    )

    # Flush statistics
    agg.flush_count_mean = float(np.mean([m.flush_count for m in per_case_metrics]))
    all_reasons = {
        reason
        for m in per_case_metrics
        for reason in m.flush_reason_counts.keys()
    }
    for reason in sorted(all_reasons):
        reason_values = [
            m.flush_reason_counts.get(reason, 0)
            for m in per_case_metrics
        ]
        agg.flush_reason_counts_mean[reason] = float(np.mean(reason_values))

    flush_accum_means = [
        m.flush_accumulated_ctu_distribution.get("mean")
        for m in per_case_metrics
        if m.flush_accumulated_ctu_distribution
        and m.flush_accumulated_ctu_distribution.get("mean") is not None
    ]
    if flush_accum_means:
        agg.flush_accumulated_ctu_distribution = compute_distribution(flush_accum_means)

    flush_interval_means = [
        m.flush_interval_ms_distribution.get("mean")
        for m in per_case_metrics
        if m.flush_interval_ms_distribution
        and m.flush_interval_ms_distribution.get("mean") is not None
    ]
    if flush_interval_means:
        agg.flush_interval_ms_distribution = compute_distribution(flush_interval_means)

    # Virtual-time throughput
    duration_values = [
        m.virtual_time_duration_ms
        for m in per_case_metrics
        if m.virtual_time_duration_ms is not None
    ]
    if duration_values:
        agg.virtual_time_duration_ms_mean = float(np.mean(duration_values))
    agg.trace_ctu_mean = float(np.mean([m.trace_ctu for m in per_case_metrics]))
    trace_ctu_per_s_values = [
        m.trace_ctu_per_s
        for m in per_case_metrics
        if m.trace_ctu_per_s is not None
    ]
    if trace_ctu_per_s_values:
        agg.trace_ctu_per_s_mean = float(np.mean(trace_ctu_per_s_values))
    summary_ctu_per_s_values = [
        m.summary_ctu_per_s
        for m in per_case_metrics
        if m.summary_ctu_per_s is not None
    ]
    if summary_ctu_per_s_values:
        agg.summary_ctu_per_s_mean = float(np.mean(summary_ctu_per_s_values))

    # Overhead attribution
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

    # M10: Compliance
    agg.schema_failure_count = sum(
        m.schema_failure_count for m in per_case_metrics
    )
    agg.summary_event_count = sum(m.update_count for m in per_case_metrics)
    if agg.summary_event_count > 0:
        agg.schema_failure_rate_mean = (
            agg.schema_failure_count / agg.summary_event_count
        )
        agg.compliance_rate_mean = float(1 - agg.schema_failure_rate_mean)
    
    # Totals
    agg.faithfulness_valid_event_count = sum(
        m.faithfulness_valid_event_count for m in per_case_metrics
    )
    agg.excluded_from_faithfulness_count = sum(
        m.excluded_from_faithfulness_count for m in per_case_metrics
    )
    agg.schema_failures = sum(m.schema_failure_count for m in per_case_metrics)
    agg.stub_trace_count = sum(1 for m in per_case_metrics if m.stub_trace_detected)
    
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
        "--manifests",
        type=Path,
        default=Path("data/manifests"),
        help="Manifests directory"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Explicit manifest path for dataset integrity checks"
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

    manifest_path = args.manifest
    if manifest_path is None:
        if args.manifests.exists():
            manifest_candidates = sorted(args.manifests.glob("*.manifest.jsonl"))
            if len(manifest_candidates) == 1:
                manifest_path = manifest_candidates[0]
            elif len(manifest_candidates) > 1:
                print("WARNING: Multiple manifests found. Use --manifest to select one.")
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
            "variant_config_hash": compute_file_hash(variant_config_path),
            "tokengate_config_hash": expected_tokengate_hash,
        }
        
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
                    case_id,
                    variant_id,
                    run_log_path,
                    trace_file,
                    extractor,
                    expected_tokengate_config_hash=expected_tokengate_hash,
                    manifest_hash_valid=manifest_info["manifest_hash_valid"]
                    if manifest_info else None,
                    manifest_trace_dataset_hash=manifest_info["computed_hash"]
                    if manifest_info else None,
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
            "extractor_config_hash": compute_file_hash(args.configs / "extractor.yaml"),
            "metrics_config_hash": compute_file_hash(args.configs / "metrics.yaml"),
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


if __name__ == "__main__":
    sys.exit(main())
