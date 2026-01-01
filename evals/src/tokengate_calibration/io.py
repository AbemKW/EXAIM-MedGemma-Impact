"""I/O helpers for TokenGate calibration artifacts."""

import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from .models import CaseMetrics, Policy, PolicyMetrics


def resolve_manifest_path(manifest_pattern: str) -> Path:
    """Resolve a manifest path or glob pattern to a concrete Path."""
    manifest_path = Path(manifest_pattern)
    if "*" in str(manifest_path):
        import glob

        manifest_files = glob.glob(str(manifest_path))
        if not manifest_files:
            raise FileNotFoundError(f"No manifest files found matching: {manifest_pattern}")
        return Path(manifest_files[0])
    return manifest_path


def load_manifest_entries(manifest_path: Path) -> Tuple[List[dict], str]:
    """Load manifest entries and extract mas_run_id."""
    trace_entries: List[dict] = []
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

    return trace_entries, mas_run_id


def verify_trace_hashes(
    trace_entries: List[dict],
    traces_dir: Path,
    log: Callable[[str], None],
) -> None:
    """Verify trace hashes against manifest entries."""
    for trace_entry in trace_entries:
        case_id = trace_entry["case_id"]
        trace_file = traces_dir / trace_entry["file"]

        if not trace_file.exists():
            log(f"WARNING: Trace file not found: {trace_file}")
            continue

        # Hash uncompressed content (matching make_traces.py behavior)
        try:
            with gzip.open(trace_file, "rt", encoding="utf-8") as tf:
                content = tf.read()
            content_bytes = content.encode("utf-8")
            actual_hash = hashlib.sha256(content_bytes).hexdigest()
            expected_hash = trace_entry.get("sha256", "").replace("sha256:", "")
            if expected_hash and actual_hash != expected_hash:
                log(
                    f"WARNING: Trace hash mismatch for {case_id}: expected {expected_hash[:8]}, got {actual_hash[:8]}"
                )
                log("  This may indicate the trace file was modified after manifest creation.")
                log("  Continuing calibration, but results may not be reproducible.")
        except (OSError, gzip.BadGzipFile, UnicodeDecodeError) as exc:
            log(f"WARNING: Failed to verify hash for {case_id}: {exc}")


def load_config(config_path: Path) -> dict:
    """Load calibration configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_trace_dataset_hash(manifest_path: Path) -> str:
    """
    Compute trace_dataset_hash per schema definition.
    
    Schema: SHA256 of JSON: {mas_run_id, case_list_hash, sorted [(case_id, trace_sha256)]}
    
    Uses the same computation as integrity.compute_manifest_hash for consistency.
    """
    from ..metrics.integrity import compute_manifest_hash, load_manifest_provenance
    
    # Use the validated implementation from integrity.py
    manifest_info = load_manifest_provenance(manifest_path)
    return manifest_info["computed_hash"]


def compute_config_hash(config: dict) -> str:
    """Compute SHA256 hash of canonicalized sweep configuration."""
    # Create canonical copy (exclude non-deterministic fields)
    canonical_config = {
        "parameter_grid": config["parameter_grid"],
        "validity_constraints": config.get("validity_constraints", {}),
        "constraints": config.get("constraints", {}),
        "selection": config.get("selection", {}),
        "spam": config.get("spam", {}),
    }

    canonical_json = json.dumps(canonical_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_exaid_commit(repo_root: Path) -> str:
    """Get current EXAID git commit hash."""
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


def generate_calibration_run_id(
    trace_dataset_hash: str,
    config_hash: str,
    exaid_commit: str,
) -> str:
    """Generate deterministic calibration run ID."""
    return f"calib_{trace_dataset_hash[:8]}_{config_hash[:8]}_{exaid_commit[:8]}"


def write_config_copy(output_path: Path, config: dict) -> None:
    """Write calibration configuration YAML."""
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def write_calibration_results_csv(
    output_path: Path, policy_metrics_list: List[PolicyMetrics]
) -> None:
    """Write calibration results CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "policy_id",
                "min_words",
                "max_words",
                "silence_timer_ms",
                "max_wait_timeout_ms",
                "ttff_content_p50_ms",
                "ttff_content_p95_ms",
                "ttff_trace_p50_ms",
                "flush_count_mean",
                "chunk_size_p50",
                "chunk_size_p90",
                "chunk_size_p95",
                "chunk_size_max",
                "worst_wait_p95_ms",
                "worst_wait_max_ms",
                "spam_pct_mean",
                "timer_flush_pct_mean",
                "timer_under_min_pct_mean",
                "constraint_violations",
                "weighted_score",
            ]
        )

        # Rows
        for pm in policy_metrics_list:
            violations_str = "; ".join(pm.constraint_violations) if pm.constraint_violations else ""
            writer.writerow(
                [
                    pm.policy_id,
                    pm.min_words,
                    pm.max_words,
                    pm.silence_timer_ms,
                    pm.max_wait_timeout_ms,
                    pm.ttff_content_p50_ms,
                    pm.ttff_content_p95_ms,
                    pm.ttff_trace_p50_ms,
                    pm.flush_count_mean,
                    pm.chunk_size_p50,
                    pm.chunk_size_p90,
                    pm.chunk_size_p95,
                    pm.chunk_size_max,
                    pm.worst_wait_p95_ms,
                    pm.worst_wait_max_ms,
                    pm.spam_pct_mean,
                    pm.timer_flush_pct_mean,
                    pm.timer_under_min_pct_mean,
                    violations_str,
                    pm.weighted_score,
                ]
            )


def write_per_case_jsonl(output_path: Path, case_metrics_list: List[CaseMetrics]) -> None:
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
                        "is_end_of_trace": e.is_end_of_trace,
                    }
                    for e in cm.flush_events
                ],
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
    selection_metadata: Optional[dict] = None,
) -> None:
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
            "config_hash8": config_hash[:8],
        },
        "policy_validity": {
            "valid_policies_count": valid_policies_count,
            "invalid_policies_count": invalid_policies_count,
            "invalid_reasons": [
                {
                    "policy_id": policy.policy_id,
                    "min_words": policy.min_words,
                    "max_words": policy.max_words,
                    "silence_timer_ms": policy.silence_timer_ms,
                    "max_wait_timeout_ms": policy.max_wait_timeout_ms,
                    "reason": reason,
                }
                for policy, reason in invalid_reasons[:10]
            ],
        },
        "constraint_filtering": {
            "total_policies": len(policy_metrics_list),
            "survivor_count": len(survivor_metrics),
            "rejected_count": len(policy_metrics_list) - len(survivor_metrics),
        },
        "selection": {
            "method": selection_method,
            "selection_mode": selection_method,  # Alias for compatibility
            "selected_policy_id": selected_policy.policy_id if selected_policy else None,
            "selected_parameters": {
                "min_words": selected_policy.min_words if selected_policy else None,
                "max_words": selected_policy.max_words if selected_policy else None,
                "silence_timer_ms": selected_policy.silence_timer_ms if selected_policy else None,
                "max_wait_timeout_ms": selected_policy.max_wait_timeout_ms if selected_policy else None,
            }
            if selected_policy
            else None,
        },
        "weighted_scores": {
            pm.policy_id: pm.weighted_score
            for pm in survivor_metrics
            if pm.weighted_score is not None
        },
    }

    # Add normalization bounds (computed bounds if available, otherwise config bounds)
    if computed_bounds is not None:
        summary["normalization_bounds"] = computed_bounds
        summary["dropped_metrics"] = dropped_metrics if dropped_metrics else []
    else:
        summary["normalization_bounds"] = (
            config.get("selection", {})
            .get("weighted_score", {})
            .get("normalization_bounds", {})
        )
        summary["dropped_metrics"] = []

    # Add selection metadata if available
    if selection_metadata:
        summary["selection"]["metadata"] = selection_metadata

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_chosen_params_yaml(output_path: Path, selected_policy: PolicyMetrics) -> None:
    """Write chosen TokenGate parameters YAML."""
    params = {
        "token_gate": {
            "min_words": selected_policy.min_words,
            "max_words": selected_policy.max_words,
            "silence_timer": selected_policy.silence_timer_ms / 1000.0,  # Convert to seconds
            "max_wait_timeout": selected_policy.max_wait_timeout_ms / 1000.0,  # Convert to seconds
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
    utopia_rankings: Optional[Dict[str, Tuple[int, float]]] = None,
) -> None:
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
            f.write(
                "3-objective Pareto frontier analysis with utopia-distance selection was used. "
            )
            f.write(
                "Objectives: minimize TTFF (Time To First Flush), minimize flush count (BufferAgent calls), maximize chunk size. "
            )
            f.write(
                "First, non-dominated points (Pareto frontier) were identified in the 3D objective space. "
            )
            f.write(
                "Then, the policy with minimum dimension-normalized Euclidean distance to the utopia point (1, 1, 1) in goodness space was selected. "
            )
            f.write(
                "Normalization bounds were computed from survivor policies using percentile-based methods (P05/P95, or min/max for small-N cases).\n\n"
            )
        elif selection_method == "weighted_fallback":
            f.write(
                "Weighted objective function was used as fallback (Pareto frontier was empty or invalid). "
            )
            f.write("Weights were renormalized to account for any dropped metrics.\n\n")
        elif selection_method == "lexicographic_fallback":
            f.write(
                "Lexicographic tie-breaking was used as fallback (all metrics were dropped due to insufficient variance). "
            )
            f.write(
                "Selection order: lower flush_count_mean, higher chunk_size_p50, lower ttff_content_p50_ms, smallest policy_id.\n\n"
            )
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
                f.write(
                    f"- **{metric_name}**: [{lo}{unit}, {hi}{unit}] (method: {method}) {status}\n"
                )
            f.write("\n")

            if dropped:
                f.write(
                    f"**Dropped Metrics:** {', '.join(dropped)} - These metrics had insufficient variance (hi - lo < epsilon) and were excluded from Pareto dominance and distance computation.\n\n"
                )
        else:
            f.write("Normalization bounds were not computed (using config defaults).\n\n")

        f.write("## Top 5 Policies\n\n")
        f.write(
            "| Policy ID | min_words | max_words | silence_timer_ms | max_wait_timeout_ms | TTFF (p50) | Chunk Size (p50) | Weighted Score |\n"
        )
        f.write(
            "|-----------|-----------|-----------|-------------------|---------------------|------------|------------------|----------------|\n"
        )
        for pm in top_policies[:5]:
            ttff = f"{pm.ttff_content_p50_ms:.1f}" if pm.ttff_content_p50_ms else "N/A"
            chunk = f"{pm.chunk_size_p50:.1f}" if pm.chunk_size_p50 else "N/A"
            score = f"{pm.weighted_score:.4f}" if pm.weighted_score else "N/A"
            f.write(
                f"| {pm.policy_id} | {pm.min_words} | {pm.max_words} | {pm.silence_timer_ms} | {pm.max_wait_timeout_ms} | {ttff} | {chunk} | {score} |\n"
            )

        f.write("\n## Spam Sensitivity Analysis\n\n")
        f.write("Spam metrics recomputed for different α values. Includes:\n")
        f.write("- Selected policy\n")
        if top_5_by_utopia:
            f.write("- Top 5 policies by utopia distance\n")
        f.write("- Top 5 policies by weighted score\n\n")

        f.write(
            "| Policy ID | Rank (Utopia) | Utopia Dist | Rank (Weighted) | α=0.5 | α=0.6 | α=0.7 | α=0.8 |\n"
        )
        f.write(
            "|-----------|---------------|-------------|-----------------|-------|-------|-------|-------|\n"
        )

        # Sort policies: selected first, then by utopia rank, then by weighted rank
        def get_sort_key(policy_id: str) -> Tuple[int, int, int, str]:
            utopia_rank = (
                utopia_rankings.get(policy_id, (999, float("inf")))[0]
                if utopia_rankings
                else 999
            )
            weighted_rank = next(
                (i + 1 for i, pm in enumerate(top_policies) if pm.policy_id == policy_id), 999
            )
            is_selected = 0 if policy_id == selected_policy.policy_id else 1
            return (is_selected, utopia_rank, weighted_rank, policy_id)

        sorted_policy_ids = sorted(spam_sensitivity.keys(), key=get_sort_key)

        for policy_id in sorted_policy_ids:
            alpha_results = spam_sensitivity[policy_id]

            # Get rankings
            utopia_info = utopia_rankings.get(policy_id, (None, None)) if utopia_rankings else (
                None,
                None,
            )
            utopia_rank = f"{utopia_info[0]}" if utopia_info[0] is not None else "-"
            utopia_dist = f"{utopia_info[1]:.4f}" if utopia_info[1] is not None else "-"

            weighted_rank = next(
                (i + 1 for i, pm in enumerate(top_policies) if pm.policy_id == policy_id),
                None,
            )
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


def write_spam_sensitivity_json(output_path: Path, spam_sensitivity: dict) -> None:
    """Write spam sensitivity JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spam_sensitivity, f, indent=2)

