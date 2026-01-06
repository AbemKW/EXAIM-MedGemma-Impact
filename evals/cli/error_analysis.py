#!/usr/bin/env python3
"""
EXAIM Evaluation - Error Analysis Extracts

Extracts trace excerpts, gate decisions, and summary outputs for outlier cases.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from ..src.deterministic.io import read_run_log, write_jsonl_deterministic
from ..src.traces.trace_text import build_window_text, load_trace_chunks_for_case


def load_per_case_metrics(metrics_path: Path) -> List[dict]:
    records = []
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("metrics_type") == "per_case":
                records.append(record)
    return records


def find_case_file(base_dir: Path, case_id: str, variant_id: Optional[str]) -> Optional[Path]:
    if variant_id:
        base_dir = base_dir / variant_id
    candidates = [
        base_dir / f"{case_id}.jsonl.gz",
        base_dir / f"{case_id}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_excerpt(text: str, max_chars: int) -> dict:
    cleaned = " ".join(text.split())
    length = len(cleaned)
    if length <= max_chars:
        return {"excerpt": cleaned, "length": length, "truncated": False}
    return {
        "excerpt": cleaned[:max_chars].rstrip() + "...[truncated]",
        "length": length,
        "truncated": True,
    }


def extract_case_details(
    case_id: str,
    variant_id: str,
    runs_dir: Path,
    traces_dir: Path,
    max_excerpt_chars: int,
) -> Optional[dict]:
    run_log_path = find_case_file(runs_dir, case_id, variant_id)
    if run_log_path is None:
        print(f"WARNING: Missing run log for {case_id} ({variant_id})")
        return None
    try:
        records = read_run_log(run_log_path)
    except Exception as exc:
        print(f"WARNING: Failed to read run log for {case_id} ({variant_id}): {exc}")
        return None

    trace_file = find_case_file(traces_dir, case_id, None)
    trace_chunks = []
    if trace_file:
        try:
            trace_chunks = load_trace_chunks_for_case(trace_file)
        except Exception as exc:
            print(f"WARNING: Failed to load trace for {case_id}: {exc}")
            return None
    summary_events = []
    buffer_decisions = []
    for record in records:
        record_type = record.get("record_type")
        if record_type == "summary_event":
            summary_events.append(record)
        elif record_type == "buffer_decision":
            buffer_decisions.append(record)

    summary_events.sort(key=lambda e: e.get("event_index", 0))
    buffer_decisions.sort(key=lambda d: d.get("decision_index", 0))

    trace_excerpts = []
    for event in summary_events:
        window_text = ""
        if trace_chunks:
            window_text = build_window_text(
                trace_chunks,
                event.get("start_seq", 0),
                event.get("end_seq", 0),
            )
        excerpt_info = make_excerpt(window_text, max_excerpt_chars)
        trace_excerpts.append({
            "source": "summary_event",
            "event_id": event.get("event_id"),
            "event_index": event.get("event_index"),
            "start_seq": event.get("start_seq"),
            "end_seq": event.get("end_seq"),
            "window_text_excerpt": excerpt_info["excerpt"],
            "window_text_length": excerpt_info["length"],
            "window_text_truncated": excerpt_info["truncated"],
        })

    for decision in buffer_decisions:
        window_text = ""
        if trace_chunks:
            window_text = build_window_text(
                trace_chunks,
                decision.get("start_seq", 0),
                decision.get("end_seq", 0),
            )
        excerpt_info = make_excerpt(window_text, max_excerpt_chars)
        trace_excerpts.append({
            "source": "buffer_decision",
            "decision_id": decision.get("decision_id"),
            "decision_index": decision.get("decision_index"),
            "start_seq": decision.get("start_seq"),
            "end_seq": decision.get("end_seq"),
            "window_text_excerpt": excerpt_info["excerpt"],
            "window_text_length": excerpt_info["length"],
            "window_text_truncated": excerpt_info["truncated"],
        })

    gate_decisions = [
        {
            "decision_id": decision.get("decision_id"),
            "decision_index": decision.get("decision_index"),
            "timestamp": decision.get("timestamp"),
            "start_seq": decision.get("start_seq"),
            "end_seq": decision.get("end_seq"),
            "input_ctu": decision.get("input_ctu"),
            "decision": decision.get("decision"),
            "filter_results": decision.get("filter_results"),
            "latency_ms": decision.get("latency_ms"),
        }
        for decision in buffer_decisions
    ]

    summary_outputs = [
        {
            "event_id": event.get("event_id"),
            "event_index": event.get("event_index"),
            "timestamp": event.get("timestamp"),
            "start_seq": event.get("start_seq"),
            "end_seq": event.get("end_seq"),
            "schema_ok": event.get("schema_ok"),
            "schema_error": event.get("schema_error"),
            "trigger_type": event.get("trigger_type"),
            "limits_ok": event.get("limits_ok"),
            "failure_mode": event.get("failure_mode"),
            "latency_ms": event.get("latency_ms"),
            "summary_ctu": event.get("summary_ctu"),
            "summary_semantics_text": event.get("summary_semantics_text"),
            "summary_content": event.get("summary_content"),
        }
        for event in summary_events
    ]

    return {
        "case_id": case_id,
        "variant_id": variant_id,
        "trace_excerpts": trace_excerpts,
        "gate_decisions": gate_decisions,
        "summary_outputs": summary_outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EXAID Error Analysis Extracts"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("data/metrics/per_case.metrics.jsonl"),
        help="Per-case metrics JSONL path"
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
        default=Path("data/metrics/error_analysis"),
        help="Output directory for error analysis extracts"
    )
    parser.add_argument(
        "--max-excerpt-chars",
        type=int,
        default=500,
        help="Maximum characters for trace excerpts"
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"ERROR: Metrics file not found: {args.metrics}")
        return 1

    per_case_records = load_per_case_metrics(args.metrics)
    flagged_records = [
        record for record in per_case_records
        if record.get("outlier_flags")
    ]
    print(
        f"Found {len(flagged_records)} flagged records out of "
        f"{len(per_case_records)} total cases."
    )

    extracts = []
    for record in flagged_records:
        case_id = record.get("case_id")
        variant_id = record.get("variant_id")
        if not case_id or not variant_id:
            continue
        detail = extract_case_details(
            case_id,
            variant_id,
            args.runs,
            args.traces,
            args.max_excerpt_chars,
        )
        if detail:
            detail["outlier_flags"] = record.get("outlier_flags", [])
            detail["outlier_latency_spike"] = record.get("outlier_latency_spike", False)
            detail["outlier_excessive_flushes"] = record.get(
                "outlier_excessive_flushes",
                False,
            )
            detail["outlier_low_coverage"] = record.get("outlier_low_coverage", False)
            extracts.append(detail)

    output_path = args.output / "error_analysis.jsonl"
    write_jsonl_deterministic(extracts, output_path)
    print(f"Wrote {len(extracts)} records to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
