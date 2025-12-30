#!/usr/bin/env python3

import json
from pathlib import Path

from evals.cli.error_analysis import (
    extract_case_details,
    find_case_file,
    load_per_case_metrics,
    make_excerpt,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_load_per_case_metrics_filters_records(tmp_path):
    metrics_path = tmp_path / "per_case.metrics.jsonl"
    records = [
        {"metrics_type": "per_case", "case_id": "case-a"},
        {"metrics_type": "aggregate", "variants": {}},
    ]
    write_jsonl(metrics_path, records)

    per_case = load_per_case_metrics(metrics_path)
    assert len(per_case) == 1
    assert per_case[0]["case_id"] == "case-a"


def test_find_case_file_prefers_existing(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    path = runs_dir / "case-1.jsonl"
    path.write_text("{}", encoding="utf-8")

    result = find_case_file(runs_dir, "case-1", None)
    assert result == path


def test_make_excerpt_truncates():
    text = " ".join(["token"] * 50)
    excerpt = make_excerpt(text, max_chars=30)
    assert excerpt["truncated"] is True
    assert excerpt["length"] > 30
    assert excerpt["excerpt"].endswith("...[truncated]")


def test_extract_case_details_missing_run_log(tmp_path):
    result = extract_case_details(
        case_id="case-missing",
        variant_id="V0",
        runs_dir=tmp_path,
        traces_dir=tmp_path,
        max_excerpt_chars=100,
    )
    assert result is None


def test_extract_case_details_handles_malformed_run_log(tmp_path):
    runs_dir = tmp_path / "runs" / "V0"
    runs_dir.mkdir(parents=True)
    run_log = runs_dir / "case-bad.jsonl"
    run_log.write_text("{not-json}\n", encoding="utf-8")

    result = extract_case_details(
        case_id="case-bad",
        variant_id="V0",
        runs_dir=tmp_path / "runs",
        traces_dir=tmp_path,
        max_excerpt_chars=100,
    )
    assert result is None


def test_extract_case_details_returns_payload(tmp_path):
    runs_dir = tmp_path / "runs" / "V0"
    traces_dir = tmp_path / "traces"
    runs_dir.mkdir(parents=True)
    traces_dir.mkdir()

    run_log = runs_dir / "case-123.jsonl"
    write_jsonl(run_log, [
        {"record_type": "run_meta"},
        {
            "record_type": "summary_event",
            "event_index": 0,
            "event_id": "case-123-V0-se-000",
            "start_seq": 0,
            "end_seq": 1,
            "schema_ok": True,
            "summary_semantics_text": "Summary text",
        },
        {
            "record_type": "buffer_decision",
            "decision_index": 0,
            "decision_id": "case-123-V0-bd-000",
            "start_seq": 0,
            "end_seq": 1,
            "decision": "summarize",
        },
    ])

    trace_file = traces_dir / "case-123.jsonl"
    write_jsonl(trace_file, [
        {
            "record_type": "stream_delta",
            "seq": 0,
            "delta_text": "Patient reports pain.",
            "agent_id": "doctor",
            "event_subtype": "message",
        },
        {
            "record_type": "stream_delta",
            "seq": 1,
            "delta_text": "Follow-up noted.",
            "agent_id": "doctor",
            "event_subtype": "message",
        },
    ])

    result = extract_case_details(
        case_id="case-123",
        variant_id="V0",
        runs_dir=tmp_path / "runs",
        traces_dir=tmp_path / "traces",
        max_excerpt_chars=200,
    )

    assert result is not None
    assert result["case_id"] == "case-123"
    assert len(result["trace_excerpts"]) == 2
    assert result["gate_decisions"][0]["decision"] == "summarize"
    assert result["summary_outputs"][0]["summary_semantics_text"] == "Summary text"
