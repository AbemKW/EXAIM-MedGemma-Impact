#!/usr/bin/env python3

import json
import warnings

import pytest

from evals.src.metrics import (
    compute_distribution,
    compute_flush_statistics,
    compute_manifest_hash,
    compute_overhead_attribution,
    compute_virtual_time_throughput,
    load_manifest_provenance,
)


def test_compute_distribution_empty():
    assert compute_distribution([]) is None


def test_compute_distribution_single_value():
    result = compute_distribution([42.0])
    assert result["min"] == 42.0
    assert result["max"] == 42.0
    assert result["mean"] == 42.0
    assert result["p99"] == 42.0


def test_compute_distribution_small_values_within_range():
    values = [1.0, 2.0, 3.0]
    result = compute_distribution(values)
    assert result["min"] == 1.0
    assert result["max"] == 3.0
    assert result["p90"] <= result["max"]
    assert result["p90"] >= result["min"]


def test_compute_flush_statistics_non_monotonic_warns():
    flushes = [
        {"flush_index": 0, "timestamp": "2024-01-01T00:00:01Z", "accumulated_ctu": 10},
        {"flush_index": 1, "timestamp": "2024-01-01T00:00:00Z", "accumulated_ctu": 12},
    ]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        stats = compute_flush_statistics(flushes)
        assert any(str(w.message) == "Flush timestamps are not monotonic" for w in captured)
    assert stats["flush_count"] == 2


def test_compute_virtual_time_throughput_zero_duration():
    trace_chunks = [
        {"t_rel_ms": 100},
        {"t_rel_ms": 100},
    ]
    result = compute_virtual_time_throughput(trace_chunks, trace_ctu=20, summary_ctu=5)
    assert result["virtual_time_duration_ms"] == 0
    assert result["trace_ctu_per_s"] is None
    assert result["summary_ctu_per_s"] is None


def test_compute_overhead_attribution_mapping():
    result = compute_overhead_attribution([10.0, 20.0], [5.0])
    assert result["content_plane_latency_ms"] == 30.0
    assert result["control_plane_latency_ms"] == 5.0


def test_load_manifest_provenance_valid(tmp_path):
    manifest_path = tmp_path / "dataset.manifest.jsonl"
    trace_entries = [("case-1", "sha256:aaa"), ("case-2", "sha256:bbb")]
    manifest_hash = compute_manifest_hash("mas_run_1", "sha256:cases", trace_entries)
    records = [
        {"record_type": "manifest_meta", "schema_version": "2.0.0", "dataset_id": "ds-1", "mas_run_id": "mas_run_1"},
        {
            "record_type": "provenance",
            "mac_commit": "abc",
            "case_list_hash": "sha256:cases",
            "trace_dataset_hash": manifest_hash,
            "config_hash": "sha256:cfg",
        },
    ]
    for case_id, sha in trace_entries:
        records.append({"record_type": "trace_entry", "case_id": case_id, "sha256": sha})
    records.append({"record_type": "summary", "total_cases": 2})
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    info = load_manifest_provenance(manifest_path)
    assert info["manifest_hash_valid"] is True
    assert info["dataset_id"] == "ds-1"
