#!/usr/bin/env python3
"""
V3 calibration helpers.

Derives the fixed V3 chunk size from V0 TokenGate regular flushes using a
deterministic calibration subset, then stores the frozen scalar in a report.
End-of-trace and calibration-only turn_end flushes are excluded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import os
import warnings
from statistics import median
from typing import Iterable

from .config.config_loader import get_configs_dir, get_evals_root, load_variant_config
from .deterministic.io import compute_file_hash
from .tokengate_calibration.io import get_exaid_commit
from .utils.hashing import compute_tokengate_config_hash


@dataclass(frozen=True)
class V3CalibrationInputs:
    case_list_path: Path
    v0_run_logs: list[Path]
    subset_count: int = 40
    excluded_flush_reasons: tuple[str, ...] = ("end_of_trace", "turn_end")


def read_case_list(case_list_path: Path) -> list[str]:
    """Read ordered case IDs from a case list JSONL manifest."""
    case_ids: list[str] = []
    with case_list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            case_id = record.get("case_id")
            if not case_id:
                raise ValueError(f"Missing case_id in {case_list_path}: {line}")
            case_ids.append(case_id)
    return case_ids


def collect_v0_flush_ctu(
    v0_run_logs: Iterable[Path],
    subset_case_ids: set[str],
    excluded_flush_reasons: tuple[str, ...],
) -> dict[str, list[int]]:
    """
    Collect deduplicated V0 TokenGate flush CTU values per case.

    Scans V0 run logs for tokengate_flush records that match the requested
    subset_case_ids and are not excluded by trigger_reason. Duplicate flushes
    are detected and skipped using (case_id, start_seq, end_seq, accumulated_ctu,
    trigger_reason, text_hash) as the composite key.
    """
    per_case: dict[str, list[int]] = {case_id: [] for case_id in subset_case_ids}
    seen_flushes: set[tuple[str, int, int, int, str, str]] = set()
    for run_log in v0_run_logs:
        with run_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("record_type") != "tokengate_flush":
                    continue
                if record.get("variant_id") != "V0":
                    continue
                case_id = record.get("case_id")
                if case_id not in subset_case_ids:
                    continue
                trigger_reason = record.get("trigger_reason")
                if trigger_reason in excluded_flush_reasons:
                    continue
                accumulated_ctu = record.get("accumulated_ctu")
                if accumulated_ctu is None:
                    raise ValueError(
                        "Missing accumulated_ctu in V0 tokengate_flush record: "
                        f"case_id={record.get('case_id')} start_seq={record.get('start_seq')} "
                        f"end_seq={record.get('end_seq')}"
                    )
                if accumulated_ctu <= 0:
                    raise ValueError(
                        "Non-positive accumulated_ctu in V0 tokengate_flush record: "
                        f"case_id={record.get('case_id')} accumulated_ctu={accumulated_ctu}"
                    )
                start_seq = record.get("start_seq")
                end_seq = record.get("end_seq")
                text_hash = record.get("text_hash") or ""
                if start_seq is None or end_seq is None:
                    raise ValueError(
                        "Missing seq bounds in V0 tokengate_flush record: "
                        f"case_id={record.get('case_id')} accumulated_ctu={accumulated_ctu}"
                    )
                dedupe_key = (
                    case_id,
                    int(start_seq),
                    int(end_seq),
                    int(accumulated_ctu),
                    trigger_reason or "",
                    text_hash,
                )
                if dedupe_key in seen_flushes:
                    continue
                seen_flushes.add(dedupe_key)
                per_case[case_id].append(int(accumulated_ctu))
    return per_case


def _extract_v0_run_meta(run_log: Path) -> dict[str, str] | None:
    """Return the first V0 run_meta record from a run log, if present."""
    with run_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_type") != "run_meta":
                continue
            if record.get("variant_id") != "V0":
                continue
            return {
                "trace_dataset_hash": record.get("trace_dataset_hash", ""),
                "tokengate_config_hash": record.get("tokengate_config_hash", ""),
            }
    return None


def compute_v3_chunk_size(inputs: V3CalibrationInputs) -> dict:
    """
    Compute deterministic V3 chunk size and return a calibration report payload.

    The report includes chunk_size_ctu, per-case medians, overall median,
    and provenance fields (trace_dataset_hash, tokengate_config_hash,
    exaid_commit, run log hashes). V0 run_meta consistency is validated
    across all provided run logs.
    """
    case_ids = read_case_list(inputs.case_list_path)
    subset_count = min(inputs.subset_count, len(case_ids))
    subset_case_ids = case_ids[:subset_count]
    if not subset_case_ids:
        raise ValueError("Calibration subset is empty; check case list.")

    run_meta: dict[str, str] = {}
    run_log_hashes: dict[str, str] = {}
    for run_log in inputs.v0_run_logs:
        run_log_hashes[str(run_log)] = compute_file_hash(run_log)
        meta = _extract_v0_run_meta(run_log)
        # Capture the first available run_meta, but continue hashing all run logs.
        if meta and not run_meta:
            run_meta = meta

    if not run_meta:
        raise ValueError("Unable to locate V0 run_meta in provided run logs.")

    for run_log in inputs.v0_run_logs:
        meta = _extract_v0_run_meta(run_log)
        if not meta:
            raise ValueError(f"Missing V0 run_meta record in {run_log}")
        if meta.get("trace_dataset_hash") != run_meta["trace_dataset_hash"]:
            raise ValueError(
                "trace_dataset_hash mismatch across V0 run logs: "
                f"{meta.get('trace_dataset_hash')} != {run_meta['trace_dataset_hash']}"
            )
        if meta.get("tokengate_config_hash") != run_meta["tokengate_config_hash"]:
            raise ValueError(
                "tokengate_config_hash mismatch across V0 run logs: "
                f"{meta.get('tokengate_config_hash')} != {run_meta['tokengate_config_hash']}"
            )

    per_case_ctu = collect_v0_flush_ctu(
        inputs.v0_run_logs,
        set(subset_case_ids),
        inputs.excluded_flush_reasons,
    )

    missing_cases = [case_id for case_id in subset_case_ids if not per_case_ctu[case_id]]
    if missing_cases:
        raise ValueError(
            "Missing V0 TokenGate flush data for calibration cases: "
            + ", ".join(missing_cases)
        )

    per_case_medians = {
        case_id: median(per_case_ctu[case_id]) for case_id in subset_case_ids
    }
    overall_median = median(list(per_case_medians.values()))
    chunk_size_ctu = int(math.ceil(overall_median))

    return {
        "chunk_size_ctu": chunk_size_ctu,
        "calibration_subset": {
            "strategy": "first_n_cases",
            "count": subset_count,
            "case_list_path": str(inputs.case_list_path),
            "case_ids": subset_case_ids,
        },
        "per_case_median_ctu": per_case_medians,
        "overall_median_ctu": overall_median,
        "excluded_flush_reasons": list(inputs.excluded_flush_reasons),
        "v0_run_logs": [str(path) for path in inputs.v0_run_logs],
        "v0_run_log_hashes": run_log_hashes,
        "trace_dataset_hash": run_meta["trace_dataset_hash"],
        "tokengate_config_hash": run_meta["tokengate_config_hash"],
        "exaid_commit": get_exaid_commit(get_evals_root().parent),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_v3_calibration_report(report_path: Path, report: dict) -> None:
    """Write V3 calibration report JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_v3_calibration_report(report_path: Path) -> dict:
    """Load V3 calibration report JSON."""
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_v3_chunk_size(
    config: dict,
    trace_dataset_hash: str | None = None,
    configs_dir: Path | None = None,
) -> int:
    """
    Resolve the V3 chunk size from configuration or a calibration report.

    Resolution order:
    1) If fixed_trigger.chunk_size_ctu is set, return it (int) and skip report.
    2) Otherwise, load v3_calibration.calibration_report and validate:
       - trace_dataset_hash matches (if provided and not "unknown")
       - tokengate_config_hash matches the active configs dir
       - exaid_commit matches (unless EXAID_ALLOW_COMMIT_MISMATCH=1)
    """
    fixed_trigger = config.get("fixed_trigger", {})
    chunk_size_ctu = fixed_trigger.get("chunk_size_ctu")
    if chunk_size_ctu is not None:
        return int(chunk_size_ctu)

    v3_calibration = config.get("v3_calibration", {})
    report_path = v3_calibration.get("calibration_report")
    if not report_path:
        raise ValueError(
            "V3 chunk_size_ctu is not set and no calibration_report is configured."
        )

    evals_root = get_evals_root()
    resolved_path = (evals_root / report_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"V3 calibration report not found: {resolved_path}. "
            "Run evals.cli.calibrate_v3 to generate it."
        )
    report = load_v3_calibration_report(resolved_path)
    report_value = report.get("chunk_size_ctu")
    if report_value is None:
        raise ValueError(f"Missing chunk_size_ctu in V3 calibration report: {resolved_path}")
    if int(report_value) <= 0:
        raise ValueError(
            f"Invalid chunk_size_ctu in V3 calibration report: {report_value}"
        )
    report_trace_hash = report.get("trace_dataset_hash")
    if (
        trace_dataset_hash
        and trace_dataset_hash != "unknown"
        and report_trace_hash
        and trace_dataset_hash != report_trace_hash
    ):
        raise ValueError(
            f"trace_dataset_hash mismatch for V3 calibration: {trace_dataset_hash} != {report_trace_hash}"
        )
    tokengate_hash = report.get("tokengate_config_hash")
    if tokengate_hash:
        expected_tokengate_hash = compute_tokengate_config_hash(
            load_variant_config("V0", configs_dir or get_configs_dir())
        )
        if tokengate_hash != expected_tokengate_hash:
            raise ValueError(
                "tokengate_config_hash mismatch for V3 calibration: "
                f"{tokengate_hash} != {expected_tokengate_hash}"
            )
    report_commit = report.get("exaid_commit")
    if report_commit:
        current_commit = get_exaid_commit(get_evals_root().parent)
        if report_commit != current_commit:
            if os.getenv("EXAID_ALLOW_COMMIT_MISMATCH", "").lower() in {"1", "true", "yes"}:
                warnings.warn(
                    "EXAID commit mismatch for V3 calibration: "
                    f"{report_commit} != {current_commit}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "EXAID commit mismatch for V3 calibration: "
                    f"{report_commit} != {current_commit}. "
                    "Set EXAID_ALLOW_COMMIT_MISMATCH=1 to override."
                )
    return int(report_value)
