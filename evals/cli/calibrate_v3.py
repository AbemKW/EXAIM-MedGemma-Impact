#!/usr/bin/env python3
"""
EXAID Evaluation - V3 Chunk Size Calibration CLI

Compute the deterministic V3 chunk_size_ctu from V0 TokenGate regular flushes.
Excludes end-of-trace and calibration-only turn_end flushes.
"""

import argparse
from pathlib import Path

from evals.src.v3_calibration import (
    V3CalibrationInputs,
    compute_v3_chunk_size,
    write_v3_calibration_report,
)


def expand_v0_run_logs(paths: list[Path]) -> list[Path]:
    """
    Expand V0 run log paths, handling directories and files.
    
    If a path is a directory, finds all .jsonl.gz files in it.
    If a path is a file, uses it directly.
    Returns a sorted list of resolved paths.
    """
    expanded: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved.is_dir():
            # Find all .jsonl.gz files in directory
            log_files = sorted(resolved.glob("*.jsonl.gz"))
            if not log_files:
                raise ValueError(
                    f"No .jsonl.gz files found in directory: {resolved}"
                )
            expanded.extend(log_files)
        elif resolved.is_file():
            expanded.append(resolved)
        else:
            raise ValueError(f"Path does not exist: {resolved}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expanded = []
    for path in expanded:
        if path not in seen:
            seen.add(path)
            unique_expanded.append(path)
    
    return sorted(unique_expanded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute V3 chunk_size_ctu from V0 TokenGate flush logs."
    )
    parser.add_argument(
        "--case-list",
        type=Path,
        required=True,
        help="Path to frozen, ordered case list JSONL.",
    )
    parser.add_argument(
        "--v0-run-log",
        type=Path,
        action="append",
        required=True,
        help=(
            "Path to V0 run log JSONL file or directory containing V0 run logs. "
            "If a directory is provided, all .jsonl.gz files in it will be used. "
            "Can be specified multiple times for multiple files/directories."
        ),
    )
    parser.add_argument(
        "--subset-count",
        type=int,
        default=40,
        help="Number of cases to use from the ordered case list (default: 40).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for V3 calibration report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Expand directories to files
    v0_run_logs = expand_v0_run_logs(args.v0_run_log)
    
    if not v0_run_logs:
        raise ValueError("No V0 run log files found. Check --v0-run-log paths.")
    
    print(f"Found {len(v0_run_logs)} V0 run log file(s)")
    
    inputs = V3CalibrationInputs(
        case_list_path=args.case_list,
        v0_run_logs=v0_run_logs,
        subset_count=args.subset_count,
    )
    report = compute_v3_chunk_size(inputs)
    write_v3_calibration_report(args.output, report)
    print(f"Wrote V3 calibration report: {args.output}")
    print(f"chunk_size_ctu: {report['chunk_size_ctu']}")


if __name__ == "__main__":
    main()
