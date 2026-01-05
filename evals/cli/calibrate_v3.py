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
        help="Path to V0 run log JSONL (repeatable).",
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
    inputs = V3CalibrationInputs(
        case_list_path=args.case_list,
        v0_run_logs=[path.resolve() for path in args.v0_run_log],
        subset_count=args.subset_count,
    )
    report = compute_v3_chunk_size(inputs)
    write_v3_calibration_report(args.output, report)
    print(f"Wrote V3 calibration report: {args.output}")
    print(f"chunk_size_ctu: {report['chunk_size_ctu']}")


if __name__ == "__main__":
    main()
