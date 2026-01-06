#!/usr/bin/env python3
"""Command-line entrypoint for deterministic variant replay."""

import argparse
import sys
from pathlib import Path

from ..src.variants.runner import run_variants


def main() -> int:
    print("EXAIM evaluation pipeline (legacy artifact namespace: exaid.*)")
    parser = argparse.ArgumentParser(
        description="EXAIM Deterministic Variant Replay Engine",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("data/traces"),
        help="Input traces directory (default: data/traces)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/runs"),
        help="Output runs directory (default: data/runs)",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs"),
        help="Configs directory (default: configs)",
    )
    parser.add_argument(
        "--variant",
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Run only specific variant (default: all)",
    )
    parser.add_argument(
        "--case",
        type=str,
        help="Run only specific case ID",
    )
    parser.add_argument(
        "--eval-run-id",
        type=str,
        help="Evaluation run batch ID (default: auto-generated)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Path to manifest file (for trace_dataset_hash computation). "
            "If not provided, will attempt to auto-detect."
        ),
    )

    args = parser.parse_args()
    return run_variants(args)


if __name__ == "__main__":
    sys.exit(main())
