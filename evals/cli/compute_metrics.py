#!/usr/bin/env python3
"""Command-line entrypoint for metrics computation."""

import argparse
import sys
from pathlib import Path

from ..src.metrics.runner import run_metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EXAID Metrics Computation",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/runs"),
        help="Run logs directory",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("data/traces"),
        help="Traces directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metrics"),
        help="Output metrics directory",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs"),
        help="Configs directory",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        default=Path("data/manifests"),
        help="Manifests directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Explicit manifest path for dataset integrity checks",
    )
    parser.add_argument(
        "--variant",
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Compute for specific variant only",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of bootstrap samples for CI",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    return run_metrics(args)


if __name__ == "__main__":
    sys.exit(main())
