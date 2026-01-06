#!/usr/bin/env python3
"""Command-line entrypoint for timed trace generation."""

import argparse
import sys
from pathlib import Path

from ..src.traces.generation import run_generation


def main() -> int:
    print("EXAIM evaluation pipeline (legacy artifact namespace: exaid.*)")
    parser = argparse.ArgumentParser(
        description="Generate timed MAS traces from clinical cases using MAC",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mas_generation.yaml"),
        help="MAS generation configuration file",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/traces"),
        help="Output traces directory",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        default=Path("data/manifests"),
        help="Output manifests directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and case list without running MAC",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of cases to process",
    )
    parser.add_argument(
        "--stub-mode",
        action="store_true",
        help="Use stub traces instead of running MAC (for testing)",
    )

    args = parser.parse_args()
    return run_generation(args)


if __name__ == "__main__":
    sys.exit(main())
