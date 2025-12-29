#!/usr/bin/env python3
"""
EXAID Evaluation - TokenGate Calibration CLI

CLI wrapper responsibilities: argument parsing, entrypoint wiring, and user-facing
logging, while reusable calibration logic lives in evals/src/* modules.

Usage:
    python -m evals.cli.calibrate_tokengate \
        --traces data/traces/ \
        --manifest data/manifests/exaid_traces_*.manifest.jsonl \
        --config configs/calibration_sweep.yaml \
        --output data/calibration/
"""

import argparse
from pathlib import Path

from ..src.tokengate_calibration_runner import run_calibration_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="EXAID TokenGate Calibration")
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Input traces directory",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Manifest file pattern (e.g., data/manifests/exaid_traces_*.manifest.jsonl)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Calibration sweep configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="Allow stub traces (for testing only)",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Verify determinism by running same policy twice",
    )

    args = parser.parse_args()

    run_calibration_sync(
        traces_dir=args.traces,
        manifest_pattern=args.manifest,
        config_path=args.config,
        output_root=args.output,
        allow_stub=args.allow_stub,
        verify_determinism=args.verify_determinism,
    )


if __name__ == "__main__":
    main()
