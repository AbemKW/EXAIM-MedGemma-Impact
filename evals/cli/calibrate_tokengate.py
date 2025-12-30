#!/usr/bin/env python3
"""
EXAID Evaluation - TokenGate Calibration CLI

CLI wrapper responsibilities: argument parsing, entrypoint wiring, and user-facing
logging, while reusable calibration logic lives in evals/src/tokengate_calibration/ package.

Usage:
    python -m evals.cli.calibrate_tokengate \
        --traces data/traces/ \
        --manifest data/manifests/exaid_traces_*.manifest.jsonl \
        --config configs/calibration_sweep.yaml \
        --output data/calibration/
"""

import argparse
from pathlib import Path

from ..src.tokengate_calibration.runner import run_calibration_sync


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
    
    # Resolve relative paths relative to evals root
    evals_root = Path(__file__).resolve().parents[1]  # cli -> evals
    
    def resolve_path(path: Path) -> Path:
        """Resolve relative paths relative to evals root."""
        if path.is_absolute():
            return path
        # Try current directory first, then evals root
        if path.exists():
            return path
        evals_path = evals_root / path
        if evals_path.exists():
            return evals_path
        return path  # Return as-is if neither exists (let downstream handle error)
    
    args.traces = resolve_path(args.traces)
    args.config = resolve_path(args.config)
    args.output = resolve_path(args.output)

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
