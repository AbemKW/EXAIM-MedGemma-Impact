#!/usr/bin/env python3
"""
EXAID Evaluation - TokenGate Calibration Visualization CLI

Generate professional 3D Pareto frontier plot or heatmaps from calibration results.

Usage:
    python -m evals.cli.plot_calibration \
        --calibration-dir data/calibration/calib_sha256:c_6bea5ea6_a800a8de \
        --output data/metrics/figures/tokengate_pareto_frontier_3d.svg
    
    python -m evals.cli.plot_calibration \
        --calibration-dir data/calibration/calib_sha256:c_6bea5ea6_a800a8de \
        --heatmaps \
        --output-dir data/metrics/figures
"""

import argparse
import sys
from pathlib import Path

from ..src.tokengate_calibration.heatmap_visualization import (
    create_all_policies_visualization,
    create_metric_heatmaps,
    create_parameter_heatmap,
)
from ..src.tokengate_calibration.visualization import create_3d_pareto_plot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualizations from TokenGate calibration results"
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        required=True,
        help="Calibration output directory (contains calibration_results.csv and calibration_summary.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output SVG file path for 3D plot (default: data/metrics/figures/tokengate_pareto_frontier_3d.svg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for heatmaps (default: data/metrics/figures)",
    )
    parser.add_argument(
        "--heatmaps",
        action="store_true",
        help="Generate heatmap visualizations instead of 3D plot",
    )
    parser.add_argument(
        "--parameter-heatmap",
        action="store_true",
        help="Also generate parameter space heatmaps",
    )
    parser.add_argument(
        "--all-policies",
        action="store_true",
        help="Generate visualization showing all individual policies (parameter space + jittered metrics)",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=20.0,
        help="Elevation angle for 3D view (default: 20.0)",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=45.0,
        help="Azimuth angle for 3D view (default: 45.0)",
    )

    args = parser.parse_args()

    # Resolve relative paths relative to evals root
    evals_root = Path(__file__).resolve().parents[1]  # cli -> evals

    def resolve_path(path: Path) -> Path:
        """Resolve relative paths relative to evals root."""
        if path.is_absolute():
            return path
        # Try current directory first, then evals root
        current_dir_path = Path.cwd() / path
        if current_dir_path.exists():
            return current_dir_path.resolve()
        evals_path = evals_root / path
        if evals_path.exists():
            return evals_path.resolve()
        # Also try as string to handle special characters like colons
        path_str = str(path)
        if Path(path_str).exists():
            return Path(path_str).resolve()
        return Path(path_str).resolve()  # Return resolved path (let downstream handle error if doesn't exist)

    # Handle calibration directory resolution - may contain special characters
    calibration_dir_input = args.calibration_dir
    if calibration_dir_input.is_absolute():
        calibration_dir_resolved = calibration_dir_input
    else:
        # Try relative to current directory first
        current_dir_path = Path.cwd() / calibration_dir_input
        if current_dir_path.exists():
            calibration_dir_resolved = current_dir_path.resolve()
        else:
            # Try relative to evals root
            evals_path = evals_root / calibration_dir_input
            if evals_path.exists():
                calibration_dir_resolved = evals_path.resolve()
            else:
                # Last resort: try to find by listing parent directory and matching
                # This handles cases where directory name contains special characters
                parent_dir = evals_root / "data" / "calibration"
                if parent_dir.exists():
                    # Try to match the directory name
                    dir_name = str(calibration_dir_input).split("/")[-1].split("\\")[-1]
                    matching_dirs = [d for d in parent_dir.iterdir() if d.is_dir() and dir_name in d.name]
                    if matching_dirs:
                        calibration_dir_resolved = matching_dirs[0].resolve()
                    else:
                        calibration_dir_resolved = evals_path.resolve()
                else:
                    calibration_dir_resolved = evals_path.resolve()
    
    # Final check: if still doesn't exist, try listing parent and matching
    if not calibration_dir_resolved.exists():
        # Extract the directory name from the input
        dir_name_parts = str(calibration_dir_input).replace("\\", "/").split("/")
        dir_name = dir_name_parts[-1] if dir_name_parts else str(calibration_dir_input)
        
        # Try to find it in the calibration parent directory
        parent_dir = evals_root / "data" / "calibration"
        if parent_dir.exists():
            # List all directories and try to match
            for d in parent_dir.iterdir():
                if d.is_dir():
                    # Match if the directory name contains key parts (handles special characters)
                    if "calib_sha256" in d.name and dir_name.split("_")[-1] in d.name:
                        calibration_dir_resolved = d.resolve()
                        break
    
    args.calibration_dir = calibration_dir_resolved

    # Set default output path if not provided
    if args.output is None:
        args.output = evals_root / "data" / "metrics" / "figures" / "tokengate_pareto_frontier_3d.svg"
    else:
        args.output = resolve_path(args.output)

    # Validate calibration directory exists
    if not args.calibration_dir.exists():
        print(f"ERROR: Calibration directory not found: {args.calibration_dir}", file=sys.stderr)
        print(f"Attempted to resolve from: {calibration_dir_input}", file=sys.stderr)
        # Try to list available directories
        parent_dir = evals_root / "data" / "calibration"
        if parent_dir.exists():
            print(f"Available directories in {parent_dir}:", file=sys.stderr)
            for d in sorted(parent_dir.iterdir()):
                if d.is_dir():
                    print(f"  - {d.name}", file=sys.stderr)
        sys.exit(1)

    # Check for required files
    csv_file = args.calibration_dir / "calibration_results.csv"
    json_file = args.calibration_dir / "calibration_summary.json"
    
    if not csv_file.exists():
        print(f"ERROR: Calibration results CSV not found: {csv_file}", file=sys.stderr)
        sys.exit(1)
    
    if not json_file.exists():
        print(f"ERROR: Calibration summary JSON not found: {json_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading calibration data from: {args.calibration_dir}")

    try:
        if args.heatmaps:
            # Generate heatmaps
            if args.output_dir is None:
                args.output_dir = evals_root / "data" / "metrics" / "figures"
            else:
                args.output_dir = resolve_path(args.output_dir)
            
            print(f"Generating metric heatmaps...")
            create_metric_heatmaps(
                calibration_dir=args.calibration_dir,
                output_dir=args.output_dir,
            )
            
            if args.parameter_heatmap:
                print(f"Generating parameter heatmaps...")
                create_parameter_heatmap(
                    calibration_dir=args.calibration_dir,
                    output_dir=args.output_dir,
                    x_param="min_words",
                    y_param="max_words",
                    metric_to_color="chunk_size_p50",
                )
                create_parameter_heatmap(
                    calibration_dir=args.calibration_dir,
                    output_dir=args.output_dir,
                    x_param="silence_timer_ms",
                    y_param="max_wait_timeout_ms",
                    metric_to_color="ttff_content_p50_ms",
                )
            
            if args.all_policies:
                print(f"Generating all-policies visualization...")
                create_all_policies_visualization(
                    calibration_dir=args.calibration_dir,
                    output_dir=args.output_dir,
                )
        else:
            # Generate 3D plot
            print(f"Generating 3D Pareto plot...")
            print(f"View angles: elev={args.elev}, azim={args.azim}")
            
            create_3d_pareto_plot(
                calibration_dir=args.calibration_dir,
                output_path=args.output,
                elev=args.elev,
                azim=args.azim,
            )
            print(f"✓ Figure saved to: {args.output}")
    except Exception as e:
        print(f"ERROR: Failed to generate visualization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

