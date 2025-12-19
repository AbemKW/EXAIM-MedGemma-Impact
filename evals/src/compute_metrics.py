#!/usr/bin/env python3
"""
EXAID Evaluation - Metrics Computation Script

Computes evaluation metrics from run logs:
- Concept coverage at different token budgets
- Redundancy at different tau thresholds
- Generates figures for the conference paper

Key Constants (frozen for paper):
- tau_list = [0.85, 0.90, 0.95]  # Redundancy thresholds
- B = [250, 500, 1000, 2000]     # Token budgets

Output:
- data/metrics/*.jsonl           # Metric records
- data/metrics/figures/coverage_vs_budget.pdf

Usage:
    python compute_metrics.py --runs data/runs/ --output data/metrics/
    python compute_metrics.py --variant V3
"""

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import yaml

# Try to import matplotlib (may not be available in all environments)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# FROZEN CONSTANTS FOR CONFERENCE PAPER
# =============================================================================

# Redundancy detection thresholds (tau)
TAU_LIST = [0.85, 0.90, 0.95]

# Token budget levels (B)
BUDGET_LIST = [250, 500, 1000, 2000]

# =============================================================================


def load_metrics_config() -> dict:
    """Load metrics configuration."""
    config_path = Path("configs/metrics.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def generate_metric_id(variant_id: str) -> str:
    """Generate a unique metric ID."""
    date_str = datetime.now().strftime("%Y%m%d")
    return f"met-{variant_id.lower()}-{date_str}"


def open_run_file(file_path: Path):
    """Open a run file, handling gzip compression."""
    if str(file_path).endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8")
    return open(file_path, "r", encoding="utf-8")


def read_runs(runs_dir: Path, variant_id: Optional[str] = None) -> Iterator[dict]:
    """
    Read run records from directory.
    
    Args:
        runs_dir: Path to runs directory
        variant_id: Optional variant filter
        
    Yields:
        Run records
    """
    if not runs_dir.exists():
        print(f"Runs directory does not exist: {runs_dir}")
        return
    
    # Determine variant directories to process
    if variant_id:
        variant_dirs = [runs_dir / variant_id]
    else:
        variant_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    
    for variant_dir in sorted(variant_dirs):
        if not variant_dir.exists():
            continue
            
        patterns = ["*.jsonl", "*.jsonl.gz"]
        for pattern in patterns:
            for run_file in sorted(variant_dir.glob(pattern)):
                with open_run_file(run_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)


def compute_coverage_at_budget(run: dict, budget: int) -> float:
    """
    Compute concept coverage at a given token budget.
    
    TODO: Implement actual coverage computation
    
    Args:
        run: Run record
        budget: Token budget
        
    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    # STUB: Return placeholder coverage
    # Real implementation would:
    # 1. Select summaries up to token budget
    # 2. Extract concepts from selected summaries
    # 3. Compare to total concepts in trace
    # 4. Return coverage ratio
    
    total_tokens = sum(s.get("token_count", 0) for s in run.get("summaries", []))
    if total_tokens == 0:
        return 0.0
    
    # Placeholder: linear interpolation based on budget
    coverage = min(1.0, budget / max(total_tokens, 1))
    return coverage


def compute_redundancy_at_tau(run: dict, tau: float) -> float:
    """
    Compute redundancy at a given similarity threshold.
    
    TODO: Implement actual redundancy computation using semantic similarity
    
    Args:
        run: Run record
        tau: Similarity threshold
        
    Returns:
        Redundancy ratio (0.0 to 1.0)
    """
    # STUB: Return placeholder redundancy
    # Real implementation would:
    # 1. Compute pairwise semantic similarity between summaries
    # 2. Count pairs with similarity >= tau
    # 3. Return fraction of redundant pairs
    
    n_summaries = len(run.get("summaries", []))
    if n_summaries < 2:
        return 0.0
    
    # Placeholder: lower redundancy at higher thresholds
    redundancy = 0.2 * (1.0 - tau)
    return redundancy


def aggregate_metrics(runs: list[dict], variant_id: str) -> dict:
    """
    Aggregate metrics across all runs for a variant.
    
    Args:
        runs: List of run records
        variant_id: Variant identifier
        
    Returns:
        Metrics record conforming to exaid.metrics.schema.json
    """
    if not runs:
        return None
    
    # Compute coverage statistics
    coverage_values = [r.get("concept_coverage", {}).get("coverage_ratio", 0.0) for r in runs]
    
    coverage_stats = {
        "mean": float(np.mean(coverage_values)) if coverage_values else 0.0,
        "std": float(np.std(coverage_values)) if coverage_values else 0.0,
        "median": float(np.median(coverage_values)) if coverage_values else 0.0,
        "min": float(np.min(coverage_values)) if coverage_values else 0.0,
        "max": float(np.max(coverage_values)) if coverage_values else 0.0,
    }
    
    # Compute redundancy at each tau threshold
    redundancy_stats = {}
    for tau in TAU_LIST:
        redundancy_values = [compute_redundancy_at_tau(r, tau) for r in runs]
        key = f"tau_{tau:.2f}".replace(".", "_").replace("_0_", "_0.")
        # Fix key format for schema
        key = f"tau_{tau}"
        redundancy_stats[key] = {
            "mean": float(np.mean(redundancy_values)) if redundancy_values else 0.0,
            "std": float(np.std(redundancy_values)) if redundancy_values else 0.0,
        }
    
    # Compute coverage at each budget level
    budget_stats = {}
    for budget in BUDGET_LIST:
        coverage_at_budget = [compute_coverage_at_budget(r, budget) for r in runs]
        budget_stats[f"B_{budget}"] = {
            "coverage_mean": float(np.mean(coverage_at_budget)) if coverage_at_budget else 0.0,
            "coverage_std": float(np.std(coverage_at_budget)) if coverage_at_budget else 0.0,
        }
    
    # Compute timing statistics
    timing_values = [r.get("timing", {}).get("total_ms", 0) for r in runs]
    
    # Build metrics record
    metrics_record = {
        "schema_name": "exaid.metrics",
        "schema_version": "1.0.0",
        "metric_id": generate_metric_id(variant_id),
        "variant_id": variant_id,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(runs),
        "coverage": coverage_stats,
        "redundancy": redundancy_stats,
        "token_budget": budget_stats,
        "timing": {
            "mean_run_ms": float(np.mean(timing_values)) if timing_values else 0.0,
            "total_compute_ms": int(sum(timing_values)),
        }
    }
    
    return metrics_record


def write_metrics(metrics: dict, output_dir: Path) -> Path:
    """Write metrics record to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{metrics['variant_id'].lower()}_metrics.jsonl"
    output_path = output_dir / filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
    
    return output_path


def generate_coverage_figure(
    all_metrics: dict[str, dict],
    output_path: Path
) -> bool:
    """
    Generate coverage vs budget figure.
    
    TODO: Implement actual figure generation with proper styling
    
    Args:
        all_metrics: Dict mapping variant_id to metrics record
        output_path: Output file path
        
    Returns:
        True if figure was generated, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping figure generation")
        return False
    
    if not all_metrics:
        print("WARNING: No metrics to plot")
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Color palette (colorblind-friendly)
    colors = plt.cm.Set2.colors
    
    # Plot each variant
    for idx, (variant_id, metrics) in enumerate(sorted(all_metrics.items())):
        budget_data = metrics.get("token_budget", {})
        
        budgets = []
        coverages = []
        stds = []
        
        for budget in BUDGET_LIST:
            key = f"B_{budget}"
            if key in budget_data:
                budgets.append(budget)
                coverages.append(budget_data[key].get("coverage_mean", 0.0))
                stds.append(budget_data[key].get("coverage_std", 0.0))
        
        if budgets:
            color = colors[idx % len(colors)]
            ax.errorbar(
                budgets, coverages, yerr=stds,
                marker='o', label=variant_id, color=color,
                capsize=3, capthick=1, linewidth=2, markersize=8
            )
    
    # Styling
    ax.set_xlabel("Token Budget (B)", fontsize=12)
    ax.set_ylabel("Concept Coverage", fontsize=12)
    ax.set_title("Coverage vs Token Budget by Variant", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(BUDGET_LIST) * 1.1)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    
    print(f"Generated figure: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute EXAID evaluation metrics from run logs"
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("data/runs"),
        help="Input runs directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metrics"),
        help="Output metrics directory"
    )
    parser.add_argument(
        "--variant",
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Compute metrics for specific variant only"
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXAID Metrics Computation (STUB)")
    print("=" * 60)
    print()
    print("NOTE: This is a STUB implementation.")
    print("Actual metric computation requires:")
    print("- Semantic similarity computation")
    print("- scispaCy concept extraction")
    print()
    print(f"Frozen constants:")
    print(f"  tau_list = {TAU_LIST}")
    print(f"  budget_list = {BUDGET_LIST}")
    print()
    
    # Load configuration
    config = load_metrics_config()
    
    # Collect runs by variant
    variants = [args.variant] if args.variant else ["V0", "V1", "V2", "V3", "V4"]
    all_metrics = {}
    
    for variant_id in variants:
        print(f"Processing variant: {variant_id}")
        
        # Read runs for this variant
        runs = list(read_runs(args.runs, variant_id))
        print(f"  Found {len(runs)} runs")
        
        if not runs:
            print(f"  Skipping (no runs)")
            continue
        
        # Compute metrics
        metrics = aggregate_metrics(runs, variant_id)
        
        if metrics:
            # Write metrics
            output_path = write_metrics(metrics, args.output)
            print(f"  Written: {output_path}")
            all_metrics[variant_id] = metrics
    
    # Generate figures
    if not args.no_figures and all_metrics:
        print()
        print("Generating figures...")
        figures_dir = args.output / "figures"
        coverage_figure_path = figures_dir / "coverage_vs_budget.pdf"
        generate_coverage_figure(all_metrics, coverage_figure_path)
    
    print()
    print("=" * 60)
    print(f"COMPLETE: Computed metrics for {len(all_metrics)} variants")
    print("=" * 60)
    print()
    print("Output files:")
    print(f"  Metrics: {args.output}/*.jsonl")
    print(f"  Figures: {args.output}/figures/")
    print()
    print("TODO for full implementation:")
    print("- Integrate scispaCy for concept extraction")
    print("- Implement semantic similarity for redundancy")
    print("- Add statistical significance tests")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


