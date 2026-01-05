"""Visualization functions for TokenGate calibration results."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from .models import PolicyMetrics
from .selection import build_pareto_frontier_3d, normalize_to_goodness


def load_calibration_data(calibration_dir: Path) -> Tuple[pd.DataFrame, dict]:
    """
    Load calibration results CSV and summary JSON from calibration output directory.
    
    Args:
        calibration_dir: Path to calibration output directory
        
    Returns:
        Tuple of (DataFrame with policy metrics, summary dict)
    """
    csv_path = calibration_dir / "calibration_results.csv"
    json_path = calibration_dir / "calibration_summary.json"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Calibration results CSV not found: {csv_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Calibration summary JSON not found: {json_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Load summary JSON
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    return df, summary


def classify_policies(df: pd.DataFrame, summary: dict) -> Dict[str, List[str]]:
    """
    Classify policies into rejected, survivors, pareto, and selected.
    
    Args:
        df: DataFrame with policy metrics
        summary: Summary dict from calibration_summary.json
        
    Returns:
        Dict with keys: 'rejected', 'survivors', 'pareto', 'selected'
        Values are lists of policy_id strings
    """
    # Rejected: policies with constraint violations
    rejected = df[df["constraint_violations"].notna() & (df["constraint_violations"] != "")]["policy_id"].tolist()
    
    # Survivors: policies without violations
    survivors = df[df["constraint_violations"].isna() | (df["constraint_violations"] == "")]["policy_id"].tolist()
    
    # Selected: from summary JSON
    selected_id = summary.get("selection", {}).get("selected_policy_id")
    selected = [selected_id] if selected_id and selected_id in df["policy_id"].values else []
    
    # Pareto frontier will be computed separately (needs normalization)
    pareto = []  # Will be populated later
    
    return {
        "rejected": rejected,
        "survivors": survivors,
        "pareto": pareto,
        "selected": selected,
    }


def normalize_to_goodness_space(
    df: pd.DataFrame, bounds: dict, dropped_metrics: List[str]
) -> pd.DataFrame:
    """
    Normalize metrics to [0,1] goodness space.
    
    Args:
        df: DataFrame with policy metrics
        bounds: Normalization bounds dict from summary
        dropped_metrics: List of dropped metric names
        
    Returns:
        DataFrame with added columns: ttff_goodness, flush_count_goodness, chunk_size_goodness
    """
    df = df.copy()
    
    # Metric order: [TTFF, flush_count, chunk_size]
    metric_order = ["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]
    metric_lower_is_better = [True, True, False]  # TTFF and flush_count: lower is better; chunk_size: higher is better
    
    goodness_cols = []
    for i, metric_name in enumerate(metric_order):
        if metric_name in dropped_metrics:
            # Skip dropped metrics
            continue
            
        goodness_col = f"{metric_name}_goodness"
        goodness_cols.append(goodness_col)
        
        # Get bounds
        if metric_name == "ttff_content_p50_ms":
            lo = bounds.get(metric_name, {}).get("lo_ms")
            hi = bounds.get(metric_name, {}).get("hi_ms")
        else:
            lo = bounds.get(metric_name, {}).get("lo")
            hi = bounds.get(metric_name, {}).get("hi")
        
        if lo is None or hi is None:
            # Missing bounds - skip
            df[goodness_col] = np.nan
            continue
        
        # Normalize to goodness space
        values = df[metric_name].values
        goodness_values = [
            normalize_to_goodness(val, lo, hi, metric_lower_is_better[i])
            if pd.notna(val) else np.nan
            for val in values
        ]
        df[goodness_col] = goodness_values
    
    return df


def compute_pareto_frontier_from_df(
    df: pd.DataFrame,
    bounds: dict,
    dropped_metrics: List[str],
    survivor_ids: List[str],
) -> List[str]:
    """
    Compute Pareto frontier from DataFrame.
    
    Args:
        df: DataFrame with policy metrics and goodness columns
        bounds: Normalization bounds dict
        dropped_metrics: List of dropped metric names
        survivor_ids: List of survivor policy IDs
        
    Returns:
        List of policy IDs in Pareto frontier
    """
    # Filter to survivors only
    survivor_df = df[df["policy_id"].isin(survivor_ids)].copy()
    
    if len(survivor_df) == 0:
        return []
    
    # Metric order: [TTFF, flush_count, chunk_size]
    metric_order = ["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]
    
    # Determine active dimensions (exclude dropped metrics)
    active_dimensions = [
        i for i, metric_name in enumerate(metric_order) if metric_name not in dropped_metrics
    ]
    
    if not active_dimensions:
        # All metrics dropped - return empty frontier
        return []
    
    # Build points list: (policy_metrics, goodness_vector)
    # Build compressed vectors (only active dimensions), matching selection.py pattern
    points = []
    for _, row in survivor_df.iterrows():
        # Create PolicyMetrics object for compatibility
        pm = PolicyMetrics(
            policy_id=row["policy_id"],
            min_words=int(row["min_words"]),
            max_words=int(row["max_words"]),
            silence_timer_ms=int(row["silence_timer_ms"]),
            max_wait_timeout_ms=int(row["max_wait_timeout_ms"]),
            ttff_content_p50_ms=row.get("ttff_content_p50_ms"),
            flush_count_mean=row.get("flush_count_mean"),
            chunk_size_p50=row.get("chunk_size_p50"),
        )
        
        # Build compressed goodness vector (only active dimensions)
        goodness_vector = []
        missing_metric = False
        for i in active_dimensions:
            metric_name = metric_order[i]
            goodness_col = f"{metric_name}_goodness"
            if goodness_col not in row or pd.isna(row[goodness_col]):
                missing_metric = True
                break
            goodness_vector.append(row[goodness_col])
        
        if not missing_metric:
            points.append((pm, goodness_vector))
    
    if not points:
        return []
    
    # After compressing vectors to active dimensions, remap active_dimensions to [0, 1, ..., k-1]
    # This matches the pattern in selection.py
    compressed_active_dimensions = list(range(len(points[0][1])))
    
    # Compute Pareto frontier
    frontier = build_pareto_frontier_3d(points, compressed_active_dimensions)
    
    # Extract policy IDs
    pareto_ids = [pm.policy_id for pm, _ in frontier]
    
    return pareto_ids


def create_3d_pareto_plot(
    calibration_dir: Path,
    output_path: Path,
    elev: float = 20,
    azim: float = 45,
) -> None:
    """
    Create professional 3D Pareto frontier plot following tutorial principles.
    
    Args:
        calibration_dir: Path to calibration output directory
        output_path: Path to save SVG figure
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
    """
    # Load data
    df, summary = load_calibration_data(calibration_dir)
    
    print(f"Total policies loaded: {len(df)}")
    
    # Get normalization bounds and dropped metrics
    bounds = summary.get("normalization_bounds", {})
    dropped_metrics = summary.get("dropped_metrics", [])
    
    # Normalize to goodness space
    df = normalize_to_goodness_space(df, bounds, dropped_metrics)
    
    # Classify policies
    classifications = classify_policies(df, summary)
    print(f"Rejected policies: {len(classifications['rejected'])}")
    print(f"Survivor policies: {len(classifications['survivors'])}")
    
    # Compute Pareto frontier
    pareto_ids = compute_pareto_frontier_from_df(
        df, bounds, dropped_metrics, classifications["survivors"]
    )
    classifications["pareto"] = pareto_ids
    print(f"Pareto frontier policies: {len(pareto_ids)}")
    
    # Check for duplicate coordinates (overlapping points) before plotting
    df_plot_all = df[
        df[["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]].notna().all(axis=1)
    ].copy()
    if len(df_plot_all) > 0:
        coords = df_plot_all[["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]]
        duplicates = coords.duplicated().sum()
        unique_coords = len(coords.drop_duplicates())
        print(f"Policies with duplicate coordinates (overlapping): {duplicates}")
        print(f"Unique coordinate combinations: {unique_coords} out of {len(coords)} total")
        
        # Show metric ranges
        print(f"Metric ranges:")
        print(f"  TTFF: {df_plot_all['ttff_content_p50_ms'].min():.1f} to {df_plot_all['ttff_content_p50_ms'].max():.1f} ms")
        print(f"  Flush Count: {df_plot_all['flush_count_mean'].min():.1f} to {df_plot_all['flush_count_mean'].max():.1f}")
        print(f"  Chunk Size: {df_plot_all['chunk_size_p50'].min():.1f} to {df_plot_all['chunk_size_p50'].max():.1f} words")
    
    # Get selected policy info
    selected_id = summary.get("selection", {}).get("selected_policy_id")
    selection_metadata = summary.get("selection", {}).get("metadata", {})
    utopia_distance = selection_metadata.get("utopia_distance")
    
    # Create figure with professional settings
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    
    # Set font to Times New Roman, 20pt
    font_props = {
        "family": "Times New Roman",
        "size": 20,
    }
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    
    # Color palette (minimal, professional)
    colors = {
        "rejected": "#CCCCCC",  # Light gray
        "survivors": "#888888",  # Medium gray
        "pareto": "#008080",  # Teal
        "selected": "#DC143C",  # Crimson
    }
    
    alphas = {
        "rejected": 0.3,
        "survivors": 0.5,
        "pareto": 0.7,
        "selected": 1.0,
    }
    
    marker_sizes = {
        "rejected": 30,  # Increased from 20 for better visibility
        "survivors": 50,  # Increased from 40
        "pareto": 80,  # Increased from 60
        "selected": 250,  # Increased from 200
    }
    
    zorders = {
        "rejected": 1,
        "survivors": 2,
        "pareto": 3,
        "selected": 5,
    }
    
    # Plot in order (bottom to top)
    # 1. Rejected policies (filter out NaN values for plotting)
    rejected_df = df[df["policy_id"].isin(classifications["rejected"])].copy()
    # Filter out rows with any NaN in the three metrics
    rejected_df_plot = rejected_df[
        rejected_df[["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]].notna().all(axis=1)
    ]
    if len(rejected_df_plot) > 0:
        ax.scatter(
            rejected_df_plot["ttff_content_p50_ms"],
            rejected_df_plot["flush_count_mean"],
            rejected_df_plot["chunk_size_p50"],
            c=colors["rejected"],
            alpha=alphas["rejected"],
            s=marker_sizes["rejected"],
            zorder=zorders["rejected"],
            label=f"Rejected Policies ({len(rejected_df_plot)}/{len(rejected_df)})",
        )
    
    # 2. Survivors (non-Pareto)
    survivors_non_pareto = [
        pid for pid in classifications["survivors"] if pid not in pareto_ids
    ]
    survivors_df = df[df["policy_id"].isin(survivors_non_pareto)].copy()
    # Filter out rows with any NaN in the three metrics
    survivors_df_plot = survivors_df[
        survivors_df[["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]].notna().all(axis=1)
    ]
    if len(survivors_df_plot) > 0:
        ax.scatter(
            survivors_df_plot["ttff_content_p50_ms"],
            survivors_df_plot["flush_count_mean"],
            survivors_df_plot["chunk_size_p50"],
            c=colors["survivors"],
            alpha=alphas["survivors"],
            s=marker_sizes["survivors"],
            zorder=zorders["survivors"],
            label=f"Survivors ({len(survivors_df_plot)}/{len(survivors_df)})",
        )
    
    # 3. Pareto frontier
    pareto_df = df[df["policy_id"].isin(pareto_ids)].copy()
    # Filter out rows with any NaN in the three metrics
    pareto_df_plot = pareto_df[
        pareto_df[["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]].notna().all(axis=1)
    ]
    if len(pareto_df_plot) > 0:
        ax.scatter(
            pareto_df_plot["ttff_content_p50_ms"],
            pareto_df_plot["flush_count_mean"],
            pareto_df_plot["chunk_size_p50"],
            c=colors["pareto"],
            alpha=alphas["pareto"],
            s=marker_sizes["pareto"],
            zorder=zorders["pareto"],
            label=f"Pareto Frontier ({len(pareto_df_plot)}/{len(pareto_df)})",
        )
    
    # 4. Selected policy
    if selected_id and selected_id in df["policy_id"].values:
        selected_df = df[df["policy_id"] == selected_id]
        if len(selected_df) > 0:
            row = selected_df.iloc[0]
            ax.scatter(
                [row["ttff_content_p50_ms"]],
                [row["flush_count_mean"]],
                [row["chunk_size_p50"]],
                c=colors["selected"],
                alpha=alphas["selected"],
                s=marker_sizes["selected"],
                zorder=zorders["selected"],
                edgecolors="black",
                linewidths=2,
                label="Selected Policy",
            )
            
            # Add annotation
            if utopia_distance is not None:
                annotation_text = f"Selected Policy\n(Utopia Distance: {utopia_distance:.4f})"
            else:
                annotation_text = "Selected Policy"
            
            ax.text(
                row["ttff_content_p50_ms"],
                row["flush_count_mean"],
                row["chunk_size_p50"],
                annotation_text,
                fontsize=18,
                color=colors["selected"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=colors["selected"]),
            )
    
    # Set axis labels
    ax.set_xlabel("Time to First Flush (ms)", **font_props)
    ax.set_ylabel("Flush Count", **font_props)
    ax.set_zlabel("Chunk Size (words)", **font_props)
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Set axis limits based on actual data ranges (don't force equal aspect ratio)
    # Get all plotted data points
    all_x = []
    all_y = []
    all_z = []
    
    # Collect data from all plotted categories
    for df_plot in [rejected_df_plot, survivors_df_plot, pareto_df_plot]:
        if len(df_plot) > 0:
            all_x.extend(df_plot["ttff_content_p50_ms"].dropna().tolist())
            all_y.extend(df_plot["flush_count_mean"].dropna().tolist())
            all_z.extend(df_plot["chunk_size_p50"].dropna().tolist())
    
    # Add selected policy if it exists
    if selected_id and selected_id in df["policy_id"].values:
        selected_df = df[df["policy_id"] == selected_id]
        if len(selected_df) > 0:
            row = selected_df.iloc[0]
            if pd.notna(row["ttff_content_p50_ms"]):
                all_x.append(row["ttff_content_p50_ms"])
            if pd.notna(row["flush_count_mean"]):
                all_y.append(row["flush_count_mean"])
            if pd.notna(row["chunk_size_p50"]):
                all_z.append(row["chunk_size_p50"])
    
    if len(all_x) > 0 and len(all_y) > 0 and len(all_z) > 0:
        # Add padding (5% on each side)
        x_padding = (max(all_x) - min(all_x)) * 0.05 if max(all_x) != min(all_x) else max(all_x) * 0.05
        y_padding = (max(all_y) - min(all_y)) * 0.05 if max(all_y) != min(all_y) else max(all_y) * 0.05
        z_padding = (max(all_z) - min(all_z)) * 0.05 if max(all_z) != min(all_z) else max(all_z) * 0.05
        
        ax.set_xlim(min(all_x) - x_padding, max(all_x) + x_padding)
        ax.set_ylim(min(all_y) - y_padding, max(all_y) + y_padding)
        ax.set_zlim(min(all_z) - z_padding, max(all_z) + z_padding)
    
    # Add subtle grids
    ax.grid(True, linewidth=0.75, alpha=0.25)  # Major grid
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.75
    ax.xaxis._axinfo["grid"]["alpha"] = 0.25
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.75
    ax.yaxis._axinfo["grid"]["alpha"] = 0.25
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.75
    ax.zaxis._axinfo["grid"]["alpha"] = 0.25
    
    # Set background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")
    
    # Legend: outside plot area, above
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        frameon=False,
        prop=font_props,
    )
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as SVG
    plt.savefig(output_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close()

