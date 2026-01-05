"""Heatmap visualizations for TokenGate calibration results."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .visualization import classify_policies, load_calibration_data, normalize_to_goodness_space


def create_metric_heatmaps(
    calibration_dir: Path,
    output_dir: Path,
    bins: int = 20,
) -> None:
    """
    Create 2D heatmaps showing policy density in metric space.
    
    Creates three heatmaps:
    1. TTFF vs Flush Count
    2. TTFF vs Chunk Size
    3. Flush Count vs Chunk Size
    
    Args:
        calibration_dir: Path to calibration output directory
        output_dir: Directory to save heatmap figures
        bins: Number of bins for histogram (default: 20)
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
    
    # Get selected policy info
    selected_id = summary.get("selection", {}).get("selected_policy_id")
    
    # Filter to policies with all three metrics
    df_plot = df[
        df[["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]].notna().all(axis=1)
    ].copy()
    
    print(f"Policies with all metrics: {len(df_plot)}")
    
    # Create separate DataFrames for each category
    rejected_df = df_plot[df_plot["policy_id"].isin(classifications["rejected"])].copy()
    survivors_df = df_plot[df_plot["policy_id"].isin(classifications["survivors"])].copy()
    
    # Get selected policy
    selected_df = None
    if selected_id and selected_id in df_plot["policy_id"].values:
        selected_df = df_plot[df_plot["policy_id"] == selected_id].copy()
    
    # Set font to Times New Roman, 20pt
    font_props = {
        "family": "Times New Roman",
        "size": 20,
    }
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    
    # Create three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    # Define metric pairs for the three heatmaps
    metric_pairs = [
        ("ttff_content_p50_ms", "flush_count_mean", "Time to First Flush (ms)", "Flush Count"),
        ("ttff_content_p50_ms", "chunk_size_p50", "Time to First Flush (ms)", "Chunk Size (words)"),
        ("flush_count_mean", "chunk_size_p50", "Flush Count", "Chunk Size (words)"),
    ]
    
    for ax_idx, (x_metric, y_metric, x_label, y_label) in enumerate(metric_pairs):
        ax = axes[ax_idx]
        
        # Get data ranges
        all_x = df_plot[x_metric].dropna()
        all_y = df_plot[y_metric].dropna()
        
        if len(all_x) == 0 or len(all_y) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.05 if x_max != x_min else x_max * 0.05
        y_padding = (y_max - y_min) * 0.05 if y_max != y_min else y_max * 0.05
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Create bins
        x_bins = np.linspace(x_min, x_max, bins + 1)
        y_bins = np.linspace(y_min, y_max, bins + 1)
        
        # Compute 2D histogram for all policies
        H_all, x_edges, y_edges = np.histogram2d(
            df_plot[x_metric].dropna(),
            df_plot[y_metric].dropna(),
            bins=[x_bins, y_bins]
        )
        
        # Compute histograms for each category
        H_rejected = np.zeros_like(H_all)
        if len(rejected_df) > 0:
            H_rejected, _, _ = np.histogram2d(
                rejected_df[x_metric].dropna(),
                rejected_df[y_metric].dropna(),
                bins=[x_bins, y_bins]
            )
        
        H_survivors = np.zeros_like(H_all)
        if len(survivors_df) > 0:
            H_survivors, _, _ = np.histogram2d(
                survivors_df[x_metric].dropna(),
                survivors_df[y_metric].dropna(),
                bins=[x_bins, y_bins]
            )
        
        # Create combined heatmap (rejected = light gray, survivors = darker)
        # Use a colormap that shows density
        H_combined = H_all.T  # Transpose for imshow
        
        # Create custom colormap: white -> light gray -> dark gray
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#FFFFFF', '#E0E0E0', '#C0C0C0', '#808080', '#404040']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('density', colors_list, N=n_bins)
        
        # Plot heatmap
        im = ax.imshow(
            H_combined,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            aspect='auto',
            cmap=cmap,
            interpolation='nearest',
            alpha=0.7
        )
        
        # Overlay rejected policies in light red tint
        if np.any(H_rejected > 0):
            H_rejected_norm = H_rejected.T / (H_all.max() + 1)  # Normalize
            ax.imshow(
                H_rejected_norm,
                extent=[x_min, x_max, y_min, y_max],
                origin='lower',
                aspect='auto',
                cmap='Reds',
                alpha=0.3,
                interpolation='nearest'
            )
        
        # Overlay survivors in blue tint
        if np.any(H_survivors > 0):
            H_survivors_norm = H_survivors.T / (H_all.max() + 1)  # Normalize
            ax.imshow(
                H_survivors_norm,
                extent=[x_min, x_max, y_min, y_max],
                origin='lower',
                aspect='auto',
                cmap='Blues',
                alpha=0.3,
                interpolation='nearest'
            )
        
        # Plot selected policy if it exists
        if selected_df is not None and len(selected_df) > 0:
            row = selected_df.iloc[0]
            if pd.notna(row[x_metric]) and pd.notna(row[y_metric]):
                ax.scatter(
                    [row[x_metric]],
                    [row[y_metric]],
                    c='#DC143C',
                    s=300,
                    marker='*',
                    edgecolors='black',
                    linewidths=2,
                    zorder=10,
                    label='Selected Policy'
                )
        
        # Set labels
        ax.set_xlabel(x_label, **font_props)
        ax.set_ylabel(y_label, **font_props)
        
        # Add grid
        ax.grid(True, linewidth=0.5, alpha=0.3, linestyle='--')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Policy Count', **font_props)
        cbar.ax.tick_params(labelsize=16)
    
    # Add overall title
    fig.suptitle('TokenGate Calibration: Policy Density Heatmaps', **font_props, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    output_path = output_dir / "tokengate_calibration_heatmaps.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"✓ Heatmaps saved to: {output_path}")


def create_parameter_heatmap(
    calibration_dir: Path,
    output_dir: Path,
    x_param: str = "min_words",
    y_param: str = "max_words",
    metric_to_color: str = "chunk_size_p50",
) -> None:
    """
    Create heatmap showing how parameter combinations map to metric values.
    
    Args:
        calibration_dir: Path to calibration output directory
        output_dir: Directory to save heatmap figure
        x_param: Parameter for x-axis (default: "min_words")
        y_param: Parameter for y-axis (default: "max_words")
        metric_to_color: Metric to use for coloring (default: "chunk_size_p50")
    """
    # Load data
    df, summary = load_calibration_data(calibration_dir)
    
    # Filter to policies with the metric
    df_plot = df[df[metric_to_color].notna()].copy()
    
    if len(df_plot) == 0:
        print(f"No policies with metric {metric_to_color}")
        return
    
    # Get selected policy
    selected_id = summary.get("selection", {}).get("selected_policy_id")
    
    # Set font to Times New Roman, 20pt
    font_props = {
        "family": "Times New Roman",
        "size": 20,
    }
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Get unique parameter values
    x_values = sorted(df_plot[x_param].unique())
    y_values = sorted(df_plot[y_param].unique())
    
    # Create matrix for heatmap
    heatmap_data = np.full((len(y_values), len(x_values)), np.nan)
    
    # Fill matrix with metric values (average if multiple policies at same parameter combination)
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            policies_at_point = df_plot[
                (df_plot[x_param] == x_val) & (df_plot[y_param] == y_val)
            ]
            if len(policies_at_point) > 0:
                # Average metric value for this parameter combination
                heatmap_data[i, j] = policies_at_point[metric_to_color].mean()
    
    # Create heatmap
    im = ax.imshow(
        heatmap_data,
        extent=[min(x_values) - 0.5, max(x_values) + 0.5, min(y_values) - 0.5, max(y_values) + 0.5],
        origin='lower',
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_to_color.replace('_', ' ').title(), **font_props)
    cbar.ax.tick_params(labelsize=16)
    
    # Mark selected policy
    if selected_id and selected_id in df_plot["policy_id"].values:
        selected_row = df_plot[df_plot["policy_id"] == selected_id].iloc[0]
        if pd.notna(selected_row[x_param]) and pd.notna(selected_row[y_param]):
            ax.scatter(
                [selected_row[x_param]],
                [selected_row[y_param]],
                c='red',
                s=400,
                marker='*',
                edgecolors='black',
                linewidths=2,
                zorder=10,
                label='Selected Policy'
            )
    
    # Set labels
    ax.set_xlabel(x_param.replace('_', ' ').title(), **font_props)
    ax.set_ylabel(y_param.replace('_', ' ').title(), **font_props)
    
    # Set ticks to parameter values
    ax.set_xticks(x_values)
    ax.set_yticks(y_values)
    
    # Add grid
    ax.grid(True, linewidth=0.5, alpha=0.3, linestyle='--')
    
    # Add title
    ax.set_title(f'Parameter Space: {x_param} vs {y_param}\n(Colored by {metric_to_color})', **font_props)
    
    # Add legend if selected policy exists
    if selected_id and selected_id in df_plot["policy_id"].values:
        ax.legend(loc='upper right', prop=font_props)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    output_path = output_dir / f"tokengate_parameter_heatmap_{x_param}_vs_{y_param}.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"✓ Parameter heatmap saved to: {output_path}")


def create_all_policies_visualization(
    calibration_dir: Path,
    output_dir: Path,
    jitter_amount: float = 0.02,
) -> None:
    """
    Create visualizations showing ALL 625 policies individually.
    
    Creates:
    1. Parameter space scatter plots (all policies unique here)
    2. Jittered metric space plots (separates overlapping points)
    3. Performance comparison across parameter combinations
    
    Args:
        calibration_dir: Path to calibration output directory
        output_dir: Directory to save figures
        jitter_amount: Amount of jitter for overlapping points (as fraction of range)
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
    
    # Get selected policy
    selected_id = summary.get("selection", {}).get("selected_policy_id")
    
    # Filter to policies with all metrics
    df_plot = df[
        df[["ttff_content_p50_ms", "flush_count_mean", "chunk_size_p50"]].notna().all(axis=1)
    ].copy()
    
    print(f"Policies with all metrics: {len(df_plot)}")
    
    # Set font to Times New Roman, 20pt
    font_props = {
        "family": "Times New Roman",
        "size": 20,
    }
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    
    # ========================================================================
    # 1. Parameter Space Visualization (all 625 policies are unique here)
    # ========================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 16), dpi=300)
    axes1 = axes1.flatten()
    
    param_pairs = [
        ("min_words", "max_words", 0),
        ("silence_timer_ms", "max_wait_timeout_ms", 1),
        ("min_words", "silence_timer_ms", 2),
        ("max_words", "max_wait_timeout_ms", 3),
    ]
    
    for x_param, y_param, ax_idx in param_pairs:
        ax = axes1[ax_idx]
        
        # Plot rejected policies
        rejected_mask = df_plot["policy_id"].isin(classifications["rejected"])
        if rejected_mask.any():
            scatter1 = ax.scatter(
                df_plot.loc[rejected_mask, x_param],
                df_plot.loc[rejected_mask, y_param],
                c=df_plot.loc[rejected_mask, "chunk_size_p50"],
                cmap="Reds",
                alpha=0.4,
                s=30,
                edgecolors="none",
                label=f"Rejected ({rejected_mask.sum()})",
            )
        
        # Plot survivors
        survivors_mask = df_plot["policy_id"].isin(classifications["survivors"])
        if survivors_mask.any():
            scatter2 = ax.scatter(
                df_plot.loc[survivors_mask, x_param],
                df_plot.loc[survivors_mask, y_param],
                c=df_plot.loc[survivors_mask, "chunk_size_p50"],
                cmap="Blues",
                alpha=0.6,
                s=50,
                edgecolors="none",
                label=f"Survivors ({survivors_mask.sum()})",
            )
        
        # Plot selected policy
        if selected_id and selected_id in df_plot["policy_id"].values:
            selected_row = df_plot[df_plot["policy_id"] == selected_id].iloc[0]
            ax.scatter(
                [selected_row[x_param]],
                [selected_row[y_param]],
                c="red",
                s=500,
                marker="*",
                edgecolors="black",
                linewidths=2,
                zorder=10,
                label="Selected",
            )
        
        ax.set_xlabel(x_param.replace("_", " ").title(), **font_props)
        ax.set_ylabel(y_param.replace("_", " ").title(), **font_props)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", prop={"size": 14})
    
    fig1.suptitle("All 625 Policies in Parameter Space\n(Colored by Chunk Size)", **font_props, y=0.995)
    plt.tight_layout()
    output_path1 = output_dir / "tokengate_all_policies_parameter_space.svg"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path1, format="svg", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Parameter space plot saved to: {output_path1}")
    
    # ========================================================================
    # 2. Jittered Metric Space Plots (separates overlapping points)
    # ========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    metric_pairs = [
        ("ttff_content_p50_ms", "flush_count_mean", "Time to First Flush (ms)", "Flush Count", 0),
        ("ttff_content_p50_ms", "chunk_size_p50", "Time to First Flush (ms)", "Chunk Size (words)", 1),
        ("flush_count_mean", "chunk_size_p50", "Flush Count", "Chunk Size (words)", 2),
    ]
    
    for x_metric, y_metric, x_label, y_label, ax_idx in metric_pairs:
        ax = axes2[ax_idx]
        
        # Calculate jitter amounts (as fraction of range)
        x_range = df_plot[x_metric].max() - df_plot[x_metric].min()
        y_range = df_plot[y_metric].max() - df_plot[y_metric].min()
        
        # Use fixed random seed for reproducibility
        rng = np.random.RandomState(42)
        x_jitter = rng.normal(0, x_range * jitter_amount, len(df_plot))
        y_jitter = rng.normal(0, y_range * jitter_amount, len(df_plot))
        
        # Apply jitter
        x_jittered = df_plot[x_metric].values + x_jitter
        y_jittered = df_plot[y_metric].values + y_jitter
        
        # Plot rejected
        rejected_mask = df_plot["policy_id"].isin(classifications["rejected"]).values
        if rejected_mask.any():
            ax.scatter(
                x_jittered[rejected_mask],
                y_jittered[rejected_mask],
                c="#CCCCCC",
                alpha=0.3,
                s=20,
                edgecolors="none",
                label=f"Rejected ({rejected_mask.sum()})",
            )
        
        # Plot survivors
        survivors_mask = df_plot["policy_id"].isin(classifications["survivors"]).values
        if survivors_mask.any():
            ax.scatter(
                x_jittered[survivors_mask],
                y_jittered[survivors_mask],
                c="#008080",
                alpha=0.6,
                s=40,
                edgecolors="none",
                label=f"Survivors ({survivors_mask.sum()})",
            )
        
        # Plot selected
        if selected_id and selected_id in df_plot["policy_id"].values:
            selected_idx = df_plot[df_plot["policy_id"] == selected_id].index[0]
            selected_pos = df_plot.index.get_loc(selected_idx)
            ax.scatter(
                [x_jittered[selected_pos]],
                [y_jittered[selected_pos]],
                c="red",
                s=300,
                marker="*",
                edgecolors="black",
                linewidths=2,
                zorder=10,
                label="Selected",
            )
        
        ax.set_xlabel(x_label, **font_props)
        ax.set_ylabel(y_label, **font_props)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", prop={"size": 14})
    
    fig2.suptitle("All 625 Policies in Metric Space (with Jitter)\n(Shows all policies, overlapping points separated)", **font_props, y=1.02)
    plt.tight_layout()
    output_path2 = output_dir / "tokengate_all_policies_metric_space_jittered.svg"
    plt.savefig(output_path2, format="svg", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Jittered metric space plot saved to: {output_path2}")
    
    # ========================================================================
    # 3. Performance Distribution by Parameter
    # ========================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes3 = axes3.flatten()
    
    params = ["min_words", "max_words", "silence_timer_ms", "max_wait_timeout_ms"]
    metrics_to_plot = ["chunk_size_p50", "ttff_content_p50_ms", "flush_count_mean"]
    metric_labels = ["Chunk Size (words)", "TTFF (ms)", "Flush Count"]
    metric_colors = ["#008080", "#800080", "#DC143C"]
    
    for param_idx, param in enumerate(params):
        ax = axes3[param_idx]
        
        # Group by parameter value and show metric distributions
        for metric, label, color in zip(metrics_to_plot, metric_labels, metric_colors):
            param_values = sorted(df_plot[param].unique())
            metric_means = []
            metric_stds = []
            
            for pval in param_values:
                subset = df_plot[df_plot[param] == pval]
                if len(subset) > 0:
                    metric_means.append(subset[metric].mean())
                    metric_stds.append(subset[metric].std())
                else:
                    metric_means.append(np.nan)
                    metric_stds.append(np.nan)
            
            # Normalize metric for plotting (0-1 scale)
            metric_min = df_plot[metric].min()
            metric_max = df_plot[metric].max()
            metric_range = metric_max - metric_min
            if metric_range > 0:
                metric_normalized = (np.array(metric_means) - metric_min) / metric_range
            else:
                metric_normalized = np.array(metric_means)
            
            ax.plot(
                param_values,
                metric_normalized,
                marker="o",
                label=label,
                linewidth=2,
                markersize=8,
                color=color,
            )
        
        ax.set_xlabel(param.replace("_", " ").title(), **font_props)
        ax.set_ylabel("Normalized Metric Value", **font_props)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", prop={"size": 12})
        ax.set_title(f"Performance vs {param.replace('_', ' ').title()}", **font_props)
    
    fig3.suptitle("Performance Trends Across Parameter Values\n(All 625 policies)", **font_props, y=0.995)
    plt.tight_layout()
    output_path3 = output_dir / "tokengate_all_policies_performance_trends.svg"
    plt.savefig(output_path3, format="svg", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Performance trends plot saved to: {output_path3}")

