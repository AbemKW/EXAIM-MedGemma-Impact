"""Efficiency Frontier with Uncertainty Visualization

Shows the trade-off between Interruption Frequency (Update Frequency M1) and Faithfulness (M6b).
- Scatter plot with vertical error bars (95% CI) for Faithfulness
- IEEE-ready styling
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_per_case_metrics(per_case_path: Path) -> Dict[str, List[Dict]]:
    """Load per-case metrics from JSONL file.
    
    Args:
        per_case_path: Path to per_case.metrics.jsonl
        
    Returns:
        Dictionary mapping variant_id to list of per-case metric records
    """
    variant_data = defaultdict(list)
    
    with open(per_case_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            variant_id = record.get('variant_id')
            if variant_id:
                variant_data[variant_id].append(record)
    
    return dict(variant_data)


def compute_variant_statistics(variant_records: List[Dict]) -> Dict:
    """Compute mean and 95% CI for a variant.
    
    Args:
        variant_records: List of per-case metric records for one variant
        
    Returns:
        Dictionary with mean_x, mean_y, ci95_y, n_cases
    """
    n_cases = len(variant_records)
    assert n_cases >= 30, f"Insufficient cases: {n_cases} < 30"
    
    # Extract x (m1_update_count) and y (m6b_mean)
    x_values = [r['m1_update_count'] for r in variant_records]
    y_values = [r['m6b_mean'] for r in variant_records]
    
    # Compute means
    mean_x = np.mean(x_values)
    mean_y = np.mean(y_values)
    
    # Compute 95% CI for y (Faithfulness)
    sd_y = np.std(y_values, ddof=1)  # Sample standard deviation
    se_y = sd_y / np.sqrt(n_cases)   # Standard error
    ci95_y = 1.96 * se_y             # 95% CI (normal approximation)
    
    return {
        'mean_x': mean_x,
        'mean_y': mean_y,
        'ci95_y': ci95_y,
        'n_cases': n_cases
    }


def generate_efficiency_frontier(metrics_data: Dict, output_dir: Path, per_case_path: Path = None) -> None:
    """Generate Efficiency Frontier with uncertainty visualization.
    
    Args:
        metrics_data: Dictionary containing metrics data from aggregate.metrics.json (for sanity check)
        output_dir: Directory to save the figure
        per_case_path: Path to per_case.metrics.jsonl (if None, uses default location)
    """
    variants = ['V0', 'V1', 'V2', 'V3', 'V4']
    
    # Load per-case metrics
    if per_case_path is None:
        # Default: per_case.metrics.jsonl is in the same directory as aggregate.metrics.json
        per_case_path = output_dir.parent / 'per_case.metrics.jsonl'
    
    variant_records = load_per_case_metrics(per_case_path)
    
    # Compute statistics for each variant
    variant_stats = {}
    for variant in variants:
        if variant not in variant_records:
            raise ValueError(f"Variant {variant} not found in per_case.metrics.jsonl")
        variant_stats[variant] = compute_variant_statistics(variant_records[variant])
    
    # Prepare data points for plotting
    points = []
    for variant in variants:
        stats = variant_stats[variant]
        points.append((stats['mean_x'], stats['mean_y'], variant))
    
    # Global Churkin Protocol: Typography & Readability
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    })
    
    # Canvas - Appropriate Aspect Ratio
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate axis ranges
    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]
    updates_min, updates_max = min(all_x), max(all_x)
    faithfulness_min, faithfulness_max = min(all_y), max(all_y)
    
    # Calculate ranges for proportional offsets
    updates_range = updates_max - updates_min
    faithfulness_range = faithfulness_max - faithfulness_min
    
    # Tight axis limits: data + small fixed margin
    # X-axis: Tighten to approximately 7-49 (min-1 to max+2) for high ROI
    updates_margin_left = 1.0  # Small margin on left
    updates_margin_right = 2.0  # Slightly larger margin on right
    faithfulness_margin = 0.02
    
    # Set axis limits - tighter x-range for better visual separation
    ax.set_xlim([updates_min - updates_margin_left, updates_max + updates_margin_right])
    ax.set_ylim([faithfulness_min - faithfulness_margin, faithfulness_max + faithfulness_margin])
    
    # Lighten border: thinner spines and remove top/right spines
    # Data should be the darkest thing - spines should be very subtle
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)  # Thinner spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Churkin Protocol: Color Palette
    variant_colors = {
        'V0': '#5E2D79',      # Purple (Full EXAIM - optimal)
        'V1': '#7B68A8',       # Lighter purple (Baseline)
        'V2': '#DC143C',       # Crimson/Red (No Buffer - high noise)
        'V3': '#4A90A4',       # Teal (Fixed)
        'V4': '#008080'        # Darker teal (No Novelty)
    }
    
    # Plot each variant with error bars
    for variant in variants:
        stats = variant_stats[variant]
        x = stats['mean_x']
        y = stats['mean_y']
        ci95 = stats['ci95_y']
        color = variant_colors[variant]
        
        # Normalize visual weight: error bars ≤ marker edges, caps small but visible
        # Markers should be visually primary, uncertainty secondary
        if variant == 'V0':
            # V0: Largest marker, error bars thinner than marker edge
            ax.errorbar(x, y, yerr=ci95,
                       fmt='o',
                       color=color,
                       ecolor=color,
                       elinewidth=2.0,  # Error bar ≤ marker edge (2.5)
                       capsize=2.5,  # Small but visible caps
                       capthick=1.5,  # Caps thinner than error bar
                       markersize=14,  # Marker larger than caps (caps ~2.5pt, marker ~14pt)
                       markerfacecolor=color,
                       markeredgecolor='black',
                       markeredgewidth=2.5,  # Marker edge > error bar
                       alpha=1.0,
                       zorder=10,
                       label=None)
        elif variant == 'V2':
            # V2: Red dot, error bars thinner than marker edge
            ax.errorbar(x, y, yerr=ci95,
                       fmt='o',
                       color=color,
                       ecolor=color,
                       elinewidth=1.5,  # Error bar ≤ marker edge (2.0)
                       capsize=2.0,  # Small but visible
                       capthick=1.0,
                       markersize=11,  # Marker larger than caps
                       markerfacecolor=color,
                       markeredgecolor='black',
                       markeredgewidth=2.0,  # Marker edge > error bar
                       alpha=0.8,
                       zorder=5,
                       label=None)
        else:
            # V1, V3, V4: Standard baselines, error bars thinner than marker edge
            ax.errorbar(x, y, yerr=ci95,
                       fmt='o',
                       color=color,
                       ecolor=color,
                       elinewidth=1.0,  # Error bar ≤ marker edge (1.5)
                       capsize=1.5,  # Small but visible
                       capthick=0.75,
                       markersize=9,  # Marker larger than caps
                       markerfacecolor=color,
                       markeredgecolor='black',
                       markeredgewidth=1.5,  # Marker edge > error bar
                       alpha=0.75,
                       zorder=5,
                       label=None)
        
        # Add label - Consistent positioning for all variants (principles: consistency)
        # All labels positioned below their points with consistent small offset
        label_offset_x = updates_range * 0.015  # Small consistent right offset
        label_offset_y = faithfulness_range * -0.015  # Small consistent down offset (negative)
        
        # Font size and weight vary by importance, but positioning is consistent
        fontsize = 20 if variant == 'V0' else (18 if variant == 'V2' else 14)
        fontweight = 'bold' if variant in ['V0', 'V2'] else 'normal'
        
        ax.text(x + label_offset_x, y + label_offset_y, variant,
               fontsize=fontsize, fontweight=fontweight, color=color,
               ha='left', va='top',  # Top-align since label is below point
               zorder=11 if variant == 'V0' else 6)
    
    # Set axis labels
    ax.set_xlabel('Interruption Frequency (Updates/Case)',
                  fontweight='bold', fontsize=24, labelpad=15)
    ax.set_ylabel('Faithfulness',
                  fontweight='bold', fontsize=24, labelpad=15)
    
    # Set ticks
    updates_tick_spacing = 5.0
    updates_ticks = np.arange(
        np.floor((updates_min - updates_margin_left) / updates_tick_spacing) * updates_tick_spacing,
        np.ceil((updates_max + updates_margin_right) / updates_tick_spacing) * updates_tick_spacing + updates_tick_spacing,
        updates_tick_spacing
    )
    ax.set_xticks(updates_ticks)
    ax.tick_params(axis='x', labelsize=20)
    
    y_tick_spacing = 0.02
    y_ticks = np.arange(
        np.floor((faithfulness_min - faithfulness_margin) / y_tick_spacing) * y_tick_spacing,
        np.ceil((faithfulness_max + faithfulness_margin) / y_tick_spacing) * y_tick_spacing + y_tick_spacing,
        y_tick_spacing
    )
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=20)
    
    # Add grids - Major only, very light (data should be darkest)
    ax.grid(True, which='major', linestyle='--', alpha=0.15, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # Add preferred-direction cue (tiny but important for figure comprehension)
    # Small, unobtrusive annotation in top-left corner
    preferred_text = "Preferred: ↓ Updates, ↑ Faithfulness"
    # Use axes coordinates (0-1) to position in top-left corner
    ax.text(0.02, 0.98,  # 2% from left, 98% from bottom (top-left)
            preferred_text,
            fontsize=12,  # Small font
            color='gray',
            alpha=0.6,  # Unobtrusive
            ha='left',
            va='top',  # Anchor from top
            transform=ax.transAxes,  # Use axes coordinates, not data coordinates
            zorder=2,  # Above grid, below data
            style='italic')  # Subtle styling
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2)
    
    # Export
    raw_dir = output_dir / 'raw' / 'efficiency_frontier'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    svg_path = raw_dir / 'efficiency_frontier.svg'
    pdf_path = raw_dir / 'efficiency_frontier.pdf'
    
    plt.savefig(svg_path, bbox_inches='tight', pad_inches=0.1, format='svg')
    print(f"Efficiency Frontier (raw) saved as {svg_path}")
    
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1, format='pdf')
    print(f"Efficiency Frontier (raw) saved as {pdf_path}")
    
    plt.close()
