"""Figure 2: Efficiency Frontier Scatter Plot (Cognitive Load Trade-Off)

Shows the trade-off between Interruption Frequency (Updates/Case) and Information Value (Faithfulness).
V0 achieves optimal balance: High Signal / Low Noise (Top-Right zone).
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np


def generate_figure2(metrics_data: Dict, output_dir: Path) -> None:
    """Generate Figure 2: Efficiency Frontier Scatter Plot
    
    Args:
        metrics_data: Dictionary containing metrics data from aggregate.metrics.json
        output_dir: Directory to save the figure (will create raw/figure2/ subdirectory)
    """
    variants = ['V0', 'V1', 'V2', 'V3', 'V4']
    
    # Extract metrics
    # X-axis: Update Frequency (Updates/Case) - will be INVERTED
    # Y-axis: Information Value (Faithfulness - M6b)
    updates = [metrics_data['variants'][v]['m1_update_count_mean'] for v in variants]
    faithfulness = [metrics_data['variants'][v]['m6b']['mean'] for v in variants]
    
    # Data points mapping
    data_points = {
        'V0': {'updates': updates[0], 'faithfulness': faithfulness[0], 'label': 'V0 (EXAIM)'},
        'V1': {'updates': updates[1], 'faithfulness': faithfulness[1], 'label': 'V1 (Baseline)'},
        'V2': {'updates': updates[2], 'faithfulness': faithfulness[2], 'label': 'V2 (No Buffer)'},
        'V3': {'updates': updates[3], 'faithfulness': faithfulness[3], 'label': 'V3 (Fixed)'},
        'V4': {'updates': updates[4], 'faithfulness': faithfulness[4], 'label': 'V4 (No Novelty)'},
    }
    
    # Global Churkin Protocol: Typography & Readability
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'axes.labelsize': 24,  # Global: 24pt for axis labels
        'axes.titlesize': 24,
        'xtick.labelsize': 20,  # Global: 20pt for tick labels
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    })
    
    # Global Churkin Protocol: Canvas - Standard Aspect Ratio
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate axis ranges
    updates_min, updates_max = min(updates), max(updates)
    faithfulness_min, faithfulness_max = min(faithfulness), max(faithfulness)
    
    # Add padding
    updates_range = updates_max - updates_min
    faithfulness_range = faithfulness_max - faithfulness_min
    updates_padding = updates_range * 0.1
    faithfulness_padding = faithfulness_range * 0.1
    
    # Set axis limits
    # X-axis will be INVERTED: Lower updates (better) on RIGHT, Higher updates (worse) on LEFT
    ax.set_xlim([updates_max + updates_padding, updates_min - updates_padding])
    ax.set_ylim([faithfulness_min - faithfulness_padding, faithfulness_max + faithfulness_padding])
    
    # Global Churkin Protocol: Stroke Width - All spines use linewidth=2.0+
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    
    # Draw Optimal Zone: Shaded rectangle in Top-Right corner (Low Updates, High Faithfulness)
    # Top-Right means: Low X (updates) and High Y (faithfulness)
    optimal_zone_x = updates_min - updates_padding  # Right side (low updates)
    optimal_zone_width = (updates_max - updates_min) * 0.3  # Width of zone
    optimal_zone_y = faithfulness_max - faithfulness_padding - (faithfulness_range * 0.3)  # Top area
    optimal_zone_height = faithfulness_range * 0.3
    
    # Churkin Protocol: Purple fill for optimal zone
    optimal_rect = Rectangle((optimal_zone_x - optimal_zone_width, optimal_zone_y), 
                            optimal_zone_width, optimal_zone_height,
                            facecolor='#5E2D79', alpha=0.2, edgecolor='#5E2D79', 
                            linewidth=2.0, linestyle='--', zorder=0)
    ax.add_patch(optimal_rect)
    
    # Label the Optimal Zone with bold purple text
    ax.text(optimal_zone_x - optimal_zone_width/2, optimal_zone_y + optimal_zone_height/2,
           'High Signal / Low Noise', fontsize=20, fontweight='bold', 
           ha='center', va='center', color='#5E2D79', zorder=1)
    
    # Draw crosshairs: Dashed lines at x=15 (Updates) and y=0.38 (Faithfulness)
    # Note: x=15 in inverted axis means we need to find where 15 maps to
    crosshair_x = 15.0
    crosshair_y = 0.38
    
    # Vertical crosshair at x=15 (Updates)
    if updates_min <= crosshair_x <= updates_max:
        ax.axvline(x=crosshair_x, color='gray', linestyle='--', linewidth=1.5, 
                  alpha=0.5, zorder=1)
    
    # Horizontal crosshair at y=0.38 (Faithfulness)
    if faithfulness_min <= crosshair_y <= faithfulness_max:
        ax.axhline(y=crosshair_y, color='gray', linestyle='--', linewidth=1.5, 
                  alpha=0.5, zorder=1)
    
    # Churkin Protocol: Color Palette (Harmonious Purple/Teal)
    variant_colors = {
        'V0': '#5E2D79',      # Purple (Full EXAIM - optimal)
        'V1': '#7B68A8',       # Lighter purple (Baseline)
        'V2': '#DC143C',       # Crimson/Red (No Buffer - high noise)
        'V3': '#4A90A4',       # Teal (Fixed)
        'V4': '#008080'        # Darker teal (No Novelty)
    }
    
    # Plot data points with Churkin Protocol colors
    # V0: Purple Star (Large) - High contrast
    ax.scatter(data_points['V0']['updates'], data_points['V0']['faithfulness'],
              s=400,  # Large marker
              marker='*',
              color=variant_colors['V0'],  # Purple
              edgecolor='black',
              linewidth=2.0,
              zorder=10,
              label=data_points['V0']['label'])
    
    # V1: Lighter purple circle
    ax.scatter(data_points['V1']['updates'], data_points['V1']['faithfulness'],
              s=200,  # Medium marker
              marker='o',
              color=variant_colors['V1'],  # Lighter purple
              edgecolor='black',
              linewidth=2.0,
              zorder=5,
              label=data_points['V1']['label'])
    
    # V2: Crimson/Red circle (high noise zone)
    ax.scatter(data_points['V2']['updates'], data_points['V2']['faithfulness'],
              s=200,
              marker='o',
              color=variant_colors['V2'],  # Crimson
              edgecolor='black',
              linewidth=2.0,
              zorder=5,
              label=data_points['V2']['label'])
    
    # V3: Teal circle (if data exists)
    if data_points['V3']['updates'] > 0:
        ax.scatter(data_points['V3']['updates'], data_points['V3']['faithfulness'],
                  s=200,
                  marker='o',
                  color=variant_colors['V3'],  # Teal
                  edgecolor='black',
                  linewidth=2.0,
                  zorder=5,
                  label=data_points['V3']['label'])
    
    # V4: Darker teal circle
    ax.scatter(data_points['V4']['updates'], data_points['V4']['faithfulness'],
              s=200,
              marker='o',
              color=variant_colors['V4'],  # Darker teal
              edgecolor='black',
              linewidth=2.0,
              zorder=5,
              label=data_points['V4']['label'])
    
    # Set axis labels
    # X-axis: Inverted - Label shows "Efficiency (Low Interruption Frequency) →"
    ax.set_xlabel('Efficiency (Low Interruption Frequency) →', 
                  fontweight='bold', fontsize=24, labelpad=15)
    ax.set_ylabel('Information Value (Faithfulness)', 
                  fontweight='bold', fontsize=24, labelpad=15)
    
    # Set ticks
    # X-axis: Updates (inverted, so higher values on left)
    updates_ticks = np.arange(np.floor(updates_min - updates_padding), 
                             np.ceil(updates_max + updates_padding) + 1, 5)
    ax.set_xticks(updates_ticks)
    ax.tick_params(axis='x', labelsize=20)
    
    # Y-axis: Faithfulness
    y_tick_spacing = 0.02
    y_ticks = np.arange(np.floor((faithfulness_min - faithfulness_padding) / y_tick_spacing) * y_tick_spacing,
                       np.ceil((faithfulness_max + faithfulness_padding) / y_tick_spacing) * y_tick_spacing + y_tick_spacing,
                       y_tick_spacing)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=20)
    
    # Add grid
    ax.grid(True, which='major', linestyle='--', alpha=0.2, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='lower left', frameon=False, fontsize=18, handletextpad=0.5)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2)
    
    # Churkin Protocol: Separate Raw Assets from Production Assets
    raw_dir = output_dir / 'raw' / 'figure2'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Churkin Protocol: Export Format
    svg_path = raw_dir / 'figure2.svg'
    pdf_path = raw_dir / 'figure2.pdf'
    
    plt.savefig(svg_path, bbox_inches='tight', pad_inches=0.1, format='svg')
    print(f"Figure 2 (raw) saved as {svg_path}")
    
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1, format='pdf')
    print(f"Figure 2 (raw) saved as {pdf_path}")
    
    plt.close()
