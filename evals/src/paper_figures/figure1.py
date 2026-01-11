"""Figure 1: Core Ablation Cluster Bar Chart

Shows Coverage, Faithfulness, and Redundancy across all variants (V0-V4).
This is the "hero figure" comparing all ablation variants across critical metrics.
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def generate_figure1(metrics_data: Dict, output_dir: Path) -> None:
    """Generate Figure 1: Core Ablation Cluster Bar Chart
    
    Args:
        metrics_data: Dictionary containing metrics data from aggregate.metrics.json
        output_dir: Directory to save the figure (will create raw/figure1/ subdirectory)
    """
    variants = ['V0', 'V1', 'V2', 'V3', 'V4']
    variant_labels = ['V0\n(Full EXAIM)', 'V1\n(Turn-Based)', 'V2\n(No BufferAgent)', 
                      'V3\n(Fixed-Chunk)', 'V4\n(No Novelty)']
    
    # Extract metrics
    coverage = [metrics_data['variants'][v]['m4_trace_coverage']['mean'] for v in variants]
    faithfulness = [metrics_data['variants'][v]['m6b']['mean'] for v in variants]
    redundancy = [metrics_data['variants'][v]['m3_redundancy']['jaccard_mean'] for v in variants]
    
    # Churkin Protocol: Typography & Readability (Shrink Test)
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    })
    
    # Churkin Protocol: Canvas & Resolution
    # Increased width to prevent overlapping X-axis labels
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up bar positions with tight grouping
    x = np.arange(len(variants))
    # Final refinement: Increase bar width to 0.28 for substantial bars (occupying ~75% of space between ticks)
    width = 0.28
    
    # Churkin Protocol: Color Palette (Harmonious & Minimal)
    # Purple/Teal palette - distinguish positive (Coverage/Faithfulness) from negative (Redundancy)
    colors = {
        'coverage': '#5E2D79',      # Purple (positive metric)
        'faithfulness': '#7B68A8',   # Lighter purple (positive metric)
        'redundancy': '#008080'      # Teal (negative metric - lower is better)
    }
    
    # Create bars with proper z-order and styling
    # Global fix: thinner outlines (linewidth=0.5)
    bars1 = ax.bar(x - width, coverage, width, label='Coverage (M4)', 
                   color=colors['coverage'], edgecolor='black', linewidth=0.5,
                   zorder=3)
    bars2 = ax.bar(x, faithfulness, width, label='Faithfulness (M6b)', 
                   color=colors['faithfulness'], edgecolor='black', linewidth=0.5,
                   zorder=3)
    bars3 = ax.bar(x + width, redundancy, width, label='Redundancy (M3)', 
                   color=colors['redundancy'], edgecolor='black', linewidth=0.5,
                   zorder=3)
    
    # Customize axes
    ax.set_xlabel('Variant', fontweight='bold', fontsize=20, labelpad=10)
    ax.set_ylabel('Metric Value', fontweight='bold', fontsize=20, labelpad=12)
    ax.set_xticks(x)
    # Figure 1 fix: DO NOT rotate labels, use \n for line breaks, rotation=0
    ax.set_xticklabels(variant_labels, fontsize=18, ha='center', rotation=0)
    
    # Add spacing between x-axis and labels
    ax.tick_params(axis='x', which='major', pad=12)
    ax.tick_params(axis='y', which='major', pad=8)
    
    # Figure 1 fix: Increase headroom to 0.60 so highest number isn't too close to top
    y_lower = 0
    y_upper = 0.60
    
    ax.set_ylim([y_lower, y_upper])
    
    # Set y-ticks with appropriate spacing (0 to 0.6)
    y_tick_spacing = 0.05
    ax.set_yticks(np.arange(y_lower, y_upper + y_tick_spacing, y_tick_spacing))
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Churkin Protocol: Grid System
    # Major & Minor grids with transparency
    ax.grid(True, axis='y', which='major', linestyle='-', alpha=0.25, 
            linewidth=0.75, zorder=1)
    ax.grid(True, axis='y', which='minor', linestyle='--', alpha=0.15, 
            linewidth=0.25, zorder=1)
    ax.set_axisbelow(True)
    
    # Add minor ticks for better grid granularity
    ax.yaxis.set_minor_locator(MultipleLocator(y_tick_spacing / 2))
    
    # Churkin Protocol: The Legend
    # Figure 1 fix: Ensure legend is placed above plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
              ncol=3, frameon=False, fontsize=18, columnspacing=1.5, handletextpad=0.5)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            # Increase padding to 3 (0.03 in data units) so labels don't sit on edge
            label_y = height + 0.03  # Padding=3 equivalent
            # Increase font size for better readability - following Churkin Protocol (larger source fonts)
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Adjust layout with extra space for legend above and padding
    plt.tight_layout(rect=[0, 0, 1, 0.92], pad=1.2)
    
    # Churkin Protocol: Separate Raw Assets from Production Assets
    raw_dir = output_dir / 'raw' / 'figure1'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Churkin Protocol: Export Format
    # Save as PDF (for LaTeX) and SVG (for vector editing/Word)
    svg_path = raw_dir / 'figure1.svg'
    pdf_path = raw_dir / 'figure1.pdf'
    
    plt.savefig(svg_path, bbox_inches='tight', pad_inches=0.1, format='svg')
    print(f"Figure 1 (raw) saved as {svg_path}")
    
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1, format='pdf')
    print(f"Figure 1 (raw) saved as {pdf_path}")
    
    plt.close()
