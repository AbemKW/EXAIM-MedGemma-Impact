"""Computational Overhead Bar Chart

Compares Latency (s) vs Relative Token Usage for V0 (Full EXAIM) vs V1 (Turn-Based).
Justifies the "modest computational overhead" mentioned in the abstract and results.
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def generate_computational_overhead(metrics_data: Dict, output_dir: Path) -> None:
    """Generate Computational Overhead Bar Chart
    
    Args:
        metrics_data: Dictionary containing metrics data from aggregate.metrics.json
        output_dir: Directory to save the figure (will create raw/computational_overhead/ subdirectory)
    """
    # Extract latency data (convert ms to seconds)
    v0_latency_ms = metrics_data['variants']['V0']['m8_latency_ms_mean']['summary']
    v1_latency_ms = metrics_data['variants']['V1']['m8_latency_ms_mean']['summary']
    
    v0_latency_s = v0_latency_ms / 1000.0
    v1_latency_s = v1_latency_ms / 1000.0
    
    # Extract token usage (calculate relative to V1)
    v0_token_total = metrics_data['variants']['V0']['m9_usage_ctu_mean']['total']
    v1_token_total = metrics_data['variants']['V1']['m9_usage_ctu_mean']['total']
    
    v0_token_relative = v0_token_total / v1_token_total
    v1_token_relative = 1.0  # Baseline
    
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
    fig, ax1 = plt.subplots(figsize=(10, 10))
    
    # Prepare data
    variants = ['V0\n(Full EXAIM)', 'V1\n(Turn-Based)']
    latency_data = [v0_latency_s, v1_latency_s]
    token_data = [v0_token_relative, v1_token_relative]
    
    x = np.arange(len(variants))
    width = 0.3  # Figure 3 fix: Reduced bar width to 0.3
    
    # Churkin Protocol: Color Palette (Harmonious & Minimal)
    color_latency = '#5E2D79'  # Purple
    color_token = '#008080'     # Teal
    
    # Create dual-axis plot
    # Global fix: thinner outlines (linewidth=0.5)
    # Left axis: Latency (seconds)
    bars1 = ax1.bar(x - width/2, latency_data, width,
                   label='Avg. Summary Latency (s)',
                   color=color_latency,
                   edgecolor='black',
                   linewidth=0.5,
                   zorder=3)
    
    # Right axis: Relative Token Usage
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, token_data, width,
                   label='Relative Token Usage (×)',
                   color=color_token,
                   edgecolor='black',
                   linewidth=0.5,
                   zorder=3)
    
    # Bar charts MUST start at 0 - no zoom-out logic
    latency_max = max(latency_data)
    latency_upper = latency_max * 1.15  # 15% padding at top
    latency_lower = 0  # Always start at 0
    
    ax1.set_ylim([latency_lower, latency_upper])
    ax1.set_ylabel('Avg. Summary Latency (s)', fontweight='bold', fontsize=20, 
                   labelpad=12, color=color_latency)
    # Figure 3 fix: Color left Y-axis ticks, label, and spine purple
    ax1.tick_params(axis='y', labelsize=18, colors=color_latency)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_color(color_latency)
    ax1.spines['left'].set_color(color_latency)
    ax1.spines['left'].set_linewidth(2)
    
    # Bar charts MUST start at 0 - no zoom-out logic
    token_max = max(token_data)
    token_upper = token_max * 1.15  # 15% padding at top
    token_lower = 0  # Always start at 0
    
    ax2.set_ylim([token_lower, token_upper])
    ax2.set_ylabel('Relative Token Usage (×)', fontweight='bold', fontsize=20,
                   labelpad=12, color=color_token)
    # Figure 3 fix: Color right Y-axis ticks, label, and spine teal
    ax2.tick_params(axis='y', labelsize=18, colors=color_token)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label1.set_color(color_token)
    ax2.spines['right'].set_color(color_token)
    ax2.spines['right'].set_linewidth(2)
    
    # X-axis setup
    ax1.set_xlabel('Variant', fontweight='bold', fontsize=20, labelpad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=18, ha='center')
    ax1.tick_params(axis='x', which='major', pad=8)
    
    # Churkin Protocol: Grid System
    ax1.grid(True, axis='y', which='major', linestyle='-', alpha=0.25,
            linewidth=0.75, zorder=1, color=color_latency)
    ax1.grid(True, axis='y', which='minor', linestyle='--', alpha=0.15,
            linewidth=0.25, zorder=1, color=color_latency)
    ax1.set_axisbelow(True)
    
    # Grid for Token Usage axis (secondary)
    ax2.grid(True, axis='y', which='major', linestyle='-', alpha=0.25,
            linewidth=0.75, zorder=1, color=color_token)
    ax2.grid(True, axis='y', which='minor', linestyle='--', alpha=0.15,
            linewidth=0.25, zorder=1, color=color_token)
    
    # Set tick spacing (starting from 0)
    latency_tick_spacing = 0.1
    token_tick_spacing = 1.0
    
    ax1.set_yticks(np.arange(latency_lower, latency_upper + latency_tick_spacing, latency_tick_spacing))
    ax2.set_yticks(np.arange(token_lower, token_upper + token_tick_spacing, token_tick_spacing))
    
    # Add minor ticks
    ax1.yaxis.set_minor_locator(MultipleLocator(latency_tick_spacing / 2))
    ax2.yaxis.set_minor_locator(MultipleLocator(token_tick_spacing / 2))
    
    # Churkin Protocol: The Legend
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
              loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=2, frameon=False, fontsize=18, columnspacing=1.5, handletextpad=0.5)
    
    # Add value labels on bars
    def add_value_labels(bars, ax, format_str='{:.2f}'):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   format_str.format(height),
                   ha='center', va='bottom', fontsize=16, fontweight='bold',
                   color=bar.get_facecolor())
    
    add_value_labels(bars1, ax1, '{:.2f}')
    add_value_labels(bars2, ax2, '{:.1f}×')
    
    # Figure 3 fix: Remove annotation - keep chart strictly data-focused
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.88], pad=1.2)
    
    # Churkin Protocol: Separate Raw Assets from Production Assets
    # Code writes to raw/ folder (safe to overwrite anytime)
    # Human edits raw/ and saves to final/ folder
    # LaTeX reads only from final/ folder
    raw_dir = output_dir / 'raw' / 'computational_overhead'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Churkin Protocol: Export Format
    svg_path = raw_dir / 'computational_overhead.svg'
    pdf_path = raw_dir / 'computational_overhead.pdf'
    
    plt.savefig(svg_path, bbox_inches='tight', pad_inches=0.1, format='svg')
    print(f"Computational Overhead (raw) saved as {svg_path}")
    
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1, format='pdf')
    print(f"Computational Overhead (raw) saved as {pdf_path}")
    
    plt.close()

