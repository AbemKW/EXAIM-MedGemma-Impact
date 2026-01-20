"""Saturation Plot: Budget-Constrained Coverage

Shows how coverage (information density) saturates with increasing summary budget.
V0 (EXAIM) achieves high density early but avoids inefficient divergence at high budgets.
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def generate_saturation_plot(metrics_data: Dict, output_dir: Path) -> None:
    """Generate Saturation Plot for Budget-Constrained Coverage
    
    Args:
        metrics_data: Dictionary containing metrics data from aggregate.metrics.json
        output_dir: Directory to save the figure (will create raw/saturation/ subdirectory)
    """
    variants = ['V0', 'V1', 'V2', 'V3', 'V4']
    budgets = [250, 500, 1000, 2000]
    
    # Extract coverage data for each variant and budget
    coverage_data = {}
    for variant in variants:
        coverage_data[variant] = [
            metrics_data['variants'][variant]['m7b_coverage_by_budget_mean'][f'ctu_{budget}']
            for budget in budgets
        ]
    
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
    
    # Churkin Protocol: Canvas & Resolution (Square for consistency)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Churkin Protocol: Harmonious Color Palette (Purple/Teal Tones)
    line_styles = {
        'V0': {'color': '#5E2D79', 'linestyle': '-', 'marker': 'o', 'linewidth': 3, 'markersize': 8, 'label': 'V0 (EXAIM)'},
        'V2': {'color': '#008080', 'linestyle': '--', 'marker': 's', 'linewidth': 2.5, 'markersize': 8, 'label': 'V2 (No Buffer)'},
        'V1': {'color': '#7B68A8', 'linestyle': '-', 'marker': '^', 'linewidth': 1.5, 'markersize': 6, 'label': 'V1 (Turn-Based)'},
        'V3': {'color': '#4A4A4A', 'linestyle': '-', 'marker': 'x', 'linewidth': 1.5, 'markersize': 6, 'label': 'V3 (Fixed-Chunk)'},
        'V4': {'color': '#A0A0A0', 'linestyle': '-', 'marker': '*', 'linewidth': 1.5, 'markersize': 6, 'label': 'V4 (No Novelty)'}
    }
    
    # Plot each variant
    for variant in variants:
        ax.plot(budgets, coverage_data[variant], 
                color=line_styles[variant]['color'],
                linestyle=line_styles[variant]['linestyle'],
                marker=line_styles[variant]['marker'],
                linewidth=line_styles[variant]['linewidth'],
                markersize=line_styles[variant]['markersize'],
                label=line_styles[variant]['label'],
                zorder=3 if variant in ['V0', 'V2'] else 2)
    
    # X-axis: Logarithmic scale for budgets
    ax.set_xscale('log')
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    
    # Force plain numbers on x-axis, disable minor tick labels
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    
    # Labels
    ax.set_xlabel('Summary Budget (CTU)', fontweight='bold', fontsize=20, labelpad=10)
    ax.set_ylabel('Coverage (M7b)', fontweight='bold', fontsize=20, labelpad=12)
    
    # Y-axis limits with some padding
    y_min = min(min(coverage_data[v]) for v in variants)
    y_max = max(max(coverage_data[v]) for v in variants)
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim([max(0, y_min - y_padding), y_max + y_padding])
    
    # Churkin Protocol: Grid System
    ax.grid(True, axis='both', which='major', linestyle='-', alpha=0.25, 
            linewidth=0.75, zorder=1)
    ax.grid(True, axis='both', which='minor', linestyle='--', alpha=0.15, 
            linewidth=0.25, zorder=1)
    ax.set_axisbelow(True)
    
    # Churkin Protocol: Legend (Outside/Above the plot area)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), 
              ncol=3, frameon=False, fontsize=16, columnspacing=1.5)
    
    # Tight layout with extra space for legend
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Churkin Protocol: Separate Raw Assets from Production Assets
    raw_dir = output_dir / 'raw' / 'saturation'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Churkin Protocol: Export Format
    svg_path = raw_dir / 'saturation.svg'
    pdf_path = raw_dir / 'saturation.pdf'
    
    plt.savefig(svg_path, bbox_inches='tight', pad_inches=0.1, format='svg')
    print(f"Saturation plot (raw) saved as {svg_path}")
    
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1, format='pdf')
    print(f"Saturation plot (raw) saved as {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    # For standalone testing
    import sys
    if len(sys.argv) > 1:
        metrics_path = Path(sys.argv[1])
    else:
        metrics_path = Path('evals/data/metrics/aggregate.metrics.json')
    
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    output_dir = metrics_path.parent / 'figures'
    generate_saturation_plot(metrics_data, output_dir)