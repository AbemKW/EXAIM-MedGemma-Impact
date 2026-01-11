"""Generate all paper figures from aggregate metrics data.

Usage:
    python -m evals.src.paper_figures.generate_all
"""

import json
from pathlib import Path

from . import figure1, figure2, figure3


def main():
    """Generate all paper figures."""
    # Path to aggregate metrics JSON (relative to workspace root)
    workspace_root = Path(__file__).parent.parent.parent.parent.parent
    metrics_path = workspace_root / 'evals' / 'data' / 'metrics' / 'aggregate.metrics.json'
    
    # Alternative: try relative to current working directory
    if not metrics_path.exists():
        metrics_path = Path('evals/data/metrics/aggregate.metrics.json')
    
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found. Tried:\n"
            f"  - {workspace_root / 'evals' / 'data' / 'metrics' / 'aggregate.metrics.json'}\n"
            f"  - {Path('evals/data/metrics/aggregate.metrics.json').absolute()}\n"
            f"Please run from workspace root or specify correct path."
        )
    
    # Load metrics data
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    # Output directory for figures
    output_dir = metrics_path.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Figure 1: Core Ablation Cluster Bar Chart...")
    figure1.generate_figure1(metrics_data, output_dir)
    
    print("\nGenerating Figure 2: Efficiency Frontier Scatter Plot...")
    figure2.generate_figure2(metrics_data, output_dir)
    
    print("\nGenerating Figure 3: Price of Transparency...")
    figure3.generate_figure3(metrics_data, output_dir)
    
    print("\nAll figures generated successfully!")
    print(f"Figures saved to: {output_dir}")


if __name__ == '__main__':
    main()



