#!/bin/bash
# EXAID Evaluation - Metrics Computation Script

set -e

echo "========================================"
echo "EXAID Evaluation - Metrics Computation"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

VARIANT_ARG=""
if [ -n "$1" ]; then
    VARIANT_ARG="--variant $1"
    echo "Computing metrics for variant: $1"
else
    echo "Computing metrics for all variants"
fi
echo ""

if [ ! -d "data/runs" ] || [ -z "$(ls -A data/runs 2>/dev/null)" ]; then
    echo "WARNING: No runs found in data/runs/"
    echo "Run ./scripts/02_run_variants.sh first"
    echo ""
fi

python src/compute_metrics.py \
    --runs data/runs \
    --output data/metrics \
    $VARIANT_ARG

echo ""
echo "========================================"
echo "Metrics computation complete"
echo "========================================"
echo ""
echo "Output files:"
echo "  Metrics:  data/metrics/*.jsonl"
echo "  Figures:  data/metrics/figures/"
echo ""
echo "Key output: data/metrics/figures/coverage_vs_budget.pdf"
