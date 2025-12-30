#!/bin/bash
# EXAID Evaluation - Variant Runner Script

set -e

echo "========================================"
echo "EXAID Evaluation - Variant Runner"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

VARIANT_ARG=""
if [ -n "$1" ]; then
    VARIANT_ARG="--variant $1"
    echo "Running variant: $1"
else
    echo "Running all variants: V0, V1, V2, V3, V4"
fi
echo ""

if [ ! -d "data/traces" ] || [ -z "$(ls -A data/traces 2>/dev/null)" ]; then
    echo "WARNING: No traces found in data/traces/"
    echo "Run ./scripts/01_make_traces.sh first"
    echo ""
fi

python -m evals.cli.run_variants \
    --traces data/traces \
    --output data/runs \
    $VARIANT_ARG

echo ""
echo "========================================"
echo "Variant runs complete"
echo "========================================"
echo ""
echo "Output: data/runs/V{0-4}/"
echo ""
echo "Next step: ./scripts/03_compute_metrics.sh"
