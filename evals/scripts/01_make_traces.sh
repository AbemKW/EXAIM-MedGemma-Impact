#!/bin/bash
# EXAID Evaluation - Trace Generation Script

set -e

echo "========================================"
echo "EXAID Evaluation - Trace Generation"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

if [ -f "configs/mas_generation.yaml" ]; then
    echo "Using config: configs/mas_generation.yaml"
else
    echo "WARNING: Config not found, using defaults"
fi
echo ""

python src/make_traces.py \
    --config configs/mas_generation.yaml \
    --cases data/cases \
    --output data/traces

echo ""
echo "========================================"
echo "Trace generation complete"
echo "========================================"
echo ""
echo "Output: data/traces/"
echo ""
echo "Next step: ./scripts/02_run_variants.sh"
