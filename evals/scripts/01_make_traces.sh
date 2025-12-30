#!/bin/bash
# =============================================================================
# EXAID Evaluation - Trace Generation Script
# =============================================================================
# Generates MAS traces from clinical cases using MAC as upstream trace generator.
#
# MAC Integration Notes:
# - MAC is used as an instrumentation-only trace generator
# - MAC controls its own decoding parameters internally (EXAID does not override)
# - Traces are frozen and replayed by EXAID variants
#
# Token Accounting:
# - All traces use CTU (Character-Normalized Token Units): ceil(len(text) / 4)
# - Provider token counts are logged separately as usage metadata only
# =============================================================================

set -e

echo "========================================"
echo "EXAID Evaluation - Trace Generation"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# Verify MAC submodule is present
# -----------------------------------------------------------------------------
MAC_PATH="/app/third_party/mac"
if [ ! -d "$MAC_PATH" ]; then
    # Check local path for development outside Docker
    if [ -d "../third_party/mac" ]; then
        MAC_PATH="../third_party/mac"
        echo "Using local MAC path: $MAC_PATH"
    else
        echo "WARNING: MAC submodule not found at $MAC_PATH"
        echo "Running in stub mode for testing..."
        echo ""
    fi
fi

# -----------------------------------------------------------------------------
# Check configuration files
# -----------------------------------------------------------------------------
if [ -f "configs/mas_generation.yaml" ]; then
    echo "MAS config: configs/mas_generation.yaml"
else
    echo "ERROR: MAS config not found: configs/mas_generation.yaml"
    exit 1
fi

if [ -f "configs/dataset.yaml" ]; then
    echo "Dataset config: configs/dataset.yaml"
else
    echo "ERROR: Dataset config not found: configs/dataset.yaml"
    exit 1
fi

echo ""

# -----------------------------------------------------------------------------
# Run trace generation
# -----------------------------------------------------------------------------
echo "Starting trace generation..."
echo ""

# Check if we should run in stub mode (for testing without MAC API)
STUB_FLAG=""
if [ "${EXAID_STUB_MODE:-}" = "1" ] || [ "${EXAID_STUB_MODE:-}" = "true" ]; then
    STUB_FLAG="--stub-mode"
    echo "NOTE: Running in stub mode (EXAID_STUB_MODE=1)"
    echo ""
fi

python -m evals.cli.make_traces \
    --config configs/mas_generation.yaml \
    --dataset-config configs/dataset.yaml \
    --output data/traces \
    --manifest data/manifests/dataset_manifest.jsonl \
    $STUB_FLAG

echo ""
echo "========================================"
echo "Trace generation complete"
echo "========================================"
echo ""
echo "Output:"
echo "  Traces: data/traces/"
echo "  Manifest: data/manifests/dataset_manifest.jsonl"
echo ""
echo "Token Accounting:"
echo "  All traces use CTU (Character-Normalized Token Units)"
echo "  CTU = ceil(len(text) / 4)"
echo ""
echo "Next step: ./scripts/02_run_variants.sh"
