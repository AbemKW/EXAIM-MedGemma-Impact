#!/bin/bash
# =============================================================================
# EXAID Evaluation - TokenGate Calibration Smoke Test Script
# Quick validation of calibration pipeline after reorganization
# =============================================================================
# Runs minimal calibration configs to verify the pipeline works correctly.
# Much faster than full calibration (4-36 policies vs 625).
#
# Usage:
#   ./scripts/calibrate_tokengate_smoke.sh [quick|minimal|small]
#
# Options:
#   quick    - Absolute minimum (4 policies, ~1-2 min)
#   minimal  - Minimal coverage (16 policies, ~5-10 min) [default]
#   small    - Small coverage (36 policies, ~15-20 min)
# =============================================================================

set -e

SMOKE_TYPE="${1:-minimal}"

echo "========================================"
echo "EXAID Evaluation - TokenGate Calibration Smoke Test"
echo "========================================"
echo ""
echo "Smoke test type: $SMOKE_TYPE"
echo ""

cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# Select smoke test config
# -----------------------------------------------------------------------------
case "$SMOKE_TYPE" in
    quick)
        CONFIG_FILE="configs/smoke_tests/smoke_test_quick.yaml"
        POLICY_COUNT="4"
        ;;
    minimal)
        CONFIG_FILE="configs/smoke_tests/smoke_test_minimal.yaml"
        POLICY_COUNT="16"
        ;;
    small)
        CONFIG_FILE="configs/smoke_tests/smoke_test_small.yaml"
        POLICY_COUNT="36"
        ;;
    *)
        echo "ERROR: Unknown smoke test type: $SMOKE_TYPE"
        echo "Valid options: quick, minimal, small"
        exit 1
        ;;
esac

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Smoke test config not found: $CONFIG_FILE"
    exit 1
fi

echo "Config: $CONFIG_FILE"
echo "Policies: $POLICY_COUNT"
echo ""

# -----------------------------------------------------------------------------
# Check manifest file
# -----------------------------------------------------------------------------
MANIFEST_PATTERN="data/manifests/exaid_traces_*.manifest.jsonl"
MANIFEST_FILES=$(ls $MANIFEST_PATTERN 2>/dev/null || echo "")
if [ -z "$MANIFEST_FILES" ]; then
    echo "ERROR: No manifest files found matching: $MANIFEST_PATTERN"
    echo "Please run trace generation first: ./scripts/make_traces.sh"
    exit 1
fi

MANIFEST_FILE=$(echo $MANIFEST_FILES | awk '{print $1}')
echo "Manifest: $MANIFEST_FILE"
echo ""

# -----------------------------------------------------------------------------
# Check traces directory
# -----------------------------------------------------------------------------
if [ ! -d "data/traces" ]; then
    echo "ERROR: Traces directory not found: data/traces"
    echo "Please run trace generation first: ./scripts/make_traces.sh"
    exit 1
fi

TRACE_COUNT=$(ls data/traces/*.trace.jsonl.gz 2>/dev/null | wc -l)
if [ "$TRACE_COUNT" -eq 0 ]; then
    echo "ERROR: No trace files found in data/traces/"
    echo "Please run trace generation first: ./scripts/make_traces.sh"
    exit 1
fi

echo "Found $TRACE_COUNT trace files"
echo ""

# -----------------------------------------------------------------------------
# Run smoke test
# -----------------------------------------------------------------------------
echo "Starting TokenGate calibration smoke test..."
echo ""
echo "This will evaluate $POLICY_COUNT parameter combinations across all traces."
echo "This is a smoke test - results may differ from full calibration."
echo ""

# Check if we should allow stub traces (for testing only)
ALLOW_STUB_FLAG=""
if [ "${EXAID_ALLOW_STUB:-}" = "1" ] || [ "${EXAID_ALLOW_STUB:-}" = "true" ]; then
    ALLOW_STUB_FLAG="--allow-stub"
    echo "NOTE: Allowing stub traces (EXAID_ALLOW_STUB=1)"
    echo ""
fi

# Run TokenGate calibration via the evals.cli.calibrate_tokengate module
python -m evals.cli.calibrate_tokengate \
    --traces data/traces \
    --manifest "$MANIFEST_FILE" \
    --config "$CONFIG_FILE" \
    --output data/calibration_smoke \
    $ALLOW_STUB_FLAG

echo ""
echo "========================================"
echo "TokenGate calibration smoke test complete"
echo "========================================"
echo ""
echo "Output directory: data/calibration_smoke/calib_*/"
echo ""
echo "Key artifacts:"
echo "  - calibration_results.csv: All policy results"
echo "  - calibration_summary.json: Summary and selected policy"
echo "  - chosen_tokengate_params.yaml: Selected parameters"
echo "  - calibration_report.md: Detailed report"
echo ""
echo "NOTE: This was a smoke test with relaxed constraints."
echo "For production calibration, use: ./scripts/calibrate_tokengate.sh"
echo ""


