#!/bin/bash
# =============================================================================
# EXAID Evaluation - TokenGate Calibration Script
# Phase 5: TokenGate Trigger Calibration
# =============================================================================
# Calibrates TokenGate trigger parameters (min_words, max_words, silence_timer,
# max_wait_timeout) by systematically evaluating literature-informed parameter
# combinations across frozen v2.0.0 traces.
#
# Output:
#   evals/data/calibration/calib_<hash8>_<hash8>_<hash8>/
#     - calibration_results.csv
#     - calibration_per_case.jsonl
#     - calibration_summary.json
#     - chosen_tokengate_params.yaml
#     - calibration_report.md
#     - calibration_config.yaml
#     - spam_sensitivity.json
# =============================================================================

set -e

echo "========================================"
echo "EXAID Evaluation - TokenGate Calibration"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# Check configuration files
# -----------------------------------------------------------------------------
if [ -f "configs/calibration_sweep.yaml" ]; then
    echo "Calibration config: configs/calibration_sweep.yaml"
else
    echo "ERROR: Calibration config not found: configs/calibration_sweep.yaml"
    exit 1
fi

# Find manifest file
MANIFEST_PATTERN="data/manifests/exaid_traces_*.manifest.jsonl"
MANIFEST_FILES=$(ls $MANIFEST_PATTERN 2>/dev/null || echo "")
if [ -z "$MANIFEST_FILES" ]; then
    echo "ERROR: No manifest files found matching: $MANIFEST_PATTERN"
    echo "Please run trace generation first: ./scripts/01_make_traces.sh"
    exit 1
fi

# Use first manifest file found
MANIFEST_FILE=$(echo $MANIFEST_FILES | awk '{print $1}')
echo "Manifest: $MANIFEST_FILE"
echo ""

# -----------------------------------------------------------------------------
# Check traces directory
# -----------------------------------------------------------------------------
if [ ! -d "data/traces" ]; then
    echo "ERROR: Traces directory not found: data/traces"
    echo "Please run trace generation first: ./scripts/01_make_traces.sh"
    exit 1
fi

TRACE_COUNT=$(ls data/traces/*.trace.jsonl.gz 2>/dev/null | wc -l)
if [ "$TRACE_COUNT" -eq 0 ]; then
    echo "ERROR: No trace files found in data/traces/"
    echo "Please run trace generation first: ./scripts/01_make_traces.sh"
    exit 1
fi

echo "Found $TRACE_COUNT trace files"
echo ""

# -----------------------------------------------------------------------------
# Run calibration
# -----------------------------------------------------------------------------
echo "Starting TokenGate calibration..."
echo ""
echo "This will evaluate 625 parameter combinations across all traces."
echo "This may take a significant amount of time..."
echo ""

# Check if we should allow stub traces (for testing only)
ALLOW_STUB_FLAG=""
if [ "${EXAID_ALLOW_STUB:-}" = "1" ] || [ "${EXAID_ALLOW_STUB:-}" = "true" ]; then
    ALLOW_STUB_FLAG="--allow-stub"
    echo "NOTE: Allowing stub traces (EXAID_ALLOW_STUB=1)"
    echo ""
fi

python -m evals.cli.calibrate_tokengate \
    --traces data/traces \
    --manifest "$MANIFEST_FILE" \
    --config configs/calibration_sweep.yaml \
    --output data/calibration \
    $ALLOW_STUB_FLAG

echo ""
echo "========================================"
echo "TokenGate calibration complete"
echo "========================================"
echo ""
echo "Output directory: data/calibration/calib_*/"
echo ""
echo "Key artifacts:"
echo "  - calibration_results.csv: All policy results"
echo "  - calibration_summary.json: Summary and selected policy"
echo "  - chosen_tokengate_params.yaml: Selected parameters"
echo "  - calibration_report.md: Detailed report"
echo ""
