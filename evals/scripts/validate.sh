#!/bin/bash
# EXAID Evaluation - Validation Script
# Validates all artifact files against JSON schemas

set -e

echo "========================================"
echo "EXAID Evaluation - Validation"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

OVERALL_STATUS=0
FILES_FOUND=0

validate_dir() {
    local dir="$1"
    local pattern="$2"
    local description="$3"

    echo "Validating: $description"

    if [ ! -d "$dir" ]; then
        echo "  Directory not found: $dir (skipping)"
        return 0
    fi

    local files=$(find "$dir" -maxdepth 2 -name "$pattern" 2>/dev/null | head -20)

    if [ -z "$files" ]; then
        echo "  No files matching $pattern (OK in scaffold mode)"
        return 0
    fi

    FILES_FOUND=1

    if python -m evals.cli.validate_logs "$dir"/$pattern 2>/dev/null; then
        echo "  PASSED"
    else
        echo "  FAILED"
        OVERALL_STATUS=1
    fi

    echo ""
}

validate_dir "data/manifests" "*.jsonl" "Manifests"
validate_dir "data/traces" "*.jsonl*" "Traces"
validate_dir "data/runs" "*.jsonl*" "Runs (all variants)"
validate_dir "data/metrics" "*.jsonl" "Metrics"

echo "========================================"
if [ $FILES_FOUND -eq 0 ]; then
    echo "RESULT: PASSED (scaffold mode - no data files yet)"
    echo ""
    echo "This is expected for a fresh scaffold."
    echo "Run the following to generate data:"
    echo "  ./scripts/make_traces.sh"
    echo "  ./scripts/run_variants.sh"
    echo "  ./scripts/compute_metrics.sh"
    exit 0
elif [ $OVERALL_STATUS -eq 0 ]; then
    echo "RESULT: PASSED - All validations successful"
    exit 0
else
    echo "RESULT: FAILED - Validation errors found"
    exit 1
fi
