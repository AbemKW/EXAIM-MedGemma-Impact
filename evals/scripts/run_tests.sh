#!/bin/bash
# EXAID Evaluation - Test Runner
# 
# Run all unit tests for the evaluation module.
# Usage: ./scripts/run_tests.sh [pytest-args]
#
# Examples:
#   ./scripts/run_tests.sh                    # Run all tests
#   ./scripts/run_tests.sh -v                 # Verbose
#   ./scripts/run_tests.sh -k "ordering"      # Run tests matching "ordering"
#   ./scripts/run_tests.sh --collect-only     # List tests without running

set -e

cd "$(dirname "$0")/.." || exit 1

echo "=============================================="
echo "EXAID Evaluation - Test Runner"
echo "=============================================="
echo ""
echo "Working directory: $(pwd)"
echo ""

# Run pytest from tests/ and src/ directories
# The pytest.ini configures test discovery
python -m pytest tests/ src/ "$@"

echo ""
echo "=============================================="
echo "Tests complete!"
echo "=============================================="








