#!/usr/bin/env python3
"""
Validate frozen traces before stoplist generation or metrics.

Paper hook: "Trace files were validated to contain canonical message chunks
before any downstream processing (Section 3.1)"

This script validates that:
    1. All trace files exist and are readable
    2. Each trace contains at least one canonical message chunk
    3. JSON parsing succeeds for all records
    4. Statistics are collected for audit

Dependencies:
    - trace_text.py (canonical text construction)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from trace_text import (
    build_canonical_trace_text,
    TraceParsingError,
    TraceParsingStats,
)


def validate_all_traces(
    traces_dir: Path,
    min_chunks: int = 1,
    verbose: bool = False
) -> dict:
    """
    Validate all trace files have canonical message content.
    
    Paper hook: "Trace validation ensures all 40 cases contain canonical
    message chunks before stoplist generation or metrics (Section 3.1)"
    
    Args:
        traces_dir: Path to data/traces/
        min_chunks: Minimum message chunks required per file
        verbose: Print per-file details
        
    Returns:
        Validation report dict
        
    Raises:
        TraceParsingError: If any trace fails validation
    """
    trace_files = sorted(traces_dir.glob("*.jsonl.gz"))
    
    if not trace_files:
        # Also check for uncompressed files
        trace_files = sorted(traces_dir.glob("*.jsonl"))
    
    if not trace_files:
        raise TraceParsingError(f"No trace files found in {traces_dir}")
    
    report = {
        "n_files": len(trace_files),
        "valid_files": 0,
        "total_message_chunks": 0,
        "total_excluded_chunks": 0,
        "files_with_missing_subtype": 0,
        "per_file_stats": {}
    }
    
    for trace_file in trace_files:
        case_id = trace_file.stem.replace(".jsonl", "")
        
        if verbose:
            print(f"Validating {case_id}...", end=" ")
        
        try:
            _, stats = build_canonical_trace_text(trace_file, fail_on_empty=True)
        except TraceParsingError as e:
            raise TraceParsingError(f"Validation failed for {case_id}: {e}")
        
        if stats.included_message_chunks < min_chunks:
            raise TraceParsingError(
                f"{case_id} has only {stats.included_message_chunks} message chunks "
                f"(minimum: {min_chunks})"
            )
        
        report["valid_files"] += 1
        report["total_message_chunks"] += stats.included_message_chunks
        report["total_excluded_chunks"] += (
            stats.excluded_orchestrator_summary +
            stats.excluded_system_note +
            stats.excluded_other_subtype
        )
        
        if stats.missing_event_subtype > 0:
            report["files_with_missing_subtype"] += 1
        
        report["per_file_stats"][case_id] = {
            "message_chunks": stats.included_message_chunks,
            "excluded": (
                stats.excluded_orchestrator_summary +
                stats.excluded_system_note +
                stats.excluded_other_subtype
            ),
            "missing_subtype": stats.missing_event_subtype
        }
        
        if verbose:
            print(f"OK ({stats.included_message_chunks} chunks)")
    
    return report


def save_validation_report(report: dict, output_path: Path):
    """Save validation report to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Validate frozen traces for EXAID evaluation"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("data/traces"),
        help="Input traces directory (default: data/traces)"
    )
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=1,
        help="Minimum message chunks required per file (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output validation report JSON file (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-file validation details"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXAID Trace Validation")
    print("=" * 60)
    print()
    print(f"Traces directory: {args.traces}")
    print(f"Minimum chunks per file: {args.min_chunks}")
    print()
    
    try:
        report = validate_all_traces(
            args.traces,
            min_chunks=args.min_chunks,
            verbose=args.verbose
        )
    except TraceParsingError as e:
        print(f"VALIDATION FAILED: {e}")
        return 1
    
    print()
    print("=" * 60)
    print("VALIDATION PASSED")
    print("=" * 60)
    print()
    print(f"  Total files: {report['n_files']}")
    print(f"  Valid files: {report['valid_files']}")
    print(f"  Total message chunks: {report['total_message_chunks']}")
    print(f"  Total excluded chunks: {report['total_excluded_chunks']}")
    print(f"  Files with missing event_subtype: {report['files_with_missing_subtype']}")
    
    if args.output:
        save_validation_report(report, args.output)
        print(f"\nReport saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




