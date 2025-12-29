#!/usr/bin/env python3
"""
EXAID Trace Validation Script (v2.0.0)

Validates timed trace files for schema compliance and integrity.

Validation Rules:
1. Global seq strictly increasing across entire trace
2. t_emitted_ms non-decreasing across all stream_delta records
3. t_rel_ms consistency:
   - For stream_delta: t_rel_ms == t_emitted_ms - t0 (always >= 0)
   - For turn_boundary: t_rel_ms == t_ms - t0 (MAY be negative!)
4. Turn boundaries: each turn has matching start/end
5. Boundary time consistency:
   - turn_start.t_ms <= first_delta.t_emitted_ms for that turn
   - turn_end.t_ms >= last_delta.t_emitted_ms for that turn
   - NOTE: ±2ms tolerance allowed (BOUNDARY_TIME_EPSILON_MS) due to
     millisecond resolution; violations within epsilon are warnings, not errors
6. content_hash matches recomputed hash for turn deltas (warning only, never an error)

Stub Mode Warning:
- Traces with stub_mode=true are flagged and should not be used for evaluation

Usage (from evals/ directory or Docker container):
    python -m evals.src.validate_traces --traces data/traces/ --verbose
"""

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Import trace_text module for canonical text validation
try:
    from trace_text import build_canonical_trace_text, TraceParsingError
except ImportError:
    # Fallback if running from different directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from trace_text import build_canonical_trace_text, TraceParsingError

# Boundary timestamp tolerance (milliseconds)
# Due to millisecond resolution and async timing, boundary timestamps may be
# off by 1-2ms from delta timestamps. This is cosmetic and does not affect
# evaluation semantics.
BOUNDARY_TIME_EPSILON_MS = 2


@dataclass
class ValidationStats:
    """Statistics from trace validation."""
    total_records: int = 0
    trace_meta_count: int = 0
    stream_delta_count: int = 0
    turn_boundary_count: int = 0
    turn_count: int = 0
    
    # Validation results
    seq_violations: int = 0
    timestamp_violations: int = 0
    t_rel_violations: int = 0
    boundary_mismatches: int = 0
    boundary_time_violations: int = 0  # turn_start > first_delta or turn_end < last_delta (> epsilon)
    boundary_time_warnings: int = 0    # boundary time off by <= BOUNDARY_TIME_EPSILON_MS (cosmetic)
    content_hash_mismatches: int = 0
    missing_fields: List[str] = field(default_factory=list)
    
    # Stub mode detection
    is_stub_mode: bool = False
    
    # Canonical text validation
    canonical_text_empty: bool = False
    canonical_text_length: int = 0


class TraceValidationError(Exception):
    """Raised when trace validation fails."""
    pass


def load_trace_records(trace_path: Path) -> List[dict]:
    """
    Load all records from a trace file.
    
    Args:
        trace_path: Path to .jsonl or .jsonl.gz file
        
    Returns:
        List of record dicts
    """
    records = []
    
    if trace_path.suffix == ".gz":
        open_fn = gzip.open
    else:
        open_fn = open
    
    with open_fn(trace_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise TraceValidationError(
                    f"JSON parse error at line {line_num}: {e}"
                )
    
    return records


def validate_trace_file(
    trace_path: Path,
    verbose: bool = False
) -> Tuple[bool, ValidationStats]:
    """
    Validate a single trace file.
    
    Validation Rules:
    1. First record must be trace_meta with required fields
    2. Global seq strictly increasing across ALL records
    3. t_emitted_ms non-decreasing across stream_delta records
    4. t_rel_ms consistency:
       - For stream_delta: t_rel_ms == t_emitted_ms - t0 (always >= 0)
       - For turn_boundary: t_rel_ms == t_ms - t0 (MAY be negative!)
    5. Turn boundary pairs: each turn has start and end
    6. Boundary time consistency:
       - turn_start.t_ms <= first_delta.t_emitted_ms for that turn
       - turn_end.t_ms >= last_delta.t_emitted_ms for that turn
    7. content_hash matches recomputed hash for turn deltas (warning only, never an error)
    
    Args:
        trace_path: Path to trace file
        verbose: Print detailed output
        
    Returns:
        Tuple of (is_valid, stats)
    """
    stats = ValidationStats()
    errors = []
    warnings = []  # Cosmetic issues within tolerance
    
    # Load records
    try:
        records = load_trace_records(trace_path)
    except TraceValidationError as e:
        errors.append(str(e))
        return False, stats
    
    if not records:
        errors.append("Trace file is empty")
        return False, stats
    
    stats.total_records = len(records)
    
    # Validate first record is trace_meta
    first_record = records[0]
    if first_record.get("record_type") != "trace_meta":
        errors.append(f"First record must be trace_meta, got: {first_record.get('record_type')}")
        return False, stats
    
    stats.trace_meta_count = 1
    
    # Check for stub_mode
    if first_record.get("stub_mode", False):
        stats.is_stub_mode = True
        if verbose:
            print("    WARNING: Trace was generated in stub mode (not real MAC execution)")
    
    # Validate canonical trace text is non-empty for real traces
    if not stats.is_stub_mode:
        try:
            canonical_text, parsing_stats = build_canonical_trace_text(
                trace_path, fail_on_empty=False
            )
            stats.canonical_text_length = len(canonical_text)
            if not canonical_text.strip():
                stats.canonical_text_empty = True
                errors.append(
                    f"Canonical trace text is empty (no message chunks found). "
                    f"Stats: total_records={parsing_stats.total_records}, "
                    f"chunk_records={parsing_stats.chunk_records}, "
                    f"included_message_chunks={parsing_stats.included_message_chunks}"
                )
        except TraceParsingError as e:
            # If parsing fails, it's already an error
            stats.canonical_text_empty = True
            errors.append(f"Failed to build canonical trace text: {e}")
        except Exception as e:
            # Other errors (e.g., import issues) should not fail validation
            if verbose:
                warnings.append(f"Could not validate canonical text: {e}")
    
    # Extract t0 from trace_meta
    t0 = first_record.get("t0_emitted_ms")
    if t0 is None:
        errors.append("trace_meta missing t0_emitted_ms")
        stats.missing_fields.append("t0_emitted_ms")
        return False, stats
    
    # Validate trace_meta required fields
    required_meta_fields = [
        "schema_version", "case_id", "mas_run_id", "mac_commit", "model", "created_at"
    ]
    for field_name in required_meta_fields:
        if field_name not in first_record:
            stats.missing_fields.append(f"trace_meta.{field_name}")
    
    # Tracking for validation
    prev_seq = -1
    prev_t_emitted = -1
    turn_starts: Dict[int, dict] = {}  # turn_id -> start boundary record
    turn_ends: Dict[int, dict] = {}    # turn_id -> end boundary record
    turn_deltas: Dict[int, List[str]] = {}  # turn_id -> list of delta_text
    turn_first_delta_t: Dict[int, int] = {}  # turn_id -> first delta t_emitted_ms
    turn_last_delta_t: Dict[int, int] = {}   # turn_id -> last delta t_emitted_ms
    
    # Process remaining records
    for idx, record in enumerate(records[1:], 2):
        record_type = record.get("record_type")
        
        if record_type == "stream_delta":
            stats.stream_delta_count += 1
            
            # Check required fields
            required_fields = ["case_id", "seq", "turn_id", "agent_id", "delta_text", "t_emitted_ms", "t_rel_ms"]
            for field_name in required_fields:
                if field_name not in record:
                    stats.missing_fields.append(f"stream_delta[{idx}].{field_name}")
            
            seq = record.get("seq", -1)
            t_emitted = record.get("t_emitted_ms", 0)
            t_rel = record.get("t_rel_ms", -1)
            turn_id = record.get("turn_id", 0)
            delta_text = record.get("delta_text", "")
            
            # Validate seq strictly increasing
            if seq <= prev_seq:
                stats.seq_violations += 1
                if verbose:
                    errors.append(f"Line {idx}: seq {seq} not > prev {prev_seq}")
            prev_seq = seq
            
            # Validate t_emitted non-decreasing
            if t_emitted < prev_t_emitted:
                stats.timestamp_violations += 1
                if verbose:
                    errors.append(f"Line {idx}: t_emitted {t_emitted} < prev {prev_t_emitted}")
            prev_t_emitted = t_emitted
            
            # Validate t_rel_ms for deltas (always >= 0)
            expected_t_rel = t_emitted - t0
            if t_rel != expected_t_rel:
                stats.t_rel_violations += 1
                if verbose:
                    errors.append(f"Line {idx}: t_rel {t_rel} != expected {expected_t_rel}")
            
            # Accumulate deltas for content hash verification
            if turn_id not in turn_deltas:
                turn_deltas[turn_id] = []
            turn_deltas[turn_id].append(delta_text)
            
            # Track first/last delta times for boundary validation
            if turn_id not in turn_first_delta_t:
                turn_first_delta_t[turn_id] = t_emitted
            turn_last_delta_t[turn_id] = t_emitted
            
        elif record_type == "turn_boundary":
            stats.turn_boundary_count += 1
            
            # Check required fields
            required_fields = ["case_id", "turn_id", "agent_id", "boundary", "seq", "t_ms", "t_rel_ms"]
            for field_name in required_fields:
                if field_name not in record:
                    stats.missing_fields.append(f"turn_boundary[{idx}].{field_name}")
            
            seq = record.get("seq", -1)
            boundary = record.get("boundary")
            turn_id = record.get("turn_id", 0)
            t_ms = record.get("t_ms", 0)
            t_rel = record.get("t_rel_ms", 0)
            
            # Validate seq strictly increasing (boundaries also have seq)
            if seq <= prev_seq:
                stats.seq_violations += 1
                if verbose:
                    errors.append(f"Line {idx}: boundary seq {seq} not > prev {prev_seq}")
            prev_seq = seq
            
            # Validate t_rel_ms for boundaries (MAY be negative!)
            expected_t_rel = t_ms - t0
            if t_rel != expected_t_rel:
                stats.t_rel_violations += 1
                if verbose:
                    errors.append(f"Line {idx}: boundary t_rel {t_rel} != expected {expected_t_rel}")
            
            if boundary == "start":
                turn_starts[turn_id] = record
                stats.turn_count += 1
            elif boundary == "end":
                turn_ends[turn_id] = record
                
                # Check for matching start
                if turn_id not in turn_starts:
                    stats.boundary_mismatches += 1
                    if verbose:
                        errors.append(f"Line {idx}: turn {turn_id} end without start")
                
                # Verify content_hash (warning only, never an error)
                content_hash = record.get("content_hash")
                if content_hash and turn_id in turn_deltas:
                    deltas = turn_deltas[turn_id]
                    content = "".join(deltas)
                    expected_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
                    if content_hash != expected_hash:
                        stats.content_hash_mismatches += 1
                        if verbose:
                            warnings.append(f"Line {idx}: content_hash mismatch for turn {turn_id}")
                
                # Remove from tracking (turn completed)
                if turn_id in turn_starts:
                    del turn_starts[turn_id]
        
        elif record_type == "trace_meta":
            # Should only appear once at the start
            errors.append(f"Line {idx}: unexpected trace_meta record")
    
    # Check for unclosed turns
    if turn_starts:
        for turn_id in turn_starts:
            stats.boundary_mismatches += 1
            if verbose:
                errors.append(f"Turn {turn_id} has start but no end")
    
    # Check boundary time consistency (after all records processed)
    for turn_id, end_record in turn_ends.items():
        if turn_id in turn_first_delta_t and turn_id in turn_last_delta_t:
            # Get start record
            # Note: turn_starts dict is cleared as ends are processed, so we need to check end record's turn
            # We'll do this by re-parsing, but for now check end boundaries
            first_delta_t = turn_first_delta_t[turn_id]
            last_delta_t = turn_last_delta_t[turn_id]
            end_t_ms = end_record.get("t_ms", 0)
            
            # turn_end.t_ms should be >= last_delta.t_emitted_ms
            # Allow small epsilon for millisecond resolution timing
            if end_t_ms < last_delta_t:
                diff_ms = last_delta_t - end_t_ms
                if diff_ms <= BOUNDARY_TIME_EPSILON_MS:
                    # Within tolerance - warning only (cosmetic)
                    stats.boundary_time_warnings += 1
                    if verbose:
                        warnings.append(f"Turn {turn_id}: end.t_ms {end_t_ms} < last_delta.t {last_delta_t} (diff={diff_ms}ms, within epsilon)")
                else:
                    # Beyond tolerance - error
                    stats.boundary_time_violations += 1
                    if verbose:
                        errors.append(f"Turn {turn_id}: end.t_ms {end_t_ms} < last_delta.t {last_delta_t} (diff={diff_ms}ms)")
    
    # Re-check turn starts (need to re-parse for this validation)
    # For simplicity, we'll do a second pass to validate turn_start <= first_delta
    turn_start_times = {}
    for idx, record in enumerate(records[1:], 2):
        if record.get("record_type") == "turn_boundary" and record.get("boundary") == "start":
            turn_id = record.get("turn_id", 0)
            turn_start_times[turn_id] = record.get("t_ms", 0)
    
    for turn_id, start_t_ms in turn_start_times.items():
        if turn_id in turn_first_delta_t:
            first_delta_t = turn_first_delta_t[turn_id]
            # turn_start.t_ms should be <= first_delta.t_emitted_ms
            # Allow small epsilon for millisecond resolution timing
            if start_t_ms > first_delta_t:
                diff_ms = start_t_ms - first_delta_t
                if diff_ms <= BOUNDARY_TIME_EPSILON_MS:
                    # Within tolerance - warning only (cosmetic)
                    stats.boundary_time_warnings += 1
                    if verbose:
                        warnings.append(f"Turn {turn_id}: start.t_ms {start_t_ms} > first_delta.t {first_delta_t} (diff={diff_ms}ms, within epsilon)")
                else:
                    # Beyond tolerance - error
                    stats.boundary_time_violations += 1
                    if verbose:
                        errors.append(f"Turn {turn_id}: start.t_ms {start_t_ms} > first_delta.t {first_delta_t} (diff={diff_ms}ms)")
    
    # Determine overall validity
    # Note: content_hash_mismatches are warnings only, not errors
    # canonical_text_empty is an error for real traces (not stubs)
    is_valid = (
        stats.seq_violations == 0 and
        stats.timestamp_violations == 0 and
        stats.t_rel_violations == 0 and
        stats.boundary_mismatches == 0 and
        stats.boundary_time_violations == 0 and
        len(stats.missing_fields) == 0 and
        len(errors) == 0 and
        not stats.canonical_text_empty  # Real traces must have non-empty canonical text
    )
    
    if verbose and errors:
        for error in errors[:10]:  # Limit output
            print(f"    ERROR: {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")
    
    if verbose and warnings:
        for warning in warnings[:5]:  # Limit output
            print(f"    WARNING: {warning}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more warnings")
    
    return is_valid, stats


def validate_all_traces(
    traces_dir: Path,
    verbose: bool = False
) -> Dict:
    """
    Validate all trace files in a directory.
    
    Args:
        traces_dir: Path to traces directory
        verbose: Print per-file details
        
    Returns:
        Validation report dict
    """
    # Find trace files
    trace_files = sorted(traces_dir.glob("*.jsonl.gz"))
    if not trace_files:
        trace_files = sorted(traces_dir.glob("*.jsonl"))
    
    if not trace_files:
        raise TraceValidationError(f"No trace files found in {traces_dir}")
    
    report = {
        "n_files": len(trace_files),
        "valid_files": 0,
        "invalid_files": 0,
        "stub_mode_files": 0,
        "total_records": 0,
        "total_stream_deltas": 0,
        "total_turns": 0,
        "total_seq_violations": 0,
        "total_timestamp_violations": 0,
        "total_t_rel_violations": 0,
        "total_boundary_mismatches": 0,
        "total_boundary_time_violations": 0,
        "total_boundary_time_warnings": 0,  # Cosmetic timing (within epsilon)
        "total_content_hash_mismatches": 0,
        "total_canonical_text_empty": 0,  # Real traces with empty canonical text
        "files_with_errors": [],
        "files_with_warnings": [],  # Files with cosmetic timing issues
        "stub_mode_files_list": [],
        "per_file_stats": {}
    }
    
    for trace_file in trace_files:
        case_id = trace_file.stem.replace(".trace.jsonl", "").replace(".jsonl", "")
        
        if verbose:
            print(f"Validating {case_id}...", end=" ")
        
        try:
            is_valid, stats = validate_trace_file(trace_file, verbose=verbose)
        except TraceValidationError as e:
            is_valid = False
            stats = ValidationStats()
            if verbose:
                print(f"FAILED: {e}")
            report["files_with_errors"].append({"case_id": case_id, "error": str(e)})
            report["invalid_files"] += 1
            continue
        
        if is_valid:
            report["valid_files"] += 1
            stub_marker = " [STUB]" if stats.is_stub_mode else ""
            if verbose:
                print(f"OK ({stats.stream_delta_count} deltas, {stats.turn_count} turns){stub_marker}")
        else:
            report["invalid_files"] += 1
            report["files_with_errors"].append({
                "case_id": case_id,
                "seq_violations": stats.seq_violations,
                "timestamp_violations": stats.timestamp_violations,
                "t_rel_violations": stats.t_rel_violations,
                "boundary_mismatches": stats.boundary_mismatches,
                "boundary_time_violations": stats.boundary_time_violations,
                "content_hash_mismatches": stats.content_hash_mismatches,
                "missing_fields": stats.missing_fields[:5]  # Limit
            })
            if verbose:
                print(f"FAILED")
        
        # Track stub mode files
        if stats.is_stub_mode:
            report["stub_mode_files"] += 1
            report["stub_mode_files_list"].append(case_id)
        
        # Aggregate stats
        report["total_records"] += stats.total_records
        report["total_stream_deltas"] += stats.stream_delta_count
        report["total_turns"] += stats.turn_count
        report["total_seq_violations"] += stats.seq_violations
        report["total_timestamp_violations"] += stats.timestamp_violations
        report["total_t_rel_violations"] += stats.t_rel_violations
        report["total_boundary_mismatches"] += stats.boundary_mismatches
        report["total_boundary_time_violations"] += stats.boundary_time_violations
        report["total_boundary_time_warnings"] += stats.boundary_time_warnings
        report["total_content_hash_mismatches"] += stats.content_hash_mismatches
        if stats.canonical_text_empty:
            report["total_canonical_text_empty"] += 1
        
        # Track files with warnings (but still valid)
        if stats.boundary_time_warnings > 0:
            report["files_with_warnings"].append(case_id)
        
        report["per_file_stats"][case_id] = {
            "is_valid": is_valid,
            "is_stub_mode": stats.is_stub_mode,
            "records": stats.total_records,
            "stream_deltas": stats.stream_delta_count,
            "turns": stats.turn_count,
            "seq_violations": stats.seq_violations,
            "timestamp_violations": stats.timestamp_violations,
            "boundary_time_violations": stats.boundary_time_violations,
            "canonical_text_empty": stats.canonical_text_empty,
            "canonical_text_length": stats.canonical_text_length
        }
    
    return report


def save_validation_report(report: dict, output_path: Path):
    """Save validation report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Validate timed traces for EXAID evaluation (v2.0.0)"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("data/traces"),
        help="Input traces directory (default: data/traces)"
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
    print("EXAID Trace Validation (v2.0.0)")
    print("=" * 60)
    print()
    print(f"Traces directory: {args.traces}")
    print()
    print("Validation Rules:")
    print("  1. Global seq strictly increasing")
    print("  2. t_emitted_ms non-decreasing for stream_delta")
    print("  3. t_rel_ms == t_emitted_ms - t0_emitted_ms")
    print("  4. Turn boundary start/end pairs match")
    print("  5. Boundary time consistency:")
    print("     - turn_start.t_ms <= first_delta.t_emitted_ms")
    print("     - turn_end.t_ms >= last_delta.t_emitted_ms")
    print("     - ±2ms tolerance allowed (warnings within tolerance)")
    print("  6. content_hash matches recomputed hash (warning only)")
    print("  7. Canonical trace text is non-empty for real traces (not stubs)")
    print()
    
    try:
        report = validate_all_traces(
            args.traces,
            verbose=args.verbose
        )
    except TraceValidationError as e:
        print(f"VALIDATION FAILED: {e}")
        return 1
    
    # Determine overall result
    all_valid = report["invalid_files"] == 0
    
    print()
    print("=" * 60)
    if all_valid:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 60)
    print()
    print(f"  Total files: {report['n_files']}")
    print(f"  Valid files: {report['valid_files']}")
    print(f"  Invalid files: {report['invalid_files']}")
    print()
    print(f"  Total records: {report['total_records']}")
    print(f"  Total stream_deltas: {report['total_stream_deltas']}")
    print(f"  Total turns: {report['total_turns']}")
    print()
    print("Violations (errors):")
    print(f"  seq violations: {report['total_seq_violations']}")
    print(f"  timestamp violations: {report['total_timestamp_violations']}")
    print(f"  t_rel violations: {report['total_t_rel_violations']}")
    print(f"  boundary mismatches: {report['total_boundary_mismatches']}")
    print(f"  boundary time violations: {report['total_boundary_time_violations']}")
    if report['total_canonical_text_empty'] > 0:
        print(f"  canonical text empty: {report['total_canonical_text_empty']}")
    
    # Show warnings (non-blocking issues)
    has_warnings = (
        report["total_content_hash_mismatches"] > 0 or
        report["total_boundary_time_warnings"] > 0
    )
    if has_warnings:
        print()
        print("Warnings (non-blocking):")
        if report["total_content_hash_mismatches"] > 0:
            print(f"  content_hash mismatches: {report['total_content_hash_mismatches']}")
        if report["total_boundary_time_warnings"] > 0:
            print(f"  boundary time warnings (within {BOUNDARY_TIME_EPSILON_MS}ms tolerance): {report['total_boundary_time_warnings']}")
            print(f"  Files with boundary time warnings: {len(report['files_with_warnings'])}")
            for warn_file in report["files_with_warnings"][:5]:
                print(f"    - {warn_file}")
            if len(report["files_with_warnings"]) > 5:
                print(f"    ... and {len(report['files_with_warnings']) - 5} more")
    
    # Stub mode warning
    if report["stub_mode_files"] > 0:
        print()
        print(f"WARNING: {report['stub_mode_files']} file(s) generated in STUB MODE")
        print("  Stub traces must NOT be used for evaluation!")
        for stub_file in report["stub_mode_files_list"][:5]:
            print(f"    - {stub_file}")
        if len(report["stub_mode_files_list"]) > 5:
            print(f"    ... and {len(report['stub_mode_files_list']) - 5} more")
    
    if report["files_with_errors"]:
        print()
        print("Files with errors:")
        for entry in report["files_with_errors"][:10]:
            print(f"  - {entry.get('case_id', 'unknown')}")
        if len(report["files_with_errors"]) > 10:
            print(f"  ... and {len(report['files_with_errors']) - 10} more")
    
    if args.output:
        save_validation_report(report, args.output)
        print(f"\nReport saved to: {args.output}")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
