#!/usr/bin/env python3
"""
EXAID Evaluation - Log Validation Script

Validates JSONL and JSONL.gz files against the appropriate JSON schema.
Supports validation of manifests, traces, runs, and metrics files.

Usage:
    python validate_logs.py <file_pattern> [--schema <schema_name>]
    python validate_logs.py data/manifests/*.jsonl
    python validate_logs.py data/traces/*.jsonl.gz
    python validate_logs.py data/runs/*/*.jsonl.gz
    python validate_logs.py data/metrics/*.jsonl
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Optional

import jsonschema
from jsonschema import Draft202012Validator


# Schema name to file mapping
SCHEMA_MAP = {
    "exaid.manifest": "exaid.manifest.schema.json",
    "exaid.trace": "exaid.trace.schema.json",
    "exaid.run": "exaid.run.schema.json",
    "exaid.metrics": "exaid.metrics.schema.json",
}

# Directory patterns to infer schema type
DIR_SCHEMA_MAP = {
    "manifests": "exaid.manifest",
    "traces": "exaid.trace",
    "runs": "exaid.run",
    "metrics": "exaid.metrics",
}


def get_schemas_dir() -> Path:
    """Get the schemas directory path."""
    # Relative to this script's location
    return Path(__file__).parent.parent / "schemas"


def load_schema(schema_name: str) -> dict:
    """Load a JSON schema by name."""
    schema_file = SCHEMA_MAP.get(schema_name)
    if not schema_file:
        raise ValueError(f"Unknown schema: {schema_name}")
    
    schema_path = get_schemas_dir() / schema_file
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_schema_from_path(file_path: Path) -> Optional[str]:
    """Infer the schema type from the file path."""
    parts = file_path.parts
    for part in parts:
        if part in DIR_SCHEMA_MAP:
            return DIR_SCHEMA_MAP[part]
    return None


def infer_schema_from_record(record: dict) -> Optional[str]:
    """Infer the schema type from the record's schema_name field."""
    return record.get("schema_name")


def open_file(file_path: Path):
    """Open a file, handling gzip compression."""
    if file_path.suffix == ".gz" or str(file_path).endswith(".jsonl.gz"):
        return gzip.open(file_path, "rt", encoding="utf-8")
    return open(file_path, "r", encoding="utf-8")


def validate_file(
    file_path: Path,
    schema_name: Optional[str] = None,
    verbose: bool = False
) -> tuple[int, int, list[str]]:
    """
    Validate a JSONL file against a schema.
    
    Returns:
        Tuple of (valid_count, invalid_count, error_messages)
    """
    valid_count = 0
    invalid_count = 0
    errors = []
    
    # Try to infer schema from path if not provided
    if not schema_name:
        schema_name = infer_schema_from_path(file_path)
    
    schema = None
    validator = None
    
    try:
        with open_file(file_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    errors.append(f"Line {line_num}: JSON parse error: {e}")
                    continue
                
                # Infer schema from record if not yet determined
                record_schema = schema_name or infer_schema_from_record(record)
                if not record_schema:
                    invalid_count += 1
                    errors.append(
                        f"Line {line_num}: Cannot determine schema "
                        "(no schema_name field and cannot infer from path)"
                    )
                    continue
                
                # Load schema if needed (lazy loading)
                if schema is None or record_schema != schema_name:
                    try:
                        schema = load_schema(record_schema)
                        validator = Draft202012Validator(schema)
                        schema_name = record_schema
                    except (ValueError, FileNotFoundError) as e:
                        invalid_count += 1
                        errors.append(f"Line {line_num}: Schema error: {e}")
                        continue
                
                # Validate the record
                validation_errors = list(validator.iter_errors(record))
                extra_errors = []
                if (
                    record_schema == "exaid.run"
                    and record.get("record_type") == "summary_event"
                ):
                    required_summary_fields = [
                        "trigger_type",
                        "summary_history_event_ids",
                        "summarizer_input_hash",
                        "limits_ok",
                        "failure_mode",
                    ]
                    for field in required_summary_fields:
                        if field not in record:
                            extra_errors.append(
                                f"Line {line_num}: {field}: missing required field"
                            )

                if validation_errors or extra_errors:
                    invalid_count += 1
                    for error in validation_errors:
                        path = ".".join(str(p) for p in error.absolute_path)
                        errors.append(
                            f"Line {line_num}: {path or 'root'}: {error.message}"
                        )
                    errors.extend(extra_errors)
                else:
                    valid_count += 1
                    if verbose:
                        print(f"  Line {line_num}: OK")
                        
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    
    return valid_count, invalid_count, errors


def validate_files(
    patterns: list[str],
    schema_name: Optional[str] = None,
    verbose: bool = False,
    strict: bool = False
) -> int:
    """
    Validate multiple files matching the given patterns.
    
    Returns:
        Exit code (0 for success, 1 for validation errors)
    """
    from glob import glob
    
    total_valid = 0
    total_invalid = 0
    total_files = 0
    total_empty = 0
    all_errors = []
    
    for pattern in patterns:
        # Expand glob pattern
        matching_files = glob(pattern, recursive=True)
        
        if not matching_files:
            print(f"No files matching pattern: {pattern}")
            # In non-strict mode, empty patterns are OK
            if strict:
                all_errors.append(f"No files matching pattern: {pattern}")
            continue
        
        for file_path_str in matching_files:
            file_path = Path(file_path_str)
            total_files += 1
            
            print(f"Validating: {file_path}")
            
            # Check if file is empty
            try:
                if file_path.stat().st_size == 0:
                    total_empty += 1
                    print(f"  Empty file (skipped)")
                    continue
                    
                # For gzipped files, check if content is empty
                if str(file_path).endswith(".gz"):
                    with gzip.open(file_path, "rt") as f:
                        content = f.read().strip()
                        if not content:
                            total_empty += 1
                            print(f"  Empty gzipped file (skipped)")
                            continue
            except Exception:
                pass  # Let validate_file handle the error
            
            valid, invalid, errors = validate_file(
                file_path, schema_name, verbose
            )
            
            total_valid += valid
            total_invalid += invalid
            all_errors.extend(errors)
            
            if errors:
                print(f"  {invalid} invalid, {valid} valid")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"    {error}")
                if len(errors) > 5:
                    print(f"    ... and {len(errors) - 5} more errors")
            else:
                print(f"  {valid} valid records")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Files processed: {total_files}")
    print(f"Empty files skipped: {total_empty}")
    print(f"Valid records: {total_valid}")
    print(f"Invalid records: {total_invalid}")
    
    if total_invalid > 0:
        print(f"\nValidation FAILED with {total_invalid} errors")
        return 1
    elif total_valid == 0 and total_files > 0 and total_empty == total_files:
        # All files were empty - this is OK for scaffolding
        print("\nValidation PASSED (all files empty - scaffold mode)")
        return 0
    else:
        print("\nValidation PASSED")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate EXAID evaluation log files against JSON schemas"
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        help="File patterns to validate (glob patterns supported)"
    )
    parser.add_argument(
        "--schema",
        choices=list(SCHEMA_MAP.keys()),
        help="Force a specific schema for all files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show validation status for each record"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no files match a pattern"
    )
    
    args = parser.parse_args()
    
    if not args.patterns:
        print("No file patterns provided. Use --help for usage information.")
        print("\nExample usage:")
        print("  python validate_logs.py data/manifests/*.jsonl")
        print("  python validate_logs.py data/traces/*.jsonl.gz")
        print("  python validate_logs.py 'data/runs/*/*.jsonl.gz'")
        return 0
    
    return validate_files(
        args.patterns,
        schema_name=args.schema,
        verbose=args.verbose,
        strict=args.strict
    )


if __name__ == "__main__":
    sys.exit(main())

