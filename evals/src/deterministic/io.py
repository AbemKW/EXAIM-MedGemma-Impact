#!/usr/bin/env python3
"""
Deterministic I/O utilities.

Paper hook: "Run logs written with deterministic serialization for
byte-stable verification (Section 3.2)"

Requirements for byte-identical outputs:
    1. Records written in deterministic order
    2. JSON: sort_keys=True, separators=(',',':')
    3. gzip: mtime=0 (no timestamp in archive)
    4. Proper file handle management (no leaks)

Dependencies:
    - Python 3.10+
    - gzip (stdlib)
    - json (stdlib)
"""

import gzip
import json
from pathlib import Path
from typing import Any


def write_run_log_deterministic(records: list[dict], output_path: Path):
    """
    Write run log with deterministic gzip and JSON.
    
    Paper hook: "Run logs serialized with sort_keys=True, compact separators,
    and gzip mtime=0 for byte-stable verification (Section 3.2)"
    
    FIXED: Proper file handle management (no leaks).
    
    Requirements:
        - Records written in deterministic order (by record_type, then index)
        - JSON: sort_keys=True, separators=(',',':')
        - gzip: mtime=0
        
    Args:
        records: List of record dicts to write
        output_path: Output file path (should end in .jsonl.gz)
    """
    # Sort records deterministically
    def record_sort_key(r: dict) -> tuple:
        type_order = {
            "run_meta": 0,
            "tokengate_flush": 1,
            "buffer_decision": 2,
            "summary_event": 3
        }
        record_type = r.get("record_type", "")
        order = type_order.get(record_type, 99)
        
        # Secondary sort by index field
        index = r.get("event_index", r.get("decision_index", r.get("flush_index", 0)))
        
        return (order, index)
    
    sorted_records = sorted(records, key=record_sort_key)
    
    # Serialize to JSON lines with deterministic formatting
    json_lines = []
    for record in sorted_records:
        line = json.dumps(
            record,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False
        )
        json_lines.append(line)
    
    content = "\n".join(json_lines) + "\n"
    content_bytes = content.encode("utf-8")
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # FIXED: Proper context manager nesting (no handle leaks)
    with open(output_path, "wb") as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            mtime=0  # Deterministic modification time
        ) as gz_file:
            gz_file.write(content_bytes)


def read_run_log(run_log_path: Path) -> list[dict]:
    """
    Read run log, handling both gzip and plain JSONL.
    
    Args:
        run_log_path: Path to run log file
        
    Returns:
        List of record dicts
    """
    open_fn = gzip.open if str(run_log_path).endswith(".gz") else open
    records = []
    
    with open_fn(run_log_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    return records


def write_json_deterministic(data: Any, output_path: Path):
    """
    Write JSON file with deterministic formatting.
    
    Args:
        data: Data to serialize
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = json.dumps(
        data,
        sort_keys=True,
        indent=2,  # Indent for readability
        ensure_ascii=False
    )
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")


def write_jsonl_deterministic(records: list[dict], output_path: Path):
    """
    Write JSONL file with deterministic formatting (no gzip).
    
    Args:
        records: List of record dicts
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(
                record,
                sort_keys=True,
                separators=(',', ':'),
                ensure_ascii=False
            )
            f.write(line)
            f.write("\n")


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA-256 hash of file contents.

    Args:
        file_path: Path to file

    Returns:
        Hash string prefixed with "sha256:"
    """
    if not file_path.exists():
        return "sha256:" + "0" * 64

    import hashlib

    h = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return f"sha256:{h.hexdigest()}"


def verify_determinism(file_path: Path, expected_hash: str) -> bool:
    """
    Verify file matches expected hash.
    
    Args:
        file_path: Path to file
        expected_hash: Expected hash string
        
    Returns:
        True if hashes match
    """
    actual_hash = compute_file_hash(file_path)
    return actual_hash == expected_hash


class RunLogBuilder:
    """
    Builder for constructing run logs with proper record ordering.
    
    Ensures all records are collected and written in deterministic order.
    """
    
    def __init__(self):
        self.run_meta: dict = {}
        self.tokengate_flushes: list[dict] = []
        self.buffer_decisions: list[dict] = []
        self.summary_events: list[dict] = []
    
    def set_run_meta(self, meta: dict):
        """Set run_meta record."""
        meta["record_type"] = "run_meta"
        self.run_meta = meta
    
    def add_tokengate_flush(self, flush: dict):
        """Add tokengate_flush record."""
        flush["record_type"] = "tokengate_flush"
        self.tokengate_flushes.append(flush)
    
    def add_buffer_decision(self, decision: dict):
        """Add buffer_decision record."""
        decision["record_type"] = "buffer_decision"
        self.buffer_decisions.append(decision)
    
    def add_summary_event(self, event: dict):
        """Add summary_event record."""
        event["record_type"] = "summary_event"
        self.summary_events.append(event)
    
    def build(self) -> list[dict]:
        """
        Build final record list in deterministic order.
        
        Returns:
            List of all records in correct order
        """
        records = []
        
        if self.run_meta:
            records.append(self.run_meta)
        
        # Sort each type by index
        for flush in sorted(self.tokengate_flushes, key=lambda x: x.get("flush_index", 0)):
            records.append(flush)
        
        for decision in sorted(self.buffer_decisions, key=lambda x: x.get("decision_index", 0)):
            records.append(decision)
        
        for event in sorted(self.summary_events, key=lambda x: x.get("event_index", 0)):
            records.append(event)
        
        return records
    
    def write(self, output_path: Path):
        """
        Write run log to file.
        
        Args:
            output_path: Output file path
        """
        records = self.build()
        write_run_log_deterministic(records, output_path)




