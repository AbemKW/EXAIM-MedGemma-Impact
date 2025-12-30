"""Dataset integrity and provenance helpers."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from trace_text import iter_trace_records

from config_loader import compute_file_hash


def load_trace_meta(trace_file: Path) -> Optional[dict]:
    """Load trace_meta record from a trace file."""
    for record in iter_trace_records(trace_file):
        if record.get("record_type") == "trace_meta":
            return record
    return None


def compute_manifest_hash(
    mas_run_id: str,
    case_list_hash: str,
    trace_entries: List[Tuple[str, str]],
) -> str:
    """Compute manifest trace_dataset_hash per schema."""
    canonical = {
        "mas_run_id": mas_run_id,
        "case_list_hash": case_list_hash,
        "traces": sorted(trace_entries),
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical_json.encode()).hexdigest()}"


def safe_file_hash(path: Path) -> Optional[str]:
    """Compute file hash if it exists, else return None."""
    if path.exists():
        return compute_file_hash(path)
    return None


def load_manifest_provenance(manifest_path: Path) -> dict:
    """Load manifest provenance and validate its hash."""
    manifest_meta = {}
    provenance = {}
    trace_entries = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_type = record.get("record_type")
            if record_type == "manifest_meta":
                manifest_meta = record
            elif record_type == "provenance":
                provenance = record
            elif record_type == "trace_entry":
                trace_entries.append((record.get("case_id", ""), record.get("sha256", "")))

    mas_run_id = manifest_meta.get("mas_run_id", "")
    case_list_hash = provenance.get("case_list_hash", "")
    computed_hash = compute_manifest_hash(mas_run_id, case_list_hash, trace_entries)
    manifest_hash = provenance.get("trace_dataset_hash")

    return {
        "dataset_id": manifest_meta.get("dataset_id"),
        "mas_run_id": mas_run_id,
        "case_list_hash": case_list_hash,
        "config_hash": provenance.get("config_hash"),
        "manifest_hash": manifest_hash,
        "computed_hash": computed_hash,
        "manifest_hash_valid": manifest_hash == computed_hash,
        "stub_mode": manifest_meta.get("stub_mode", False),
        "manifest_path": str(manifest_path),
    }


def get_git_commit(repo_root: Path) -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
