"""Hashing helpers shared across evaluation CLI workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

from ..metrics.integrity import load_manifest_provenance
from ..deterministic.utils import compute_text_hash


def compute_config_hash(*config_paths: Path) -> str:
    """
    Compute SHA256 hash of concatenated config files.

    Args:
        config_paths: Paths to config files

    Returns:
        Hash in format "sha256:xxx"
    """
    hasher = hashlib.sha256()
    for path in sorted(config_paths):  # Sort for determinism
        if path.exists():
            with open(path, "rb") as f:
                hasher.update(f.read())
    return f"sha256:{hasher.hexdigest()}"


def compute_trace_dataset_hash_from_entries(
    mas_run_id: str,
    case_list_hash: str,
    trace_entries: List[Tuple[str, str]],
) -> str:
    """
    Compute trace_dataset_hash for eval_run_id derivation.

    Definition: SHA256 of canonical manifest fields (NOT raw file bytes).

    Args:
        mas_run_id: MAS generation campaign ID
        case_list_hash: SHA256 of case list file
        trace_entries: List of (case_id, trace_sha256) tuples

    Returns:
        Hash in format "sha256:xxx"
    """
    canonical = {
        "mas_run_id": mas_run_id,
        "case_list_hash": case_list_hash,
        "traces": sorted(trace_entries),  # Sort for determinism
    }

    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    hash_digest = hashlib.sha256(canonical_json.encode()).hexdigest()
    return f"sha256:{hash_digest}"


def compute_trace_dataset_hash_from_manifest(
    traces_root: Path,
    manifest_path: Optional[Path] = None,
) -> str:
    """
    Compute trace_dataset_hash per schema definition using manifest provenance.

    Schema: SHA256 of JSON: {mas_run_id, case_list_hash, sorted [(case_id, trace_sha256)]}

    Args:
        traces_root: Directory that contains trace files
        manifest_path: Optional path to manifest file

    Returns:
        Hash in format "sha256:xxxx"
    """
    manifest_candidate = manifest_path if manifest_path and manifest_path.exists() else None

    if not manifest_candidate:
        import glob

        possible_manifest_dirs = [
            traces_root.parent / "manifests",
            traces_root.parent.parent / "manifests",
            traces_root,
        ]
        for manifest_dir in possible_manifest_dirs:
            if not manifest_dir.exists():
                continue
            matches = sorted(glob.glob(str(manifest_dir / "*.manifest.jsonl")))
            if matches:
                manifest_candidate = Path(matches[0])
                break

    if not manifest_candidate:
        raise ValueError(
            "Cannot compute trace_dataset_hash without a manifest file. "
            "Please provide --manifest or place a manifest in a known location."
        )

    try:
        manifest_info = load_manifest_provenance(manifest_candidate)
    except Exception as exc:
        raise ValueError(
            f"Failed to load manifest provenance from {manifest_candidate}: {exc}"
        ) from exc
    if not manifest_info.get("manifest_hash_valid", False):
        raise ValueError(
            "Manifest trace_dataset_hash does not match computed value: "
            f"{manifest_info.get('manifest_hash')} != {manifest_info.get('computed_hash')}"
        )

    return manifest_info["computed_hash"]


def compute_tokengate_config_hash(variant_config: dict) -> str:
    """Compute deterministic hash of TokenGate config."""
    token_gate_config = variant_config.get("components", {}).get("token_gate", {})
    payload = json.dumps(
        token_gate_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return compute_text_hash(payload)
