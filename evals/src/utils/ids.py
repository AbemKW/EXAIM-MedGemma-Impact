"""Deterministic ID helpers for trace generation and evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def generate_mas_run_id(
    mac_commit: str,
    model: str,
    decoding: Dict[str, Any],
    case_list_hash: str,
) -> str:
    """
    Generate deterministic mas_run_id from ALL generation inputs.

    Format: mas_<mac8>_<model>_<decoding8>_<cases8>

    This ID is input-derived only (no date) for true determinism.
    Same inputs = same ID regardless of when generation runs.

    Args:
        mac_commit: Full MAC commit hash
        model: Model name (e.g., "gpt-4o-mini")
        decoding: Decoding parameters dict
        case_list_hash: SHA256 hash of case list (sha256:xxx format)

    Returns:
        Deterministic run ID
    """
    # Canonicalize decoding params for hash
    decoding_canonical = json.dumps(
        {
            "temperature": decoding.get("temperature", 1.0),
            "top_p": decoding.get("top_p"),
            "max_tokens": decoding.get("max_tokens"),
            "seed": decoding.get("seed"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    decoding_hash = hashlib.sha256(decoding_canonical.encode()).hexdigest()[:8]

    # Slugify model name (lowercase, remove dashes/dots, max 12 chars)
    model_slug = model.lower().replace("-", "").replace(".", "").replace(" ", "")[:12]

    # Extract hash portion from case_list_hash (remove "sha256:" prefix)
    cases_hash = case_list_hash.replace("sha256:", "")[:8]

    return f"mas_{mac_commit[:8]}_{model_slug}_{decoding_hash}_{cases_hash}"


def generate_dataset_id(
    mac_commit: str,
    decoding: Dict[str, Any],
    case_list_hash: str,
) -> str:
    """
    Generate deterministic dataset_id from generation inputs.

    Format: exaid_traces_<mac8>_<decoding8>_<cases8>

    Args:
        mac_commit: Full MAC commit hash
        decoding: Decoding parameters dict
        case_list_hash: SHA256 hash of case list

    Returns:
        Deterministic dataset ID
    """
    decoding_canonical = json.dumps(
        {
            "temperature": decoding.get("temperature", 1.0),
            "top_p": decoding.get("top_p"),
            "max_tokens": decoding.get("max_tokens"),
            "seed": decoding.get("seed"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    decoding_hash = hashlib.sha256(decoding_canonical.encode()).hexdigest()[:8]

    cases_hash = case_list_hash.replace("sha256:", "")[:8]

    return f"exaid_traces_{mac_commit[:8]}_{decoding_hash}_{cases_hash}"
