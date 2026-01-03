#!/usr/bin/env python3
"""
Centralized configuration loader for EXAID evaluation.

Paper hook: "Configuration loading is centralized to ensure drift-proof
evaluation across all phases (Section 3.1)"

INVARIANT: All modules MUST use this loader for extractor configuration.
This ensures:
    1. Single source of truth (extractor.yaml)
    2. Consistent path resolution for stoplist files
    3. Proper hash computation for provenance

Dependencies:
    - yaml
    - pathlib
    - hashlib
"""

from pathlib import Path
from typing import Optional

import yaml

from ..deterministic.io import compute_file_hash


def get_evals_root() -> Path:
    """
    Get the evals directory root.
    
    Works from any execution context by finding the directory
    containing this file and going up to evals/.
    """
    # This file is in evals/src/config/
    return Path(__file__).resolve().parents[2]


def get_configs_dir() -> Path:
    """Get the configs directory path."""
    return get_evals_root() / "configs"


def load_extractor_config(
    configs_dir: Optional[Path] = None,
    resolve_paths: bool = True,
    include_hashes: bool = True
) -> dict:
    """
    Load extractor configuration from extractor.yaml.
    
    Paper hook: "Extractor configuration loaded via centralized loader
    with path resolution and hash computation (Section 6.1)"
    
    INVARIANT: This is the ONLY function that should load extractor config.
    
    Args:
        configs_dir: Configs directory (default: auto-detect)
        resolve_paths: Resolve stoplist paths to absolute paths
        include_hashes: Include file hashes for provenance
        
    Returns:
        Extractor configuration dict with resolved paths and hashes
    """
    if configs_dir is None:
        configs_dir = get_configs_dir()
    
    config_path = configs_dir / "extractor.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Extractor config not found: {config_path}. "
            "Run from evals/ directory or specify --configs path."
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    
    config = full_config.get("concept_extractor", {})
    
    # Resolve stoplist paths
    if resolve_paths:
        stop_entities_file = config.get("stop_entities_file")
        stop_cuis_file = config.get("stop_cuis_file")
        
        if stop_entities_file:
            # Handle relative paths
            stop_entities_path = Path(stop_entities_file)
            if not stop_entities_path.is_absolute():
                # Resolve relative to evals/ working directory
                evals_root = get_evals_root()
                stop_entities_path = evals_root / stop_entities_file
            config["stop_entities_file"] = str(stop_entities_path)
        
        if stop_cuis_file:
            stop_cuis_path = Path(stop_cuis_file)
            if not stop_cuis_path.is_absolute():
                # Resolve relative to evals/ working directory
                evals_root = get_evals_root()
                stop_cuis_path = evals_root / stop_cuis_file
            config["stop_cuis_file"] = str(stop_cuis_path)
    
    # Add file hashes for provenance
    if include_hashes:
        stop_entities_path = Path(config.get("stop_entities_file", ""))
        stop_cuis_path = Path(config.get("stop_cuis_file", ""))
        
        config["_stoplists_provenance"] = {
            "stop_entities_hash": compute_file_hash(stop_entities_path),
            "stop_cuis_hash": compute_file_hash(stop_cuis_path),
            "config_path": str(config_path),
        }
        
        # Add DF report hash if exists
        df_report_path = configs_dir / "stoplist_df_report.csv"
        if df_report_path.exists():
            config["_stoplists_provenance"]["stoplist_df_report_hash"] = compute_file_hash(df_report_path)
    
    return config


def load_extractor_config_for_stoplist_generation(
    configs_dir: Optional[Path] = None
) -> dict:
    """
    Load extractor config with stoplists DISABLED for generation.
    
    Paper hook: "Stoplist generation disables existing stoplists to
    prevent circular filtering (Section 6.1)"
    
    INVARIANT: This function MUST only be used by generate_stoplists.py.
    All other modules use load_extractor_config().
    
    Args:
        configs_dir: Configs directory
        
    Returns:
        Extractor config with stoplist files set to None
    """
    config = load_extractor_config(
        configs_dir,
        resolve_paths=False,  # Don't resolve since we're disabling
        include_hashes=False  # No hashes needed for generation
    )
    
    # DISABLE stoplists for non-circular generation
    config["stop_entities_file"] = None
    config["stop_cuis_file"] = None
    
    return config


def get_stoplists_provenance(configs_dir: Optional[Path] = None) -> dict:
    """
    Get stoplist provenance info for run_meta logging.
    
    Returns:
        Dict with stop_entities_hash, stop_cuis_hash, stoplist_df_report_hash
    """
    if configs_dir is None:
        configs_dir = get_configs_dir()
    
    provenance = {
        "stop_entities_hash": compute_file_hash(configs_dir / "stop_entities.txt"),
        "stop_cuis_hash": compute_file_hash(configs_dir / "stop_cuis.txt"),
        "stoplists_generated_at": None,
        "stoplists_generated_by_commit": None,
    }
    
    df_report_path = configs_dir / "stoplist_df_report.csv"
    if df_report_path.exists():
        provenance["stoplist_df_report_hash"] = compute_file_hash(df_report_path)
    
    return provenance


def load_variant_config(variant_id: str, configs_dir: Optional[Path] = None) -> dict:
    """
    Load variant configuration.
    
    Args:
        variant_id: V0, V1, V2, V3, or V4
        configs_dir: Configs directory
        
    Returns:
        Variant configuration dict
    """
    if configs_dir is None:
        configs_dir = get_configs_dir()
    
    config_path = configs_dir / "variants" / f"{variant_id}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Variant config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_metrics_config(configs_dir: Optional[Path] = None) -> dict:
    """Load metrics configuration."""
    if configs_dir is None:
        configs_dir = get_configs_dir()
    
    config_path = configs_dir / "metrics.yaml"
    
    if not config_path.exists():
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}



