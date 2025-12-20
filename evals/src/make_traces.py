#!/usr/bin/env python3
"""
EXAID Evaluation - Trace Generation Script

Generates MAS traces from clinical cases using the MAC (Multi-Agent Conversation)
framework as an upstream trace generator.

MAC Integration:
- MAC is used as an instrumentation-only trace generator
- MAC controls its own decoding parameters internally (EXAID does not override)
- Traces are frozen and replayed by EXAID variants

Token Accounting:
- All traces use Character-Normalized Token Units (CTU): ceil(len(text) / 4)
- CTU is vendor-agnostic and deterministic
- Provider token counts are logged separately as usage metadata only

Usage:
    python make_traces.py --config configs/mas_generation.yaml --dataset-config configs/dataset.yaml
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import yaml

# =============================================================================
# CTU (Character-Normalized Token Units)
# =============================================================================

def compute_ctu(text: str) -> int:
    """
    Compute Character-Normalized Token Units (CTU).
    
    CTU = ceil(len(text) / 4)
    
    CTU is a model-agnostic, character-normalized text unit used as a deterministic
    proxy for text volume. The ~4 characters per unit heuristic reflects commonly
    observed token densities across contemporary subword tokenizers (OpenAI, LLaMA,
    Gemini, Claude) and is used here strictly as a normalization constant.
    
    CTU is NOT a tokenizer—it avoids proprietary tokenizer dependencies and ensures:
    - Offline computation (no API calls required)
    - Deterministic replay (same text always produces same CTU count)
    - Reproducibility for reviewers (vendor-independent)
    
    For non-text or empty emissions, returns 0.
    
    Args:
        text: The text content to measure
        
    Returns:
        CTU count (integer)
    """
    if not text:
        return 0
    return math.ceil(len(text) / 4)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load MAS generation configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_config(config_path: Path) -> dict:
    """Load dataset configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Case Selection
# =============================================================================

def get_all_mac_case_ids(mac_module_path: str) -> list[str]:
    """
    Get all case IDs from the MAC dataset.
    
    Returns:
        List of case IDs in the format "case-{case_url}"
    """
    dataset_path = Path(mac_module_path) / "dataset" / "rare_disease_302.json"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"MAC dataset not found: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    case_ids = []
    for case in data.get("Cases", []):
        case_url = case.get("Case URL")
        if case_url:
            # Normalize to lowercase - schema requires [a-z0-9-]+ pattern
            case_ids.append(f"case-{str(case_url).lower()}")
    
    return case_ids


def select_mac_cases(dataset_config: dict, all_case_ids: list[str]) -> list[str]:
    """
    Select MAC cases based on dataset configuration.
    
    Selection is deterministic using a seeded random generator.
    
    Args:
        dataset_config: Dataset configuration dict
        all_case_ids: List of all available case IDs
        
    Returns:
        Sorted list of selected case IDs
    """
    mac_sel = dataset_config.get("mac_selection", {})
    mode = mac_sel.get("mode", "fixed_subset")
    
    if mode == "all":
        return sorted(all_case_ids)
    
    # Check for explicit case list override
    explicit_cases = mac_sel.get("explicit_cases")
    if explicit_cases:
        return sorted(explicit_cases)
    
    # Fixed subset with seeded random selection
    seed = mac_sel.get("seed", 42)
    n_cases = mac_sel.get("n_cases", 40)
    
    rng = random.Random(seed)
    sorted_ids = sorted(all_case_ids)
    n_to_select = min(n_cases, len(sorted_ids))
    selected = rng.sample(sorted_ids, n_to_select)
    
    return sorted(selected)


# =============================================================================
# Deterministic Run ID Generation
# =============================================================================

def generate_mas_run_id(
    mac_commit: str,
    model: str,
    case_ids: list[str]
) -> str:
    """
    Generate a deterministic MAS run ID.
    
    The run ID is a hash of:
    - MAC commit hash
    - Model name
    - Ordered case list
    
    Args:
        mac_commit: Git commit hash of MAC submodule
        model: Base model name
        case_ids: Sorted list of case IDs
        
    Returns:
        Deterministic run ID in format "mas-{hash16}"
    """
    payload = json.dumps({
        "mac_commit": mac_commit,
        "model": model,
        "cases": case_ids  # Already sorted
    }, sort_keys=True)
    
    hash_digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"mas-{hash_digest}"


def compute_case_list_hash(case_ids: list[str]) -> str:
    """
    Compute a SHA-256 hash of the ordered case list.
    
    Args:
        case_ids: Sorted list of case IDs
        
    Returns:
        Hash in format "sha256:{hash64}"
    """
    payload = json.dumps(case_ids, sort_keys=True)
    hash_digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"sha256:{hash_digest}"


# =============================================================================
# Trace ID Generation
# =============================================================================

def generate_trace_id(case_id: str, sequence: int) -> str:
    """
    Generate a unique trace ID.
    
    Args:
        case_id: Case identifier (e.g., "case-37470964")
        sequence: Sequence number within the case
        
    Returns:
        Trace ID in format "trc-{case_number}-{seq:03d}"
    """
    # Extract the numeric part from case_id and ensure lowercase
    # Schema requires [a-z0-9-]+ pattern
    case_num = case_id.replace("case-", "").lower()
    return f"trc-{case_num}-{sequence:03d}"


# =============================================================================
# MAC Integration
# =============================================================================

def load_mac_case(mac_module_path: str, case_id: str, stage: str = "inital") -> Optional[dict]:
    """
    Load a specific case from the MAC dataset.
    
    Args:
        mac_module_path: Path to MAC module
        case_id: Case ID in format "case-{case_url}"
        stage: "inital" or "follow_up"
        
    Returns:
        Case data dict or None if not found
    """
    dataset_path = Path(mac_module_path) / "dataset" / "rare_disease_302.json"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    case_url = case_id.replace("case-", "")
    
    for case in data.get("Cases", []):
        if str(case.get("Case URL")) == case_url:
            presentation_key = "Initial Presentation" if stage == "inital" else "Follow-up Presentation"
            return {
                "case_id": case_id,
                "case_url": case.get("Case URL"),
                "type": case.get("Type"),
                "final_name": case.get("Final Name"),
                "presentation": case.get(presentation_key, ""),
                "stage": stage
            }
    
    return None


def run_mac_case(
    mac_module_path: str,
    case_data: dict,
    mas_config: dict,
) -> tuple[list[dict], Optional[str]]:
    """
    Run MAC for a single case and capture agent emissions.
    
    IMPORTANT: This function does NOT modify MAC's internal behavior.
    MAC controls its own decoding parameters (temperature, sampling) internally.
    EXAID only captures the agent message emissions.
    
    Args:
        mac_module_path: Path to MAC module
        case_data: Case data from load_mac_case
        mas_config: MAS generation configuration
        
    Returns:
        Tuple of (list of message dicts, error_message or None)
    """
    try:
        # Add MAC to path
        sys.path.insert(0, mac_module_path)
        
        from autogen import (
            GroupChat,
            GroupChatManager,
            AssistantAgent,
            config_list_from_json,
        )
        from utils.prompts import (
            get_doc_system_message,
            get_supervisor_system_message,
            get_inital_message,
        )
        
        # Load MAC's own config
        config_path = Path(mac_module_path) / "configs" / "config_list.json"
        
        # Check if config exists, if not we'll create a minimal stub
        if not config_path.exists():
            # For trace generation without actual API calls, we need the config
            # In production, this should be provided
            return [], f"MAC config not found: {config_path}"
        
        # Use MAC's internal configuration
        # Note: We do NOT override temperature or other decoding parameters
        # Model priority: gpt-4o-mini (preferred) -> gpt-4-turbo -> gpt-3.5-turbo
        config_list = config_list_from_json(
            env_or_file=str(config_path),
            filter_dict={"tags": ["x_gpt4o_mini"]}
        )
        
        if not config_list:
            # Fall back to GPT-4 turbo if gpt-4o-mini not configured
            config_list = config_list_from_json(
                env_or_file=str(config_path),
                filter_dict={"tags": ["x_gpt4_turbo"]}
            )
        
        if not config_list:
            # Fall back to GPT-3.5 as last resort
            config_list = config_list_from_json(
                env_or_file=str(config_path),
                filter_dict={"tags": ["x_gpt35_turbo"]}
            )
        
        if not config_list:
            return [], "No valid model configuration found in MAC config"
        
        # MAC's internal model config - we do not override these
        model_config = {
            "cache_seed": None,
            "temperature": 1,  # MAC's default - we do NOT override this
            "config_list": config_list,
            "timeout": 300,
        }
        
        stage = case_data.get("stage", "inital")
        num_doctors = 3
        n_round = 13
        
        # Create agents using MAC's prompts
        docs = []
        for index in range(num_doctors):
            name = f"Doctor{index}"
            doc_system_message = get_doc_system_message(
                doctor_name=name, stage=stage
            )
            doc = AssistantAgent(
                name=name,
                llm_config=model_config,
                system_message=doc_system_message,
            )
            docs.append(doc)
        
        supervisor_system_message = get_supervisor_system_message(
            stage=stage, use_specialist=False
        )
        
        supervisor = AssistantAgent(
            name="Supervisor",
            llm_config=model_config,
            system_message=supervisor_system_message,
        )
        
        agents = docs + [supervisor]
        
        groupchat = GroupChat(
            agents=agents,
            messages=[],
            max_round=n_round,
            speaker_selection_method="auto",
            admin_name="Supervisor",
            select_speaker_auto_verbose=False,
            allow_repeat_speaker=True,
            send_introductions=False,
            max_retries_for_selecting_speaker=n_round // (1 + num_doctors),
        )
        
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=model_config,
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        )
        
        initial_message = get_inital_message(
            patient_history=case_data.get("presentation", ""),
            stage=stage
        )
        
        # Run the conversation
        output = supervisor.initiate_chat(
            manager,
            message=initial_message,
        )
        
        # Extract messages from chat history
        messages = []
        for msg in output.chat_history:
            agent_name = msg.get("name", "Supervisor")
            content = msg.get("content", "")
            messages.append({
                "agent_id": agent_name.lower().replace(" ", "_"),
                "content": content,
                "role": msg.get("role", "assistant"),
            })
        
        return messages, None
        
    except ImportError as e:
        return [], f"Failed to import MAC dependencies: {e}"
    except Exception as e:
        return [], f"MAC execution failed: {e}"
    finally:
        # Remove MAC from path
        if mac_module_path in sys.path:
            sys.path.remove(mac_module_path)


def generate_stub_traces(case_id: str, case_data: dict) -> list[dict]:
    """
    Generate stub traces for demonstration/testing without MAC execution.
    
    This is used when MAC is not available or for scaffold testing.
    
    Args:
        case_id: Case identifier
        case_data: Case data dict
        
    Returns:
        List of message dicts
    """
    # Create minimal stub messages
    presentation = case_data.get("presentation", "No presentation available")
    
    messages = [
        {
            "agent_id": "supervisor",
            "content": f"[STUB] Analyzing case: {case_id}\nPresentation: {presentation[:200]}...",
            "role": "assistant"
        },
        {
            "agent_id": "doctor0",
            "content": "[STUB] Initial differential diagnosis based on presentation...",
            "role": "assistant"
        },
        {
            "agent_id": "doctor1",
            "content": "[STUB] Additional considerations from specialist perspective...",
            "role": "assistant"
        },
        {
            "agent_id": "supervisor",
            "content": '[STUB] Consensus reached. {"Most Likely Diagnosis": "Pending MAC integration"}',
            "role": "assistant"
        },
    ]
    
    return messages


# =============================================================================
# Trace Generation
# =============================================================================

def generate_traces_for_case(
    case_id: str,
    case_data: dict,
    messages: list[dict],
    mas_run_id: str,
    mac_commit: str,
    base_model: str,
    status: str = "success",
    failure_reason: Optional[str] = None
) -> Iterator[dict]:
    """
    Generate trace records for a case from MAC messages.
    
    Args:
        case_id: Case identifier
        case_data: Original case data
        messages: List of agent messages from MAC
        mas_run_id: Deterministic run ID
        mac_commit: MAC commit hash
        base_model: Base model name
        status: Execution status ("success" or "failed")
        failure_reason: Reason for failure if status is "failed"
        
    Yields:
        Trace records conforming to exaid.trace schema
    """
    timestamp = datetime.now(timezone.utc)
    
    if status == "failed" or not messages:
        # Generate stub trace for failed case
        yield {
            "schema_name": "exaid.trace",
            "schema_version": "1.0.0",
            "trace_id": generate_trace_id(case_id, 0),
            "case_id": case_id,
            "agent_id": "_system",
            "sequence_num": 0,
            "timestamp": timestamp.isoformat(),
            "content": f"[FAILED] {failure_reason or 'Case execution failed'}",
            "text_units_ctu": 0,
            "metadata": {
                "status": "failed",
                "failure_reason": failure_reason or "Unknown error",
                "mas_run_id": mas_run_id,
                "mac_commit": mac_commit,
                "model": base_model
            }
        }
        return
    
    # Generate traces for each message
    for seq, msg in enumerate(messages):
        content = msg.get("content", "")
        agent_id = msg.get("agent_id", "unknown")
        
        trace = {
            "schema_name": "exaid.trace",
            "schema_version": "1.0.0",
            "trace_id": generate_trace_id(case_id, seq),
            "case_id": case_id,
            "agent_id": agent_id,
            "sequence_num": seq,
            "timestamp": timestamp.isoformat(),
            "content": content,
            "text_units_ctu": compute_ctu(content),
            "metadata": {
                "mas_run_id": mas_run_id,
                "mac_commit": mac_commit,
                "model": base_model,
                "status": "success"
            }
        }
        
        yield trace


# =============================================================================
# File I/O
# =============================================================================

def write_traces(
    traces: Iterator[dict],
    output_path: Path,
    compress: bool = True
) -> int:
    """
    Write traces to a JSONL file.
    
    Args:
        traces: Iterator of trace records
        output_path: Output file path
        compress: Whether to gzip compress the output
        
    Returns:
        Number of traces written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    open_func = gzip.open if compress else open
    mode = "wt" if compress else "w"
    
    with open_func(output_path, mode, encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")
            count += 1
    
    return count


def write_manifest(
    manifest_path: Path,
    mas_run_id: str,
    mac_commit: str,
    base_model: str,
    dataset_config: dict,
    selected_cases: list[str],
    all_case_ids: list[str]
) -> None:
    """
    Write dataset manifest with generation metadata.
    
    Args:
        manifest_path: Path to manifest file
        mas_run_id: Deterministic run ID
        mac_commit: MAC commit hash
        base_model: Base model name
        dataset_config: Dataset configuration
        selected_cases: List of selected case IDs
        all_case_ids: List of all available case IDs
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    mac_sel = dataset_config.get("mac_selection", {})
    
    manifest = {
        "schema_name": "exaid.manifest",
        "schema_version": "1.0.0",
        "experiment_id": f"exp-mac-traces-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": f"sha256:{hashlib.sha256(json.dumps(dataset_config, sort_keys=True).encode()).hexdigest()}",
        "mas_generation_config": {
            "mas_run_id": mas_run_id,
            "mac_commit": mac_commit,
            "base_model": base_model,
            "decoding_note": "Controlled by MAC internally; EXAID does not override",
            "text_unit": {
                "name": "CTU",
                "definition": "ceil(len(text) / 4)",
                "applies_to": "input_and_output"
            },
            "selection_mode": mac_sel.get("mode", "fixed_subset"),
            "selection_seed": mac_sel.get("seed", 42),
            "n_cases_selected": len(selected_cases),
            "n_cases_available": len(all_case_ids),
            "case_list_hash": compute_case_list_hash(selected_cases),
            "ordered_case_list": selected_cases
        }
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest) + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate MAS traces from clinical cases using MAC"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mas_generation.yaml"),
        help="MAS generation configuration file"
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset configuration file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/traces"),
        help="Output traces directory"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/dataset_manifest.jsonl"),
        help="Output manifest file"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable gzip compression"
    )
    parser.add_argument(
        "--stub-mode",
        action="store_true",
        help="Use stub traces instead of running MAC (for testing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXAID Trace Generation - MAC Integration")
    print("=" * 70)
    print()
    
    # Load configurations
    mas_config = {}
    if args.config.exists():
        mas_config = load_config(args.config)
        print(f"Loaded MAS config: {args.config}")
    else:
        print(f"WARNING: MAS config not found: {args.config}")
    
    dataset_config = {}
    if args.dataset_config.exists():
        dataset_config = load_dataset_config(args.dataset_config)
        print(f"Loaded dataset config: {args.dataset_config}")
    else:
        print(f"WARNING: Dataset config not found: {args.dataset_config}")
    
    # Get MAC configuration
    mac_config = mas_config.get("mac", {})
    mac_module_path = mac_config.get("module_path", "/app/third_party/mac")
    mac_commit = mac_config.get("commit", "unknown")
    base_model = mac_config.get("base_model", "gpt-4o-mini")
    mac_enabled = mac_config.get("enabled", False)
    
    print()
    print("Configuration:")
    print(f"  MAC module path: {mac_module_path}")
    print(f"  MAC commit: {mac_commit}")
    print(f"  Base model: {base_model}")
    print(f"  MAC enabled: {mac_enabled}")
    print()
    
    # Check if MAC is available
    mac_available = Path(mac_module_path).exists() and mac_enabled and not args.stub_mode
    
    if not mac_available:
        if args.stub_mode:
            print("NOTE: Running in stub mode (--stub-mode flag)")
        elif not mac_enabled:
            print("NOTE: MAC integration disabled in config")
        else:
            print(f"NOTE: MAC module not found at {mac_module_path}")
        print("Using stub traces for demonstration.")
        print()
    
    # Get all case IDs
    try:
        if mac_available or Path(mac_module_path).exists():
            all_case_ids = get_all_mac_case_ids(mac_module_path)
        else:
            # Generate stub case IDs for testing
            all_case_ids = [f"case-stub-{i:03d}" for i in range(10)]
        print(f"Total cases available: {len(all_case_ids)}")
    except Exception as e:
        print(f"ERROR: Failed to load case IDs: {e}")
        all_case_ids = [f"case-stub-{i:03d}" for i in range(10)]
    
    # Select cases based on configuration
    selected_cases = select_mac_cases(dataset_config, all_case_ids)
    print(f"Cases selected: {len(selected_cases)}")
    
    mac_sel = dataset_config.get("mac_selection", {})
    print(f"  Selection mode: {mac_sel.get('mode', 'fixed_subset')}")
    print(f"  Selection seed: {mac_sel.get('seed', 42)}")
    print()
    
    # Generate deterministic run ID
    mas_run_id = generate_mas_run_id(mac_commit, base_model, selected_cases)
    print(f"MAS Run ID: {mas_run_id}")
    print()
    
    # Process cases
    total_traces = 0
    total_cases = 0
    failed_cases = 0
    
    on_failure = mac_sel.get("on_failure", "log_stub")
    
    print("Processing cases:")
    print("-" * 50)
    
    for case_id in selected_cases:
        total_cases += 1
        print(f"  [{total_cases}/{len(selected_cases)}] {case_id}...", end=" ")
        
        # Load case data
        case_data = None
        if mac_available or Path(mac_module_path).exists():
            try:
                case_data = load_mac_case(mac_module_path, case_id, stage="inital")
            except Exception as e:
                case_data = None
        
        if case_data is None:
            case_data = {
                "case_id": case_id,
                "presentation": "Stub case for testing",
                "stage": "inital"
            }
        
        # Run MAC or generate stub traces
        messages = []
        error_message = None
        
        if mac_available:
            messages, error_message = run_mac_case(mac_module_path, case_data, mas_config)
        else:
            messages = generate_stub_traces(case_id, case_data)
        
        # Handle failures
        status = "success"
        if error_message:
            failed_cases += 1
            status = "failed"
            if on_failure == "raise":
                print(f"FAILED: {error_message}")
                raise RuntimeError(f"Case {case_id} failed: {error_message}")
            else:
                print(f"FAILED (stub logged)")
        else:
            print(f"OK ({len(messages)} messages)")
        
        # Generate traces
        traces = generate_traces_for_case(
            case_id=case_id,
            case_data=case_data,
            messages=messages,
            mas_run_id=mas_run_id,
            mac_commit=mac_commit,
            base_model=base_model,
            status=status,
            failure_reason=error_message
        )
        
        # Determine output path
        # Sanitize case_id for filename (replace non-alphanumeric with dash)
        safe_case_id = re.sub(r'[^a-z0-9-]', '-', case_id.lower())
        output_file = args.output / f"{safe_case_id}.jsonl"
        if not args.no_compress:
            output_file = output_file.with_suffix(".jsonl.gz")
        
        # Write traces
        count = write_traces(traces, output_file, compress=not args.no_compress)
        total_traces += count
    
    print("-" * 50)
    print()
    
    # Write manifest
    print(f"Writing manifest: {args.manifest}")
    write_manifest(
        manifest_path=args.manifest,
        mas_run_id=mas_run_id,
        mac_commit=mac_commit,
        base_model=base_model,
        dataset_config=dataset_config,
        selected_cases=selected_cases,
        all_case_ids=all_case_ids
    )
    
    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"  Cases processed: {total_cases}")
    print(f"  Cases failed: {failed_cases}")
    print(f"  Total traces: {total_traces}")
    print(f"  Output directory: {args.output}")
    print(f"  Manifest: {args.manifest}")
    print()
    print("Text Unit Accounting:")
    print("  All traces use CTU (Character-Normalized Token Units)")
    print("  Definition: ceil(len(text) / 4)")
    print("  Provider token counts logged separately as usage metadata only")
    print()
    
    if failed_cases > 0:
        print(f"WARNING: {failed_cases} cases failed and have stub traces")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
