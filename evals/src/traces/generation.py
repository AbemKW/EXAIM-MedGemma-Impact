#!/usr/bin/env python3
"""
EXAIM Evaluation - Timed Trace Generation Script (v2.0.0)

Generates timed MAS traces from clinical cases using the MAC (Multi-Agent Conversation)
framework with streaming instrumentation.

Trace Content:
- delta_text: Raw stream delta/chunk as emitted by LLM
- t_emitted_ms: Absolute emission timestamp (ms since epoch)
- t_rel_ms: Relative to t0 (first delta)
- agent_id, turn_id, seq: Attribution and ordering

Timing Semantics:
- t0_emitted_ms: Anchor is first stream_delta emission timestamp
- t_rel_ms for deltas: Always >= 0 (t_emitted_ms - t0)
- t_rel_ms for boundaries: May be NEGATIVE if boundary occurs before t0

Run ID Generation:
- mas_run_id: Input-derived, deterministic from MAC commit + model + decoding + case list
- dataset_id: Same pattern for dataset identification

Critical Constraints:
- MAC behavior is UNCHANGED (instrumentation-only)
- Global seq ordering across entire trace (strictly increasing)
- Deterministic sorting before seq assignment

Usage (from repo root or Docker container):
    python -m evals.cli.make_traces --config configs/mas_generation.yaml --dry-run
    python -m evals.cli.make_traces --config configs/mas_generation.yaml --limit 1
"""

import gzip
import hashlib
import json
import os
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..utils.hashing import (
    compute_config_hash as _compute_config_hash,
    compute_trace_dataset_hash_from_entries,
)
from ..utils.ids import generate_dataset_id as _generate_dataset_id
from ..utils.ids import generate_mas_run_id as _generate_mas_run_id

# =============================================================================
# Type Aliases
# =============================================================================
TurnTrace = Any  # From MAC instrumentation


# =============================================================================
# Deterministic ID Generation
# =============================================================================

def generate_mas_run_id(
    mac_commit: str,
    model: str,
    decoding: Dict[str, Any],
    case_list_hash: str
) -> str:
    """Generate deterministic mas_run_id from ALL generation inputs."""
    return _generate_mas_run_id(mac_commit, model, decoding, case_list_hash)


def generate_dataset_id(
    mac_commit: str,
    decoding: Dict[str, Any],
    case_list_hash: str
) -> str:
    """Generate deterministic dataset_id from generation inputs."""
    return _generate_dataset_id(mac_commit, decoding, case_list_hash)


def compute_trace_dataset_hash(
    mas_run_id: str,
    case_list_hash: str,
    trace_entries: List[Tuple[str, str]]
) -> str:
    """Compute trace_dataset_hash for eval_run_id derivation."""
    return compute_trace_dataset_hash_from_entries(mas_run_id, case_list_hash, trace_entries)


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


def compute_config_hash(*config_paths: Path) -> str:
    """Compute SHA256 hash of concatenated config files."""
    return _compute_config_hash(*config_paths)


def get_exaim_commit() -> str:
    """Get current EXAIM repository commit hash (canonical name)."""
    # Get repo root (go up from evals/src/traces/)
    repo_root = Path(__file__).resolve().parents[2]  # src -> evals -> repo root
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

# Backward compatibility alias
def get_exaid_commit() -> str:
    """Legacy alias for get_exaim_commit()."""
    return get_exaim_commit()


# =============================================================================
# Case Selection
# =============================================================================

def get_all_mac_case_ids(mac_module_path: str) -> List[str]:
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


def select_mac_cases(dataset_config: dict, all_case_ids: List[str]) -> List[str]:
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
# Case List File Operations
# =============================================================================

def write_case_list(case_ids: List[str], output_path: Path) -> str:
    """
    Write ordered case list to JSONL file.
    
    Format: Each line is {"index": N, "case_id": "case-XXXXX"}
    NO _meta line - hash is stored in manifest only.
    
    Args:
        case_ids: Sorted list of case IDs
        output_path: Path to output file
        
    Returns:
        SHA256 hash of file bytes (format: "sha256:xxx")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    for idx, case_id in enumerate(case_ids):
        record = {"index": idx, "case_id": case_id}
        lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    
    content = "\n".join(lines) + "\n"
    content_bytes = content.encode("utf-8")
    
    with open(output_path, "wb") as f:
        f.write(content_bytes)
    
    hash_digest = hashlib.sha256(content_bytes).hexdigest()
    return f"sha256:{hash_digest}"


# =============================================================================
# MAC Integration with Safe Monkeypatching
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
        if str(case.get("Case URL")).lower() == case_url:
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


def run_mac_case_instrumented(
    mac_module_path: str,
    case_data: dict,
    mas_config: dict,
) -> Tuple[List[Any], Optional[str]]:
    """
    Run MAC for a single case with streaming instrumentation.
    
    CRITICAL: Uses reversible monkeypatching with try/finally restore.
    
    IMPORTANT: This function does NOT modify MAC's internal behavior.
    MAC controls its own decoding parameters (temperature, sampling) internally.
    EXAIM only captures the delta/chunk emission timing.
    
    Args:
        mac_module_path: Path to MAC module
        case_data: Case data from load_mac_case
        mas_config: MAS generation configuration
        
    Returns:
        Tuple of (list of TurnTrace objects, error_message or None)
    """
    original_methods = {}  # Store originals for restoration
    
    try:
        # Add MAC to path
        sys.path.insert(0, mac_module_path)
        
        # CRITICAL: Install OpenAI patch BEFORE importing autogen
        # Import from the instrumentation package (not individual files)
        from instrumentation import (
            install_openai_patch,
            get_trace_collector,
            set_current_agent_id,
        )
        
        install_openai_patch()
        
        # Get and RESET collector for this case (avoid cross-case contamination)
        collector = get_trace_collector()
        collector.clear()
        
        # Suppress autogen's overly strict API key format warning
        import logging
        autogen_logger = logging.getLogger("autogen.oai.client")
        autogen_logger.setLevel(logging.ERROR)
        
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
        
        # Load EXAIM-owned model configuration
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "configs" / "mac_model_config.json"
        
        if not config_path.exists():
            return [], f"EXAIM model config not found: {config_path}"
        
        # Load JSON and inject API key from environment
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return [], "OPENAI_API_KEY environment variable not set"
        
        api_key = api_key.strip()
        if not api_key:
            return [], "OPENAI_API_KEY environment variable is empty"
        
        for config_entry in config_data:
            if "api_key" not in config_entry or not config_entry.get("api_key"):
                config_entry["api_key"] = api_key
        
        # Write to temporary file for autogen
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(config_data, tmp_file)
            tmp_config_path = tmp_file.name
        
        try:
            config_list = config_list_from_json(
                env_or_file=tmp_config_path,
                filter_dict={"tags": ["x_gpt4o_mini"]}
            )
            
            if not config_list:
                config_list = config_list_from_json(
                    env_or_file=tmp_config_path,
                    filter_dict={"tags": ["x_gpt4_turbo"]}
                )
            
            if not config_list:
                config_list = config_list_from_json(
                    env_or_file=tmp_config_path,
                    filter_dict={"tags": ["x_gpt35_turbo"]}
                )
            
            if not config_list:
                return [], "No valid model configuration found"
        finally:
            try:
                os.unlink(tmp_config_path)
            except OSError:
                pass
        
        # MAC's internal model config
        model_config = {
            "cache_seed": None,
            "temperature": 1,  # MAC's default
            "config_list": config_list,
            "timeout": 300,
        }
        
        stage = case_data.get("stage", "inital")
        num_doctors = 3
        n_round = 13
        
        # Create agents
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
        
        # SAFE MONKEYPATCHING: Store originals and patch with closure safety
        for agent in agents:
            original_methods[agent.name] = agent.generate_reply
            
            # Capture in closure explicitly to avoid late-binding issues
            def make_wrapper(agent_name: str, original_fn):
                def wrapper(*args, **kwargs):
                    set_current_agent_id(agent_name.lower())
                    return original_fn(*args, **kwargs)
                return wrapper
            
            agent.generate_reply = make_wrapper(agent.name, original_methods[agent.name])
        
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
        supervisor.initiate_chat(
            manager,
            message=initial_message,
        )
        
        # Retrieve timed turns from collector
        turns = collector.get_all_turns()
        return turns, None
        
    except ImportError as e:
        return [], f"Failed to import MAC dependencies: {e}"
    except Exception as e:
        return [], f"MAC execution failed: {e}"
    finally:
        # ALWAYS restore original methods (even on exception)
        for agent_name, original_fn in original_methods.items():
            for agent in agents if 'agents' in dir() else []:
                if agent.name == agent_name:
                    agent.generate_reply = original_fn
        
        # Remove MAC from path
        if mac_module_path in sys.path:
            sys.path.remove(mac_module_path)
        
        instrumentation_path = os.path.join(mac_module_path, "instrumentation")
        if instrumentation_path in sys.path:
            sys.path.remove(instrumentation_path)


# =============================================================================
# Trace Serialization with Deterministic Global Order
# =============================================================================

def serialize_to_trace_records(
    turns: List[Any],
    case_id: str
) -> Tuple[List[dict], int, int, int]:
    """
    Convert collector turns to globally-ordered trace records.
    
    CRITICAL: Explicit sort by time with deterministic tie-breakers.
    seq is global across entire trace (strictly increasing).
    
    RAW CAPTURE: No derived units (ws_units, ctu) are stored.
    Stream emissions are delta/chunks (may be fragments, not tokenizer tokens).
    
    Timing Semantics:
    - t0 = first stream_delta t_emitted_ms
    - Delta t_rel_ms: Always >= 0 (by definition)
    - Boundary t_rel_ms: May be NEGATIVE if boundary occurs before t0
    
    Args:
        turns: List of TurnTrace objects from collector
        case_id: Case identifier
        
    Returns:
        Tuple of (records, t0_emitted_ms, total_deltas, total_turns)
    """
    if not turns:
        return [], 0, 0, 0
    
    # Step 1: Flatten all events into a single list with sort keys
    events = []
    
    for turn in turns:
        turn_id = turn.turn_id
        agent_id = turn.agent_id
        
        # Turn start event
        events.append({
            "sort_key": (turn.t_start_ms, 0, turn_id, 0),  # (time, type_priority, turn, sub_seq)
            "type": "turn_start",
            "turn_id": turn_id,
            "agent_id": agent_id,
            "t_ms": turn.t_start_ms
        })
        
        # Delta events (stream chunks, not tokenizer tokens)
        for i, emission in enumerate(turn.token_emissions):
            events.append({
                "sort_key": (emission.t_emitted_ms, 1, turn_id, i),
                "type": "delta",
                "turn_id": turn_id,
                "agent_id": agent_id,
                "delta_text": emission.token,
                "t_emitted_ms": emission.t_emitted_ms
            })
        
        # Turn end event
        events.append({
            "sort_key": (turn.t_end_ms, 2, turn_id, 0),
            "type": "turn_end",
            "turn_id": turn_id,
            "agent_id": agent_id,
            "t_ms": turn.t_end_ms,
            "content": turn.content
        })
    
    # Step 2: Sort by sort_key (deterministic)
    events.sort(key=lambda e: e["sort_key"])
    
    # Step 3: Find t0 (first delta's emission time)
    t0 = None
    for event in events:
        if event["type"] == "delta":
            t0 = event["t_emitted_ms"]
            break
    
    if t0 is None:
        # No deltas found, use first turn start
        t0 = events[0]["t_ms"] if events else 0
    
    # Step 4: Assign global seq and build records
    records = []
    global_seq = 0
    total_deltas = 0
    turn_delta_texts = {}  # turn_id -> list of delta_text for content hash
    turn_first_delta_t = {}  # turn_id -> first delta t_emitted_ms
    turn_last_delta_t = {}   # turn_id -> last delta t_emitted_ms
    
    for event in events:
        turn_id = event["turn_id"]
        agent_id = event["agent_id"]
        
        if event["type"] == "delta":
            t_emitted = event["t_emitted_ms"]
            records.append({
                "record_type": "stream_delta",
                "case_id": case_id,
                "seq": global_seq,
                "turn_id": turn_id,
                "agent_id": agent_id,
                "delta_text": event["delta_text"],
                "t_emitted_ms": t_emitted,
                "t_rel_ms": t_emitted - t0  # Always >= 0 for deltas by definition
            })
            total_deltas += 1
            
            # Accumulate for content hash
            if turn_id not in turn_delta_texts:
                turn_delta_texts[turn_id] = []
            turn_delta_texts[turn_id].append(event["delta_text"])
            
            # Track first/last delta times for boundary validation
            if turn_id not in turn_first_delta_t:
                turn_first_delta_t[turn_id] = t_emitted
            turn_last_delta_t[turn_id] = t_emitted
            
        elif event["type"] == "turn_start":
            t_ms = event["t_ms"]
            # t_rel_ms may be NEGATIVE for boundaries before t0
            records.append({
                "record_type": "turn_boundary",
                "case_id": case_id,
                "turn_id": turn_id,
                "agent_id": agent_id,
                "boundary": "start",
                "seq": global_seq,
                "t_ms": t_ms,
                "t_rel_ms": t_ms - t0  # May be negative!
            })
            
        elif event["type"] == "turn_end":
            t_ms = event["t_ms"]
            
            # Compute content hash from accumulated deltas
            deltas = turn_delta_texts.get(turn_id, [])
            content = "".join(deltas)
            content_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
            
            # t_rel_ms may be negative for boundaries (though end is typically after t0)
            records.append({
                "record_type": "turn_boundary",
                "case_id": case_id,
                "turn_id": turn_id,
                "agent_id": agent_id,
                "boundary": "end",
                "seq": global_seq,
                "t_ms": t_ms,
                "t_rel_ms": t_ms - t0,  # Typically positive for end boundaries
                "content_hash": content_hash,
                "content_hash_input": "UTF-8 bytes of concatenated delta_text in seq order for this turn"
            })
        
        global_seq += 1
    
    total_turns = len(turns)
    
    return records, t0, total_deltas, total_turns


# =============================================================================
# Trace File I/O
# =============================================================================

def write_trace_file(
    case_id: str,
    records: List[dict],
    t0_emitted_ms: int,
    total_deltas: int,
    total_turns: int,
    mas_run_id: str,
    mac_commit: str,
    exaid_commit: str,
    model: str,
    decoding: Dict[str, Any],
    output_path: Path,
    stub_mode: bool = False
) -> Tuple[int, str]:
    """
    Write trace records to gzip JSONL file.
    
    First record is trace_meta with provenance.
    
    Timing Semantics:
    - t0_emitted_ms: First stream_delta emission timestamp
    - Delta t_rel_ms: Always >= 0
    - Boundary t_rel_ms: May be negative if boundary occurs before t0
    
    Args:
        case_id: Case identifier
        records: List of stream_delta and turn_boundary records
        t0_emitted_ms: Anchor timestamp (first delta)
        total_deltas: Total delta count
        total_turns: Total turn count
        mas_run_id: MAS run ID
        mac_commit: MAC commit hash
        exaid_commit: EXAIM commit hash (legacy field name for artifact compatibility)
        model: Model name
        decoding: Decoding parameters
        output_path: Output file path
        stub_mode: True if trace generated with stub mode (not real MAC)
        
    Returns:
        Tuple of (record_count, sha256_hash)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build trace_meta record
    trace_meta = {
        "record_type": "trace_meta",
        "schema_version": "2.0.0",
        "case_id": case_id,
        "mas_run_id": mas_run_id,
        "mac_commit": mac_commit,
        "exaid_commit": exaid_commit,
        "model": model,
        "decoding": decoding,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "t0_emitted_ms": t0_emitted_ms,
        "t0_definition": "t_emitted_ms of first stream_delta record",
        "total_turns": total_turns,
        "total_deltas": total_deltas
    }
    
    # Add stub_mode flag if true
    if stub_mode:
        trace_meta["stub_mode"] = True
    
    # Write with deterministic gzip (mtime=0)
    lines = []
    lines.append(json.dumps(trace_meta, sort_keys=True, separators=(",", ":")))
    for record in records:
        lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    
    content = "\n".join(lines) + "\n"
    content_bytes = content.encode("utf-8")
    
    # Compute hash before compression
    hash_digest = hashlib.sha256(content_bytes).hexdigest()
    
    # Write with gzip (mtime=0 for determinism)
    with gzip.GzipFile(output_path, mode='wb', mtime=0) as f:
        f.write(content_bytes)
    
    return len(records) + 1, f"sha256:{hash_digest}"  # +1 for trace_meta


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file bytes."""
    with open(file_path, "rb") as f:
        hash_digest = hashlib.sha256(f.read()).hexdigest()
    return f"sha256:{hash_digest}"


# =============================================================================
# Manifest Writer
# =============================================================================

def write_manifest(
    output_path: Path,
    dataset_id: str,
    mas_run_id: str,
    mac_fork_url: str,
    mac_commit: str,
    exaid_commit: str,
    model: str,
    decoding: Dict[str, Any],
    case_list_hash: str,
    config_hash: str,
    trace_entries: List[Dict[str, Any]],
    total_deltas: int,
    total_turns: int,
    successful_cases: int,
    failed_cases: int,
    stub_mode: bool = False
) -> None:
    """
    Write dataset manifest with full provenance.
    
    Multi-record JSONL format per schema v2.0.0.
    
    Args:
        output_path: Manifest file path
        stub_mode: True if traces generated with stub mode (not real MAC)
        ... (see schema for field descriptions)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute trace_dataset_hash
    trace_hashes = [(e["case_id"], e["sha256"]) for e in trace_entries]
    trace_dataset_hash = compute_trace_dataset_hash(mas_run_id, case_list_hash, trace_hashes)
    
    records = []
    
    # manifest_meta
    manifest_meta = {
        "record_type": "manifest_meta",
        "schema_version": "2.0.0",
        "dataset_id": dataset_id,
        "mas_run_id": mas_run_id,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Add stub_mode flag if true
    if stub_mode:
        manifest_meta["stub_mode"] = True
    
    records.append(manifest_meta)
    
    # provenance
    records.append({
        "record_type": "provenance",
        "mac_fork_url": mac_fork_url,
        "mac_commit": mac_commit,
        "exaid_commit": exaid_commit,
        "model": model,
        "decoding": decoding,
        "case_list_hash": case_list_hash,
        "case_list_hash_input": "SHA256 of case_list.jsonl file bytes",
        "config_hash": config_hash,
        "trace_dataset_hash": trace_dataset_hash,
        "trace_dataset_hash_definition": "SHA256 of JSON: {mas_run_id, case_list_hash, sorted [(case_id, trace_sha256)]}"
    })
    
    # trace_entry records
    for entry in trace_entries:
        records.append({
            "record_type": "trace_entry",
            "case_id": entry["case_id"],
            "file": entry["file"],
            "sha256": entry["sha256"],
            "delta_count": entry["delta_count"],
            "turn_count": entry["turn_count"]
        })
    
    # summary
    records.append({
        "record_type": "summary",
        "total_cases": len(trace_entries),
        "total_deltas": total_deltas,
        "total_turns": total_turns,
        "successful_cases": successful_cases,
        "failed_cases": failed_cases
    })
    
    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


# =============================================================================
# Stub Generation (for testing without MAC)
# =============================================================================

def generate_stub_turns(case_id: str, case_data: dict) -> List[Any]:
    """
    Generate stub turns for testing without MAC execution.
    
    Creates minimal TurnTrace-like objects for scaffold testing.
    """
    import time
    from dataclasses import dataclass, field
    from typing import List as ListType
    
    @dataclass
    class StubTokenEmission:
        token: str
        t_emitted_ms: int
        seq: int
    
    @dataclass
    class StubTurnTrace:
        turn_id: int
        agent_id: str
        content: str
        t_start_ms: int
        t_end_ms: int
        duration_ms: int
        token_emissions: ListType = field(default_factory=list)
    
    base_time = int(time.time() * 1000)
    turns = []
    
    # Stub messages
    messages = [
        ("supervisor", f"[STUB] Analyzing case: {case_id}"),
        ("doctor0", "[STUB] Initial differential diagnosis..."),
        ("doctor1", "[STUB] Additional considerations..."),
        ("supervisor", '[STUB] Consensus reached. {"Most Likely Diagnosis": "Pending"}'),
    ]
    
    for i, (agent, content) in enumerate(messages):
        t_start = base_time + i * 1000
        emissions = []
        
        # Split content into "tokens"
        words = content.split()
        for j, word in enumerate(words):
            token = word + " " if j < len(words) - 1 else word
            emissions.append(StubTokenEmission(
                token=token,
                t_emitted_ms=t_start + 50 + j * 30,
                seq=j
            ))
        
        t_end = t_start + 50 + len(words) * 30 + 100
        
        turns.append(StubTurnTrace(
            turn_id=i + 1,
            agent_id=agent,
            content=content,
            t_start_ms=t_start,
            t_end_ms=t_end,
            duration_ms=t_end - t_start,
            token_emissions=emissions
        ))
    
    return turns


# =============================================================================
# Main Entry Point
# =============================================================================

def run_generation(args) -> int:
    # Resolve relative paths relative to evals root
    evals_root = Path(__file__).resolve().parents[2]  # traces -> src -> evals

    def resolve_path(path: Path) -> Path:
        """Resolve relative paths relative to evals root."""
        if path.is_absolute():
            return path
        # Try current directory first, then evals root
        if path.exists():
            return path
        evals_path = evals_root / path
        if evals_path.exists():
            return evals_path
        return path  # Return as-is if neither exists (let downstream handle error)

    args.config = resolve_path(args.config)
    args.dataset_config = resolve_path(args.dataset_config)
    args.output = resolve_path(args.output)
    args.manifests = resolve_path(args.manifests)
    print("=" * 70)
    print("EXAIM Timed Trace Generation (v2.0.0)")
    print("=" * 70)
    print()
    
    # Load configurations
    mas_config = {}
    if args.config.exists():
        mas_config = load_config(args.config)
        print(f"Loaded MAS config: {args.config}")
    else:
        print(f"ERROR: MAS config not found: {args.config}")
        return 1
    
    dataset_config = {}
    if args.dataset_config.exists():
        dataset_config = load_dataset_config(args.dataset_config)
        print(f"Loaded dataset config: {args.dataset_config}")
    else:
        print(f"ERROR: Dataset config not found: {args.dataset_config}")
        return 1
    
    # Get MAC configuration
    mac_config = mas_config.get("mac", {})
    mac_module_path = mac_config.get("module_path", "/app/third_party/mac")
    mac_fork_url = mac_config.get("fork_url", "https://github.com/AbemKW/mac-streaming-traces")
    mac_commit = mac_config.get("commit", "unknown")
    model = mac_config.get("base_model", "gpt-4o-mini")
    decoding = mac_config.get("decoding", {"temperature": 1.0})
    mac_enabled = mac_config.get("enabled", False)
    
    # Get EXAIM commit
    exaim_commit = get_exaim_commit()
    exaid_commit = exaim_commit  # Legacy name for artifact compatibility
    
    print()
    print("Configuration:")
    print(f"  MAC module path: {mac_module_path}")
    print(f"  MAC fork URL: {mac_fork_url}")
    print(f"  MAC commit: {mac_commit}")
    print(f"  EXAIM commit: {exaid_commit[:8]}...")
    print(f"  Model: {model}")
    print(f"  Decoding: {json.dumps(decoding)}")
    print(f"  MAC enabled: {mac_enabled}")
    print()
    
    # Check MAC availability
    mac_available = Path(mac_module_path).exists() and mac_enabled and not args.stub_mode
    
    if not mac_available and not args.dry_run:
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
            all_case_ids = [f"case-stub-{i:03d}" for i in range(10)]
        print(f"Total cases available: {len(all_case_ids)}")
    except Exception as e:
        print(f"ERROR: Failed to load case IDs: {e}")
        all_case_ids = [f"case-stub-{i:03d}" for i in range(10)]
    
    # Select cases
    selected_cases = select_mac_cases(dataset_config, all_case_ids)
    
    # Apply limit if specified
    if args.limit:
        selected_cases = selected_cases[:args.limit]
    
    print(f"Cases selected: {len(selected_cases)}")
    
    mac_sel = dataset_config.get("mac_selection", {})
    print(f"  Selection mode: {mac_sel.get('mode', 'fixed_subset')}")
    print(f"  Selection seed: {mac_sel.get('seed', 42)}")
    print()
    
    # Write case list and compute hash
    config_hash = compute_config_hash(args.config, args.dataset_config)
    
    # Compute case list hash from in-memory content (for initial ID generation)
    # Note: This should match the file hash, but we verify below
    case_list_content = "\n".join([
        json.dumps({"index": i, "case_id": cid}, sort_keys=True, separators=(",", ":"))
        for i, cid in enumerate(selected_cases)
    ]) + "\n"
    case_list_hash = f"sha256:{hashlib.sha256(case_list_content.encode('utf-8')).hexdigest()}"
    
    # Generate deterministic IDs (will regenerate if hash mismatch detected)
    mas_run_id = generate_mas_run_id(mac_commit, model, decoding, case_list_hash)
    dataset_id = generate_dataset_id(mac_commit, decoding, case_list_hash)
    
    # Write case list file
    case_list_path = args.manifests / f"{dataset_id}.case_list.jsonl"
    actual_case_list_hash = write_case_list(selected_cases, case_list_path)
    
    # Verify hash matches - if not, regenerate IDs and rename file
    if actual_case_list_hash != case_list_hash:
        print(f"WARNING: Case list hash mismatch detected!")
        print(f"  Expected: {case_list_hash[:30]}...")
        print(f"  Actual:   {actual_case_list_hash[:30]}...")
        print(f"  Regenerating IDs with actual hash...")
        
        # Regenerate IDs with actual hash
        case_list_hash = actual_case_list_hash
        mas_run_id = generate_mas_run_id(mac_commit, model, decoding, case_list_hash)
        dataset_id = generate_dataset_id(mac_commit, decoding, case_list_hash)
        
        # Rename file to match new dataset_id
        new_case_list_path = args.manifests / f"{dataset_id}.case_list.jsonl"
        if case_list_path != new_case_list_path:
            case_list_path.rename(new_case_list_path)
            case_list_path = new_case_list_path
            print(f"  Renamed case list file to: {case_list_path}")
    
    print(f"MAS Run ID: {mas_run_id}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Case list hash: {case_list_hash[:30]}...")
    print(f"Config hash: {config_hash[:30]}...")
    print(f"Case list written: {case_list_path}")
    print()
    
    # DRY RUN: Stop here if requested
    if args.dry_run:
        print()
        print("=" * 70)
        print("DRY RUN COMPLETE")
        print("=" * 70)
        print()
        print("Configuration validated successfully.")
        print(f"Would process {len(selected_cases)} cases.")
        print()
        print("Files that would be created:")
        print(f"  - {case_list_path}")
        print(f"  - {args.manifests}/{dataset_id}.manifest.jsonl")
        for case_id in selected_cases[:3]:
            safe_id = re.sub(r'[^a-z0-9-]', '-', case_id.lower())
            print(f"  - {args.output}/{safe_id}.trace.jsonl.gz")
        if len(selected_cases) > 3:
            print(f"  - ... and {len(selected_cases) - 3} more trace files")
        print()
        return 0
    
    # Process cases
    trace_entries = []
    total_deltas = 0
    total_turns = 0
    successful_cases = 0
    failed_cases = 0
    
    on_failure = mac_sel.get("on_failure", "log_stub")
    
    print("Processing cases:")
    print("-" * 50)
    
    for idx, case_id in enumerate(selected_cases):
        print(f"  [{idx + 1}/{len(selected_cases)}] {case_id}...", end=" ", flush=True)
        
        # Load case data
        case_data = None
        if mac_available or Path(mac_module_path).exists():
            try:
                case_data = load_mac_case(mac_module_path, case_id, stage="inital")
            except Exception:
                case_data = None
        
        if case_data is None:
            case_data = {
                "case_id": case_id,
                "presentation": "Stub case for testing",
                "stage": "inital"
            }
        
        # Run MAC or generate stub traces
        turns = []
        error_message = None
        
        if mac_available:
            turns, error_message = run_mac_case_instrumented(mac_module_path, case_data, mas_config)
        else:
            turns = generate_stub_turns(case_id, case_data)
        
        # Handle failures
        if error_message:
            failed_cases += 1
            if on_failure == "raise":
                print(f"FAILED: {error_message}")
                raise RuntimeError(f"Case {case_id} failed: {error_message}")
            else:
                print(f"FAILED (stub): {error_message}")
                turns = generate_stub_turns(case_id, case_data)
        else:
            successful_cases += 1
        
        # Serialize turns to trace records
        records, t0, case_deltas, case_turns = serialize_to_trace_records(turns, case_id)
        
        print(f"OK ({case_turns} turns, {case_deltas} deltas)")
        
        # Write trace file
        safe_case_id = re.sub(r'[^a-z0-9-]', '-', case_id.lower())
        output_file = args.output / f"{safe_case_id}.trace.jsonl.gz"
        
        _, trace_hash = write_trace_file(
            case_id=case_id,
            records=records,
            t0_emitted_ms=t0,
            total_deltas=case_deltas,
            total_turns=case_turns,
            mas_run_id=mas_run_id,
            mac_commit=mac_commit,
            exaid_commit=exaid_commit,
            model=model,
            decoding=decoding,
            output_path=output_file,
            stub_mode=args.stub_mode
        )
        
        # Track entry for manifest
        trace_entries.append({
            "case_id": case_id,
            "file": output_file.name,
            "sha256": trace_hash,
            "delta_count": case_deltas,
            "turn_count": case_turns
        })
        
        total_deltas += case_deltas
        total_turns += case_turns
    
    print("-" * 50)
    print()
    
    # Write manifest
    manifest_path = args.manifests / f"{dataset_id}.manifest.jsonl"
    print(f"Writing manifest: {manifest_path}")
    
    write_manifest(
        output_path=manifest_path,
        dataset_id=dataset_id,
        mas_run_id=mas_run_id,
        mac_fork_url=mac_fork_url,
        mac_commit=mac_commit,
        exaid_commit=exaid_commit,
        model=model,
        decoding=decoding,
        case_list_hash=case_list_hash,
        config_hash=config_hash,
        trace_entries=trace_entries,
        total_deltas=total_deltas,
        total_turns=total_turns,
        successful_cases=successful_cases,
        failed_cases=failed_cases,
        stub_mode=args.stub_mode
    )
    
    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"  Cases processed: {len(selected_cases)}")
    print(f"  Successful: {successful_cases}")
    print(f"  Failed: {failed_cases}")
    print(f"  Total turns: {total_turns}")
    print(f"  Total deltas: {total_deltas}")
    print(f"  Output directory: {args.output}")
    print(f"  Manifest: {manifest_path}")
    print()
    
    if failed_cases > 0:
        print(f"WARNING: {failed_cases} cases failed and have stub traces")
    
    return 0
