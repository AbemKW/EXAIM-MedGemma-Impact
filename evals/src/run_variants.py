#!/usr/bin/env python3
"""
EXAID Evaluation - Variant Runner Script

Replays traces through different summarizer variants and logs the results.
Implements the data model and logging structure for evaluation runs.

Current behavior:
- Reads trace files from data/traces/
- Processes each trace through variant configurations V0-V4
- Outputs run logs to data/runs/V{n}/
- Includes concept extractor configuration in run metadata

Usage:
    python run_variants.py --traces data/traces/ --output data/runs/
    python run_variants.py --variant V3 --traces data/traces/
"""

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import yaml


# Concept extractor configuration (scispaCy baseline)
DEFAULT_CONCEPT_EXTRACTOR_CONFIG = {
    "model": "en_core_sci_sm",
    "entity_types_kept": ["ALL"],
    "stop_entities_file": "configs/stop_entities.txt",
    "min_entity_len": 3,
    "normalization": {
        "lowercase": True,
        "strip_whitespace": True,
        "canonicalize_punctuation": True
    }
}


def load_config(config_path: Path) -> dict:
    """Load a YAML configuration file."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_summarizer_config() -> dict:
    """Load the summarizer configuration."""
    config_path = Path("configs/summarizer.yaml")
    return load_config(config_path)


def load_variant_config(variant_id: str) -> dict:
    """Load a variant configuration."""
    config_path = Path(f"configs/variants/{variant_id}.yaml")
    return load_config(config_path)


def load_stop_entities(stop_file: Path) -> set:
    """Load stop entities from file."""
    if not stop_file.exists():
        return set()
    
    entities = set()
    with open(stop_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                entities.add(line.lower())
    return entities


def normalize_entity(entity: str) -> str:
    """Normalize an entity string."""
    # Lowercase
    entity = entity.lower()
    # Strip whitespace
    entity = entity.strip()
    # Canonicalize punctuation (remove trailing punctuation)
    entity = entity.rstrip(".,;:!?")
    return entity


def extract_concepts_stub(text: str, config: dict) -> list[str]:
    """
    STUB: Extract concepts from text using scispaCy.
    
    TODO: Implement actual scispaCy extraction
    
    This stub returns placeholder concepts.
    Real implementation will:
    1. Load en_core_sci_sm model
    2. Process text through NER pipeline
    3. Filter by entity_types_kept
    4. Apply stop entity filtering
    5. Apply min_entity_len filtering
    6. Normalize and deduplicate
    
    Args:
        text: Input text
        config: Concept extractor configuration
        
    Returns:
        List of extracted concept strings
    """
    # Placeholder: return stub concepts
    # Real implementation would use spaCy/scispaCy
    stub_concepts = ["placeholder_concept_1", "placeholder_concept_2"]
    
    # Apply normalization and filtering (demonstrating the pipeline)
    min_len = config.get("min_entity_len", 3)
    normalized = []
    for concept in stub_concepts:
        norm = normalize_entity(concept)
        if len(norm) >= min_len:
            normalized.append(norm)
    
    return list(set(normalized))


def generate_run_id(variant_id: str, trace_id: str) -> str:
    """Generate a unique run ID."""
    trace_part = trace_id.replace("trc-", "")
    return f"run-{variant_id.lower()}-{trace_part}"


def open_trace_file(file_path: Path):
    """Open a trace file, handling gzip compression."""
    if str(file_path).endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8")
    return open(file_path, "r", encoding="utf-8")


def read_traces(traces_dir: Path) -> Iterator[tuple[Path, list[dict]]]:
    """
    Read trace files from directory.
    
    Yields:
        Tuples of (file_path, list_of_traces)
    """
    if not traces_dir.exists():
        print(f"Traces directory does not exist: {traces_dir}")
        return
    
    patterns = ["*.jsonl", "*.jsonl.gz"]
    for pattern in patterns:
        for trace_file in sorted(traces_dir.glob(pattern)):
            traces = []
            with open_trace_file(trace_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        traces.append(json.loads(line))
            if traces:
                yield trace_file, traces


def generate_stub_summary(trace: dict, summary_idx: int, history_k: int) -> dict:
    """
    STUB: Generate a summary from a trace.
    
    TODO: Replace with actual EXAID summarizer integration
    
    Args:
        trace: Trace record
        summary_idx: Summary sequence index
        history_k: Number of historical summaries to consider
        
    Returns:
        Summary content dict
    """
    return {
        "status_action": f"[STUB] Processing trace from {trace.get('agent_id', 'unknown')}",
        "key_findings": "[STUB] Placeholder findings",
        "differential_rationale": "[STUB] Placeholder rationale",
        "uncertainty_confidence": "[STUB] Moderate confidence",
        "recommendation_next_step": "[STUB] Continue analysis",
        "agent_contributions": f"[STUB] {trace.get('agent_id', 'unknown')}: primary"
    }


def run_variant(
    variant_id: str,
    traces: list[dict],
    summarizer_config: dict,
    variant_config: dict
) -> dict:
    """
    Run a variant on a set of traces.
    
    Args:
        variant_id: Variant identifier (V0, V1, etc.)
        traces: List of trace records
        summarizer_config: Base summarizer configuration
        variant_config: Variant-specific configuration
        
    Returns:
        Run record conforming to exaid.run.schema.json
    """
    if not traces:
        return None
    
    first_trace = traces[0]
    trace_id = first_trace.get("trace_id", "unknown")
    case_id = first_trace.get("case_id", "unknown")
    
    # Determine history_k (variant override or default)
    history_k = variant_config.get("summarizer", {}).get("history_k_override")
    if history_k is None:
        history_k = summarizer_config.get("history_k", 3)
    
    # Check if summarization is enabled for this variant
    summarization_enabled = variant_config.get("summarization", {}).get("enabled", True)
    
    # Generate summaries
    summaries = []
    start_time = datetime.now(timezone.utc)
    
    if summarization_enabled:
        # Generate summaries (stub implementation)
        for idx, trace in enumerate(traces):
            summary = {
                "summary_idx": idx,
                "content": generate_stub_summary(trace, idx, history_k),
                "token_count": 50,  # Placeholder
                "trigger_trace_idx": idx
            }
            summaries.append(summary)
    else:
        # V0 baseline: no summarization, pass through raw traces
        pass
    
    end_time = datetime.now(timezone.utc)
    duration_ms = int((end_time - start_time).total_seconds() * 1000)
    
    # Extract concepts from traces (for coverage computation)
    all_trace_text = " ".join(t.get("content", "") for t in traces)
    extracted_concepts = extract_concepts_stub(all_trace_text, DEFAULT_CONCEPT_EXTRACTOR_CONFIG)
    
    # Build run record
    run_record = {
        "schema_name": "exaid.run",
        "schema_version": "1.0.0",
        "run_id": generate_run_id(variant_id, trace_id),
        "trace_id": trace_id,
        "case_id": case_id,
        "variant_id": variant_id,
        "timestamp": end_time.isoformat(),
        "summaries": summaries,
        "timing": {
            "total_ms": duration_ms,
            "avg_summary_ms": duration_ms / len(summaries) if summaries else 0
        },
        "run_meta": {
            "concept_extractor": DEFAULT_CONCEPT_EXTRACTOR_CONFIG,
            "summarizer_config": {
                "history_k": history_k
            }
        },
        "concept_coverage": {
            "total_concepts_in_trace": len(extracted_concepts),
            "concepts_in_summaries": len(extracted_concepts),  # Stub: same as total
            "coverage_ratio": 1.0 if extracted_concepts else 0.0,  # Stub
            "extracted_concepts": extracted_concepts
        }
    }
    
    return run_record


def write_run(run_record: dict, output_dir: Path, compress: bool = True) -> Path:
    """Write a run record to file."""
    variant_dir = output_dir / run_record["variant_id"]
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{run_record['case_id']}.jsonl"
    if compress:
        filename += ".gz"
    
    output_path = variant_dir / filename
    
    open_func = gzip.open if compress else open
    mode = "wt" if compress else "w"
    
    with open_func(output_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(run_record) + "\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run EXAID summarizer variants on traces"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("data/traces"),
        help="Input traces directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/runs"),
        help="Output runs directory"
    )
    parser.add_argument(
        "--variant",
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Run only a specific variant (default: all)"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable gzip compression"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXAID Variant Runner (STUB)")
    print("=" * 60)
    print()
    print("NOTE: This is a STUB implementation.")
    print("Actual EXAID summarizer integration is pending.")
    print()
    
    # Load configurations
    summarizer_config = load_summarizer_config()
    print(f"Summarizer config: history_k={summarizer_config.get('history_k', 3)}")
    print()
    
    # Determine variants to run
    variants = [args.variant] if args.variant else ["V0", "V1", "V2", "V3", "V4"]
    
    # Load variant configs
    variant_configs = {}
    for v in variants:
        variant_configs[v] = load_variant_config(v)
        print(f"Loaded variant config: {v}")
    print()
    
    # Process traces
    total_runs = 0
    
    for trace_file, traces in read_traces(args.traces):
        print(f"Processing: {trace_file.name} ({len(traces)} traces)")
        
        for variant_id in variants:
            run_record = run_variant(
                variant_id,
                traces,
                summarizer_config,
                variant_configs[variant_id]
            )
            
            if run_record:
                output_path = write_run(
                    run_record,
                    args.output,
                    compress=not args.no_compress
                )
                total_runs += 1
                print(f"  {variant_id}: {len(run_record['summaries'])} summaries -> {output_path.name}")
    
    print()
    print("=" * 60)
    print(f"COMPLETE: {total_runs} runs across {len(variants)} variants")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Integrate actual EXAID summarizer")
    print("2. Implement scispaCy concept extraction")
    print("3. Run: python src/compute_metrics.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


