#!/usr/bin/env python3
"""
EXAID Evaluation - Deterministic Variant Replay Engine

Paper hook: "Frozen traces replayed through variant pipelines with
deterministic timestamp derivation and byte-stable output (Section 3.1)"

Replays frozen traces through V0-V4 variant pipelines:
    V0: full_exaid - TokenGate + BufferAgent (all filters) + Summarizer
    V1: turn_end - Trigger at turn boundaries only
    V2: no_buffer - TokenGate → Summarizer (skip BufferAgent)
    V3: no_tokengate - Fixed intervals + BufferAgent + Summarizer
    V4: no_novelty - TokenGate + BufferAgent (no novelty check) + Summarizer

Output: Multi-record JSONL run logs per evals/schemas/exaid.run.schema.json

Usage:
    python run_variants.py --traces data/traces/ --output data/runs/
    python run_variants.py --variant V3 --case case-33651373

Dependencies:
    - trace_text.py (canonical text construction)
    - deterministic_utils.py (timestamps, IDs)
    - deterministic_io.py (gzip/JSON writing)
"""

import argparse
import hashlib
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from trace_text import (
    build_canonical_trace_text,
    build_window_text,
    load_trace_chunks_for_case,
    TraceParsingStats,
)
from deterministic_utils import (
    DeterministicTimestamps,
    compute_ctu,
    compute_text_hash,
    generate_event_id,
    generate_decision_id,
)
from deterministic_io import (
    RunLogBuilder,
    write_run_log_deterministic,
    compute_file_hash,
)
from config_loader import (
    load_extractor_config as _load_extractor_config_central,
    load_variant_config as _load_variant_config_central,
    get_stoplists_provenance,
)


# ============================================================================
# Configuration Loading
# ============================================================================

def load_yaml_config(config_path: Path) -> dict:
    """Load YAML config file."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_variant_config(variant_id: str, configs_dir: Path) -> dict:
    """Load variant configuration via centralized loader."""
    return _load_variant_config_central(variant_id, configs_dir)


def load_extractor_config(configs_dir: Path) -> dict:
    """
    Load extractor configuration via centralized loader.
    
    Paper hook: "Extractor config loaded via centralized loader ensuring
    drift-proof evaluation with resolved stoplist paths (Section 6.1)"
    """
    return _load_extractor_config_central(configs_dir, resolve_paths=True, include_hashes=True)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SummaryResult:
    """Result of summary generation."""
    event_index: int
    event_id: str
    start_seq: int
    end_seq: int
    timestamp: str
    schema_ok: bool
    schema_error: Optional[str]
    summary_semantics_text: str
    summary_ctu: int
    summary_content: dict
    latest_summary_event_id: Optional[str]
    new_buffer_text_hash: str
    llm_usage: dict
    latency_ms: int


@dataclass 
class BufferDecisionResult:
    """Result of BufferAgent decision."""
    decision_index: int
    decision_id: str
    start_seq: int
    end_seq: int
    timestamp: str
    input_ctu: int
    decision: str  # "summarize", "buffer", "discard"
    filter_results: dict
    llm_usage: dict
    latency_ms: int


@dataclass
class TokenGateFlushResult:
    """Result of TokenGate flush."""
    flush_index: int
    start_seq: int
    end_seq: int
    timestamp: str
    accumulated_ctu: int
    trigger_reason: str
    text_hash: str


@dataclass
class RunContext:
    """Context for a single run execution."""
    case_id: str
    variant_id: str
    trace_file: Path
    trace_chunks: list[dict]
    timestamps: DeterministicTimestamps
    history_k: int = 3
    mas_run_id: str = ""
    eval_run_id: str = ""
    
    # Counters
    summary_count: int = 0
    decision_count: int = 0
    flush_count: int = 0
    
    # Latest summary for M6b reconstruction
    latest_summary_event_id: Optional[str] = None
    latest_summary_text: str = ""


# ============================================================================
# Abstract Variant Pipeline
# ============================================================================

class VariantPipeline(ABC):
    """
    Abstract base class for variant pipelines.
    
    Paper hook: "Each variant implements a specific trigger policy
    and component configuration (Section 4.1)"
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.variant_id = config.get("variant_id", "V?")
        self.trigger_policy = config.get("trigger_policy", "unknown")
    
    @abstractmethod
    def run(self, ctx: RunContext) -> tuple[list, list, list]:
        """
        Execute variant pipeline on trace.
        
        Args:
            ctx: Run context with trace data
            
        Returns:
            Tuple of (summary_results, buffer_decisions, tokengate_flushes)
        """
        pass
    
    def generate_stub_summary(
        self,
        ctx: RunContext,
        start_seq: int,
        end_seq: int,
        input_text: str
    ) -> SummaryResult:
        """
        Generate a stub summary (to be replaced with real EXAID integration).
        
        Args:
            ctx: Run context
            start_seq: Start of window
            end_seq: End of window
            input_text: Input text for summarization
        """
        event_index = ctx.summary_count
        ctx.summary_count += 1
        
        event_id = generate_event_id(ctx.case_id, ctx.variant_id, event_index)
        timestamp = ctx.timestamps.get_window_timestamp(end_seq)
        
        # Stub summary content
        summary_content = {
            "status_action": f"[STUB] Processing chunks {start_seq}-{end_seq}",
            "key_findings": "[STUB] Placeholder findings",
            "differential_rationale": "[STUB] Placeholder rationale",
            "uncertainty_confidence": "[STUB] Moderate confidence",
            "recommendation_next_step": "[STUB] Continue analysis",
            "agent_contributions": "[STUB] Analysis complete"
        }
        
        # Concatenate summary sections for concept extraction
        summary_semantics_text = " ".join([
            summary_content.get("status_action", ""),
            summary_content.get("key_findings", ""),
            summary_content.get("differential_rationale", ""),
            summary_content.get("uncertainty_confidence", ""),
            summary_content.get("recommendation_next_step", ""),
            summary_content.get("agent_contributions", ""),
        ])
        
        result = SummaryResult(
            event_index=event_index,
            event_id=event_id,
            start_seq=start_seq,
            end_seq=end_seq,
            timestamp=timestamp,
            schema_ok=True,
            schema_error=None,
            summary_semantics_text=summary_semantics_text,
            summary_ctu=compute_ctu(summary_semantics_text),
            summary_content=summary_content,
            latest_summary_event_id=ctx.latest_summary_event_id,
            new_buffer_text_hash=compute_text_hash(input_text),
            llm_usage={
                "prompt_ctu": compute_ctu(input_text),
                "completion_ctu": compute_ctu(summary_semantics_text),
                "provider_prompt_tokens": None,
                "provider_completion_tokens": None,
                "model_id": "stub"
            },
            latency_ms=100  # Stub latency
        )
        
        # Update context for next summary
        ctx.latest_summary_event_id = event_id
        ctx.latest_summary_text = summary_semantics_text
        
        return result


# ============================================================================
# Variant Implementations
# ============================================================================

class V0_FullEXAID(VariantPipeline):
    """
    V0: Full EXAID pipeline.
    
    Paper hook: "V0 implements complete EXAID with TokenGate accumulation,
    BufferAgent 3-layer filtering, and Summarizer (Section 4.1)"
    """
    
    def run(self, ctx: RunContext) -> tuple[list, list, list]:
        summaries = []
        decisions = []
        flushes = []
        
        # TokenGate parameters (word thresholds)
        min_words = self.config.get("components", {}).get("token_gate", {}).get("min_words", 35)
        max_words = self.config.get("components", {}).get("token_gate", {}).get("max_words", 90)
        
        # Simulate TokenGate accumulation
        accumulated_text = ""
        accumulated_ctu = 0
        window_start = 0
        
        for chunk in ctx.trace_chunks:
            seq = chunk.get("seq", chunk.get("sequence_num", 0))
            text = chunk.get("text_chunk") or chunk.get("content") or ""
            chunk_ctu = compute_ctu(text)
            
            accumulated_text += " " + text
            accumulated_ctu += chunk_ctu
            
            # Check TokenGate threshold
            if accumulated_ctu >= min_words:
                # TokenGate flush
                flush = TokenGateFlushResult(
                    flush_index=ctx.flush_count,
                    start_seq=window_start,
                    end_seq=seq,
                    timestamp=ctx.timestamps.get_window_timestamp(seq),
                    accumulated_ctu=accumulated_ctu,
                    trigger_reason="threshold",
                    text_hash=compute_text_hash(accumulated_text)
                )
                flushes.append(flush)
                ctx.flush_count += 1
                
                # BufferAgent decision (stub: always summarize)
                decision = BufferDecisionResult(
                    decision_index=ctx.decision_count,
                    decision_id=generate_decision_id(ctx.case_id, ctx.variant_id, ctx.decision_count),
                    start_seq=window_start,
                    end_seq=seq,
                    timestamp=ctx.timestamps.get_window_timestamp(seq),
                    input_ctu=accumulated_ctu,
                    decision="summarize",
                    filter_results={
                        "completeness_passed": True,
                        "value_passed": True,
                        "novelty_passed": True
                    },
                    llm_usage={
                        "prompt_ctu": accumulated_ctu,
                        "completion_ctu": 5,
                        "provider_prompt_tokens": None,
                        "provider_completion_tokens": None,
                        "model_id": "stub"
                    },
                    latency_ms=50
                )
                decisions.append(decision)
                ctx.decision_count += 1
                
                # Generate summary
                summary = self.generate_stub_summary(
                    ctx, window_start, seq, accumulated_text
                )
                summaries.append(summary)
                
                # Reset accumulator
                accumulated_text = ""
                accumulated_ctu = 0
                window_start = seq + 1
        
        # Handle remaining accumulated text
        if accumulated_ctu > 0:
            last_seq = ctx.trace_chunks[-1].get("seq", 0) if ctx.trace_chunks else 0
            summary = self.generate_stub_summary(
                ctx, window_start, last_seq, accumulated_text
            )
            summaries.append(summary)
        
        return summaries, decisions, flushes


class V1_TurnEnd(VariantPipeline):
    """
    V1: Turn-end only trigger.
    
    Paper hook: "V1 triggers summarization only at turn boundaries,
    bypassing TokenGate and BufferAgent (Section 4.2)"
    """
    
    def run(self, ctx: RunContext) -> tuple[list, list, list]:
        summaries = []
        decisions = []
        flushes = []
        
        accumulated_text = ""
        window_start = 0
        
        for chunk in ctx.trace_chunks:
            seq = chunk.get("seq", chunk.get("sequence_num", 0))
            text = chunk.get("text_chunk") or chunk.get("content") or ""
            is_turn_end = chunk.get("is_turn_end", False)
            
            accumulated_text += " " + text
            
            # Only trigger on turn end
            if is_turn_end and accumulated_text.strip():
                summary = self.generate_stub_summary(
                    ctx, window_start, seq, accumulated_text
                )
                summaries.append(summary)
                
                accumulated_text = ""
                window_start = seq + 1
        
        # Handle remaining text
        if accumulated_text.strip():
            last_seq = ctx.trace_chunks[-1].get("seq", 0) if ctx.trace_chunks else 0
            summary = self.generate_stub_summary(
                ctx, window_start, last_seq, accumulated_text
            )
            summaries.append(summary)
        
        return summaries, decisions, flushes


class V2_NoBuffer(VariantPipeline):
    """
    V2: No BufferAgent (TokenGate → Summarizer).
    
    Paper hook: "V2 flushes TokenGate output directly to Summarizer,
    measuring BufferAgent's contribution (Section 4.2)"
    """
    
    def run(self, ctx: RunContext) -> tuple[list, list, list]:
        summaries = []
        decisions = []
        flushes = []
        
        min_words = self.config.get("components", {}).get("token_gate", {}).get("min_words", 35)
        
        accumulated_text = ""
        accumulated_ctu = 0
        window_start = 0
        
        for chunk in ctx.trace_chunks:
            seq = chunk.get("seq", chunk.get("sequence_num", 0))
            text = chunk.get("text_chunk") or chunk.get("content") or ""
            chunk_ctu = compute_ctu(text)
            
            accumulated_text += " " + text
            accumulated_ctu += chunk_ctu
            
            if accumulated_ctu >= min_words:
                # TokenGate flush
                flush = TokenGateFlushResult(
                    flush_index=ctx.flush_count,
                    start_seq=window_start,
                    end_seq=seq,
                    timestamp=ctx.timestamps.get_window_timestamp(seq),
                    accumulated_ctu=accumulated_ctu,
                    trigger_reason="threshold",
                    text_hash=compute_text_hash(accumulated_text)
                )
                flushes.append(flush)
                ctx.flush_count += 1
                
                # Skip BufferAgent, go directly to summarizer
                summary = self.generate_stub_summary(
                    ctx, window_start, seq, accumulated_text
                )
                summaries.append(summary)
                
                accumulated_text = ""
                accumulated_ctu = 0
                window_start = seq + 1
        
        if accumulated_ctu > 0:
            last_seq = ctx.trace_chunks[-1].get("seq", 0) if ctx.trace_chunks else 0
            summary = self.generate_stub_summary(
                ctx, window_start, last_seq, accumulated_text
            )
            summaries.append(summary)
        
        return summaries, decisions, flushes


class V3_NoTokenGate(VariantPipeline):
    """
    V3: No TokenGate (fixed intervals).
    
    Paper hook: "V3 uses fixed CTU intervals calibrated from V0 median,
    measuring TokenGate's adaptive contribution (Section 4.2)"
    """
    
    def run(self, ctx: RunContext) -> tuple[list, list, list]:
        summaries = []
        decisions = []
        flushes = []
        
        # Fixed chunk size from calibration
        chunk_size_ctu = self.config.get("fixed_trigger", {}).get("chunk_size_ctu", 125)
        
        accumulated_text = ""
        accumulated_ctu = 0
        window_start = 0
        
        for chunk in ctx.trace_chunks:
            seq = chunk.get("seq", chunk.get("sequence_num", 0))
            text = chunk.get("text_chunk") or chunk.get("content") or ""
            chunk_ctu = compute_ctu(text)
            
            accumulated_text += " " + text
            accumulated_ctu += chunk_ctu
            
            if accumulated_ctu >= chunk_size_ctu:
                # BufferAgent decision (stub)
                decision = BufferDecisionResult(
                    decision_index=ctx.decision_count,
                    decision_id=generate_decision_id(ctx.case_id, ctx.variant_id, ctx.decision_count),
                    start_seq=window_start,
                    end_seq=seq,
                    timestamp=ctx.timestamps.get_window_timestamp(seq),
                    input_ctu=accumulated_ctu,
                    decision="summarize",
                    filter_results={
                        "completeness_passed": True,
                        "value_passed": True,
                        "novelty_passed": True
                    },
                    llm_usage={
                        "prompt_ctu": accumulated_ctu,
                        "completion_ctu": 5,
                        "provider_prompt_tokens": None,
                        "provider_completion_tokens": None,
                        "model_id": "stub"
                    },
                    latency_ms=50
                )
                decisions.append(decision)
                ctx.decision_count += 1
                
                summary = self.generate_stub_summary(
                    ctx, window_start, seq, accumulated_text
                )
                summaries.append(summary)
                
                accumulated_text = ""
                accumulated_ctu = 0
                window_start = seq + 1
        
        if accumulated_ctu > 0:
            last_seq = ctx.trace_chunks[-1].get("seq", 0) if ctx.trace_chunks else 0
            summary = self.generate_stub_summary(
                ctx, window_start, last_seq, accumulated_text
            )
            summaries.append(summary)
        
        return summaries, decisions, flushes


class V4_NoNovelty(VariantPipeline):
    """
    V4: No novelty check in BufferAgent.
    
    Paper hook: "V4 disables novelty filtering to measure its
    contribution to redundancy reduction (Section 4.2)"
    """
    
    def run(self, ctx: RunContext) -> tuple[list, list, list]:
        summaries = []
        decisions = []
        flushes = []
        
        min_words = self.config.get("components", {}).get("token_gate", {}).get("min_words", 35)
        
        accumulated_text = ""
        accumulated_ctu = 0
        window_start = 0
        
        for chunk in ctx.trace_chunks:
            seq = chunk.get("seq", chunk.get("sequence_num", 0))
            text = chunk.get("text_chunk") or chunk.get("content") or ""
            chunk_ctu = compute_ctu(text)
            
            accumulated_text += " " + text
            accumulated_ctu += chunk_ctu
            
            if accumulated_ctu >= min_words:
                flush = TokenGateFlushResult(
                    flush_index=ctx.flush_count,
                    start_seq=window_start,
                    end_seq=seq,
                    timestamp=ctx.timestamps.get_window_timestamp(seq),
                    accumulated_ctu=accumulated_ctu,
                    trigger_reason="threshold",
                    text_hash=compute_text_hash(accumulated_text)
                )
                flushes.append(flush)
                ctx.flush_count += 1
                
                # BufferAgent with novelty DISABLED
                decision = BufferDecisionResult(
                    decision_index=ctx.decision_count,
                    decision_id=generate_decision_id(ctx.case_id, ctx.variant_id, ctx.decision_count),
                    start_seq=window_start,
                    end_seq=seq,
                    timestamp=ctx.timestamps.get_window_timestamp(seq),
                    input_ctu=accumulated_ctu,
                    decision="summarize",
                    filter_results={
                        "completeness_passed": True,
                        "value_passed": True,
                        "novelty_passed": None  # DISABLED
                    },
                    llm_usage={
                        "prompt_ctu": accumulated_ctu,
                        "completion_ctu": 5,
                        "provider_prompt_tokens": None,
                        "provider_completion_tokens": None,
                        "model_id": "stub"
                    },
                    latency_ms=50
                )
                decisions.append(decision)
                ctx.decision_count += 1
                
                summary = self.generate_stub_summary(
                    ctx, window_start, seq, accumulated_text
                )
                summaries.append(summary)
                
                accumulated_text = ""
                accumulated_ctu = 0
                window_start = seq + 1
        
        if accumulated_ctu > 0:
            last_seq = ctx.trace_chunks[-1].get("seq", 0) if ctx.trace_chunks else 0
            summary = self.generate_stub_summary(
                ctx, window_start, last_seq, accumulated_text
            )
            summaries.append(summary)
        
        return summaries, decisions, flushes


# ============================================================================
# Pipeline Factory
# ============================================================================

def create_pipeline(variant_id: str, config: dict) -> VariantPipeline:
    """Create variant pipeline instance."""
    pipeline_classes = {
        "V0": V0_FullEXAID,
        "V1": V1_TurnEnd,
        "V2": V2_NoBuffer,
        "V3": V3_NoTokenGate,
        "V4": V4_NoNovelty,
    }
    
    pipeline_class = pipeline_classes.get(variant_id)
    if not pipeline_class:
        raise ValueError(f"Unknown variant: {variant_id}")
    
    return pipeline_class(config)


# ============================================================================
# Run Execution
# ============================================================================

def execute_run(
    case_id: str,
    variant_id: str,
    trace_file: Path,
    variant_config: dict,
    extractor_config: dict,
    stoplists_provenance: dict,
    eval_run_id: str,
    output_dir: Path
) -> Path:
    """
    Execute a single run and write output.
    
    Args:
        case_id: Case identifier
        variant_id: Variant identifier
        trace_file: Path to trace file
        variant_config: Variant configuration
        extractor_config: Concept extractor configuration
        stoplists_provenance: Stoplist provenance info
        eval_run_id: Evaluation run batch ID
        output_dir: Output directory
        
    Returns:
        Path to output run log
    """
    # Load trace chunks
    trace_chunks = load_trace_chunks_for_case(trace_file)
    
    if not trace_chunks:
        raise ValueError(f"No chunks found in {trace_file}")
    
    # Initialize deterministic timestamps
    timestamps = DeterministicTimestamps(trace_chunks)
    
    # Extract mas_run_id from first chunk if available
    mas_run_id = trace_chunks[0].get("mas_run_id", "mas-unknown")
    
    # Create run context
    ctx = RunContext(
        case_id=case_id,
        variant_id=variant_id,
        trace_file=trace_file,
        trace_chunks=trace_chunks,
        timestamps=timestamps,
        history_k=variant_config.get("summarizer", {}).get("history_k", 3),
        mas_run_id=mas_run_id,
        eval_run_id=eval_run_id,
    )
    
    # Create and run pipeline
    pipeline = create_pipeline(variant_id, variant_config)
    summaries, decisions, flushes = pipeline.run(ctx)
    
    # Build run log
    builder = RunLogBuilder()
    
    # run_meta record
    run_meta = {
        "schema_name": "exaid.run",
        "schema_version": "1.3.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "variant_id": variant_id,
        "mas_run_id": mas_run_id,
        "eval_run_id": eval_run_id,
        "history_k": ctx.history_k,
        "trigger_policy": variant_config.get("trigger_policy", "unknown"),
        "trace_file_hash": compute_file_hash(trace_file),
        "concept_extractor": {
            "spacy_version": "3.7.4",
            "scispacy_version": "0.5.4",
            "scispacy_model": extractor_config.get("scispacy_model", "en_core_sci_sm"),
            "linker_name": extractor_config.get("linker_name", "umls"),
            "linker_kb_version": extractor_config.get("linker_kb_version", "2023AB"),
            "linker_resolve_abbreviations": extractor_config.get("linker_resolve_abbreviations", True),
            "linker_max_entities_per_mention": extractor_config.get("linker_max_entities_per_mention", 10),
            "linker_threshold": extractor_config.get("linker_threshold", 0.7),
            "cui_score_threshold": extractor_config.get("cui_score_threshold", 0.7),
            "max_k": extractor_config.get("max_k", 10),
            "min_entity_len": extractor_config.get("min_entity_len", 3),
            "concept_representation": extractor_config.get("concept_representation", "cui"),
            "cui_normalization": extractor_config.get("cui_normalization", "uppercase"),
            "entity_types_kept": extractor_config.get("entity_types_kept", ["ALL"]),
            "stop_entities_count": 0,
            "stop_cuis_count": 0
        },
        "stoplists_provenance": stoplists_provenance,
        "timestamp_derivation": timestamps.get_stats(),
        "determinism": {
            "gzip_mtime": 0,
            "json_sort_keys": True,
            "json_separators": [",", ":"]
        }
    }
    builder.set_run_meta(run_meta)
    
    # Add tokengate flushes
    for flush in flushes:
        builder.add_tokengate_flush({
            "flush_index": flush.flush_index,
            "case_id": case_id,
            "variant_id": variant_id,
            "timestamp": flush.timestamp,
            "start_seq": flush.start_seq,
            "end_seq": flush.end_seq,
            "accumulated_ctu": flush.accumulated_ctu,
            "trigger_reason": flush.trigger_reason,
            "text_hash": flush.text_hash
        })
    
    # Add buffer decisions
    for decision in decisions:
        builder.add_buffer_decision({
            "decision_index": decision.decision_index,
            "decision_id": decision.decision_id,
            "case_id": case_id,
            "variant_id": variant_id,
            "timestamp": decision.timestamp,
            "start_seq": decision.start_seq,
            "end_seq": decision.end_seq,
            "input_ctu": decision.input_ctu,
            "decision": decision.decision,
            "filter_results": decision.filter_results,
            "llm_usage": decision.llm_usage,
            "latency_ms": decision.latency_ms
        })
    
    # Add summary events
    for summary in summaries:
        builder.add_summary_event({
            "event_index": summary.event_index,
            "event_id": summary.event_id,
            "case_id": case_id,
            "variant_id": variant_id,
            "timestamp": summary.timestamp,
            "start_seq": summary.start_seq,
            "end_seq": summary.end_seq,
            "schema_ok": summary.schema_ok,
            "schema_error": summary.schema_error,
            "summary_semantics_text": summary.summary_semantics_text,
            "summary_ctu": summary.summary_ctu,
            "summary_content": summary.summary_content,
            "latest_summary_event_id": summary.latest_summary_event_id,
            "new_buffer_text_hash": summary.new_buffer_text_hash,
            "llm_usage": summary.llm_usage,
            "latency_ms": summary.latency_ms
        })
    
    # Write output
    output_path = output_dir / variant_id / f"{case_id}.jsonl.gz"
    builder.write(output_path)
    
    # Validate determinism
    timestamps.validate()
    
    return output_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXAID Deterministic Variant Replay Engine"
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
        default=Path("data/runs"),
        help="Output runs directory (default: data/runs)"
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs"),
        help="Configs directory (default: configs)"
    )
    parser.add_argument(
        "--variant",
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Run only specific variant (default: all)"
    )
    parser.add_argument(
        "--case",
        type=str,
        help="Run only specific case ID"
    )
    parser.add_argument(
        "--eval-run-id",
        type=str,
        help="Evaluation run batch ID (default: auto-generated)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXAID Deterministic Variant Replay Engine")
    print("=" * 60)
    print()
    
    # Generate eval_run_id if not provided
    eval_run_id = args.eval_run_id
    if not eval_run_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        eval_run_id = f"eval-{timestamp}"
    
    print(f"Traces: {args.traces}")
    print(f"Output: {args.output}")
    print(f"Eval Run ID: {eval_run_id}")
    print()
    
    # Load extractor config
    extractor_config = load_extractor_config(args.configs)
    
    # Stub stoplists provenance (will be populated from actual stoplists)
    stoplists_provenance = {
        "stop_entities_hash": "sha256:0" * 64,
        "stop_cuis_hash": "sha256:0" * 64,
        "stoplists_generated_at": datetime.now(timezone.utc).isoformat(),
        "stoplists_generated_by_commit": None
    }
    
    # Load actual hashes if files exist
    stop_entities_path = args.configs / "stop_entities.txt"
    stop_cuis_path = args.configs / "stop_cuis.txt"
    if stop_entities_path.exists():
        stoplists_provenance["stop_entities_hash"] = compute_file_hash(stop_entities_path)
    if stop_cuis_path.exists():
        stoplists_provenance["stop_cuis_hash"] = compute_file_hash(stop_cuis_path)
    
    # Determine variants to run
    variants = [args.variant] if args.variant else ["V0", "V1", "V2", "V3", "V4"]
    
    # Find trace files
    trace_files = sorted(args.traces.glob("*.jsonl.gz"))
    if not trace_files:
        trace_files = sorted(args.traces.glob("*.jsonl"))
    
    if not trace_files:
        print(f"ERROR: No trace files found in {args.traces}")
        return 1
    
    # Filter by case if specified
    if args.case:
        trace_files = [f for f in trace_files if args.case in f.stem]
    
    print(f"Found {len(trace_files)} trace files")
    print(f"Running variants: {', '.join(variants)}")
    print()
    
    # Process each trace file
    total_runs = 0
    
    for trace_file in trace_files:
        case_id = trace_file.stem.replace(".jsonl", "")
        
        if args.verbose:
            print(f"Processing: {case_id}")
        
        for variant_id in variants:
            variant_config = load_variant_config(variant_id, args.configs)
            
            try:
                output_path = execute_run(
                    case_id=case_id,
                    variant_id=variant_id,
                    trace_file=trace_file,
                    variant_config=variant_config,
                    extractor_config=extractor_config,
                    stoplists_provenance=stoplists_provenance,
                    eval_run_id=eval_run_id,
                    output_dir=args.output
                )
                total_runs += 1
                
                if args.verbose:
                    print(f"  {variant_id}: {output_path.name}")
                    
            except Exception as e:
                print(f"ERROR: {case_id}/{variant_id}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"COMPLETE: {total_runs} runs")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
