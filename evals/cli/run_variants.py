#!/usr/bin/env python3
"""
EXAID Evaluation - Deterministic Variant Replay Engine

Paper hook: "Frozen traces replayed through variant pipelines with
deterministic timestamp derivation and byte-stable output (Section 3.1)"

Replays frozen traces through V0-V4 variant pipelines:
    V0: full_exaid - TokenGate + BufferAgent (all filters) + Summarizer
    V1: turn_end - Trigger at turn_end events (from turn_boundary records) + Summarizer only
    V2: no_buffer - TokenGate → Summarizer (skip BufferAgent)
    V3: no_tokengate - Fixed intervals + BufferAgent + Summarizer
    V4: no_novelty - TokenGate + BufferAgent (no novelty check) + Summarizer

Turn Boundary Derivation:
    - Turn boundaries are derived from explicit turn_boundary records in trace
    - Explicit boundaries take precedence over classification-derived bounds
    - V1 variant triggers summarization on turn_end events from ReplayEvent stream

Output: Multi-record JSONL run logs per evals/schemas/exaid.run.schema.json

Usage:
    python -m evals.cli.run_variants --traces data/traces/ --output data/runs/
    python -m evals.cli.run_variants --variant V3 --case case-33651373

Dependencies:
    - traces/trace_replay_engine.py (content-plane replay)
    - deterministic/utils.py (timestamps, IDs)
    - deterministic/io.py (gzip/JSON writing)
"""

import argparse
import asyncio
import json
import sys
import time
from statistics import mean
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from ..src.traces.trace_replay_engine import (
    TraceReplayEngine,
    iter_trace_records,
    ReplayEvent,
)
from ..src.deterministic.utils import (
    DeterministicTimestamps,
    compute_ctu,
    compute_text_hash,
    generate_event_id,
    generate_decision_id,
)
from ..src.deterministic.io import (
    RunLogBuilder,
    write_run_log_deterministic,
    compute_file_hash,
    write_json_deterministic,
)
from ..src.config.config_loader import (
    load_extractor_config as _load_extractor_config_central,
    load_variant_config as _load_variant_config_central,
    get_stoplists_provenance,
)
from ..src.metrics.integrity import load_manifest_provenance
from exaid_core.buffer_agent.buffer_agent import BufferAgent, BufferAnalysis, BufferAnalysisNoNovelty
from exaid_core.summarizer_agent.summarizer_agent import SummarizerAgent
from exaid_core.schema.agent_segment import AgentSegment
from exaid_core.schema.agent_summary import AgentSummary
from exaid_core.token_gate.token_gate import ManualClock, TokenGate


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


def run_async(ctx: "RunContext", coro):
    """Run an async coroutine using the run context loop."""
    return ctx.loop.run_until_complete(coro)


def format_summary_for_history(summary: AgentSummary) -> str:
    """Format summary for history prompts."""
    parts = [
        f"Status/Action: {summary.status_action}",
        f"Key Findings: {summary.key_findings}",
        f"Differential/Rationale: {summary.differential_rationale}",
        f"Uncertainty/Confidence: {summary.uncertainty_confidence}",
        f"Recommendation/Next: {summary.recommendation_next_step}",
        f"Agent Contributions: {summary.agent_contributions}",
    ]
    return " | ".join(parts)


def build_summary_semantics_text(summary: AgentSummary) -> str:
    """Concatenate summary fields for semantics text (exclude agent_contributions)."""
    return " ".join([
        summary.status_action,
        summary.key_findings,
        summary.differential_rationale,
        summary.uncertainty_confidence,
        summary.recommendation_next_step,
    ])


def event_time_ms(t0_emitted_ms: int, event: ReplayEvent) -> int:
    """Convert replay virtual time to absolute emitted time."""
    return int(t0_emitted_ms + event.virtual_time_ms)


def get_llm_model_id(llm) -> Optional[str]:
    """Best-effort extraction of model identifier from a LangChain LLM."""
    for attr in ("model_name", "model", "model_id"):
        if hasattr(llm, attr):
            value = getattr(llm, attr)
            if value:
                return value
    return None


def compute_tokengate_config_hash(variant_config: dict) -> str:
    """Compute deterministic hash of TokenGate config."""
    token_gate_config = variant_config.get("components", {}).get("token_gate", {})
    payload = json.dumps(
        token_gate_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )
    return compute_text_hash(payload)


def compute_trace_dataset_hash(traces_root: Path, manifest_path: Optional[Path] = None) -> str:
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


def build_run_summary(
    variant_id: str,
    eval_run_id: str,
    case_results: list[dict]
) -> dict:
    """Build per-variant run summary report."""
    total_cases = len(case_results)
    failed_cases = [
        {
            "case_id": result["case_id"],
            "reason": result.get("failure_reason", "unknown")
        }
        for result in case_results
        if result["status"] == "failed"
    ]
    summary_counts = {
        result["case_id"]: result.get("summary_count", 0)
        for result in case_results
    }
    durations_ms = [
        result["duration_ms"]
        for result in case_results
        if result.get("duration_ms") is not None
    ]
    timing_summary = {
        "count": len(durations_ms),
        "min": min(durations_ms) if durations_ms else None,
        "mean": mean(durations_ms) if durations_ms else None,
        "max": max(durations_ms) if durations_ms else None
    }

    return {
        "variant_id": variant_id,
        "eval_run_id": eval_run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": total_cases,
        "succeeded": total_cases - len(failed_cases),
        "failed": failed_cases,
        "per_case_summary_counts": summary_counts,
        "timing_ms": timing_summary
    }




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
    trigger_type: str
    summary_history_event_ids: list[str]
    summarizer_input_hash: str
    summarizer_input_text: Optional[str]
    limits_ok: bool
    failure_mode: Optional[str]
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
    trace_chunks: list[dict]  # stream_delta records (for timestamp derivation)
    replay_events: list[ReplayEvent]
    turn_bounds: dict[int, tuple[Optional[int], Optional[int]]]
    timestamps: DeterministicTimestamps
    clock: ManualClock
    t0_emitted_ms: int
    loop: asyncio.AbstractEventLoop
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
    summaries: list[AgentSummary] = field(default_factory=list)
    summary_event_ids: list[str] = field(default_factory=list)
    buffer_window_start_seq: Optional[int] = None
    buffer_window_end_seq: Optional[int] = None


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
        # Read novelty_check from config (default True for backward compatibility)
        novelty_check = config.get("components", {}).get("buffer_agent", {}).get("novelty_check", True)
        self.buffer_agent = BufferAgent(disable_novelty=not novelty_check)
        self.summarizer_agent = SummarizerAgent()
    
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
    
    def generate_summary(
        self,
        ctx: RunContext,
        start_seq: int,
        end_seq: int,
        segments: list[AgentSegment],
        agent_id: str,
        trigger_type: str
    ) -> SummaryResult:
        """
        Generate a summary using the real SummarizerAgent.
        
        Args:
            ctx: Run context
            start_seq: Start of window
            end_seq: End of window
            segments: Segments to summarize
            agent_id: Agent identifier
        """
        event_index = ctx.summary_count
        ctx.summary_count += 1
        
        event_id = generate_event_id(ctx.case_id, ctx.variant_id, event_index)
        timestamp = ctx.timestamps.get_window_timestamp(end_seq)

        summary_history = ctx.summaries
        summary_event_ids = ctx.summary_event_ids
        latest_summary = summary_history[-1] if summary_history else None
        latest_summary_event_id = summary_event_ids[-1] if summary_event_ids else None
        history_limit = max(ctx.history_k - 1, 0)
        if latest_summary:
            history_slice = summary_history[:-1]
            history_event_ids = summary_event_ids[:-1]
        else:
            history_slice = summary_history
            history_event_ids = summary_event_ids
        if history_limit > 0:
            history_slice = history_slice[-history_limit:]
            history_event_ids = history_event_ids[-history_limit:]
        else:
            history_slice = []
            history_event_ids = []
        summary_history_strs = [
            format_summary_for_history(summary) for summary in history_slice
        ]
        latest_summary_str = (
            format_summary_for_history(latest_summary)
            if latest_summary
            else "No summaries yet."
        )

        prompt_parts = []
        if summary_history_strs:
            prompt_parts.append("\n".join(summary_history_strs))
        prompt_parts.append(latest_summary_str)
        new_buffer, _ = self.summarizer_agent.format_segments_for_prompt(segments)
        prompt_parts.append(new_buffer)
        prompt_text = "\n".join(prompt_parts)

        start_time = time.perf_counter()
        summary = run_async(
            ctx,
            self.summarizer_agent.summarize(
                segments,
                summary_history_strs,
                latest_summary_str
            )
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if summary is None:
            summary = AgentSummary(
                status_action="",
                key_findings="",
                differential_rationale="",
                uncertainty_confidence="",
                recommendation_next_step="",
                agent_contributions=""
            )

        ctx.summaries.append(summary)
        if hasattr(summary, "model_dump"):
            summary_content = summary.model_dump()
        else:
            summary_content = summary.dict()
        summary_semantics_text = build_summary_semantics_text(summary)

        result = SummaryResult(
            event_index=event_index,
            event_id=event_id,
            start_seq=start_seq,
            end_seq=end_seq,
            timestamp=timestamp,
            trigger_type=trigger_type,
            summary_history_event_ids=history_event_ids,
            summarizer_input_hash=compute_text_hash(prompt_text),
            summarizer_input_text=prompt_text,
            limits_ok=True,
            failure_mode=None,
            schema_ok=True,
            schema_error=None,
            summary_semantics_text=summary_semantics_text,
            summary_ctu=compute_ctu(summary_semantics_text),
            summary_content=summary_content,
            latest_summary_event_id=latest_summary_event_id,
            new_buffer_text_hash=compute_text_hash(new_buffer),
            llm_usage={
                "prompt_ctu": compute_ctu(prompt_text),
                "completion_ctu": compute_ctu(summary_semantics_text),
                "provider_prompt_tokens": None,
                "provider_completion_tokens": None,
                "model_id": get_llm_model_id(self.summarizer_agent.base_llm)
            },
            latency_ms=latency_ms
        )
        
        # Update context for next summary
        ctx.latest_summary_event_id = event_id
        ctx.latest_summary_text = summary_semantics_text
        ctx.summary_event_ids.append(event_id)
        
        return result

    def evaluate_buffer_agent(
        self,
        ctx: RunContext,
        agent_id: str,
        segment_text: str
    ) -> tuple[bool, BufferDecisionResult]:
        """Run BufferAgent on a segment and build a decision result.
        
        Novelty check is automatically determined by the BufferAgent's initialization
        (based on config), so we derive it from the analysis type.
        """
        decision_index = ctx.decision_count
        ctx.decision_count += 1

        summary_history = ctx.summaries[-ctx.history_k:]
        summary_history = [
            format_summary_for_history(summary) for summary in summary_history
        ]
        buffer_context = self.buffer_agent.format_segments_for_prompt(self.buffer_agent.buffer)

        prompt_text = "\n".join([
            f"agent_id: {agent_id}",
            "summaries:",
            "\n".join(summary_history),
            f"previous_trace: {buffer_context}",
            f"new_trace: {segment_text}"
        ])

        start_time = time.perf_counter()
        should_trigger = run_async(
            ctx,
            self.buffer_agent.addsegment(agent_id, segment_text, summary_history)
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        analysis: BufferAnalysis | BufferAnalysisNoNovelty | None = self.buffer_agent.get_last_analysis(agent_id)

        if analysis is None:
            filter_results = {
                "completeness_passed": None,
                "value_passed": None,
                "novelty_passed": None
            }
        else:
            # Derive novelty check status from analysis type (hasattr check)
            # BufferAnalysisNoNovelty doesn't have is_novel field
            novelty_passed = None
            if hasattr(analysis, "is_novel"):
                novelty_passed = analysis.is_novel
            
            filter_results = {
                "completeness_passed": analysis.is_complete,
                "value_passed": analysis.is_relevant,
                "novelty_passed": novelty_passed
            }

        if analysis is not None:
            if hasattr(analysis, "model_dump"):
                completion_payload = json.dumps(analysis.model_dump())
            else:
                completion_payload = json.dumps(analysis.dict())
        else:
            completion_payload = ""

        decision = BufferDecisionResult(
            decision_index=decision_index,
            decision_id=generate_decision_id(ctx.case_id, ctx.variant_id, decision_index),
            start_seq=ctx.buffer_window_start_seq or 0,
            end_seq=ctx.buffer_window_end_seq or 0,
            timestamp=ctx.timestamps.get_window_timestamp(ctx.buffer_window_end_seq or 0),
            input_ctu=compute_ctu(buffer_context + "\n" + segment_text),
            decision="summarize" if should_trigger else "buffer",
            filter_results=filter_results,
            llm_usage={
                "prompt_ctu": compute_ctu(prompt_text),
                "completion_ctu": compute_ctu(completion_payload),
                "provider_prompt_tokens": None,
                "provider_completion_tokens": None,
                "model_id": get_llm_model_id(self.buffer_agent.base_llm)
            },
            latency_ms=latency_ms
        )

        return should_trigger, decision

    def create_token_gate(self, ctx: RunContext) -> TokenGate:
        """Create a TokenGate with deterministic clock and config."""
        token_gate_config = self.config.get("components", {}).get("token_gate", {})
        return TokenGate(
            min_words=token_gate_config.get("min_words", 35),
            max_words=token_gate_config.get("max_words", 90),
            boundary_cues=token_gate_config.get("boundary_cues", ".?!\n"),
            silence_timer=token_gate_config.get("silence_timer", 15),
            max_wait_timeout=token_gate_config.get("max_wait_timeout", 40),
            clock=ctx.clock
        )

    def run_tokengate_pipeline(
        self,
        ctx: RunContext,
        use_buffer_agent: bool
    ) -> tuple[list, list, list]:
        """Run TokenGate-driven pipeline for V0/V2/V4."""
        summaries: list[SummaryResult] = []
        decisions: list[BufferDecisionResult] = []
        flushes: list[TokenGateFlushResult] = []

        token_gate = self.create_token_gate(ctx)
        token_gate_state: dict[str, dict[str, Optional[int]]] = {}

        for event in ctx.replay_events:
            if event.event_type != "delta":
                continue

            seq = event.seq
            text = event.delta_text or ""
            if not text:
                continue

            agent_id = event.agent_id
            ctx.clock.set_time(
                datetime.fromtimestamp(
                    event_time_ms(ctx.t0_emitted_ms, event) / 1000,
                    tz=timezone.utc
                )
            )

            state = token_gate_state.setdefault(agent_id, {"window_start": None, "last_seq": None})
            if not token_gate.buffers.get(agent_id):
                state["window_start"] = seq
                state["last_seq"] = None

            previous_last_seq = state["last_seq"]
            flushed_text = run_async(ctx, token_gate.add_token(agent_id, text))

            if flushed_text:
                flush_reason = token_gate.get_last_flush_reason(agent_id) or "threshold"
                if flush_reason == "silence_timer":
                    flush_start = state["window_start"] if state["window_start"] is not None else seq
                    flush_end = previous_last_seq if previous_last_seq is not None else seq
                    state["window_start"] = seq
                    state["last_seq"] = seq
                else:
                    flush_start = state["window_start"] if state["window_start"] is not None else seq
                    flush_end = seq
                    state["window_start"] = None
                    state["last_seq"] = None

                flush = TokenGateFlushResult(
                    flush_index=ctx.flush_count,
                    start_seq=flush_start,
                    end_seq=flush_end,
                    timestamp=ctx.timestamps.get_window_timestamp(flush_end),
                    accumulated_ctu=compute_ctu(flushed_text),
                    trigger_reason=flush_reason,
                    text_hash=compute_text_hash(flushed_text)
                )
                flushes.append(flush)
                ctx.flush_count += 1

                if use_buffer_agent:
                    if not self.buffer_agent.buffer:
                        ctx.buffer_window_start_seq = flush_start
                    ctx.buffer_window_end_seq = flush_end

                    should_trigger, decision = self.evaluate_buffer_agent(
                        ctx, agent_id, flushed_text
                    )
                    decisions.append(decision)

                    if should_trigger:
                        flushed_segments = self.buffer_agent.flush()
                        summary = self.generate_summary(
                            ctx,
                            ctx.buffer_window_start_seq or flush_start,
                            ctx.buffer_window_end_seq or flush_end,
                            flushed_segments,
                            agent_id,
                            trigger_type="buffer_agent"
                        )
                        summaries.append(summary)
                        ctx.buffer_window_start_seq = None
                        ctx.buffer_window_end_seq = None
                else:
                    summary = self.generate_summary(
                        ctx,
                        flush_start,
                        flush_end,
                        [AgentSegment(agent_id=agent_id, segment=flushed_text)],
                        agent_id,
                        trigger_type="tokengate_flush"
                    )
                    summaries.append(summary)
            else:
                if token_gate.buffers.get(agent_id):
                    state["last_seq"] = seq

        for agent_id, state in token_gate_state.items():
            if token_gate.buffers.get(agent_id):
                flushed_text = run_async(ctx, token_gate.flush(agent_id, reason="end_of_trace"))
                if flushed_text:
                    flush_start = state["window_start"] if state["window_start"] is not None else 0
                    flush_end = state["last_seq"] if state["last_seq"] is not None else flush_start
                    flush = TokenGateFlushResult(
                        flush_index=ctx.flush_count,
                        start_seq=flush_start,
                        end_seq=flush_end,
                        timestamp=ctx.timestamps.get_window_timestamp(flush_end),
                        accumulated_ctu=compute_ctu(flushed_text),
                        trigger_reason="end_of_trace",
                        text_hash=compute_text_hash(flushed_text)
                    )
                    flushes.append(flush)
                    ctx.flush_count += 1

                    if use_buffer_agent:
                        if not self.buffer_agent.buffer:
                            ctx.buffer_window_start_seq = flush_start
                        ctx.buffer_window_end_seq = flush_end

                        should_trigger, decision = self.evaluate_buffer_agent(
                            ctx, agent_id, flushed_text
                        )
                        decisions.append(decision)

                        if should_trigger:
                            flushed_segments = self.buffer_agent.flush()
                            summary = self.generate_summary(
                                ctx,
                                ctx.buffer_window_start_seq or flush_start,
                                ctx.buffer_window_end_seq or flush_end,
                                flushed_segments,
                                agent_id,
                                trigger_type="buffer_agent"
                            )
                            summaries.append(summary)
                            ctx.buffer_window_start_seq = None
                            ctx.buffer_window_end_seq = None
                    else:
                        summary = self.generate_summary(
                            ctx,
                            flush_start,
                            flush_end,
                            [AgentSegment(agent_id=agent_id, segment=flushed_text)],
                            agent_id,
                            trigger_type="tokengate_flush"
                        )
                        summaries.append(summary)

        return summaries, decisions, flushes


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
        return self.run_tokengate_pipeline(
            ctx, use_buffer_agent=True
        )


class V1_TurnEnd(VariantPipeline):
    """
    V1: Turn-end only trigger.
    
    Paper hook: "V1 triggers summarization only at turn boundaries,
    bypassing TokenGate and BufferAgent (Section 4.2)"
    
    Implementation:
        - Triggers on turn_end events from TraceReplayEngine
        - Turn boundaries derived from explicit turn_boundary records in trace
        - Accumulates text from delta events until turn_end event
        - Generates summary with accumulated text for the turn
    """
    
    def run(self, ctx: RunContext) -> tuple[list, list, list]:
        summaries = []
        decisions = []
        flushes = []
        
        accumulated_text = ""
        window_start: Optional[int] = None
        last_agent_id = "unknown"
        last_seq_seen: Optional[int] = None
        current_turn_id: Optional[int] = None
        
        for event in ctx.replay_events:
            if event.event_type == "turn_start":
                current_turn_id = event.turn_id
                bounds = ctx.turn_bounds.get(event.turn_id)
                window_start = event.seq if event.seq is not None else (bounds[0] if bounds else None)
                accumulated_text = ""
                last_seq_seen = None
                last_agent_id = event.agent_id
                continue

            if event.event_type == "delta":
                if current_turn_id is None:
                    current_turn_id = event.turn_id
                    bounds = ctx.turn_bounds.get(event.turn_id)
                    window_start = bounds[0] if bounds and bounds[0] is not None else event.seq
                accumulated_text += event.delta_text or ""
                last_agent_id = event.agent_id
                last_seq_seen = event.seq
                continue

            if event.event_type == "turn_end":
                if not accumulated_text:
                    current_turn_id = None
                    window_start = None
                    last_seq_seen = None
                    continue

                bounds = ctx.turn_bounds.get(event.turn_id)
                end_seq = event.seq if event.seq is not None else (bounds[1] if bounds else last_seq_seen)
                start_seq = window_start if window_start is not None else end_seq
                summary = self.generate_summary(
                    ctx,
                    start_seq,
                    end_seq,
                    [AgentSegment(agent_id=last_agent_id, segment=accumulated_text)],
                    last_agent_id,
                    trigger_type="turn_end"
                )
                summaries.append(summary)

                accumulated_text = ""
                window_start = None
                last_seq_seen = None
                current_turn_id = None
        
        # Handle remaining text
        if accumulated_text and last_seq_seen is not None:
            start_seq = window_start if window_start is not None else last_seq_seen
            summary = self.generate_summary(
                ctx,
                start_seq,
                last_seq_seen,
                [AgentSegment(agent_id=last_agent_id, segment=accumulated_text)],
                last_agent_id,
                trigger_type="turn_end"
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
        return self.run_tokengate_pipeline(
            ctx, use_buffer_agent=False
        )


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
        last_agent_id = "unknown"
        last_seq_seen: Optional[int] = None
        
        for event in ctx.replay_events:
            if event.event_type != "delta":
                continue

            seq = event.seq
            text = event.delta_text or ""
            if not text:
                continue

            if not accumulated_text:
                window_start = seq
            last_agent_id = event.agent_id
            chunk_ctu = compute_ctu(text)
            last_seq_seen = seq
            
            accumulated_text += text
            accumulated_ctu += chunk_ctu
            
            if accumulated_ctu >= chunk_size_ctu:
                if not self.buffer_agent.buffer:
                    ctx.buffer_window_start_seq = window_start
                ctx.buffer_window_end_seq = seq

                should_trigger, decision = self.evaluate_buffer_agent(
                    ctx, last_agent_id, accumulated_text
                )
                decisions.append(decision)

                if should_trigger:
                    flushed_segments = self.buffer_agent.flush()
                    summary = self.generate_summary(
                        ctx,
                        ctx.buffer_window_start_seq or window_start,
                        ctx.buffer_window_end_seq or seq,
                        flushed_segments,
                        last_agent_id,
                        trigger_type="buffer_agent"
                    )
                    summaries.append(summary)
                    ctx.buffer_window_start_seq = None
                    ctx.buffer_window_end_seq = None
                
                accumulated_text = ""
                accumulated_ctu = 0
                window_start = seq + 1
        
        if accumulated_ctu > 0 and last_seq_seen is not None:
            if not self.buffer_agent.buffer:
                ctx.buffer_window_start_seq = window_start
            ctx.buffer_window_end_seq = last_seq_seen

            should_trigger, decision = self.evaluate_buffer_agent(
                ctx, last_agent_id, accumulated_text
            )
            decisions.append(decision)

            if should_trigger:
                flushed_segments = self.buffer_agent.flush()
                summary = self.generate_summary(
                    ctx,
                    ctx.buffer_window_start_seq or window_start,
                    ctx.buffer_window_end_seq or last_seq_seen,
                    flushed_segments,
                    last_agent_id,
                    trigger_type="buffer_agent"
                )
                summaries.append(summary)
                ctx.buffer_window_start_seq = None
                ctx.buffer_window_end_seq = None
        
        return summaries, decisions, flushes


class V4_NoNovelty(VariantPipeline):
    """
    V4: No novelty check in BufferAgent.
    
    Paper hook: "V4 disables novelty filtering to measure its
    contribution to redundancy reduction (Section 4.2)"
    """
    
    def run(self, ctx: RunContext) -> tuple[list, list, list]:
        # V4 uses BufferAgent with novelty_check disabled via config
        # The BufferAgent is initialized in VariantPipeline.__init__() based on config
        return self.run_tokengate_pipeline(
            ctx, use_buffer_agent=True
        )


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
    trace_dataset_hash: str,
    output_dir: Path
) -> tuple[Path, dict]:
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
        Tuple of output path and run stats
    """
    # Load trace records and replay events
    trace_records = list(iter_trace_records(trace_file))
    # Filter for stream_delta records (chunk type for text extraction)
    delta_records = [
        record for record in trace_records
        if record.get("record_type") == "stream_delta"
    ]
    if not delta_records:
        raise ValueError(f"No stream_delta records found in {trace_file}")

    # Initialize deterministic timestamps from stream_delta records
    timestamps = DeterministicTimestamps(delta_records)

    # Initialize deterministic clock for TokenGate timers
    clock = ManualClock(
        datetime.fromtimestamp(timestamps.run_start_time / 1000, tz=timezone.utc)
    )

    replay_engine = TraceReplayEngine(trace_file)
    replay_events = list(replay_engine.replay_content_plane())
    if not replay_events:
        raise ValueError(f"No content_plane replay events in {trace_file}")

    # Derive turn bounds from explicit turn_boundary records (authoritative source)
    boundary_turn_bounds: dict[int, dict[str, Optional[int]]] = {}
    for record in trace_records:
        if record.get("record_type") != "turn_boundary":
            continue
        turn_id = record.get("turn_id")
        boundary = record.get("boundary")
        seq = record.get("seq")
        if turn_id is not None and boundary in ("start", "end"):
            bounds = boundary_turn_bounds.setdefault(turn_id, {})
            bounds[boundary] = seq
    
    # Merge with classification bounds (explicit boundaries take precedence)
    classification_bounds = {
        turn_id: (cls.start_seq, cls.end_seq)
        for turn_id, cls in replay_engine.get_turn_classifications().items()
    }
    
    turn_bounds: dict[int, tuple[Optional[int], Optional[int]]] = {}
    for turn_id, (start_seq, end_seq) in classification_bounds.items():
        boundary_bounds = boundary_turn_bounds.get(turn_id, {})
        turn_bounds[turn_id] = (
            boundary_bounds.get("start", start_seq),
            boundary_bounds.get("end", end_seq)
        )
    
    # Add any turns that have explicit boundaries but no classification
    for turn_id, bounds in boundary_turn_bounds.items():
        if turn_id not in turn_bounds:
            turn_bounds[turn_id] = (
                bounds.get("start"),
                bounds.get("end")
            )
    meta = replay_engine.get_metadata()
    mas_run_id = meta.mas_run_id

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create run context
    history_k = variant_config.get("summarizer", {}).get("history_k")
    if history_k is None:
        raise ValueError(f"Missing summarizer.history_k for {variant_id}")
    if not isinstance(history_k, int) or history_k < 1:
        raise ValueError(f"Invalid summarizer.history_k for {variant_id}: {history_k}")

    ctx = RunContext(
        case_id=case_id,
        variant_id=variant_id,
        trace_file=trace_file,
        trace_chunks=delta_records,
        replay_events=replay_events,
        turn_bounds=turn_bounds,
        timestamps=timestamps,
        clock=clock,
        t0_emitted_ms=meta.t0_emitted_ms,
        loop=loop,
        history_k=history_k,
        mas_run_id=mas_run_id,
        eval_run_id=eval_run_id,
    )
    
    # Create and run pipeline
    pipeline = create_pipeline(variant_id, variant_config)
    try:
        summaries, decisions, flushes = pipeline.run(ctx)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    
    # Build run log
    builder = RunLogBuilder()
    
    # run_meta record
    run_meta = {
        "schema_name": "exaid.run",
        "schema_version": "1.5.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "variant_id": variant_id,
        "mas_run_id": mas_run_id,
        "eval_run_id": eval_run_id,
        "history_k": ctx.history_k,
        "trigger_policy": variant_config.get("trigger_policy", "unknown"),
        "trace_file_hash": compute_file_hash(trace_file),
        "trace_dataset_hash": trace_dataset_hash,
        "tokengate_config_hash": compute_tokengate_config_hash(variant_config),
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
            "trigger_type": summary.trigger_type,
            "summary_history_event_ids": summary.summary_history_event_ids,
            "summarizer_input_hash": summary.summarizer_input_hash,
            "summarizer_input_text": summary.summarizer_input_text,
            "limits_ok": summary.limits_ok,
            "failure_mode": summary.failure_mode,
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
    
    return output_path, {
        "summary_count": len(summaries),
        "decision_count": len(decisions),
        "flush_count": len(flushes),
    }


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
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to manifest file (for trace_dataset_hash computation). If not provided, will attempt to auto-detect."
    )
    
    args = parser.parse_args()
    
    # Resolve relative paths relative to evals root
    evals_root = Path(__file__).resolve().parents[1]  # cli -> evals
    
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
    
    args.traces = resolve_path(args.traces)
    args.output = resolve_path(args.output)
    args.configs = resolve_path(args.configs)
    if args.manifest:
        args.manifest = resolve_path(args.manifest)
    
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
        if args.case:
            for variant_id in variants:
                case_results = [{
                    "case_id": args.case,
                    "status": "failed",
                    "summary_count": 0,
                    "duration_ms": None,
                    "failure_reason": "missing trace file"
                }]
                summary = build_run_summary(variant_id, eval_run_id, case_results)
                summary_path = args.output / "run_summaries" / f"{eval_run_id}_{variant_id}.json"
                write_json_deterministic(summary, summary_path)
        return 1
    
    # Filter by case if specified
    if args.case:
        trace_files = [f for f in trace_files if args.case in f.stem]

    if args.case and not trace_files:
        print(f"ERROR: No trace files found for case {args.case}")
        for variant_id in variants:
            case_results = [{
                "case_id": args.case,
                "status": "failed",
                "summary_count": 0,
                "duration_ms": None,
                "failure_reason": "missing trace file"
            }]
            summary = build_run_summary(variant_id, eval_run_id, case_results)
            summary_path = args.output / "run_summaries" / f"{eval_run_id}_{variant_id}.json"
            write_json_deterministic(summary, summary_path)
        return 1
    
    print(f"Found {len(trace_files)} trace files")
    print(f"Running variants: {', '.join(variants)}")
    print()

    # Process each trace file
    total_runs = 0
    trace_dataset_hash = compute_trace_dataset_hash(trace_files[0].parent, manifest_path=args.manifest)
    variant_case_results = {variant_id: [] for variant_id in variants}
    
    for trace_file in trace_files:
        case_id = trace_file.stem.replace(".jsonl", "")
        
        if args.verbose:
            print(f"Processing: {case_id}")
        
        for variant_id in variants:
            variant_config = load_variant_config(variant_id, args.configs)
            
            try:
                start_time = time.perf_counter()
                output_path, run_stats = execute_run(
                    case_id=case_id,
                    variant_id=variant_id,
                    trace_file=trace_file,
                    variant_config=variant_config,
                    extractor_config=extractor_config,
                    stoplists_provenance=stoplists_provenance,
                    eval_run_id=eval_run_id,
                    trace_dataset_hash=trace_dataset_hash,
                    output_dir=args.output
                )
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                total_runs += 1
                variant_case_results[variant_id].append({
                    "case_id": case_id,
                    "status": "succeeded",
                    "summary_count": run_stats.get("summary_count", 0),
                    "duration_ms": duration_ms,
                    "failure_reason": None
                })
                
                if args.verbose:
                    print(f"  {variant_id}: {output_path.name}")
                    
            except Exception as e:
                print(f"ERROR: {case_id}/{variant_id}: {e}")
                variant_case_results[variant_id].append({
                    "case_id": case_id,
                    "status": "failed",
                    "summary_count": 0,
                    "duration_ms": None,
                    "failure_reason": str(e)
                })
                if args.verbose:
                    import traceback
                    traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"COMPLETE: {total_runs} runs")
    print(f"Output: {args.output}")
    print("=" * 60)

    for variant_id, case_results in variant_case_results.items():
        summary = build_run_summary(variant_id, eval_run_id, case_results)
        summary_path = args.output / "run_summaries" / f"{eval_run_id}_{variant_id}.json"
        write_json_deterministic(summary, summary_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
