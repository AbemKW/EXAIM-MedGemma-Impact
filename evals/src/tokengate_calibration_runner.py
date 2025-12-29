"""Orchestrate TokenGate calibration runs."""

import asyncio
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

# Ensure repo root is available for exaid_core imports if running outside Docker.
repo_root = Path(__file__).parent.parent.parent.resolve()
if (repo_root / "exaid_core").exists():
    sys.path.insert(0, str(repo_root))

from .trace_replay_engine import TraceReplayEngine, StubTraceError, TraceValidationError
from exaid_core.token_gate.token_gate import TokenGate, ManualClock

from .tokengate_calibration_grid import filter_valid_policies, generate_policy_grid
from .tokengate_calibration_io import (
    compute_config_hash,
    compute_trace_dataset_hash,
    generate_calibration_run_id,
    get_exaid_commit,
    load_config,
    load_manifest_entries,
    resolve_manifest_path,
    verify_trace_hashes,
    write_calibration_report_md,
    write_calibration_results_csv,
    write_chosen_params_yaml,
    write_config_copy,
    write_per_case_jsonl,
    write_spam_sensitivity_json,
    write_summary_json,
)
from .tokengate_calibration_metrics import (
    aggregate_policy_metrics,
    check_constraints,
    compute_all_normalization_bounds,
    compute_case_metrics,
    compute_spam_sensitivity,
)
from .tokengate_calibration_models import CaseMetrics, FlushEvent, Policy, PolicyMetrics
from .tokengate_calibration_selection import (
    compute_utopia_distances_for_all,
    compute_weighted_score,
    select_pareto_utopia,
)


async def replay_case_with_policy(
    trace_path: Path,
    policy: Policy,
    case_id: str,
    trace_meta,
    engine: Optional[TraceReplayEngine] = None,
    strict_stub_guard: bool = True,
    strict_validation: bool = True,
) -> CaseMetrics:
    """
    Replay a single case with a policy, recording flush events.

    Production-faithful replay: Matches production behavior exactly:
    - Advance clock to event time
    - Call add_token() (which internally checks silence timer before adding)
    - Call check_timers() after add_token() (matching production line 144)
    - On turn_end, call flush() with reason="turn_end" (matching production flush_agent())

    We do NOT simulate timer expiries during gaps between events, as production
    has no background/tick loop. Timers are only checked synchronously after
    each token addition or at turn boundaries.

    Args:
        trace_path: Path to trace file
        policy: Policy to evaluate
        case_id: Case identifier
        trace_meta: Trace metadata (from engine.get_metadata())
        engine: Optional pre-created TraceReplayEngine (if None, creates one)
        strict_stub_guard: Whether to enforce strict stub guard (default: True)
        strict_validation: Whether to enforce strict validation (default: True)
    """

    # Initialize TokenGate with ManualClock
    # trace_meta.t0_emitted_ms is milliseconds since epoch
    t0_datetime = datetime.fromtimestamp(trace_meta.t0_emitted_ms / 1000.0, tz=timezone.utc)
    clock = ManualClock(t0_datetime)
    token_gate = TokenGate(
        min_words=policy.min_words,
        max_words=policy.max_words,
        boundary_cues=policy.boundary_cues,
        silence_timer=policy.silence_timer_ms / 1000.0,  # Convert ms to seconds
        max_wait_timeout=policy.max_wait_timeout_ms / 1000.0,  # Convert ms to seconds
        clock=clock,
    )

    # Use provided engine or create one with specified strictness
    if engine is None:
        engine = TraceReplayEngine(
            trace_path,
            strict_stub_guard=strict_stub_guard,
            strict_validation=strict_validation,
        )

    # Track flush events
    flush_events: List[FlushEvent] = []
    first_content_delta_time_ms: Optional[int] = None
    trace_t0_ms = trace_meta.t0_emitted_ms

    # Replay content_plane stream
    events = list(engine.replay_content_plane())

    if not events:
        # No content events - return empty metrics
        return CaseMetrics(case_id=case_id, policy_id=policy.policy_id)

    # Process events matching production behavior exactly
    for event in events:
        current_time_ms = event.virtual_time_ms

        # Advance clock to event time
        # trace_t0_ms is milliseconds since epoch, current_time_ms is relative ms
        event_datetime = datetime.fromtimestamp(
            (trace_t0_ms + current_time_ms) / 1000.0, tz=timezone.utc
        )
        clock.set_time(event_datetime)

        # Process delta event
        if event.event_type == "delta" and event.delta_text:
            # Track first content delta for TTFF_content
            if first_content_delta_time_ms is None:
                first_content_delta_time_ms = current_time_ms

            agent_id = event.agent_id

            # Add token (internally checks silence timer before adding)
            flushed_text = await token_gate.add_token(agent_id, event.delta_text)

            if flushed_text:
                # Flush occurred from add_token()
                flush_reason = token_gate.get_last_flush_reason(agent_id) or "unknown"
                flush_time = token_gate.get_last_flush_time(agent_id)
                # Convert datetime to milliseconds since trace t0
                if flush_time:
                    flush_time_ms = int((flush_time.timestamp() * 1000 - trace_t0_ms))
                else:
                    flush_time_ms = current_time_ms

                chunk_words = len(flushed_text.split())

                flush_events.append(
                    FlushEvent(
                        agent_id=agent_id,
                        flush_time_ms=flush_time_ms,
                        chunk_words=chunk_words,
                        flush_reason=flush_reason,
                        is_end_of_trace=False,
                    )
                )

            # Check timers after add_token() (matching production line 144)
            timer_chunk = await token_gate.check_timers(agent_id)
            if timer_chunk:
                flush_reason = token_gate.get_last_flush_reason(agent_id) or "timer"
                flush_time = token_gate.get_last_flush_time(agent_id)
                if flush_time:
                    flush_time_ms = int((flush_time.timestamp() * 1000 - trace_t0_ms))
                else:
                    flush_time_ms = current_time_ms

                chunk_words = len(timer_chunk.split())

                flush_events.append(
                    FlushEvent(
                        agent_id=agent_id,
                        flush_time_ms=flush_time_ms,
                        chunk_words=chunk_words,
                        flush_reason=flush_reason,
                        is_end_of_trace=False,
                    )
                )

        # Process turn_end event (explicit flush, matching production flush_agent())
        elif event.event_type == "turn_end":
            agent_id = event.agent_id

            # Explicitly flush buffer at turn end (matching production flush_agent() behavior)
            flushed_text = await token_gate.flush(agent_id, reason="turn_end")

            if flushed_text:
                flush_time = token_gate.get_last_flush_time(agent_id)
                if flush_time:
                    flush_time_ms = int((flush_time.timestamp() * 1000 - trace_t0_ms))
                else:
                    flush_time_ms = current_time_ms

                chunk_words = len(flushed_text.split())

                flush_events.append(
                    FlushEvent(
                        agent_id=agent_id,
                        flush_time_ms=flush_time_ms,
                        chunk_words=chunk_words,
                        flush_reason="turn_end",  # Explicit reason, not timer-based
                        is_end_of_trace=False,  # Mid-trace turn end, not end of entire trace
                    )
                )

    # End-of-trace handling: deterministic cleanup flush without timer simulation
    # Production-faithful: we do NOT advance time or simulate timer expiries.
    # Instead, we perform a deterministic cleanup flush for any remaining buffers.
    if events:
        last_event = events[-1]
        last_time_ms = last_event.virtual_time_ms

        # Flush any remaining buffers without advancing time (production-faithful)
        for agent_id in list(token_gate.buffers.keys()):
            if agent_id in token_gate.buffers and token_gate.buffers[agent_id]:
                # Flush with deterministic reason (not timer-based)
                flushed_text = await token_gate.flush(agent_id, reason="end_of_trace")
                if flushed_text:
                    chunk_words = len(flushed_text.split())

                    flush_events.append(
                        FlushEvent(
                            agent_id=agent_id,
                            flush_time_ms=last_time_ms,  # Use last event time, don't advance
                            chunk_words=chunk_words,
                            flush_reason="end_of_trace",
                            is_end_of_trace=True,
                        )
                    )

    # Compute metrics from flush events
    return compute_case_metrics(
        case_id,
        policy.policy_id,
        flush_events,
        first_content_delta_time_ms,
        trace_t0_ms,
        policy,
    )


async def run_calibration(
    traces_dir: Path,
    manifest_pattern: str,
    config_path: Path,
    output_root: Path,
    allow_stub: bool = False,
    verify_determinism: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Run TokenGate calibration with explicit inputs and outputs."""
    _ = verify_determinism
    config = load_config(config_path)

    # Find manifest file
    manifest_path = resolve_manifest_path(manifest_pattern)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # Load manifest and extract case list
    trace_entries, mas_run_id = load_manifest_entries(manifest_path)
    if not trace_entries:
        raise ValueError(f"No trace entries found in manifest: {manifest_path}")

    # Compute reproducibility hashes
    trace_dataset_hash = compute_trace_dataset_hash(manifest_path)
    config_hash = compute_config_hash(config)
    exaid_commit = get_exaid_commit(repo_root)
    calibration_run_id = generate_calibration_run_id(
        trace_dataset_hash,
        config_hash,
        exaid_commit,
    )

    log(f"Calibration Run ID: {calibration_run_id}")
    log(f"Trace Dataset Hash: {trace_dataset_hash[:8]}")
    log(f"Config Hash: {config_hash[:8]}")
    log(f"EXAID Commit: {exaid_commit[:8]}")
    log("")

    # Generate policy grid
    log("Generating policy grid...")
    all_policies = generate_policy_grid(config)
    log(f"Total combinations: {len(all_policies)}")

    # Filter valid policies
    log("Filtering valid policies...")
    valid_policies, invalid_policies = filter_valid_policies(all_policies, config)
    log(f"Valid policies: {len(valid_policies)}")
    log(f"Invalid policies: {len(invalid_policies)}")

    # Create output directory
    output_dir = output_root / calibration_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration copy
    write_config_copy(output_dir / "calibration_config.yaml", config)

    # Verify trace hashes once before calibration (matching make_traces.py: hash uncompressed content)
    if config.get("safety", {}).get("verify_trace_hashes", True):
        log("\nVerifying trace file integrity...")
        verify_trace_hashes(trace_entries, traces_dir, log)

    # Run calibration for each valid policy
    log(
        f"\nRunning calibration for {len(valid_policies)} policies across {len(trace_entries)} cases..."
    )

    all_case_metrics: List[CaseMetrics] = []
    policy_metrics_list: List[PolicyMetrics] = []

    for policy_idx, policy in enumerate(valid_policies):
        if (policy_idx + 1) % 10 == 0:
            log(f"  Progress: {policy_idx + 1}/{len(valid_policies)} policies")

        case_metrics_for_policy: List[CaseMetrics] = []

        for trace_entry in trace_entries:
            case_id = trace_entry["case_id"]
            trace_file = traces_dir / trace_entry["file"]

            if not trace_file.exists():
                log(f"WARNING: Trace file not found: {trace_file}")
                continue

            # Load trace and replay
            try:
                engine = TraceReplayEngine(
                    trace_file,
                    strict_stub_guard=not allow_stub,
                    strict_validation=config.get("safety", {}).get("strict_validation", True),
                )
                trace_meta = engine.get_metadata()

                case_metrics = await replay_case_with_policy(
                    trace_file,
                    policy,
                    case_id,
                    trace_meta,
                    engine=engine,  # Pass the already-created engine to avoid duplicate I/O
                    strict_stub_guard=not allow_stub,
                    strict_validation=config.get("safety", {}).get("strict_validation", True),
                )
                case_metrics_for_policy.append(case_metrics)
                all_case_metrics.append(case_metrics)

            except (StubTraceError, TraceValidationError) as exc:
                log(f"ERROR: Failed to replay {case_id}: {exc}")
                if config.get("safety", {}).get("strict_validation", True):
                    raise

        # Aggregate metrics for this policy
        policy_metrics = aggregate_policy_metrics(policy, case_metrics_for_policy)

        # Check constraints
        policy_metrics.constraint_violations = check_constraints(policy_metrics, config)

        policy_metrics_list.append(policy_metrics)

    log(f"\nCalibration complete. Evaluated {len(policy_metrics_list)} policies.")

    # Filter survivors (policies that pass constraints)
    survivor_metrics = [pm for pm in policy_metrics_list if not pm.constraint_violations]
    log(f"Survivor policies: {len(survivor_metrics)}")

    # Diagnostic: Show constraint violation statistics
    violation_counts = defaultdict(int)
    if len(survivor_metrics) == 0:
        log("\n⚠️  WARNING: No policies passed all constraints!")
        log("\nConstraint violation statistics:")
        for pm in policy_metrics_list:
            for violation in pm.constraint_violations:
                # Extract constraint name from violation message
                constraint_name = violation.split("(")[0].strip()
                violation_counts[constraint_name] += 1

        for constraint, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
            pct = (count / len(policy_metrics_list)) * 100
            log(f"  {constraint}: {count}/{len(policy_metrics_list)} policies ({pct:.1f}%)")

        log("\nConsider:")
        log("  1. Relaxing constraint thresholds in config")
        log("  2. Checking if metrics are computed correctly")
        log("  3. Reviewing a sample of failed policies in the CSV output")

    # Write CSV output early (even if no survivors, for analysis)
    log("\nWriting output artifacts...")
    write_calibration_results_csv(output_dir / "calibration_results.csv", policy_metrics_list)

    if len(survivor_metrics) == 0:
        log("\n❌ Cannot select best policy: no survivors.")
        log("CSV output written for analysis. Review constraint violations and adjust thresholds.")
        # Write summary JSON with constraint stats (using dummy selected policy for structure)
        dummy_policy = policy_metrics_list[0] if policy_metrics_list else None
        write_summary_json(
            output_dir / "calibration_summary.json",
            calibration_run_id,
            trace_dataset_hash,
            config_hash,
            exaid_commit,
            mas_run_id,
            len(valid_policies),
            len(invalid_policies),
            invalid_policies[:10],
            policy_metrics_list,
            [],  # No survivors
            dummy_policy,  # Dummy for structure (will be ignored if None)
            "none",  # No selection method
            config,
            None,  # No computed bounds
            None,  # No dropped metrics
            None,  # No selection metadata
        )
        # Write per-case JSONL even if no survivors
        write_per_case_jsonl(output_dir / "calibration_per_case.jsonl", all_case_metrics)
        return

    # Pass B: Compute normalization bounds from survivors
    log("\nComputing normalization bounds from survivor policies...")
    computed_bounds, dropped_metrics = compute_all_normalization_bounds(survivor_metrics)

    if dropped_metrics:
        log(f"⚠️  Dropped metrics (insufficient variance): {dropped_metrics}")
    else:
        log("✅ All metrics have sufficient variance")

    # Compute weighted scores for survivors (using computed bounds)
    for pm in survivor_metrics:
        pm.weighted_score = compute_weighted_score(pm, config, computed_bounds, dropped_metrics)

    # Select best policy using 3-objective Pareto + utopia distance
    selected_policy, selection_method, selection_metadata = select_pareto_utopia(
        survivor_metrics,
        computed_bounds,
        dropped_metrics,
        config,
        logger=log,
    )
    log(f"\nSelected policy: {selected_policy.policy_id} (method: {selection_method})")

    # Compute utopia distances for all survivors (for ranking)
    log("\nComputing utopia distances for all survivors...")
    utopia_distances = compute_utopia_distances_for_all(
        survivor_metrics,
        computed_bounds,
        dropped_metrics,
    )

    # Get top 5 by utopia distance
    top_5_by_utopia = [pm for pm, _ in utopia_distances[:5]] if utopia_distances else []

    # Get top 5 by weighted score
    top_5_by_weighted = sorted(
        survivor_metrics,
        key=lambda pm: pm.weighted_score if pm.weighted_score is not None else -1,
        reverse=True,
    )[:5]

    # Collect policies for spam sensitivity analysis
    policies_for_spam_analysis = set()
    policies_for_spam_analysis.add(selected_policy.policy_id)  # Selected policy
    for pm in top_5_by_utopia:
        policies_for_spam_analysis.add(pm.policy_id)
    for pm in top_5_by_weighted:
        policies_for_spam_analysis.add(pm.policy_id)

    # α Sensitivity Analysis
    log("\nComputing α sensitivity analysis...")
    spam_sensitivity = {}
    alpha_values = config.get("spam", {}).get("alpha_sensitivity", [0.5, 0.6, 0.7, 0.8])

    # Compute spam sensitivity for selected policy, top 5 by utopia, and top 5 by weighted score
    for policy_metrics in survivor_metrics:
        if policy_metrics.policy_id in policies_for_spam_analysis:
            policy = Policy(
                policy_id=policy_metrics.policy_id,
                min_words=policy_metrics.min_words,
                max_words=policy_metrics.max_words,
                silence_timer_ms=policy_metrics.silence_timer_ms,
                max_wait_timeout_ms=policy_metrics.max_wait_timeout_ms,
            )
            case_metrics_for_policy = [
                cm for cm in all_case_metrics if cm.policy_id == policy.policy_id
            ]
            spam_sensitivity[policy.policy_id] = compute_spam_sensitivity(
                case_metrics_for_policy,
                policy,
                alpha_values,
            )

    # Write remaining outputs (CSV already written earlier)
    log("\nWriting remaining output artifacts...")

    write_per_case_jsonl(output_dir / "calibration_per_case.jsonl", all_case_metrics)

    summary_data = {
        "reproducibility": {
            "trace_dataset_hash": trace_dataset_hash,
            "mas_run_id": mas_run_id,
            "exaid_commit": exaid_commit,
            "calibration_config_hash": config_hash,
        },
        "policy_validity": {
            "valid_policies_count": len(valid_policies),
            "invalid_policies_count": len(invalid_policies),
            "invalid_reasons": invalid_policies[:10],
        },
        "constraint_filtering": {
            "total_policies": len(policy_metrics_list),
            "survivor_count": len(survivor_metrics),
            "rejected_count": len(policy_metrics_list) - len(survivor_metrics),
        },
        "normalization_bounds": computed_bounds if computed_bounds else {},
        "dropped_metrics": dropped_metrics if dropped_metrics else [],
    }

    write_summary_json(
        output_dir / "calibration_summary.json",
        calibration_run_id,
        trace_dataset_hash,
        config_hash,
        exaid_commit,
        mas_run_id,
        len(valid_policies),
        len(invalid_policies),
        invalid_policies,
        policy_metrics_list,
        survivor_metrics,
        selected_policy,
        selection_method,
        config,
        computed_bounds,
        dropped_metrics,
        selection_metadata,
    )

    write_chosen_params_yaml(output_dir / "chosen_tokengate_params.yaml", selected_policy)

    # Sort survivors by weighted score for report
    top_policies = sorted(
        survivor_metrics, key=lambda pm: pm.weighted_score or -1, reverse=True
    )

    # Prepare utopia distance rankings for report
    utopia_rankings = {
        pm.policy_id: (idx + 1, dist) for idx, (pm, dist) in enumerate(utopia_distances)
    }

    write_calibration_report_md(
        output_dir / "calibration_report.md",
        calibration_run_id,
        summary_data,
        selected_policy,
        selection_method,
        top_policies,
        spam_sensitivity,
        config,
        top_5_by_utopia,
        utopia_rankings,
    )

    write_spam_sensitivity_json(output_dir / "spam_sensitivity.json", spam_sensitivity)

    log(f"\nCalibration complete. Output directory: {output_dir}")
    log(f"Selected policy: {selected_policy.policy_id}")
    log(f"  min_words: {selected_policy.min_words}")
    log(f"  max_words: {selected_policy.max_words}")
    log(f"  silence_timer_ms: {selected_policy.silence_timer_ms}")
    log(f"  max_wait_timeout_ms: {selected_policy.max_wait_timeout_ms}")


def run_calibration_sync(*args, **kwargs) -> None:
    """Synchronous wrapper for run_calibration."""
    asyncio.run(run_calibration(*args, **kwargs))
