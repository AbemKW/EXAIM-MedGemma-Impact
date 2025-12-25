#!/usr/bin/env python3
"""
Generate stub trace fixtures for testing the Trace Replay Engine.

Run this script to regenerate fixtures:
    python evals/tests/fixtures/generate_stub_traces.py
"""

import gzip
import json
from pathlib import Path


def write_trace(path: Path, records: list[dict]) -> None:
    """Write records to a gzipped JSONL file."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"  Written: {path}")


def generate_stub_trace():
    """
    Generate main stub trace with various turn types.
    
    Turns:
        1. Content turn (doctor discussion) - boundary at t_rel=-5ms
        2. Control turn (just "Doctor0") - speaker selection
        3. Content turn (supervisor response)
        4. Content turn containing "Ask Doctor0" (should NOT be filtered)
        5. Content turn (doctor response)
        6. Control turn (just "TERMINATE")
        7. Content turn with suspicious text "Doctor5" (not in derived labels)
    """
    t0 = 1700000000000  # Anchor timestamp
    
    records = [
        # trace_meta
        {
            "record_type": "trace_meta",
            "schema_version": "2.0.0",
            "case_id": "case-stub-test",
            "mas_run_id": "mas_test_stub",
            "mac_commit": "1234567890abcdef",
            "model": "gpt-4o-mini",
            "created_at": "2025-12-24T12:00:00Z",
            "t0_emitted_ms": t0,
            "t0_definition": "t_emitted_ms of first stream_delta record",
            "stub_mode": False,  # Not a stub for testing purposes
            "total_turns": 7,
            "total_deltas": 12,
        },
        
        # Turn 1: Content turn with negative boundary t_rel
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 1, "agent_id": "Supervisor", "boundary": "start", "seq": 0, "t_ms": t0 - 5, "t_rel_ms": -5},
        {"record_type": "stream_delta", "case_id": "case-stub-test", "seq": 1, "turn_id": 1, "agent_id": "Supervisor", "delta_text": "Based on the patient presentation, ", "t_emitted_ms": t0, "t_rel_ms": 0},
        {"record_type": "stream_delta", "case_id": "case-stub-test", "seq": 2, "turn_id": 1, "agent_id": "Supervisor", "delta_text": "we should consider multiple differential diagnoses.", "t_emitted_ms": t0 + 50, "t_rel_ms": 50},
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 1, "agent_id": "Supervisor", "boundary": "end", "seq": 3, "t_ms": t0 + 150, "t_rel_ms": 150, "content_hash": "sha256:" + "a" * 64},
        
        # Turn 2: Control plane - speaker selection (just "Doctor0")
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 2, "agent_id": "Supervisor", "boundary": "start", "seq": 4, "t_ms": t0 + 200, "t_rel_ms": 200},
        {"record_type": "stream_delta", "case_id": "case-stub-test", "seq": 5, "turn_id": 2, "agent_id": "Supervisor", "delta_text": "Doctor0", "t_emitted_ms": t0 + 210, "t_rel_ms": 210},
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 2, "agent_id": "Supervisor", "boundary": "end", "seq": 6, "t_ms": t0 + 220, "t_rel_ms": 220, "content_hash": "sha256:" + "b" * 64},
        
        # Turn 3: Content turn (doctor response)
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 3, "agent_id": "Doctor0", "boundary": "start", "seq": 7, "t_ms": t0 + 300, "t_rel_ms": 300},
        {"record_type": "stream_delta", "case_id": "case-stub-test", "seq": 8, "turn_id": 3, "agent_id": "Doctor0", "delta_text": "I agree with the assessment. ", "t_emitted_ms": t0 + 350, "t_rel_ms": 350},
        {"record_type": "stream_delta", "case_id": "case-stub-test", "seq": 9, "turn_id": 3, "agent_id": "Doctor0", "delta_text": "The symptoms are consistent with the diagnosis.", "t_emitted_ms": t0 + 400, "t_rel_ms": 400},
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 3, "agent_id": "Doctor0", "boundary": "end", "seq": 10, "t_ms": t0 + 450, "t_rel_ms": 450, "content_hash": "sha256:" + "c" * 64},
        
        # Turn 4: Content turn containing "Ask Doctor0" (should NOT be filtered)
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 4, "agent_id": "Doctor1", "boundary": "start", "seq": 11, "t_ms": t0 + 500, "t_rel_ms": 500},
        {"record_type": "stream_delta", "case_id": "case-stub-test", "seq": 12, "turn_id": 4, "agent_id": "Doctor1", "delta_text": "Ask Doctor0 to elaborate on the treatment plan.", "t_emitted_ms": t0 + 550, "t_rel_ms": 550},
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 4, "agent_id": "Doctor1", "boundary": "end", "seq": 13, "t_ms": t0 + 600, "t_rel_ms": 600, "content_hash": "sha256:" + "d" * 64},
        
        # Turn 5: Content turn (another doctor)
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 5, "agent_id": "Doctor2", "boundary": "start", "seq": 14, "t_ms": t0 + 700, "t_rel_ms": 700},
        {"record_type": "stream_delta", "case_id": "case-stub-test", "seq": 15, "turn_id": 5, "agent_id": "Doctor2", "delta_text": "The patient history supports this approach.", "t_emitted_ms": t0 + 750, "t_rel_ms": 750},
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 5, "agent_id": "Doctor2", "boundary": "end", "seq": 16, "t_ms": t0 + 800, "t_rel_ms": 800, "content_hash": "sha256:" + "e" * 64},
        
        # Turn 6: Control plane - TERMINATE sentinel
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 6, "agent_id": "Supervisor", "boundary": "start", "seq": 17, "t_ms": t0 + 900, "t_rel_ms": 900},
        {"record_type": "stream_delta", "case_id": "case-stub-test", "seq": 18, "turn_id": 6, "agent_id": "Supervisor", "delta_text": "TERMINATE", "t_emitted_ms": t0 + 910, "t_rel_ms": 910},
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 6, "agent_id": "Supervisor", "boundary": "end", "seq": 19, "t_ms": t0 + 920, "t_rel_ms": 920, "content_hash": "sha256:" + "f" * 64},
        
        # Turn 7: Content turn with suspicious text "Doctor5" (not in derived labels)
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 7, "agent_id": "Supervisor", "boundary": "start", "seq": 20, "t_ms": t0 + 1000, "t_rel_ms": 1000},
        {"record_type": "stream_delta", "case_id": "case-stub-test", "seq": 21, "turn_id": 7, "agent_id": "Supervisor", "delta_text": "Doctor5", "t_emitted_ms": t0 + 1010, "t_rel_ms": 1010},
        {"record_type": "turn_boundary", "case_id": "case-stub-test", "turn_id": 7, "agent_id": "Supervisor", "boundary": "end", "seq": 22, "t_ms": t0 + 1020, "t_rel_ms": 1020, "content_hash": "sha256:" + "0" * 64},
    ]
    
    return records


def generate_stub_trace_delta_only_agent():
    """
    Generate trace where a delta has agent_id not present on any boundary.
    
    This tests that labels are derived from boundaries only.
    """
    t0 = 1700000000000
    
    records = [
        # trace_meta
        {
            "record_type": "trace_meta",
            "schema_version": "2.0.0",
            "case_id": "case-delta-only-agent",
            "mas_run_id": "mas_test_delta_agent",
            "mac_commit": "1234567890abcdef",
            "model": "gpt-4o-mini",
            "created_at": "2025-12-24T12:00:00Z",
            "t0_emitted_ms": t0,
            "stub_mode": False,
            "total_turns": 1,
            "total_deltas": 2,
        },
        
        # Turn 1: Boundary has "Doctor0", but delta has "RogueAgent"
        {"record_type": "turn_boundary", "case_id": "case-delta-only-agent", "turn_id": 1, "agent_id": "Doctor0", "boundary": "start", "seq": 0, "t_ms": t0, "t_rel_ms": 0},
        # This delta has a different agent_id that's not on any boundary
        {"record_type": "stream_delta", "case_id": "case-delta-only-agent", "seq": 1, "turn_id": 1, "agent_id": "RogueAgent", "delta_text": "Some content here.", "t_emitted_ms": t0 + 50, "t_rel_ms": 50},
        {"record_type": "stream_delta", "case_id": "case-delta-only-agent", "seq": 2, "turn_id": 1, "agent_id": "Doctor0", "delta_text": " More content.", "t_emitted_ms": t0 + 100, "t_rel_ms": 100},
        {"record_type": "turn_boundary", "case_id": "case-delta-only-agent", "turn_id": 1, "agent_id": "Doctor0", "boundary": "end", "seq": 3, "t_ms": t0 + 150, "t_rel_ms": 150, "content_hash": "sha256:" + "a" * 64},
    ]
    
    return records


def generate_stub_trace_with_unknown():
    """
    Generate trace with agent_id="unknown" on a boundary.
    
    Tests that "unknown" is excluded from derived labels.
    """
    t0 = 1700000000000
    
    records = [
        # trace_meta
        {
            "record_type": "trace_meta",
            "schema_version": "2.0.0",
            "case_id": "case-with-unknown",
            "mas_run_id": "mas_test_unknown",
            "mac_commit": "1234567890abcdef",
            "model": "gpt-4o-mini",
            "created_at": "2025-12-24T12:00:00Z",
            "t0_emitted_ms": t0,
            "stub_mode": False,
            "total_turns": 2,
            "total_deltas": 2,
        },
        
        # Turn 1: Normal turn
        {"record_type": "turn_boundary", "case_id": "case-with-unknown", "turn_id": 1, "agent_id": "Doctor0", "boundary": "start", "seq": 0, "t_ms": t0, "t_rel_ms": 0},
        {"record_type": "stream_delta", "case_id": "case-with-unknown", "seq": 1, "turn_id": 1, "agent_id": "Doctor0", "delta_text": "Normal content.", "t_emitted_ms": t0 + 50, "t_rel_ms": 50},
        {"record_type": "turn_boundary", "case_id": "case-with-unknown", "turn_id": 1, "agent_id": "Doctor0", "boundary": "end", "seq": 2, "t_ms": t0 + 100, "t_rel_ms": 100, "content_hash": "sha256:" + "a" * 64},
        
        # Turn 2: Turn with agent_id="unknown"
        {"record_type": "turn_boundary", "case_id": "case-with-unknown", "turn_id": 2, "agent_id": "unknown", "boundary": "start", "seq": 3, "t_ms": t0 + 200, "t_rel_ms": 200},
        {"record_type": "stream_delta", "case_id": "case-with-unknown", "seq": 4, "turn_id": 2, "agent_id": "unknown", "delta_text": "Unknown agent content.", "t_emitted_ms": t0 + 250, "t_rel_ms": 250},
        {"record_type": "turn_boundary", "case_id": "case-with-unknown", "turn_id": 2, "agent_id": "unknown", "boundary": "end", "seq": 5, "t_ms": t0 + 300, "t_rel_ms": 300, "content_hash": "sha256:" + "b" * 64},
    ]
    
    return records


def generate_stub_mode_trace():
    """
    Generate a trace with stub_mode=true.
    
    Tests the stub guard functionality.
    """
    t0 = 1700000000000
    
    records = [
        # trace_meta with stub_mode=true
        {
            "record_type": "trace_meta",
            "schema_version": "2.0.0",
            "case_id": "case-stub-mode",
            "mas_run_id": "mas_test_stub_mode",
            "mac_commit": "1234567890abcdef",
            "model": "gpt-4o-mini",
            "created_at": "2025-12-24T12:00:00Z",
            "t0_emitted_ms": t0,
            "stub_mode": True,  # This is a stub trace
            "total_turns": 1,
            "total_deltas": 1,
        },
        
        # Simple turn
        {"record_type": "turn_boundary", "case_id": "case-stub-mode", "turn_id": 1, "agent_id": "Doctor0", "boundary": "start", "seq": 0, "t_ms": t0, "t_rel_ms": 0},
        {"record_type": "stream_delta", "case_id": "case-stub-mode", "seq": 1, "turn_id": 1, "agent_id": "Doctor0", "delta_text": "Stub content.", "t_emitted_ms": t0 + 50, "t_rel_ms": 50},
        {"record_type": "turn_boundary", "case_id": "case-stub-mode", "turn_id": 1, "agent_id": "Doctor0", "boundary": "end", "seq": 2, "t_ms": t0 + 100, "t_rel_ms": 100, "content_hash": "sha256:" + "a" * 64},
    ]
    
    return records


def generate_boundary_time_violation_trace():
    """
    Generate a trace where turn_start.t_ms > first_delta.t_emitted_ms (violation).
    
    Tests boundary-time containment validation.
    """
    t0 = 1700000000000
    
    records = [
        # trace_meta
        {
            "record_type": "trace_meta",
            "schema_version": "2.0.0",
            "case_id": "case-boundary-time-violation",
            "mas_run_id": "mas_test_boundary_time",
            "mac_commit": "1234567890abcdef",
            "model": "gpt-4o-mini",
            "created_at": "2025-12-24T12:00:00Z",
            "t0_emitted_ms": t0,
            "stub_mode": False,
            "total_turns": 1,
            "total_deltas": 2,
        },
        
        # Turn 1: start boundary occurs AFTER first delta (violation)
        # Boundary at t0+100, but first delta at t0+50
        {"record_type": "turn_boundary", "case_id": "case-boundary-time-violation", "turn_id": 1, "agent_id": "Doctor0", "boundary": "start", "seq": 0, "t_ms": t0 + 100, "t_rel_ms": 100},
        {"record_type": "stream_delta", "case_id": "case-boundary-time-violation", "seq": 1, "turn_id": 1, "agent_id": "Doctor0", "delta_text": "First delta.", "t_emitted_ms": t0 + 50, "t_rel_ms": 50},
        {"record_type": "stream_delta", "case_id": "case-boundary-time-violation", "seq": 2, "turn_id": 1, "agent_id": "Doctor0", "delta_text": "Second delta.", "t_emitted_ms": t0 + 150, "t_rel_ms": 150},
        {"record_type": "turn_boundary", "case_id": "case-boundary-time-violation", "turn_id": 1, "agent_id": "Doctor0", "boundary": "end", "seq": 3, "t_ms": t0 + 200, "t_rel_ms": 200, "content_hash": "sha256:" + "a" * 64},
    ]
    
    return records


def generate_t_rel_ms_violation_trace():
    """
    Generate a trace where t_rel_ms is incorrect (violation).
    
    Tests t_rel_ms consistency validation.
    """
    t0 = 1700000000000
    
    records = [
        # trace_meta
        {
            "record_type": "trace_meta",
            "schema_version": "2.0.0",
            "case_id": "case-trel-violation",
            "mas_run_id": "mas_test_trel",
            "mac_commit": "1234567890abcdef",
            "model": "gpt-4o-mini",
            "created_at": "2025-12-24T12:00:00Z",
            "t0_emitted_ms": t0,
            "stub_mode": False,
            "total_turns": 1,
            "total_deltas": 1,
        },
        
        # Turn 1: delta has wrong t_rel_ms
        # t_emitted_ms = t0 + 50, so t_rel_ms should be 50, but it's 100
        {"record_type": "turn_boundary", "case_id": "case-trel-violation", "turn_id": 1, "agent_id": "Doctor0", "boundary": "start", "seq": 0, "t_ms": t0, "t_rel_ms": 0},
        {"record_type": "stream_delta", "case_id": "case-trel-violation", "seq": 1, "turn_id": 1, "agent_id": "Doctor0", "delta_text": "Content.", "t_emitted_ms": t0 + 50, "t_rel_ms": 100},  # Wrong! Should be 50
        {"record_type": "turn_boundary", "case_id": "case-trel-violation", "turn_id": 1, "agent_id": "Doctor0", "boundary": "end", "seq": 2, "t_ms": t0 + 100, "t_rel_ms": 100, "content_hash": "sha256:" + "a" * 64},
    ]
    
    return records


def generate_delta_only_turn_trace():
    """
    Generate a trace with a turn that has deltas but no boundaries.
    
    Tests schema violation handling.
    """
    t0 = 1700000000000
    
    records = [
        # trace_meta
        {
            "record_type": "trace_meta",
            "schema_version": "2.0.0",
            "case_id": "case-delta-only",
            "mas_run_id": "mas_test_delta_only",
            "mac_commit": "1234567890abcdef",
            "model": "gpt-4o-mini",
            "created_at": "2025-12-24T12:00:00Z",
            "t0_emitted_ms": t0,
            "stub_mode": False,
            "total_turns": 2,
            "total_deltas": 2,
        },
        
        # Turn 1: Normal turn with boundaries
        {"record_type": "turn_boundary", "case_id": "case-delta-only", "turn_id": 1, "agent_id": "Doctor0", "boundary": "start", "seq": 0, "t_ms": t0, "t_rel_ms": 0},
        {"record_type": "stream_delta", "case_id": "case-delta-only", "seq": 1, "turn_id": 1, "agent_id": "Doctor0", "delta_text": "Normal content.", "t_emitted_ms": t0 + 50, "t_rel_ms": 50},
        {"record_type": "turn_boundary", "case_id": "case-delta-only", "turn_id": 1, "agent_id": "Doctor0", "boundary": "end", "seq": 2, "t_ms": t0 + 100, "t_rel_ms": 100, "content_hash": "sha256:" + "a" * 64},
        
        # Turn 2: Deltas but NO boundaries (schema violation)
        {"record_type": "stream_delta", "case_id": "case-delta-only", "seq": 3, "turn_id": 2, "agent_id": "Doctor1", "delta_text": "Delta without boundary.", "t_emitted_ms": t0 + 200, "t_rel_ms": 200},
    ]
    
    return records


def main():
    fixtures_dir = Path(__file__).parent
    
    print("Generating stub trace fixtures...")
    print()
    
    # Main stub trace
    write_trace(
        fixtures_dir / "stub_trace.jsonl.gz",
        generate_stub_trace()
    )
    
    # Delta-only agent trace
    write_trace(
        fixtures_dir / "stub_trace_delta_only_agent.jsonl.gz",
        generate_stub_trace_delta_only_agent()
    )
    
    # Unknown agent trace
    write_trace(
        fixtures_dir / "stub_trace_with_unknown.jsonl.gz",
        generate_stub_trace_with_unknown()
    )
    
    # Stub mode trace
    write_trace(
        fixtures_dir / "stub_trace_stub_mode.jsonl.gz",
        generate_stub_mode_trace()
    )
    
    # Boundary time violation trace
    write_trace(
        fixtures_dir / "stub_trace_boundary_time_violation.jsonl.gz",
        generate_boundary_time_violation_trace()
    )
    
    # t_rel_ms violation trace
    write_trace(
        fixtures_dir / "stub_trace_trel_violation.jsonl.gz",
        generate_t_rel_ms_violation_trace()
    )
    
    # Delta-only turn trace
    write_trace(
        fixtures_dir / "stub_trace_delta_only_turn.jsonl.gz",
        generate_delta_only_turn_trace()
    )
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()

