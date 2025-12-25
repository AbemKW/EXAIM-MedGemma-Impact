#!/usr/bin/env python3
"""
EXAID Trace Replay CLI Tool

Replay a trace file and display timeline, classifications, or audit flags.

Usage:
    python cli/replay_trace.py data/traces/case-33651373.trace.jsonl.gz
    python cli/replay_trace.py --stream content_plane data/traces/case-33651373.trace.jsonl.gz
    python cli/replay_trace.py --classifications data/traces/case-33651373.trace.jsonl.gz
    python cli/replay_trace.py --audit data/traces/case-33651373.trace.jsonl.gz
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports (in case run from different directory)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trace_replay_engine import (
    TraceReplayEngine,
    TraceReplayError,
    ReplayEvent,
)


def format_time(ms: int) -> str:
    """Format milliseconds as a readable time string."""
    if ms < 0:
        return f"{ms}ms"
    elif ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def truncate_text(text: str, max_len: int = 50) -> str:
    """Truncate text for display."""
    text = text.replace("\n", " ").replace("\r", "")
    if len(text) > max_len:
        return text[:max_len-3] + "..."
    return text


def print_metadata(engine: TraceReplayEngine) -> None:
    """Print trace metadata."""
    meta = engine.get_metadata()
    labels = engine.get_derived_agent_labels()
    
    print("=" * 60)
    print("TRACE METADATA")
    print("=" * 60)
    print(f"  case_id:              {meta.case_id}")
    print(f"  schema_version:       {meta.schema_version}")
    print(f"  mas_run_id:           {meta.mas_run_id}")
    print(f"  model:                {meta.model}")
    print(f"  t0_emitted_ms:        {meta.t0_emitted_ms}")
    print(f"  stub_mode:            {meta.stub_mode}")
    print(f"  total_turns:          {meta.total_turns}")
    print(f"  total_deltas:         {meta.total_deltas}")
    print()
    print(f"  derived_agent_labels: {{{', '.join(sorted(labels))}}}")
    print()


def print_timeline(engine: TraceReplayEngine, stream: str, limit: int = None, verbose: bool = False) -> None:
    """Print replay timeline."""
    classifications = engine.get_turn_classifications()
    
    print("=" * 60)
    print(f"TIMELINE ({stream.upper()} STREAM)")
    print("=" * 60)
    
    if stream == "full":
        events = engine.replay_full()
    else:
        events = engine.replay_content_plane()
    
    count = 0
    for event in events:
        if limit and count >= limit:
            print(f"  ... (limited to {limit} events)")
            break
        
        # Get classification info for turn
        cls = classifications.get(event.turn_id)
        cls_type = cls.turn_type if cls else "unknown"
        cls_reason = cls.classification_reason if cls else ""
        
        # Format time
        time_str = f"t={format_time(event.virtual_time_ms):>8}"
        
        if event.event_type == "turn_start":
            cls_info = f"[{cls_type}]"
            if cls_type == "control_plane":
                cls_info += f" ({cls_reason})"
            print(f"  {time_str}  turn_start  turn={event.turn_id:<3} agent={event.agent_id:<12} {cls_info}")
            
        elif event.event_type == "turn_end":
            hash_preview = event.content_hash[:20] + "..." if event.content_hash else ""
            print(f"  {time_str}  turn_end    turn={event.turn_id:<3} {hash_preview}")
            
        elif event.event_type == "delta":
            if verbose:
                text = truncate_text(event.delta_text, 40)
                print(f"  {time_str}  delta       turn={event.turn_id:<3} \"{text}\"")
        
        count += 1
    
    print()
    print(f"  Total events: {count}")
    print()


def print_classifications(engine: TraceReplayEngine) -> None:
    """Print turn classifications summary."""
    classifications = engine.get_turn_classifications()
    
    print("=" * 60)
    print("TURN CLASSIFICATIONS")
    print("=" * 60)
    
    content_count = 0
    control_count = 0
    
    for turn_id in sorted(classifications.keys()):
        cls = classifications[turn_id]
        text_preview = truncate_text(cls.turn_text, 40)
        
        if cls.turn_type == "content_plane":
            content_count += 1
        else:
            control_count += 1
        
        print(f"  turn={turn_id:<3}  {cls.turn_type:<14}  {cls.classification_reason:<30}  \"{text_preview}\"")
    
    print()
    print("-" * 60)
    print(f"  Summary: {content_count} content_plane, {control_count} control_plane")
    print()


def print_audit_flags(engine: TraceReplayEngine) -> None:
    """Print audit flags."""
    flags = engine.get_audit_flags()
    classifications = engine.get_turn_classifications()
    
    print("=" * 60)
    print("AUDIT FLAGS")
    print("=" * 60)
    
    if not flags:
        print("  No audit flags. All classifications are unambiguous.")
        print()
        return
    
    for flag in flags:
        cls = classifications.get(flag.turn_id)
        print(f"  turn={flag.turn_id}:")
        print(f"    flag_type:   {flag.flag_type}")
        print(f"    details:     {flag.details[:100]}")
        if cls:
            print(f"    classified:  {cls.turn_type} (conservative - kept as content)")
            print(f"    turn_text:   \"{truncate_text(cls.turn_text, 60)}\"")
        print()
    
    print("-" * 60)
    print(f"  Total flags: {len(flags)}")
    print("  (Flagged turns are classified as content_plane for safety)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Replay a trace file and display timeline, classifications, or audit flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show metadata and timeline
    python cli/replay_trace.py data/traces/case-33651373.trace.jsonl.gz

    # Show content_plane stream only
    python cli/replay_trace.py --stream content_plane data/traces/case-33651373.trace.jsonl.gz

    # Show turn classifications
    python cli/replay_trace.py --classifications data/traces/case-33651373.trace.jsonl.gz

    # Show audit flags
    python cli/replay_trace.py --audit data/traces/case-33651373.trace.jsonl.gz

    # Verbose output with delta text
    python cli/replay_trace.py --verbose data/traces/case-33651373.trace.jsonl.gz
        """
    )
    
    parser.add_argument(
        "trace_file",
        type=Path,
        help="Path to trace JSONL(.gz) file"
    )
    
    parser.add_argument(
        "--stream",
        choices=["full", "content_plane"],
        default="full",
        help="Which stream to replay (default: full)"
    )
    
    parser.add_argument(
        "--classifications",
        action="store_true",
        help="Show turn classifications only"
    )
    
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Show audit flags only"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show delta text in timeline"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit timeline output to N events"
    )
    
    parser.add_argument(
        "--shift-to-zero",
        action="store_true",
        help="Shift timeline so minimum virtual_time = 0"
    )
    
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="Allow replaying stub traces (not recommended for evaluation)"
    )
    
    args = parser.parse_args()
    
    if not args.trace_file.exists():
        print(f"ERROR: Trace file not found: {args.trace_file}", file=sys.stderr)
        return 1
    
    try:
        engine = TraceReplayEngine(
            args.trace_file,
            strict_stub_guard=not args.allow_stub,
            shift_to_zero=args.shift_to_zero,
        )
        
        # Print metadata always (unless specific mode)
        if not args.classifications and not args.audit:
            print_metadata(engine)
        
        # Mode-specific output
        if args.classifications:
            print_classifications(engine)
        elif args.audit:
            print_audit_flags(engine)
        else:
            print_timeline(engine, args.stream, limit=args.limit, verbose=args.verbose)
            
            # Also show classification summary
            classifications = engine.get_turn_classifications()
            content = sum(1 for c in classifications.values() if c.turn_type == "content_plane")
            control = sum(1 for c in classifications.values() if c.turn_type == "control_plane")
            print(f"  Classifications: {content} content_plane, {control} control_plane")
            
            flags = engine.get_audit_flags()
            if flags:
                print(f"  Audit flags: {len(flags)} (use --audit to view)")
            print()
        
        return 0
        
    except TraceReplayError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

