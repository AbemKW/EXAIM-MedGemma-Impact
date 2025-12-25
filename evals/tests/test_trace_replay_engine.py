#!/usr/bin/env python3
"""
Unit tests for the Trace Replay Engine.

Tests cover:
    - Agent label derivation from boundaries only
    - Conservative classification (exact match + TERMINATE)
    - Audit flags for suspicious turns
    - Timing gap preservation
    - shift_to_zero behavior
    - Stub mode guard
"""

import pytest
import warnings
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trace_replay_engine import (
    TraceReplayEngine,
    TraceValidationError,
    StubTraceError,
    derive_agent_labels,
    is_suspicious_label_like,
    classify_turn,
    AuditFlag,
)


# =============================================================================
# Fixtures
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"

STUB_TRACE = FIXTURES_DIR / "stub_trace.jsonl.gz"
STUB_TRACE_DELTA_ONLY_AGENT = FIXTURES_DIR / "stub_trace_delta_only_agent.jsonl.gz"
STUB_TRACE_WITH_UNKNOWN = FIXTURES_DIR / "stub_trace_with_unknown.jsonl.gz"
STUB_TRACE_STUB_MODE = FIXTURES_DIR / "stub_trace_stub_mode.jsonl.gz"
STUB_TRACE_BOUNDARY_TIME_VIOLATION = FIXTURES_DIR / "stub_trace_boundary_time_violation.jsonl.gz"
STUB_TRACE_TREL_VIOLATION = FIXTURES_DIR / "stub_trace_trel_violation.jsonl.gz"
STUB_TRACE_DELTA_ONLY_TURN = FIXTURES_DIR / "stub_trace_delta_only_turn.jsonl.gz"


@pytest.fixture
def stub_trace_path():
    """Path to main stub trace."""
    return STUB_TRACE


@pytest.fixture
def engine(stub_trace_path):
    """Pre-initialized engine with main stub trace."""
    return TraceReplayEngine(stub_trace_path)


# =============================================================================
# Test: Agent Label Derivation (Boundaries Only)
# =============================================================================

class TestAgentLabelDerivation:
    """Tests for derive_agent_labels() - Pass 1."""
    
    def test_derive_labels_from_boundaries(self, stub_trace_path):
        """Labels derived from turn_boundary agent_id values only."""
        engine = TraceReplayEngine(stub_trace_path)
        labels = engine.get_derived_agent_labels()
        
        # Should have all agents from boundaries (normalized to lowercase)
        assert "supervisor" in labels
        assert "doctor0" in labels
        assert "doctor1" in labels
        assert "doctor2" in labels
    
    def test_derive_labels_ignores_delta_agent_ids(self):
        """Labels NOT derived from stream_delta (only boundaries)."""
        engine = TraceReplayEngine(STUB_TRACE_DELTA_ONLY_AGENT)
        labels = engine.get_derived_agent_labels()
        
        # RogueAgent appears only on delta, not boundary
        assert "rogueagent" not in labels
        assert "rogue_agent" not in labels
        
        # Doctor0 is on boundary
        assert "doctor0" in labels
    
    def test_derive_labels_excludes_unknown(self):
        """unknown/null agent_ids excluded from label set."""
        engine = TraceReplayEngine(STUB_TRACE_WITH_UNKNOWN)
        labels = engine.get_derived_agent_labels()
        
        # "unknown" should be excluded
        assert "unknown" not in labels
        
        # Doctor0 should be included
        assert "doctor0" in labels
    
    def test_derive_labels_normalized_lowercase(self, stub_trace_path):
        """Labels are normalized to lowercase."""
        labels = derive_agent_labels(stub_trace_path)
        
        # All labels should be lowercase
        for label in labels:
            assert label == label.lower()
    
    def test_derive_labels_frozen(self, stub_trace_path):
        """Returned label set is frozen (immutable)."""
        engine = TraceReplayEngine(stub_trace_path)
        labels = engine.get_derived_agent_labels()
        
        assert isinstance(labels, frozenset)


# =============================================================================
# Test: Classification (Exact Match Only)
# =============================================================================

class TestClassification:
    """Tests for conservative classification rules."""
    
    def test_exact_label_match_control_plane(self, engine):
        """Turn with exact agent name is control_plane."""
        classifications = engine.get_turn_classifications()
        
        # Turn 2 text is "Doctor0" (exact match with derived label)
        assert classifications[2].turn_type == "control_plane"
        assert "exact_label_match" in classifications[2].classification_reason
        assert "doctor0" in classifications[2].classification_reason
    
    def test_terminate_sentinel(self, engine):
        """TERMINATE is control_plane."""
        classifications = engine.get_turn_classifications()
        
        # Turn 6 is "TERMINATE"
        assert classifications[6].turn_type == "control_plane"
        assert classifications[6].classification_reason == "terminate_sentinel"
    
    def test_partial_match_stays_content(self, engine):
        """Partial match like 'Ask Doctor0' is NOT filtered."""
        classifications = engine.get_turn_classifications()
        
        # Turn 4 text is "Ask Doctor0 to elaborate on the treatment plan."
        assert classifications[4].turn_type == "content_plane"
        assert classifications[4].classification_reason == "default_content"
    
    def test_content_turns_classified_correctly(self, engine):
        """Regular content turns are content_plane."""
        classifications = engine.get_turn_classifications()
        
        # Turn 1, 3, 4, 5 are content turns
        assert classifications[1].turn_type == "content_plane"
        assert classifications[3].turn_type == "content_plane"
        assert classifications[4].turn_type == "content_plane"
        assert classifications[5].turn_type == "content_plane"
    
    def test_classification_has_reason(self, engine):
        """Every classification has a reason string."""
        classifications = engine.get_turn_classifications()
        
        for turn_id, cls in classifications.items():
            assert cls.classification_reason is not None
            assert len(cls.classification_reason) > 0
    
    def test_case_insensitive_label_match(self):
        """Label matching is case-insensitive."""
        derived_labels = frozenset({"doctor0", "supervisor"})
        audit_flags = []
        
        # "doctor0" (lowercase) matches
        cls = classify_turn(1, "doctor0", 1, "test", 0, 1, derived_labels, audit_flags)
        assert cls.turn_type == "control_plane"
        
        # "Doctor0" (mixed case) also matches
        cls = classify_turn(2, "Doctor0", 1, "test", 0, 1, derived_labels, audit_flags)
        assert cls.turn_type == "control_plane"
        
        # "DOCTOR0" (uppercase) also matches
        cls = classify_turn(3, "DOCTOR0", 1, "test", 0, 1, derived_labels, audit_flags)
        assert cls.turn_type == "control_plane"
    
    def test_terminate_case_insensitive(self):
        """TERMINATE matching is case-insensitive."""
        derived_labels = frozenset({"doctor0"})
        audit_flags = []
        
        # Various cases of TERMINATE
        for text in ["TERMINATE", "terminate", "Terminate", "TERMinate"]:
            cls = classify_turn(1, text, 1, "test", 0, 1, derived_labels, audit_flags)
            assert cls.turn_type == "control_plane", f"Failed for: {text}"
            assert cls.classification_reason == "terminate_sentinel"
    
    def test_empty_turn_classified_control_plane(self):
        """Empty or whitespace-only turns are control_plane."""
        derived_labels = frozenset({"doctor0"})
        audit_flags = []
        
        # Various empty/whitespace cases
        for text in ["", "   ", "\n", "\t", "  \n\t  "]:
            cls = classify_turn(1, text, 0, "test", 0, 1, derived_labels, audit_flags)
            assert cls.turn_type == "control_plane", f"Failed for: {repr(text)}"
            assert cls.classification_reason == "empty_turn"


# =============================================================================
# Test: Audit Flags
# =============================================================================

class TestAuditFlags:
    """Tests for suspicious turn flagging."""
    
    def test_suspicious_label_like_flagged(self, engine):
        """Unmatched label-like turn is flagged but not filtered."""
        classifications = engine.get_turn_classifications()
        flags = engine.get_audit_flags()
        
        # Turn 7 has text "Doctor5" which is not in derived labels
        # It should be content_plane (conservative) but flagged
        assert classifications[7].turn_type == "content_plane"
        
        # Should have audit flag for turn 7
        turn_7_flags = [f for f in flags if f.turn_id == 7]
        assert len(turn_7_flags) == 1
        assert turn_7_flags[0].flag_type == "suspicious_label_like_unmatched"
        assert "Doctor5" in turn_7_flags[0].details
    
    def test_matched_label_not_suspicious(self):
        """Matched labels are not flagged as suspicious."""
        derived_labels = frozenset({"doctor0"})
        
        # "Doctor0" matches, so not suspicious
        assert not is_suspicious_label_like("Doctor0", derived_labels)
    
    def test_multi_word_not_suspicious(self):
        """Multi-word text is not flagged as suspicious."""
        derived_labels = frozenset({"doctor0"})
        
        # Multiple words -> not suspicious
        assert not is_suspicious_label_like("Ask Doctor0", derived_labels)
        assert not is_suspicious_label_like("The patient needs care", derived_labels)
    
    def test_single_word_identifier_suspicious(self):
        """Single identifier-like word not in labels is suspicious."""
        derived_labels = frozenset({"doctor0"})
        
        # "Doctor5" looks like an agent but isn't in labels
        assert is_suspicious_label_like("Doctor5", derived_labels)
        assert is_suspicious_label_like("Agent99", derived_labels)
        assert is_suspicious_label_like("Supervisor2", derived_labels)
    
    def test_numeric_only_not_suspicious(self):
        """Pure numbers are not suspicious."""
        derived_labels = frozenset({"doctor0"})
        
        # Just numbers
        assert not is_suspicious_label_like("12345", derived_labels)
    
    def test_audit_flags_list_not_mutated(self, engine):
        """get_audit_flags returns a copy, not the internal list."""
        flags1 = engine.get_audit_flags()
        flags2 = engine.get_audit_flags()
        
        # Should be different list objects
        assert flags1 is not flags2


# =============================================================================
# Test: Timing
# =============================================================================

class TestTiming:
    """Tests for virtual time and timing gap preservation."""
    
    def test_content_plane_timing_gap_preserved(self, engine):
        """Timing gaps from control_plane turns preserved."""
        events = list(engine.replay_content_plane())
        
        # Find turn 3 start (should be at t=300, after control turn 2 at t=200-220)
        turn_3_starts = [e for e in events if e.turn_id == 3 and e.event_type == "turn_start"]
        assert len(turn_3_starts) == 1
        
        # The timing should be 300, NOT shifted to fill the gap
        assert turn_3_starts[0].virtual_time_ms == 300
    
    def test_shift_to_zero_considers_negative_boundaries(self, stub_trace_path):
        """shift_to_zero uses min across ALL events including negative boundaries."""
        engine = TraceReplayEngine(stub_trace_path, shift_to_zero=True)
        events = list(engine.replay_full())
        
        # Stub has boundary at t_rel_ms=-5
        # After shift, minimum should be 0
        min_time = min(e.virtual_time_ms for e in events)
        assert min_time == 0
    
    def test_shift_to_zero_preserves_relative_timing(self, stub_trace_path):
        """shift_to_zero preserves relative timing between events."""
        engine_raw = TraceReplayEngine(stub_trace_path, shift_to_zero=False)
        engine_shifted = TraceReplayEngine(stub_trace_path, shift_to_zero=True)
        
        events_raw = list(engine_raw.replay_full())
        events_shifted = list(engine_shifted.replay_full())
        
        # Same number of events
        assert len(events_raw) == len(events_shifted)
        
        # Relative differences should be preserved
        for i in range(1, len(events_raw)):
            raw_diff = events_raw[i].virtual_time_ms - events_raw[i-1].virtual_time_ms
            shifted_diff = events_shifted[i].virtual_time_ms - events_shifted[i-1].virtual_time_ms
            assert raw_diff == shifted_diff
    
    def test_negative_t_rel_ms_handled(self, engine):
        """Negative t_rel_ms values are handled correctly."""
        events = list(engine.replay_full())
        
        # First event (turn 1 start) has t_rel_ms=-5
        assert events[0].virtual_time_ms == -5
        assert events[0].event_type == "turn_start"
        assert events[0].turn_id == 1
    
    def test_content_plane_excludes_control_turns(self, engine):
        """content_plane stream excludes control_plane turn events."""
        full_events = list(engine.replay_full())
        content_events = list(engine.replay_content_plane())
        
        # content_plane should have fewer events
        assert len(content_events) < len(full_events)
        
        # No events from control_plane turns (2 and 6) in content stream
        control_turn_ids = {2, 6}
        for event in content_events:
            assert event.turn_id not in control_turn_ids


# =============================================================================
# Test: Validation
# =============================================================================

class TestValidation:
    """Tests for trace validation."""
    
    def test_stub_mode_guard_raises(self):
        """Stub mode traces raise error by default."""
        with pytest.raises(StubTraceError):
            engine = TraceReplayEngine(STUB_TRACE_STUB_MODE, strict_stub_guard=True)
            # Trigger initialization (where stub guard check happens)
            engine.get_metadata()
    
    def test_stub_mode_guard_can_be_disabled(self):
        """Stub mode guard can be disabled."""
        # Should not raise
        engine = TraceReplayEngine(STUB_TRACE_STUB_MODE, strict_stub_guard=False)
        assert engine.get_metadata().stub_mode is True
    
    def test_valid_trace_passes_validation(self, stub_trace_path):
        """Valid trace passes validation."""
        # Should not raise
        engine = TraceReplayEngine(stub_trace_path, strict_validation=True)
        # Should be able to get metadata
        meta = engine.get_metadata()
        assert meta.case_id == "case-stub-test"
    
    def test_boundary_time_violation_fails_strict(self):
        """Boundary-time violation fails validation in strict mode."""
        with pytest.raises(TraceValidationError) as exc_info:
            engine = TraceReplayEngine(STUB_TRACE_BOUNDARY_TIME_VIOLATION, strict_validation=True)
            engine.get_metadata()  # Trigger validation
        
        error_msg = str(exc_info.value)
        assert "turn_start.t_ms" in error_msg or "boundary" in error_msg.lower()
    
    def test_boundary_time_violation_warns_inspect(self):
        """Boundary-time violation warns but doesn't fail in inspect mode."""
        # Should not raise, but should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine = TraceReplayEngine(STUB_TRACE_BOUNDARY_TIME_VIOLATION, strict_validation=False)
            engine.get_metadata()  # Trigger validation
        
        # Should have warnings about boundary time
        warning_texts = [str(warning.message) for warning in w]
        assert any("turn_start.t_ms" in text or "boundary" in text.lower() for text in warning_texts)
    
    def test_trel_ms_violation_fails_strict(self):
        """t_rel_ms inconsistency fails validation in strict mode."""
        with pytest.raises(TraceValidationError) as exc_info:
            engine = TraceReplayEngine(STUB_TRACE_TREL_VIOLATION, strict_validation=True)
            engine.get_metadata()  # Trigger validation
        
        error_msg = str(exc_info.value)
        assert "t_rel_ms" in error_msg
    
    def test_trel_ms_violation_warns_inspect(self):
        """t_rel_ms inconsistency warns but doesn't fail in inspect mode."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine = TraceReplayEngine(STUB_TRACE_TREL_VIOLATION, strict_validation=False)
            engine.get_metadata()  # Trigger validation
        
        # Should have warnings about t_rel_ms
        warning_texts = [str(warning.message) for warning in w]
        assert any("t_rel_ms" in text for text in warning_texts)
    
    def test_delta_only_turn_fails_strict(self):
        """Turn with deltas but no boundaries fails in strict mode."""
        with pytest.raises(TraceValidationError) as exc_info:
            engine = TraceReplayEngine(STUB_TRACE_DELTA_ONLY_TURN, strict_validation=True)
            engine.get_metadata()  # Trigger validation
        
        error_msg = str(exc_info.value)
        assert "boundaries" in error_msg.lower() or "schema violation" in error_msg.lower()
    
    def test_delta_only_turn_warns_inspect(self):
        """Turn with deltas but no boundaries warns in inspect mode."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine = TraceReplayEngine(STUB_TRACE_DELTA_ONLY_TURN, strict_validation=False)
            engine.get_metadata()  # Trigger validation
        
        # Should have warnings about missing boundaries
        warning_texts = [str(warning.message) for warning in w]
        assert any("boundaries" in text.lower() or "schema violation" in text.lower() for text in warning_texts)


# =============================================================================
# Test: Metadata
# =============================================================================

class TestMetadata:
    """Tests for trace metadata access."""
    
    def test_get_metadata(self, engine):
        """get_metadata returns TraceMeta object."""
        meta = engine.get_metadata()
        
        assert meta.case_id == "case-stub-test"
        assert meta.schema_version == "2.0.0"
        assert meta.stub_mode is False
        assert meta.total_turns == 7
        assert meta.total_deltas == 12
    
    def test_metadata_t0(self, engine):
        """Metadata includes t0_emitted_ms."""
        meta = engine.get_metadata()
        assert meta.t0_emitted_ms == 1700000000000


# =============================================================================
# Test: Replay Events
# =============================================================================

class TestReplayEvents:
    """Tests for replay event structure."""
    
    def test_delta_events_have_text(self, engine):
        """Delta events have delta_text."""
        events = list(engine.replay_full())
        
        delta_events = [e for e in events if e.event_type == "delta"]
        assert len(delta_events) > 0
        
        for event in delta_events:
            assert event.delta_text is not None
            assert len(event.delta_text) > 0
    
    def test_boundary_events_have_boundary_field(self, engine):
        """Boundary events have boundary field."""
        events = list(engine.replay_full())
        
        start_events = [e for e in events if e.event_type == "turn_start"]
        end_events = [e for e in events if e.event_type == "turn_end"]
        
        for event in start_events:
            assert event.boundary == "start"
        
        for event in end_events:
            assert event.boundary == "end"
    
    def test_turn_end_has_content_hash(self, engine):
        """Turn end events have content_hash."""
        events = list(engine.replay_full())
        
        end_events = [e for e in events if e.event_type == "turn_end"]
        assert len(end_events) > 0
        
        for event in end_events:
            assert event.content_hash is not None
            assert event.content_hash.startswith("sha256:")
    
    def test_events_ordered_by_seq(self, engine):
        """Events are yielded in seq order."""
        events = list(engine.replay_full())
        
        seqs = [e.seq for e in events]
        assert seqs == sorted(seqs)
    
    def test_stream_mode_set_correctly(self, engine):
        """stream_mode field is set correctly."""
        full_events = list(engine.replay_full())
        content_events = list(engine.replay_content_plane())
        
        for event in full_events:
            assert event.stream_mode == "full"
        
        for event in content_events:
            assert event.stream_mode == "content_plane"


# =============================================================================
# Test: Turn Classifications
# =============================================================================

class TestTurnClassifications:
    """Tests for turn classification data structure."""
    
    def test_all_turns_classified(self, engine):
        """All turns have classifications."""
        classifications = engine.get_turn_classifications()
        
        # Should have 7 turns
        assert len(classifications) == 7
        
        for turn_id in range(1, 8):
            assert turn_id in classifications
    
    def test_classification_fields(self, engine):
        """Classifications have all required fields."""
        classifications = engine.get_turn_classifications()
        
        for turn_id, cls in classifications.items():
            assert cls.turn_id == turn_id
            assert cls.turn_type in ("content_plane", "control_plane")
            assert cls.agent_id is not None
            assert cls.turn_text is not None
            assert cls.classification_reason is not None
            assert cls.delta_count >= 0
            assert cls.start_seq is not None
            assert cls.end_seq is not None
    
    def test_classifications_dict_is_copy(self, engine):
        """get_turn_classifications returns a copy."""
        cls1 = engine.get_turn_classifications()
        cls2 = engine.get_turn_classifications()
        
        # Should be different dict objects
        assert cls1 is not cls2


# =============================================================================
# Integration Tests (Real Trace Files)
# =============================================================================

class TestIntegrationRealTraces:
    """Integration tests using real trace files from data/traces/."""
    
    @pytest.fixture
    def real_trace_dir(self):
        """Path to real traces directory."""
        # In Docker, working_dir is /app/evals, so traces are at /app/evals/data/traces
        # Locally, they're at evals/data/traces relative to test file
        test_file_dir = Path(__file__).parent  # tests/
        evals_dir = test_file_dir.parent  # evals/
        return evals_dir / "data" / "traces"
    
    @pytest.fixture
    def real_trace_files(self, real_trace_dir):
        """List of real trace files (limit to first 3 for speed)."""
        if not real_trace_dir.exists():
            pytest.skip(f"Real traces directory not found: {real_trace_dir}")
        
        trace_files = sorted(real_trace_dir.glob("*.trace.jsonl.gz"))[:3]
        if not trace_files:
            pytest.skip(f"No trace files found in {real_trace_dir}")
        
        return trace_files
    
    def test_replay_real_trace_full_stream(self, real_trace_files):
        """Replay real trace FULL stream."""
        trace_file = real_trace_files[0]
        engine = TraceReplayEngine(trace_file, strict_stub_guard=False)
        
        # Should be able to get metadata
        meta = engine.get_metadata()
        assert meta.case_id is not None
        assert meta.schema_version == "2.0.0"
        
        # Should derive labels
        labels = engine.get_derived_agent_labels()
        assert len(labels) > 0
        
        # Should replay events
        events = list(engine.replay_full())
        assert len(events) > 0
        
        # Events should be ordered by seq
        seqs = [e.seq for e in events]
        assert seqs == sorted(seqs)
    
    def test_replay_real_trace_content_plane_stream(self, real_trace_files):
        """Replay real trace content_plane stream."""
        trace_file = real_trace_files[0]
        engine = TraceReplayEngine(trace_file, strict_stub_guard=False)
        
        # Get classifications
        classifications = engine.get_turn_classifications()
        assert len(classifications) > 0
        
        # Replay content_plane stream
        content_events = list(engine.replay_content_plane())
        full_events = list(engine.replay_full())
        
        # content_plane should have fewer or equal events
        assert len(content_events) <= len(full_events)
        
        # Verify no control_plane turns in content stream
        control_turn_ids = {
            turn_id for turn_id, cls in classifications.items()
            if cls.turn_type == "control_plane"
        }
        
        for event in content_events:
            assert event.turn_id not in control_turn_ids
    
    def test_real_trace_timing_gaps_preserved(self, real_trace_files):
        """Verify timing gaps preserved in content_plane stream."""
        trace_file = real_trace_files[0]
        engine = TraceReplayEngine(trace_file, strict_stub_guard=False)
        
        full_events = list(engine.replay_full())
        content_events = list(engine.replay_content_plane())
        
        if len(content_events) < 2:
            pytest.skip("Not enough events to test timing gaps")
        
        # Find a control_plane turn in full stream
        classifications = engine.get_turn_classifications()
        control_turn_ids = {
            turn_id for turn_id, cls in classifications.items()
            if cls.turn_type == "control_plane"
        }
        
        if not control_turn_ids:
            pytest.skip("No control_plane turns in this trace")
        
        # Find events around a control turn
        control_turn_id = list(control_turn_ids)[0]
        control_events = [e for e in full_events if e.turn_id == control_turn_id]
        
        if len(control_events) < 2:
            pytest.skip("Control turn has insufficient events")
        
        control_start_time = control_events[0].virtual_time_ms
        control_end_time = control_events[-1].virtual_time_ms
        
        # Find content events before and after control turn
        before_events = [e for e in content_events if e.virtual_time_ms < control_start_time]
        after_events = [e for e in content_events if e.virtual_time_ms > control_end_time]
        
        if before_events and after_events:
            # Gap should be preserved (not compressed)
            gap_in_full = after_events[0].virtual_time_ms - before_events[-1].virtual_time_ms
            gap_expected = control_end_time - control_start_time
            
            # Gap should be at least as large as the control turn duration
            assert gap_in_full >= gap_expected
    
    def test_real_trace_classifications_complete(self, real_trace_files):
        """All turns in real trace have classifications."""
        trace_file = real_trace_files[0]
        engine = TraceReplayEngine(trace_file, strict_stub_guard=False)
        
        classifications = engine.get_turn_classifications()
        meta = engine.get_metadata()
        
        # Should have classifications for all turns
        if meta.total_turns:
            assert len(classifications) == meta.total_turns
        
        # All classifications should have valid types
        for turn_id, cls in classifications.items():
            assert cls.turn_type in ("content_plane", "control_plane")
            assert cls.classification_reason is not None
    
    def test_multiple_real_traces(self, real_trace_files):
        """Test that multiple real traces can be replayed."""
        for trace_file in real_trace_files:
            engine = TraceReplayEngine(trace_file, strict_stub_guard=False)
            
            # Should derive labels
            labels = engine.get_derived_agent_labels()
            assert len(labels) > 0
            
            # Should classify turns
            classifications = engine.get_turn_classifications()
            assert len(classifications) > 0
            
            # Should replay both streams
            full_events = list(engine.replay_full())
            content_events = list(engine.replay_content_plane())
            
            assert len(full_events) > 0
            assert len(content_events) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

