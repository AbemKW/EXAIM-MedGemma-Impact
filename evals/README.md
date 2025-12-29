# EXAID Evaluation Module

Reproducible evaluation framework for the EXAID conference paper. This module provides Docker-based reproducibility for all evaluation experiments.

## Purpose and Claims Boundary

**Performance-only evaluation of EXAID summarization middleware.**

This evaluation measures:
- Compression efficiency (CTU saved)
- Concept coverage (trace CUIs recalled in summaries)
- Faithfulness (unsupported content rates)
- Latency and resource usage

**No clinical outcome claims are made.**

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- No host Python required (evaluation runs in container)

### MAC Submodule

EXAID uses a forked MAC repository with delta/chunk-level streaming instrumentation:

- Fork URL: https://github.com/AbemKW/mac-streaming-traces
- Path: `third_party/mac`
- Purpose: Capture per-delta emission timestamps (`t_emitted_ms`) for realistic streaming replay

**Invariant:** This fork only adds transparent delta/chunk-level timing instrumentation. All MAC conversation logic, agent orchestration, speaker selection, and termination conditions remain unchanged from the original implementation.

The submodule is pinned to a specific commit for reproducibility. Initialize with:

```bash
git submodule update --init --recursive
```

**Important:** If the MAC submodule commit is updated, traces MUST be regenerated before running any evaluation. Results obtained from mismatched traces and submodule commits are invalid.

---

## Trace Format

**Trace Schema v2.0.0 - Raw stream captures without derived units.**

Traces store raw streaming data only:

| Stored in Traces | NOT Stored (Computed During Replay) |
|------------------|-------------------------------------|
| `delta_text` (raw stream delta/chunk) | `ws_units` (TokenGate, during evaluation) |
| `t_emitted_ms` (absolute timestamp) | `ctu` (metrics computation) |
| `t_rel_ms` (relative to t0) | |
| `agent_id`, `turn_id`, `seq` | |

**Rationale:** Under streaming fragmentation, per-delta whitespace counting produces unreliable results. TokenGate-size counting must use the same streaming accumulator logic during replay. Storing derived units in traces increases drift risk between capture and replay.

### Record Types

| Record Type | Purpose |
|-------------|---------|
| `trace_meta` | Provenance, t0 anchor, schema version |
| `stream_delta` | Raw stream delta/chunk with timing |
| `turn_boundary` | Turn start/end markers with content_hash |

### Timing Semantics

- `t0_emitted_ms`: Anchor timestamp (first `stream_delta` emission)
- `t_emitted_ms`: Absolute emission time (ms since epoch)
- `t_rel_ms`: Relative to t0 (`t_ms - t0_emitted_ms`)

**Important:** `t_rel_ms` for `turn_boundary` records MAY BE NEGATIVE if the boundary occurs before the first delta. This is valid because:
- `t0` is defined as the first `stream_delta` emission time
- `turn_start` boundaries often occur before the first delta of that turn
- All times remain replayable: given `t0` and `t_rel_ms`, exact timing is reconstructible

| Record Type | t_rel_ms | Notes |
|-------------|----------|-------|
| `stream_delta` | Always >= 0 | By definition, t0 = first delta |
| `turn_boundary` | May be negative | Boundaries can occur before t0 |

### Generate Timed Traces

```bash
# Dry-run (validate config without running MAC)
python src/make_traces.py --config configs/mas_generation.yaml --dry-run

# Generate one trace (for testing)
python src/make_traces.py --config configs/mas_generation.yaml --limit 1

# Generate all traces
python src/make_traces.py --config configs/mas_generation.yaml
```

### Validate Traces

**Standalone validation tool** (pre-evaluation check):

```bash
python src/validate_traces.py --traces data/traces/ --verbose
```

**Note:** The Trace Replay Engine also validates traces during replay (see "Replay Validation Guarantees" section below). Both tools share similar validation rules but serve different purposes:
- `validate_traces.py`: Standalone validation before evaluation
- Replay engine: Validation during replay with strict/inspect modes

**Validation Rules:**
1. `seq` strictly increasing across entire trace
2. `t_emitted_ms` non-decreasing for `stream_delta` records
3. `t_rel_ms` consistency:
   - For `stream_delta`: `t_rel_ms == t_emitted_ms - t0` (always >= 0)
   - For `turn_boundary`: `t_rel_ms == t_ms - t0` (may be negative!)
4. Turn boundary start/end pairs match
5. Boundary time consistency:
   - `turn_start.t_ms <= first_delta.t_emitted_ms` for that turn
   - `turn_end.t_ms >= last_delta.t_emitted_ms` for that turn
   - **Tolerance:** ±2ms allowed due to millisecond resolution; violations within tolerance are warnings (cosmetic), not errors
6. `content_hash` matches recomputed hash

**Stub Mode Warning:** Traces with `stub_mode: true` are flagged during validation and must NOT be used for evaluation.

---

## Trace Replay Engine

**Paper hook: Section 3.2**

The Trace Replay Engine provides deterministic replay of v2.0.0 traces with virtual time and turn classification. It enables downstream evaluation components (TokenGate, metrics) to consume traces in a consistent, reproducible manner.

**File Organization:**
- `src/trace_replay_engine.py` - Core library module (importable)
- `cli/replay_trace.py` - CLI tool for inspection/debugging (runnable)
- `tests/test_trace_replay_engine.py` - Unit and integration tests
- `cli/calibrate_tokengate.py` - TokenGate calibration CLI wrapper (argument parsing + orchestration)
- `src/tokengate_calibration/` - TokenGate calibration subpackage
  - `runner.py` - Calibration orchestrator (sequencing + trace replay)
  - `models.py` - Calibration dataclasses (Policy, CaseMetrics, etc.)
  - `grid.py` - Policy grid generation + validation
  - `metrics.py` - Per-case + aggregate metrics, constraints, percentiles
  - `selection.py` - Pareto/utopia selection + weighted fallback logic
  - `io.py` - Hashing, manifest/config loading, artifact I/O

**Note:** `cli/` contains runnable inspection/debug tools; `src/` contains importable library code.

### Architecture

**Two-pass design:**
1. **Pass 1**: Derive `agent_labels` from `turn_boundary` records only (authoritative source)
2. **Pass 2**: Classify turns and yield replay events

This ensures determinism: same trace always produces same label set before classification.

### Turn Classification

Turns are classified as either `content_plane` (substantive) or `control_plane` (orchestration):

| Category | Description | Classification Rules |
|----------|-------------|---------------------|
| `control_plane` | Speaker selection, orchestration signals | Exact agent label match, TERMINATE sentinel |
| `content_plane` | Substantive agent content | Everything else (conservative default) |

**Why control_plane exists**: MAC's GroupChat emits short turns for speaker selection (e.g., just "Doctor0"). These carry no semantic content and should be excluded from concept extraction and TokenGate accumulation.

### Classification Rules (Conservative, Exact-Match Only)

| Rule | Condition | Result | `classification_reason` |
|------|-----------|--------|-------------------------|
| **Exact Label Match** | `turn_text.strip().lower()` exactly matches derived agent label | `control_plane` | `"exact_label_match:{label}"` |
| **TERMINATE Sentinel** | `turn_text.strip().upper() == "TERMINATE"` | `control_plane` | `"terminate_sentinel"` |
| **Empty Turn** | `turn_text.strip() == ""` | `control_plane` | `"empty_turn"` |
| **Default** | Everything else | `content_plane` | `"default_content"` |

**Design principle**: We never filter unless certain. Partial matches like "Ask Doctor0" remain as content. Empty turns are classified as control_plane to avoid polluting semantic evaluation.

### Agent Labels Derived from Trace (Boundaries Only)

Labels are derived from the trace itself (Pass 1), not from config files:
1. Scan `turn_boundary` records only (authoritative source for speaker identity)
2. Collect unique `agent_id` values, normalize to lowercase
3. Use this frozen set for classification (Pass 2)

**Why boundaries only?** Boundaries are the authoritative source for speaker identity in the schema. Deltas may have missing or inconsistent agent_id values. This makes the derivation rule simpler and more defensible.

### Replay Streams

| Stream | Contents | Use Case |
|--------|----------|----------|
| FULL | All events including control_plane | Raw replay, timing analysis |
| content_plane | content_plane events only | TokenGate, semantic evaluation |

**Critical**: The content_plane stream preserves timing gaps from excluded control_plane turns. Virtual time is NOT compressed. This ensures TokenGate sees realistic inter-turn delays.

### Audit Flags

Suspicious turns (look label-like but don't match derived labels) are:
- Classified as `content_plane` (conservative)
- Flagged with `suspicious_label_like_unmatched` for reviewer visibility

Use `--audit` flag in CLI to view flagged turns.

### Usage

**Python API:**

```python
from pathlib import Path

# Import from evals package (if running from repo root)
from evals.src.trace_replay_engine import TraceReplayEngine

# Alternative: If running from evals/ directory, add src to path first:
# import sys
# sys.path.insert(0, "src")
# from trace_replay_engine import TraceReplayEngine

# Initialize engine
engine = TraceReplayEngine(Path("data/traces/case-33651373.trace.jsonl.gz"))

# Get metadata
meta = engine.get_metadata()
labels = engine.get_derived_agent_labels()

# Replay full stream
for event in engine.replay_full():
    print(f"t={event.virtual_time_ms}ms: {event.event_type}")

# Replay content_plane stream only
for event in engine.replay_content_plane():
    process_content(event)

# Get classifications
classifications = engine.get_turn_classifications()
for turn_id, cls in classifications.items():
    print(f"Turn {turn_id}: {cls.turn_type} ({cls.classification_reason})")

# Get audit flags
flags = engine.get_audit_flags()
for flag in flags:
    print(f"Turn {flag.turn_id}: {flag.flag_type}")
```

**CLI Tool:**

```bash
# Show metadata and timeline
python cli/replay_trace.py data/traces/case-33651373.trace.jsonl.gz

# Show content_plane stream only
python cli/replay_trace.py --stream content_plane data/traces/case-33651373.trace.jsonl.gz

# Show turn classifications
python cli/replay_trace.py --classifications data/traces/case-33651373.trace.jsonl.gz

# Show audit flags
python cli/replay_trace.py --audit data/traces/case-33651373.trace.jsonl.gz

# Shift timeline to start at t=0
python cli/replay_trace.py --shift-to-zero data/traces/case-33651373.trace.jsonl.gz
```

### Timing Semantics

**Virtual Time Derivation:**

| Mode | Formula | Use Case |
|------|---------|----------|
| Default | `virtual_time_ms = t_rel_ms` | Preserves anchor semantics |
| `shift_to_zero=True` | `virtual_time_ms = t_rel_ms - min_t_rel_ms` | Plots starting at t=0 |

**Anchor Semantics:**
- `t0_emitted_ms`: First `stream_delta` emission (absolute epoch time)
- `stream_delta.t_rel_ms`: Always >= 0 (by definition of t0)
- `turn_boundary.t_rel_ms`: May be negative (boundary before first delta)

When `shift_to_zero=True`, the minimum `t_rel_ms` is computed across **all** record types (deltas AND boundaries), ensuring all events shift together while preserving relative timing.

### Replay Validation Guarantees

**Paper hook: Section 3.2**

The replay engine enforces deterministic validation with two modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Strict** (`strict_validation=True`) | Violations → Errors → Stops execution | Production evaluation, reviewer verification |
| **Inspect** (`strict_validation=False`) | Violations → Warnings → Continues execution | Debugging, inspection, known-issue traces |

**Strict Mode Enforces:**

| Check | Severity | Action |
|-------|----------|--------|
| `seq` strictly increasing | ERROR | Raise `TraceValidationError` |
| `t_emitted_ms` non-decreasing (deltas) | ERROR | Raise `TraceValidationError` |
| `t_rel_ms` consistency (critical for virtual time) | ERROR | Raise `TraceValidationError` |
| Boundary start/end pairs match | ERROR | Raise `TraceValidationError` |
| Boundary-time containment (>epsilon) | ERROR | Raise `TraceValidationError` |
| Turns with deltas but no boundaries | ERROR | Raise `TraceValidationError` |
| `stub_mode == true` (if `strict_stub_guard=True`) | ERROR | Raise `StubTraceError` |

**Inspect Mode Allows:**

- All strict-mode violations become **warnings** instead of errors
- Execution continues, enabling analysis of problematic traces
- Boundary-time violations within epsilon (≤2ms) are always warnings
- `content_hash` mismatches are always warnings (never errors)

**Design Rationale:** Strict mode ensures data quality for evaluation reproducibility. Inspect mode enables debugging and analysis while surfacing issues via warnings. This aligns with the instrumentation-only policy: traces are frozen artifacts, and validation ensures they meet replay requirements.

### Performance Characteristics

**Memory Usage:**

The replay engine loads one trace into memory (O(n) records where n = number of records in trace).

**Typical Trace Sizes:**

- Per-case traces: ~1,000-5,000 records
- Compressed size: ~50-200 KB (gzipped JSONL)
- Memory footprint: ~2-10 MB per trace (uncompressed)

**Scalability:**

For the EXAID evaluation (40 cases), total memory usage is ~80-400 MB, which is well within reasonable limits. The two-pass architecture (label derivation → classification) ensures deterministic behavior at the cost of loading the full trace.

**Future Enhancement:** For very large traces (>100k records), a streaming-only mode could be added that processes records without full in-memory reconstruction. This is not required for the current evaluation scale.

### Empty Turn Policy

Turns with empty or whitespace-only text (`turn_text.strip() == ""`) are classified as `control_plane` with `classification_reason="empty_turn"`. This ensures:

- Empty turns are excluded from semantic evaluation (consistent with control_plane filtering)
- Timing gaps are still preserved (empty turns don't compress virtual time)
- Clear auditability (empty turns are explicitly flagged)

**Rationale:** Empty turns carry no semantic content and should not pollute TokenGate accumulation or concept extraction. Classifying them as control_plane is conservative and reduces confusion.

### Integration Points

**TokenGate Calibration:**

```python
# TokenGate consumes content_plane stream but gets FULL timing
for event in engine.replay_content_plane():
    if event.event_type == "delta":
        tokengate.accumulate(
            text=event.delta_text,
            virtual_time_ms=event.virtual_time_ms  # Gaps preserved
        )
```

**Metrics Scripts:**

```python
# M4/M5 (trace coverage) - content_plane only
content_text = "".join(
    e.delta_text for e in engine.replay_content_plane() 
    if e.event_type == "delta"
)

# Audit classification
for turn_id, cls in engine.get_turn_classifications().items():
    print(f"Turn {turn_id}: {cls.turn_type} ({cls.classification_reason})")
```

---

## Run ID Definitions

### Separation of Concerns

| Step | ID | Purpose |
|------|-----|---------|
| Trace Generation | `mas_run_id` | Trace generation campaign |
| Evaluation | `eval_run_id` | EXAID variant evaluation |

### mas_run_id (Trace Generation)

**Format:** `mas_<mac8>_<model>_<decoding8>_<cases8>`

- `mac8`: First 8 chars of MAC commit hash
- `model`: Slugified model name (lowercase, no dashes)
- `decoding8`: First 8 chars of SHA256(canonicalized decoding params)
- `cases8`: First 8 chars of case_list_hash

**Example:** `mas_1d5b227a_gpt4omini_a3b4c5d6_e7f8g9h0`

**Property:** Input-derived (no date), deterministic. Same inputs = same ID regardless of when generation runs.

### eval_run_id (Evaluation)

**Format:** `eval_<variant>_<trace_dataset_hash_8>_<exaid_commit_8>`

- `variant`: V0, V1, V2, V3, V4
- `trace_dataset_hash_8`: First 8 chars of trace_dataset_hash
- `exaid_commit_8`: First 8 chars of EXAID commit

**Example:** `eval_V0_b2c3d4e5_8d45cbb1`

**Note:** Used during evaluation; not used in trace generation.

### trace_dataset_hash

**Definition:** SHA256 of canonical manifest fields (NOT raw file bytes)

```python
canonical = {
    "mas_run_id": "...",
    "case_list_hash": "sha256:...",
    "traces": sorted([(case_id, trace_sha256), ...])
}
```

---

## Instrumentation-Only Policy

### ALLOWED (Transparent Instrumentation)

- Capture streaming deltas/chunks with timestamps
- Record agent attribution per turn
- Compute content hashes for integrity
- Write trace files with provenance

### NOT ALLOWED (Would Change MAC Behavior)

- Modify prompts or system messages
- Change speaker selection or routing logic
- Alter termination conditions
- Reorder messages or turns
- Override decoding parameters (temperature, sampling)

---

## What Gets Committed vs Not Committed

| Artifact | Committed | Reason |
|----------|-----------|--------|
| Schema files | ✅ Yes | Defines trace format |
| Config files | ✅ Yes | Reproducibility |
| Scripts | ✅ Yes | Code |
| **Timed traces** | ✅ **Yes** | Full evaluation reproducibility |
| **Manifests** | ✅ **Yes** | Provenance and integrity |
| **Case lists** | ✅ **Yes** | Public MAC case IDs |
| Run outputs | ❌ No | Regenerated by reviewers |
| Metrics | ❌ No | Regenerated by reviewers |

### Data Provenance

**Traces are derived from MAC's public rare-disease case dataset:**
- Source: [MAC (Medical multi-Agent Consultation)](https://github.com/microsoft/MAC)
- License: **CC BY 4.0** (Creative Commons Attribution 4.0)
- Content: Synthetic/anonymized rare-disease clinical vignettes
- No PHI: Safe to redistribute

The MAC submodule code is MIT-licensed.

### Reviewer Workflow

**Cloning the repository is sufficient for evaluation:**

```bash
git clone --recurse-submodules https://github.com/AbemKW/ExAID.git
cd ExAID
docker compose -f docker-compose.evals.yml build
# Traces and manifests are already included - no regeneration required
```

To **regenerate traces** (optional, requires OpenAI API key):
```bash
docker compose -f docker-compose.evals.yml run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  evals python src/make_traces.py --config configs/mas_generation.yaml
```

### .gitignore Policy

**Tracked (committed):**
```
evals/data/traces/*.trace.jsonl.gz
evals/data/manifests/*.manifest.jsonl
evals/data/manifests/*.case_list.jsonl
```

**Ignored (not committed):**
```
evals/data/runs/**/*.jsonl.gz    # Evaluation runs
evals/data/metrics/*.json        # Computed metrics
```

---

### Build and Run

```bash
# Build the evaluation container
docker compose -f docker-compose.evals.yml build

# Verify the build
docker compose -f docker-compose.evals.yml run --rm evals -c "python --version && spacy validate"

# Run validation (passes with empty directories in scaffold mode)
docker compose -f docker-compose.evals.yml run --rm evals scripts/00_validate.sh
```

---

## Complete Workflow (Phases 0-5)

Execute the following commands in order:

### Step 1: Validate Traces + Generate Stoplists

```bash
# 0a. Validate frozen traces
docker compose -f docker-compose.evals.yml run --rm evals \
  python src/validate_traces.py --traces data/traces/ --verbose

# 0b. Generate stoplists (drift-proof, non-circular)
docker compose -f docker-compose.evals.yml run --rm evals \
  python src/generate_stoplists.py --traces data/traces/ \
    --output configs/ --config configs/extractor.yaml --verbose
```

**Output:**
- `configs/stop_entities.txt` - Surface-form stoplist
- `configs/stop_cuis.txt` - CUI stoplist
- `configs/stoplist_df_report.csv` - Audit artifact with DF statistics

### Step 2: Generate Traces (if not already frozen)

```bash
docker compose -f docker-compose.evals.yml run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  evals python src/make_traces.py --config configs/mas_generation.yaml
```

### Step 3: Validate Schemas

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/00_validate.sh
```

### Step 4: Run Summarizer Variants

```bash
# Run all variants (V0-V4)
docker compose -f docker-compose.evals.yml run --rm evals scripts/02_run_variants.sh

# Run specific variant
docker compose -f docker-compose.evals.yml run --rm evals scripts/02_run_variants.sh V3

# Or use Python directly with more options
docker compose -f docker-compose.evals.yml run --rm evals \
  python src/run_variants.py --traces data/traces/ --output data/runs/ --verbose
```

### Step 5: Compute Metrics

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/03_compute_metrics.sh

# Or use Python directly
docker compose -f docker-compose.evals.yml run --rm evals \
  python src/compute_metrics.py --runs data/runs/ --traces data/traces/ \
    --output data/metrics/ --bootstrap-samples 10000 --verbose
```

**Output:**
- `data/metrics/per_case.metrics.jsonl` - Per-case metric records
- `data/metrics/aggregate.metrics.json` - Aggregate statistics with CIs

### Step 6: TokenGate Calibration (Phase 5)

**Paper hook: Section 3.3**

Calibrates TokenGate trigger parameters (`min_words`, `max_words`, `silence_timer`, `max_wait_timeout`) by systematically evaluating literature-informed parameter combinations across frozen v2.0.0 traces.

```bash
# Run calibration sweep
docker compose -f docker-compose.evals.yml run --rm evals scripts/05_calibrate_tokengate.sh

# Or use Python directly
docker compose -f docker-compose.evals.yml run --rm evals \
  python -m evals.cli.calibrate_tokengate \
    --traces data/traces/ \
    --manifest data/manifests/exaid_traces_*.manifest.jsonl \
    --config configs/calibration_sweep.yaml \
    --output data/calibration/
```

**Output Directory Structure:**
```
evals/data/calibration/calib_<hash8>_<hash8>_<hash8>/
  ├── calibration_results.csv          # All policy results
  ├── calibration_per_case.jsonl       # Per-case detailed metrics
  ├── calibration_summary.json         # Summary and selected policy
  ├── chosen_tokengate_params.yaml     # Selected parameters (frozen config)
  ├── calibration_report.md            # Detailed report
  ├── calibration_config.yaml          # Sweep configuration used
  └── spam_sensitivity.json            # α sensitivity analysis
```

**Calibration Methodology:**

1. **Parameter Grid**: 5×5×5×5 = 625 combinations
   - `min_words`: [30, 40, 50, 60, 70] (larger chunks for "bucket not pipe" behavior)
   - `max_words`: [80, 100, 120, 140, 160] (typical sentence ranges and practical chunk sizes)
   - `silence_timer_ms`: [1000, 1500, 2000, 2500, 3000] (streaming pause thresholds)
   - `max_wait_timeout_ms`: [4000, 5000, 6000, 7000, 8000] (absolute upper bound to flush)

2. **Policy Validity Filter**: Filters invalid combinations before replay
   - `min_words < max_words` (strictly)
   - `max_wait_timeout_ms >= silence_timer_ms`
   - `max_words >= min_words + k` (k=10, ensures meaningful gap for larger chunks)

3. **Production-Faithful Replay**: Deterministic replay matching production behavior exactly
   - Uses `TokenGate` with `ManualClock` for virtual time
   - Timers checked synchronously: inside `add_token()` (silence check) and after `add_token()` via `check_timers()`
   - Explicit flush at `turn_end` events (matching production `flush_agent()` behavior)
   - **No gap-based timer processing** (production has no background/tick loop)
   - Multi-agent support (per-agent buffers)

4. **Metrics Computation**: Per-case and aggregated metrics
   - TTFF (time to first flush): `ttff_content_ms` (from first content delta) and `ttff_trace_ms` (from trace t0)
   - Flush count, chunk size distribution (p50, p90, p95, max)
   - Spam percentage (policy-relative: `% flushes < α * min_words`, default α=0.7)
   - Timer flush percentage, timer under-minimum percentage

5. **Constraint Filters**: Hard requirements (reject violating policies)
   - `ttff_content_p95_ms ≤ derived` (3× global p95 time-to-reach-min_words)
   - `spam_pct_mean ≤ 10%`
   - `timer_under_min_pct_mean ≤ 20%`
   - `chunk_size_p50 ≥ 0.6 * min_words` (policy-relative)
   - `chunk_size_p50 ≥ 50 words` (absolute minimum, cost constraint)
   - `chunk_size_p95 ≤ 180 words`
   - `worst_wait_p95_ms ≤ derived` (1.5× max_wait_timeout_ms upper bound)
   - `flush_count_mean ≤ 100` (cost constraint: limits BufferAgent calls per case)

6. **Selection Rule**: 3-objective Pareto frontier + utopia-distance selection
   - **Objectives**: Minimize TTFF (Time To First Flush), minimize flush count (BufferAgent calls), maximize chunk size
   - **Normalization**: Data-driven percentile-based bounds (P05/P95) computed from survivor policies
     - Small-N handling: Uses min/max if len(survivors) < 5
     - Degenerate bounds: Metrics with insufficient variance (hi - lo < epsilon) are dropped
       - Epsilon thresholds: EPS_MS = 5.0 ms (TTFF), EPS_COUNT = 2.0 (flush_count), EPS_WORDS = 2.0 (chunk_size)
       - Relaxed thresholds prevent false positives from floating-point rounding
   - **Pareto frontier**: k-dimensional non-dominated points (k = number of active metrics, excluding dropped)
   - **Utopia distance**: Dimension-normalized Euclidean distance to utopia point (1, 1, ..., 1) in goodness space
     - Formula: `sqrt(mean((1 - goodness_i)^2))` for active dimensions
   - **Tie-breaking**: Deterministic order: lower flush_count → higher chunk_size → lower TTFF → smallest policy_id
   - **Fallbacks**: Weighted score (with renormalized weights) if Pareto frontier empty; lexicographic if all metrics dropped

7. **α Sensitivity Analysis**: Post-processing to validate spam definition
   - Recompute spam metrics for α ∈ {0.5, 0.6, 0.7, 0.8}
   - Demonstrates winner/top-k stability across α values

**Reproducibility:**
- Deterministic run ID: `calib_<trace_dataset_hash8>_<config_hash8>_<exaid_commit8>`
- All hashes logged in summary JSON
- Same inputs → same outputs (verified)

**Selected Parameters:**
The chosen parameters are written to `chosen_tokengate_params.yaml` and become the frozen TokenGate config for all later phases.

---

## Canonical Trace Text Definition

**Paper hook: Section 3.1**

Canonical trace text is defined in `src/trace_text.py:build_canonical_trace_text()`:

| Rule | Description |
|------|-------------|
| **Include** | Chunks where `event_subtype == "message"` |
| **Exclude** | `orchestrator_summary`, `system_note`, other subtypes |
| **Fallback** | If `event_subtype` missing, include unless `agent_id` in `{orchestrator, _system, system, meta}` |
| **FAIL-FAST** | Raises `TraceParsingError` if 0 message chunks found |

**Statistics tracked:**
- `included_message_chunks` - Count of message chunks included
- `excluded_orchestrator_summary` - Excluded orchestrator records
- `excluded_system_note` - Excluded system notes
- `excluded_other_subtype` - Other excluded subtypes
- `missing_event_subtype` - Chunks included via fallback (missing subtype)

**Single source of truth:** All modules MUST import from `trace_text.py`:
- `build_canonical_trace_text()` - Full trace text
- `build_window_text()` - Window text for M6a/M6b

---

## Determinism Rules

**Paper hook: Section 3.2**

All outputs are byte-stable across runs:

### Timestamps

| Event Type | Derivation |
|------------|------------|
| Chunk events | `t_emitted_ms` from trace chunk |
| Window events | `t_emitted_ms` from end_seq chunk |
| Fallback | `run_start_time + seq*1000` (tracked in `run_meta`) |

**STRICT:** No silent 1970 epoch fallback. `missing_t_emitted_ms_count` is tracked in `run_meta.timestamp_derivation`.

### Serialization

| Requirement | Value |
|-------------|-------|
| gzip mtime | `0` (deterministic) |
| JSON sort_keys | `True` |
| JSON separators | `(',', ':')` (compact) |
| Record ordering | By record_type, then index |

### Verification

```bash
# Running twice should produce identical outputs
md5sum data/runs/V0/*.jsonl.gz
```

---

## Stoplist Generation (No Leakage)

**Paper hook: Section 6.1**

> "DF computed strictly on frozen traces (n=40). Concepts appearing in >=90% of traces were stoplisted."

### Process

1. **Load metrics config** - Same extractor settings as metrics computation (drift-proof)
2. **Disable stoplists** - `stop_entities_file=None`, `stop_cuis_file=None` (non-circular)
3. **Extract concepts** - Per-case unique CUI/surface sets
4. **Compute DF** - Document frequency across all traces
5. **Apply threshold** - DF >= 90% → stoplisted

### Audit Artifact

`configs/stoplist_df_report.csv` includes:
- `concept` - CUI or surface string
- `type` - "CUI" or "SURFACE"
- `df_count` - Number of traces containing concept
- `df_fraction` - df_count / n_cases
- `status` - "STOPLISTED" or "KEPT"

Summary statistics at end:
```
# SUMMARY
# n_cases,40
# threshold,0.9
# cutoff_count,36
# stoplisted_cuis,15
# stoplisted_surfaces,42
```

---

## CUI Extraction Ordering

**Paper hook: Section 6.1**

CUI extraction follows **exact** filter→sort→topK→stoplist ordering:

```python
# Per entity mention:
1. candidates = list(ent._.kb_ents)           # Retrieve all
2. filtered = [c for c in candidates if c[1] >= threshold]  # Filter by score
3. sorted_cands = sorted(filtered, key=lambda x: x[1], reverse=True)  # Sort DESC
4. top_k = sorted_cands[:max_k]              # Take top K
5. result = [cui for cui, _ in top_k if cui.upper() not in stop_cuis]  # Stoplist
```

### CUI Normalization

| Rule | Description |
|------|-------------|
| Canonical form | Uppercase (C1234567) |
| Stoplist matching | Case-insensitive (normalized before comparison) |
| Storage | Always uppercase |

---

## Faithfulness Paradox Resolution

**Paper hook: Section 5.1**

Schema failures must be handled correctly for M6a/M6b faithfulness metrics:

| Condition | M6a/M6b Result | Effect |
|-----------|----------------|--------|
| `schema_ok=false` | **EXCLUDED** (return `None`) | Not counted in denominator |
| `schema_ok=true`, empty CUIs | **0.0** | Grounded by absence |
| `schema_ok=true`, non-empty CUIs | Computed value | Normal case |

**Reporting:**
- `faithfulness_valid_event_count` - Events included in M6a/M6b means
- `excluded_from_faithfulness_count` - Events excluded (schema failures)
- `schema_failures` - Total schema failure count (for M10)

**Coverage penalty:** Schema-failed events contribute empty concept sets, penalizing recall.

---

## Variant Ablation Suite

**Paper hook: Section 4.1-4.2**

Note: TokenGate uses whitespace-delimited word counts (not model tokenizer tokens) for its min/max thresholds.

| Variant | Trigger Policy | Components |
|---------|----------------|------------|
| **V0** | `full_exaid` | TokenGate (word thresholds) + BufferAgent (completeness/value/novelty) + Summarizer |
| **V1** | `turn_end` | Trigger at `is_turn_end=True` + Summarizer only |
| **V2** | `no_buffer` | TokenGate flush → Summarizer (skip BufferAgent) |
| **V3** | `no_tokengate` | Fixed chunk/time + BufferAgent + Summarizer |
| **V4** | `no_novelty` | TokenGate (word thresholds) + BufferAgent (completeness + value only) + Summarizer |

### V3 Calibration

V3 uses fixed CTU intervals calibrated from V0:
- **Method:** Median (not mean) of V0 flush sizes
- **Chunk size:** 125 CTU (calibrated)
- **Documented in:** `configs/variants/V3.yaml`

---

## Metrics (M1-M10)

| Metric | Description | Computation |
|--------|-------------|-------------|
| **M1** | Compression ratio | `1 - (summary_ctu / trace_ctu)` |
| **M2** | Summary count | Number of summary events |
| **M3** | Redundancy | Jaccard similarity on CUI sets between consecutive summaries |
| **M4** | Trace coverage | `|summary_cuis ∩ trace_cuis| / |trace_cuis|` |
| **M5a** | Unsupported global | `|summary_cuis - trace_cuis| / |summary_cuis|` |
| **M5b** | Unsupported per-summary | Mean per-summary unsupported rate |
| **M6a** | Window-groundedness | Unsupported fraction vs window |
| **M6b** | Contract-groundedness | Unsupported fraction vs window+latest_summary |
| **M7** | Mean latency | Mean summary generation latency (ms) |
| **M8** | LLM usage | Total prompt + completion CTU |
| **M9** | BufferAgent overhead | Decision count and CTU |
| **M10** | Schema failure rate | `schema_failures / summary_count` |

### M6b Reconstruction

M6b support set = window_text + latest_summary_semantics_text

`latest_summary_semantics_text` is reconstructed by looking up `latest_summary_event_id` within the run log.

---

## Run Log Schema (v1.3.0)

Multi-record JSONL format per `schemas/exaid.run.schema.json`:

| Record Type | Purpose |
|-------------|---------|
| `run_meta` | Provenance, config, timestamps stats |
| `tokengate_flush` | TokenGate accumulation events |
| `buffer_decision` | BufferAgent decisions (M9) |
| `summary_event` | Summary generation events (M6a/M6b) |

### Key Fields

**run_meta:**
- `concept_extractor.linker_kb_version` - UMLS snapshot (e.g., "2023AB")
- `stoplists_provenance` - SHA256 hashes of stoplist files
- `timestamp_derivation.missing_t_emitted_ms_count` - Fallback count

**summary_event:**
- `schema_ok` - Boolean for faithfulness exclusion
- `latest_summary_event_id` - For M6b reconstruction
- `llm_usage.prompt_ctu` / `completion_ctu` - Deterministic CTU
- `llm_usage.provider_prompt_tokens` - API-reported (optional)

---

## Directory Structure

```
evals/
├── README.md                      # This file
├── requirements-evals.txt         # Pinned Python dependencies
├── schemas/                       # JSON Schema definitions
│   ├── exaid.manifest.schema.json
│   ├── exaid.trace.schema.json
│   ├── exaid.run.schema.json      # Multi-record JSONL v1.3.0
│   └── exaid.metrics.schema.json
├── configs/                       # Configuration files
│   ├── extractor.yaml             # Concept extractor config
│   ├── dataset.yaml               # Dataset selection + MAC case policy
│   ├── mas_generation.yaml        # MAS trace generation (MAC config)
│   ├── summarizer.yaml            # Summarizer params (history_k=3)
│   ├── metrics.yaml               # Metric computation params
│   ├── stop_entities.txt          # Surface-form stoplist
│   ├── stop_cuis.txt              # CUI stoplist
│   ├── stoplist_df_report.csv     # Audit artifact
│   └── variants/                  # Variant-specific configs
│       ├── V0.yaml                # Full EXAID
│       ├── V1.yaml                # Turn-end only
│       ├── V2.yaml                # No BufferAgent
│       ├── V3.yaml                # No TokenGate (calibrated)
│       └── V4.yaml                # No novelty check
├── data/                          # Data artifacts
│   ├── manifests/                 # Experiment manifests
│   ├── traces/                    # Generated MAS traces (frozen)
│   ├── runs/                      # Summarizer run outputs
│   │   ├── V0/
│   │   ├── V1/
│   │   ├── V2/
│   │   ├── V3/
│   │   └── V4/
│   └── metrics/                   # Computed metrics
│       ├── per_case.metrics.jsonl
│       ├── aggregate.metrics.json
│       └── figures/               # Generated figures
├── src/                           # Python library modules (importable)
│   ├── trace_replay_engine.py     # Trace replay engine (core library)
│   ├── trace_text.py              # Canonical trace text (single source)
│   ├── validate_traces.py         # Trace validation
│   ├── generate_stoplists.py      # Stoplist generation
│   ├── deterministic_utils.py     # Timestamps, IDs, CTU
│   ├── deterministic_io.py        # gzip/JSON writing
│   ├── concept_extractor.py       # CUI extraction
│   ├── validate_logs.py           # Schema validation
│   ├── make_traces.py             # Trace generation (MAC integration)
│   ├── run_variants.py            # Variant replay engine
│   ├── compute_metrics.py         # M1-M10 metrics
│   └── test_concept_extractor.py  # Unit tests
├── cli/                           # CLI tools (runnable inspection/debug tools)
│   └── replay_trace.py            # Trace replay CLI tool
└── scripts/                       # Orchestration scripts (shell scripts)
    ├── 00_validate.sh
    ├── 01_make_traces.sh
    ├── 02_run_variants.sh
    └── 03_compute_metrics.sh
```

---

## Frozen Parameters

The following parameters are frozen for the conference paper evaluation:

### Summarizer Configuration
- `history_k: 3` - Number of historical summaries in context

### Concept Extractor Configuration
- Model: `en_core_sci_sm`
- Linker: UMLS (`linker_name: umls`)
- CUI score threshold: 0.7
- Max K: 10
- Min entity length: 3 characters
- CUI normalization: uppercase

### Metric Configuration
- Bootstrap samples: 10,000
- Confidence level: 95%
- Random seed: 42

### V3 Calibration
- Method: Median of V0 flush sizes
- Chunk size: 125 CTU

---

## Statistical Analysis

**Paper hook: Section 5.3**

### Bootstrap Confidence Intervals

95% CIs computed via 10k bootstrap resamples with fixed seed (42):

```python
def bootstrap_ci(values, n_resamples=10000, ci_level=0.95, seed=42):
    rng = np.random.RandomState(seed)
    bootstrap_means = [np.mean(rng.choice(values, len(values), replace=True)) 
                       for _ in range(n_resamples)]
    return (np.mean(values),
            np.percentile(bootstrap_means, 2.5),
            np.percentile(bootstrap_means, 97.5))
```

### Paired Tests (if needed)

- Wilcoxon signed-rank test for paired comparisons
- Paired bootstrap for effect sizes

---

## Troubleshooting

### Container build fails
```bash
docker compose -f docker-compose.evals.yml build --no-cache
```

### scispaCy model not found
```bash
docker compose -f docker-compose.evals.yml run --rm evals \
  -c "python -c 'import spacy; nlp = spacy.load(\"en_core_sci_sm\"); print(\"OK\")'"
```

### Validation fails
```bash
docker compose -f docker-compose.evals.yml run --rm evals \
  python src/validate_traces.py --traces data/traces/ --verbose
```

### Trace parsing errors
Check `TraceParsingStats` in validation output:
- `missing_event_subtype > 0` - Some chunks used fallback logic
- `excluded_orchestrator_summary > 0` - Orchestrator records excluded (expected)

### MAC submodule not found
The MAC submodule is a fork with streaming instrumentation. Initialize it with:
```bash
git submodule update --init --recursive
```
Verify the fork remote:
```bash
cd third_party/mac && git remote -v
# Should show: https://github.com/AbemKW/mac-streaming-traces
```

---

## License

MIT - See repository root for full license.

---

## Version History

- **v3.2** - Schema-robust trace parsing, fail-fast validation, faithfulness paradox fix
- **v3.1** - Multi-record JSONL run logs, CUI extraction with ordering
- **v3.0** - Initial v3.x architecture with deterministic replay
