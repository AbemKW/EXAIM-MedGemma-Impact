# EXAID Evaluation Schemas

JSON Schema definitions for EXAID evaluation data artifacts.

## Schema Overview

| Schema | Version | Purpose |
|--------|---------|---------|
| `exaid.manifest.schema.json` | 2.0.0 | Dataset manifest with case selection |
| `exaid.trace.schema.json` | 2.0.0 | Frozen MAS traces |
| `exaid.run.schema.json` | 1.5.0 | Multi-record run logs |
| `exaid.metrics.schema.json` | 2.2.0 | Computed evaluation metrics |

All schemas use JSON Schema Draft 2020-12.

---

## Run Log Schema v1.5.0

**Paper hook: Section 3.2**

The run log schema defines a **multi-record JSONL format** where each line is one of:

### Record Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| `run_meta` | Provenance and configuration | First record in file |
| `tokengate_flush` | TokenGate accumulation event | Flush timing and CTU |
| `buffer_decision` | BufferAgent decision | M9 overhead tracking |
| `summary_event` | Summary generation | M6a/M6b computation |

### Record Ordering

Records are written in deterministic order:
1. `run_meta` (exactly one)
2. `tokengate_flush` records (sorted by `flush_index`)
3. `buffer_decision` records (sorted by `decision_index`)
4. `summary_event` records (sorted by `event_index`)

---

## run_meta Record

First record in every run log file.

```json
{
  "record_type": "run_meta",
  "schema_name": "exaid.run",
  "schema_version": "1.5.0",
  "created_at": "2025-12-21T10:30:00.000Z",
  "case_id": "case-33651373",
  "variant_id": "V0",
  "mas_run_id": "mas_1d5b227a_gpt4omini_0ad5f2b4_d22060bf",
  "eval_run_id": "eval-V0-b2c3d4e5-8d45cbb1",
  "history_k": 3,
  "trigger_policy": "full_exaid",
  
  "concept_extractor": {
    "spacy_version": "3.7.4",
    "scispacy_version": "0.5.4",
    "scispacy_model": "en_core_sci_sm",
    "linker_name": "umls",
    "linker_kb_version": "2023AB",
    "linker_resolve_abbreviations": true,
    "linker_max_entities_per_mention": 10,
    "linker_threshold": 0.7,
    "cui_score_threshold": 0.7,
    "max_k": 10,
    "min_entity_len": 3,
    "concept_representation": "cui",
    "cui_normalization": "uppercase"
  },
  
  "stoplists_provenance": {
    "stop_entities_hash": "sha256:abc123...",
    "stop_cuis_hash": "sha256:def456...",
    "stoplist_df_report_hash": "sha256:ghi789...",
    "stoplists_generated_at": "2025-12-21T10:00:00Z"
  },
  
  "timestamp_derivation": {
    "method": "trace_t_emitted_ms",
    "fallback_method": "run_start_time_plus_seq",
    "derived_from_t_emitted": 45,
    "derived_from_fallback": 0,
    "missing_t_emitted_ms_count": 0
  },
  
  "determinism": {
    "gzip_mtime": 0,
    "json_sort_keys": true,
    "json_separators": [",", ":"]
  }
}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `case_id` | Clinical case identifier |
| `variant_id` | V0-V4 variant |
| `mas_run_id` | Trace generation campaign ID |
| `eval_run_id` | Evaluation batch ID |
| `history_k` | Summary history context size |
| `concept_extractor` | Full extractor configuration |
| `stoplists_provenance` | Stoplist file hashes |
| `timestamp_derivation` | How timestamps were derived |
| `determinism` | Serialization settings |

### Provenance Fields

**Paper hook: Section 6.1**

`stoplists_provenance` must be in a **single canonical location** (not duplicated):
- `stop_entities_hash` - SHA256 of surface stoplist
- `stop_cuis_hash` - SHA256 of CUI stoplist
- `stoplists_generated_at` - ISO timestamp

`concept_extractor.linker_kb_version` tracks the UMLS snapshot used (e.g., "2023AB").

---

## summary_event Record

One record per generated summary.

```json
{
  "record_type": "summary_event",
  "event_index": 0,
  "event_id": "case-33651373-V0-se-000",
  "case_id": "case-33651373",
  "variant_id": "V0",
  "timestamp": "2025-12-21T10:30:01.234Z",
  "start_seq": 0,
  "end_seq": 15,
  "schema_ok": true,
  "schema_error": null,
  "summary_semantics_text": "Patient presents with...",
  "summary_ctu": 45,
  "summary_content": {
    "status_action": "...",
    "key_findings": "...",
    "differential_rationale": "...",
    "uncertainty_confidence": "...",
    "recommendation_next_step": "...",
    "agent_contributions": "..."
  },
  "latest_summary_event_id": null,
  "new_buffer_text_hash": "sha256:...",
  "llm_usage": {
    "prompt_ctu": 120,
    "completion_ctu": 45,
    "provider_prompt_tokens": 115,
    "provider_completion_tokens": 42,
    "model_id": "gpt-4o-mini"
  },
  "latency_ms": 1250
}
```

`summary_semantics_text` concatenates clinician-facing fields for concept extraction and **excludes** `agent_contributions` to avoid inflating semantic metrics.

### M6b Reconstruction

**Paper hook: Section 5.1**

To compute M6b (contract-groundedness), you need:

1. **Window text** - Reconstructed from trace using `[start_seq, end_seq]`
2. **Latest summary text** - Looked up via `latest_summary_event_id`

```python
# M6b support reconstruction
support_text = window_text
if event.get("latest_summary_event_id"):
    latest_event = run_events[event["latest_summary_event_id"]]
    support_text += " " + latest_event.get("summary_semantics_text", "")
support_cuis = extractor.extract(support_text)
```

### Event ID Format

Deterministic, collision-free:
```
{case_id}-{variant_id}-se-{event_index:03d}
```

Example: `case-33651373-V0-se-000`

---

## buffer_decision Record

One record per BufferAgent decision.

```json
{
  "record_type": "buffer_decision",
  "decision_index": 0,
  "decision_id": "case-33651373-V0-bd-000",
  "case_id": "case-33651373",
  "variant_id": "V0",
  "timestamp": "2025-12-21T10:30:00.500Z",
  "start_seq": 0,
  "end_seq": 10,
  "input_ctu": 35,
  "decision": "summarize",
  "filter_results": {
    "completeness_passed": true,
    "value_passed": true,
    "novelty_passed": true
  },
  "llm_usage": {
    "prompt_ctu": 35,
    "completion_ctu": 5,
    "provider_prompt_tokens": null,
    "provider_completion_tokens": null,
    "model_id": "gpt-4o-mini"
  },
  "latency_ms": 500
}
```

### M9 Overhead Tracking

**Paper hook: Section 5.1**

M9 metrics computed from buffer_decision records:
- `buffer_decision_count` - Total decisions
- `buffer_decision_ctu` - Sum of `input_ctu`
- Decision breakdown by type (`summarize`, `buffer`, `discard`)

### Filter Results

For V4 (no novelty check):
```json
"filter_results": {
  "completeness_passed": true,
  "value_passed": true,
  "novelty_passed": null  // DISABLED
}
```

---

## tokengate_flush Record

One record per TokenGate flush event. Note: TokenGate accumulates text and flushes based on whitespace-delimited word counts (not model tokenizer tokens).

```json
{
  "record_type": "tokengate_flush",
  "flush_index": 0,
  "case_id": "case-33651373",
  "variant_id": "V0",
  "timestamp": "2025-12-21T10:30:00.250Z",
  "start_seq": 0,
  "end_seq": 10,
  "accumulated_ctu": 45,
  "trigger_reason": "max_words",
  "text_hash": "sha256:..."
}
```

### Trigger Reasons

| Reason | Description |
|--------|-------------|
| `max_words` | Accumulated word count exceeded max_words threshold |
| `boundary_cue` | Sentence boundary detected (after min_words reached) |
| `silence_timer` | Inactivity timeout (no tokens received for silence_timer seconds) |
| `max_wait_timeout` | Maximum wait timeout exceeded (buffer existed for max_wait_timeout seconds) |
| `end_of_trace` | Trace ended, flush remaining buffer |
| `turn_end` | Turn boundary (V1 only) |

---

## CTU vs Provider Tokens

**Paper hook: Section 6.1**

The schema separates deterministic CTU from provider-reported tokens:

| Field | Description | Used In Metrics |
|-------|-------------|-----------------|
| `prompt_ctu` | `ceil(len(prompt_text) / 4)` | Yes (M8) |
| `completion_ctu` | `ceil(len(completion_text) / 4)` | Yes (M8) |
| `provider_prompt_tokens` | API-reported (may be null) | No (logging only) |
| `provider_completion_tokens` | API-reported (may be null) | No (logging only) |

---

## Schema Failure Handling

**Paper hook: Section 5.1**

### Faithfulness Metrics (M6a/M6b)

| `schema_ok` | `summary_semantics_text` | Result |
|-------------|--------------------------|--------|
| `false` | Any | **EXCLUDE** from means |
| `true` | Empty | Include as **0.0** |
| `true` | Non-empty | Compute normally |

### Coverage Metrics (M4/M5)

Schema-failed events contribute **empty concept sets**, penalizing recall.

### M10 Schema Failure Rate

```
M10 = schema_failures / summary_count
```

---

## Validation

Validate run logs against schema:

```bash
# Using jsonschema
jsonschema --instance data/runs/V0/case-33651373.jsonl.gz \
           schemas/exaid.run.schema.json

# Or using the validation script
python -m evals.cli.validate_logs data/runs/V0/*.jsonl.gz --schema schemas/exaid.run.schema.json
```

---

## Invariants

1. **One run_meta per file** - Always first record
2. **Monotonic indices** - event_index, decision_index, flush_index are strictly increasing
3. **Deterministic event IDs** - `{case_id}-{variant_id}-{type}-{index:03d}`
4. **No duplicate hashes** - `stoplists_provenance` in run_meta only
5. **Timestamps from trace** - Derived from `t_emitted_ms`, not wall clock

---

## Migration Notes

### v1.2.0 → v1.3.0

- Added `stoplists_provenance` to run_meta
- Added `timestamp_derivation` stats
- Added `linker_kb_version` to concept_extractor
- Changed from single-record to multi-record JSONL

### v1.3.0 → v1.5.0

- Added `trace_file_hash`, `trace_dataset_hash`, `tokengate_config_hash` to run_meta
- Added `trigger_type`, `summary_history_event_ids`, `summarizer_input_hash`, `limits_ok`, `failure_mode` to summary_event
- Enhanced provenance tracking

### Backward Compatibility

Readers should check `schema_version` and handle:
- Single-record format (v1.2.x and earlier)
- Multi-record JSONL format (v1.3.0+)
