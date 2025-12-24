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

EXAID uses a forked MAC repository with token-level streaming instrumentation:

- Fork URL: https://github.com/AbemKW/mac-streaming-traces
- Path: `third_party/mac`
- Purpose: Capture per-token emission timestamps (`t_emitted_ms`) for realistic streaming replay

**Invariant:** This fork only adds transparent token-level timing instrumentation. All MAC conversation logic, agent orchestration, speaker selection, and termination conditions remain unchanged from the original implementation.

The submodule is pinned to a specific commit for reproducibility. Initialize with:

```bash
git submodule update --init --recursive
```

**Important:** If the MAC submodule commit is updated, traces MUST be regenerated before running any evaluation. Results obtained from mismatched traces and submodule commits are invalid.

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

### Phase 0: Validate Traces + Generate Stoplists

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

### Phase 1: Generate Traces (if not already frozen)

```bash
docker compose -f docker-compose.evals.yml run --rm \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  evals scripts/01_make_traces.sh
```

### Phase 2: Validate Schemas

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/00_validate.sh
```

### Phase 3: Run Summarizer Variants

```bash
# Run all variants (V0-V4)
docker compose -f docker-compose.evals.yml run --rm evals scripts/02_run_variants.sh

# Run specific variant
docker compose -f docker-compose.evals.yml run --rm evals scripts/02_run_variants.sh V3

# Or use Python directly with more options
docker compose -f docker-compose.evals.yml run --rm evals \
  python src/run_variants.py --traces data/traces/ --output data/runs/ --verbose
```

### Phase 4-5: Compute Metrics

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

1. **Load Phase 4 config** - Same extractor settings as metrics computation (drift-proof)
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
│   ├── extractor.yaml             # Concept extractor config (Phase 4)
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
│   ├── cases/                     # Input clinical cases
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
│       └── figures/               # Generated figures (Phase 6)
├── src/                           # Python modules
│   ├── trace_text.py              # Canonical trace text (single source)
│   ├── validate_traces.py         # Phase 0 validation
│   ├── generate_stoplists.py      # Phase 0 stoplist generation
│   ├── deterministic_utils.py     # Timestamps, IDs, CTU
│   ├── deterministic_io.py        # gzip/JSON writing
│   ├── concept_extractor.py       # CUI extraction
│   ├── validate_logs.py           # Schema validation
│   ├── make_traces.py             # Trace generation (MAC integration)
│   ├── run_variants.py            # Variant replay engine
│   ├── compute_metrics.py         # M1-M10 metrics
│   └── test_concept_extractor.py  # Unit tests
└── scripts/                       # Orchestration scripts
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
