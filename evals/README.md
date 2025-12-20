# EXAID Evaluation Module

Reproducible evaluation framework for the EXAID conference paper. This module provides Docker-based reproducibility for all evaluation experiments.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- No host Python required (evaluation runs in container)

### Build and Run

```bash
# Build the evaluation container
docker compose -f docker-compose.evals.yml build

# Verify the build
docker compose -f docker-compose.evals.yml run --rm evals -c "python --version && spacy validate"

# Run validation (passes with empty directories in scaffold mode)
docker compose -f docker-compose.evals.yml run --rm evals scripts/00_validate.sh
```

## How to Reproduce Tables/Figures

Execute the following commands in order:

### Step 0: Validate Schemas and Data

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/00_validate.sh
```

This validates all JSONL files against their schemas. In scaffold mode (empty data directories), this will pass with a "scaffold mode" message.

### Step 1: Generate Traces

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/01_make_traces.sh
```

Generates MAS traces using MAC as the upstream trace generator. See [MAC Trace Generation](#mac-trace-generation) for details.

### Step 2: Run Summarizer Variants

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/02_run_variants.sh
```

Runs all variants (V0-V4) on the generated traces. To run a specific variant:

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/02_run_variants.sh V3
```

### Step 3: Compute Metrics

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/03_compute_metrics.sh
```

Generates:
- `data/metrics/*.jsonl` - Metric records for each variant
- `data/metrics/figures/coverage_vs_budget.pdf` - Main paper figure

---

## MAC Trace Generation

This section documents the integration of MAC (Multi-Agent Conversation) as an upstream trace generator for EXAID evaluation. This documentation is intended to be detailed enough for use in the Methods section of the paper.

### Role of MAC

MAC is used exclusively as an **upstream trace generator**. EXAID performs **instrumentation-only** integration:

- MAC generates multi-agent diagnostic conversations from clinical cases
- EXAID captures agent message emissions as frozen traces
- EXAID variants replay these frozen traces for middleware evaluation
- **No diagnostic accuracy evaluation is performed by EXAID**

This separation ensures that:
1. MAC's reasoning, prompts, routing, and termination behavior are preserved exactly as released by the authors
2. EXAID evaluates only its own summarization middleware, not MAC's diagnostic capability
3. Results are reproducible using frozen trace files

**Runtime Configuration:** MAC is executed using a model configuration supplied by EXAID at runtime (`evals/configs/mac_model_config.json`). The MAC submodule itself is unmodified and pinned to the authors' released commit.

### Version Pinning and Decoding Parameters

MAC is integrated as a pinned git submodule at `third_party/mac/`:

| Parameter | Value |
|-----------|-------|
| Repository | `https://github.com/geteff1/Multi-agent-conversation-for-disease-diagnosis` |
| Commit | `896a5deb4d6db7a2c872630a6638da4da3b0f4d4` |
| Base Model | `gpt-4o-mini` (OpenAI) |

MAC traces are generated using OpenAI's `gpt-4o-mini` model. Decoding parameters (temperature, sampling) are controlled internally by MAC and are not overridden by EXAID. CTU remains the evaluation unit for all metrics, ensuring vendor-agnostic and reproducible measurement.

**Decoding Parameters:**

MAC was executed using the authors' released decoding parameters. **EXAID does not override agent-level temperature or sampling behavior.** Decoding parameters (temperature, top-p, etc.) are controlled internally by MAC per agent role and are not modified by the EXAID integration.

This is recorded in `configs/mas_generation.yaml`:

```yaml
mac:
  decoding:
    max_tokens: 4096  # Logged for documentation only
    note: "Decoding parameters (temperature, sampling) are controlled by MAC internally and are not overridden by EXAID"
```
API credentials are supplied via environment variables at runtime. No credentials are stored in the repository.

### Case Selection Policy

MAC provides 302 clinical cases (primary and follow-up presentations). EXAID implements a configurable case selection policy defined in `configs/dataset.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `fixed_subset` | Selection mode (`fixed_subset` or `all`) |
| `seed` | `42` | Random seed for deterministic selection |
| `n_cases` | `40` | Number of cases to select (when `mode=fixed_subset`) |
| `on_failure` | `log_stub` | Failure handling (`log_stub` or `raise`) |

**Selection Process:**
1. All 302 case IDs are sorted lexicographically
2. A seeded random generator (`random.Random(seed)`) samples `n_cases` cases
3. Selected cases are sorted again for deterministic ordering
4. The ordered case list and its hash are recorded in the manifest

This policy is deterministic: the same seed always produces the same case selection.

**Rationale for Subset Selection:**
- MAC execution is computationally expensive (~5-10 minutes per case)
- 40 cases provide sufficient statistical power for middleware evaluation
- Full 302-case runs can be enabled by setting `mode: all`
- Selection methodology is fully disclosed and reproducible

### Token Accounting and Compression Metrics

All evaluation metrics use **Character-Normalized Token Units (CTU)**, defined as:

```
CTU(text) = ceil(len(text) / 4)
```

CTU is a **model-agnostic, character-normalized text unit**—not a tokenizer. The ≈4 characters per unit heuristic reflects average token densities observed across modern LLM tokenizers (OpenAI, LLaMA, Gemini, Claude) and is used strictly as a normalization constant. This ensures:

- **Offline computation**: CTU can be computed without API calls
- **Deterministic replay**: Same text always produces same CTU count
- **Reproducibility for reviewers**: No vendor-specific tokenizer dependencies
- **Vendor-agnostic**: Works identically regardless of LLM provider

**CTU is applied uniformly to both input trace text and EXAID summary outputs; all compression and budget metrics are computed using the same unit on both sides.**

For non-text or empty emissions, `text_units_ctu` is defined as `0`.

**Provider Token Counts:**

Provider-reported token counts (when available from any LLM provider) are logged separately in trace records as `provider_token_count`. These are recorded for **usage transparency only** and are **not used in any evaluation metric**. Evaluation metrics do not depend on provider tokenizers—CTU ensures reproducibility, offline computation, and vendor independence.

This separation is enforced in the trace schema (`schemas/exaid.trace.schema.json`):

```json
"text_units_ctu": {
  "type": "integer",
  "minimum": 0,
  "description": "Character-Normalized Token Units (CTU), defined as ceil(len(text)/4). Used for all evaluation metrics."
},
"provider_token_count": {
  "type": "integer", 
  "minimum": 0,
  "description": "Provider-reported token count (usage metadata only, not used in evaluation)"
}
```

### Trace Construction Rules

For each selected case, MAC generates a conversation trace stored as a JSONL file:

**File Format:**
- One JSONL.gz file per case: `data/traces/case-{id}.jsonl.gz`
- Each line is a JSON object conforming to `exaid.trace.schema.json`

**Record Structure:**
```json
{
  "schema_name": "exaid.trace",
  "schema_version": "1.0.0",
  "trace_id": "trc-{case_id}-{seq:03d}",
  "case_id": "case-{mac_case_url}",
  "agent_id": "doctor0|doctor1|doctor2|supervisor",
  "sequence_num": 0,
  "timestamp": "2024-12-19T12:00:00Z",
  "content": "Agent message text...",
  "text_units_ctu": 123,
  "metadata": {
    "mas_run_id": "mas-abc123...",
    "mac_commit": "896a5deb...",
    "model": "gpt-4o-mini",
    "status": "success"
  }
}
```

**Trace Properties:**
- `sequence_num` is strictly increasing (0, 1, 2, ...)
- No buffering or reordering of agent emissions
- One trace record per MAC message emission
- `text_units_ctu` computed via `ceil(len(content) / 4)`
- For non-text or empty emissions, `text_units_ctu` is `0`

### Include-All and Failure Handling

EXAID implements an **include-all execution policy** within the selected subset:

1. All selected cases are executed (no early stopping)
2. Failed cases produce stub traces with `status: failed`
3. No cases are silently skipped

**Failure Trace Format:**
```json
{
  "schema_name": "exaid.trace",
  "schema_version": "1.0.0",
  "trace_id": "trc-{case_id}-000",
  "case_id": "case-{case_id}",
  "agent_id": "_system",
  "sequence_num": 0,
  "timestamp": "...",
  "content": "[FAILED] Case execution failed",
  "text_units_ctu": 0,
  "metadata": {
    "status": "failed",
    "failure_reason": "Specific error message",
    "mas_run_id": "...",
    "mac_commit": "..."
  }
}
```

This ensures:
- Downstream metrics can account for failures
- Trace file counts match selected case counts
- No hidden data loss

### Freezing and Replay Semantics

The trace generation process follows a **frozen replay model**:

1. **Generation Phase**: MAC runs once to produce frozen traces
2. **Storage**: Traces are written to `data/traces/` as JSONL.gz files
3. **Replay Phase**: EXAID variants replay the same frozen traces
4. **Provenance**: `mas_run_id` ensures trace provenance is verifiable

**`mas_run_id` Generation:**

The run ID is a deterministic hash of:
- MAC commit hash
- Model name
- Ordered case list

```python
mas_run_id = f"mas-{sha256(payload)[:16]}"
```

This ID is recorded in:
- Each trace record's `metadata.mas_run_id`
- The dataset manifest (`data/manifests/dataset_manifest.jsonl`)

### Exact Reproduction Commands

To reproduce trace generation:

```bash
# 1. Build the evaluation container
docker compose -f docker-compose.evals.yml build

# 2. Generate traces (requires API credentials for MAC)
docker compose -f docker-compose.evals.yml run --rm \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  evals scripts/01_make_traces.sh

# 3. Validate generated traces
docker compose -f docker-compose.evals.yml run --rm evals scripts/00_validate.sh
```

**For testing without API calls (stub mode):**

```bash
docker compose -f docker-compose.evals.yml run --rm \
  -e EXAID_STUB_MODE=1 \
  evals scripts/01_make_traces.sh
```

### Dataset Manifest

Trace generation produces a manifest file at `data/manifests/dataset_manifest.jsonl`:

```json
{
  "schema_name": "exaid.manifest",
  "schema_version": "1.0.0",
  "experiment_id": "exp-mac-traces-20241219-120000",
  "created_at": "2024-12-19T12:00:00Z",
  "config_hash": "sha256:...",
  "mas_generation_config": {
    "mas_run_id": "mas-abc123...",
    "mac_commit": "896a5deb4d6db7a2c872630a6638da4da3b0f4d4",
    "base_model": "gpt-4o-mini",
    "decoding_note": "Controlled by MAC internally; EXAID does not override",
    "text_unit": {
      "name": "CTU",
      "definition": "ceil(len(text) / 4)",
      "applies_to": "input_and_output"
    },
    "selection_mode": "fixed_subset",
    "selection_seed": 42,
    "n_cases_selected": 40,
    "n_cases_available": 302,
    "case_list_hash": "sha256:...",
    "ordered_case_list": ["case-34775698", "case-34989141", ...]
  }
}
```

### Documentation Version Tracking

This documentation corresponds to EXAID commit: `7dadb9143892c045b6317e297aa651a79cf4bd14`

Future changes to MAC integration should reference changes made after this commit and update this section accordingly.

---

## Directory Structure

```
evals/
├── README.md                      # This file
├── requirements-evals.txt         # Pinned Python dependencies
├── schemas/                       # JSON Schema definitions
│   ├── exaid.manifest.schema.json
│   ├── exaid.trace.schema.json
│   ├── exaid.run.schema.json
│   └── exaid.metrics.schema.json
├── configs/                       # Configuration files
│   ├── dataset.yaml               # Dataset selection + MAC case policy
│   ├── mas_generation.yaml        # MAS trace generation (MAC config)
│   ├── summarizer.yaml            # Summarizer params (history_k=3)
│   ├── metrics.yaml               # Metric computation params
│   ├── stop_entities.txt          # scispaCy stop-entity list
│   └── variants/                  # Variant-specific configs
│       ├── V0.yaml                # Baseline (no summarization)
│       ├── V1.yaml                # Fixed threshold
│       ├── V2.yaml                # LLM-triggered
│       ├── V3.yaml                # Calibrated (primary)
│       └── V4.yaml                # Extended history
├── data/                          # Data artifacts
│   ├── manifests/                 # Experiment manifests
│   ├── cases/                     # Input clinical cases
│   ├── traces/                    # Generated MAS traces
│   ├── runs/                      # Summarizer run outputs
│   │   ├── V0/
│   │   ├── V1/
│   │   ├── V2/
│   │   ├── V3/
│   │   └── V4/
│   └── metrics/                   # Computed metrics
│       └── figures/               # Generated figures
├── src/                           # Python modules
│   ├── validate_logs.py           # Schema validation
│   ├── make_traces.py             # Trace generation (MAC integration)
│   ├── run_variants.py            # Variant runner
│   └── compute_metrics.py         # Metrics computation
└── scripts/                       # Orchestration scripts
    ├── 00_validate.sh
    ├── 01_make_traces.sh
    ├── 02_run_variants.sh
    └── 03_compute_metrics.sh
```

## Frozen Parameters

The following parameters are frozen for the conference paper evaluation:

### Summarizer Configuration (`configs/summarizer.yaml`)
- `history_k: 3` - Number of historical summaries in context

### Metric Configuration (`configs/metrics.yaml`)
- `tau_list: [0.85, 0.90, 0.95]` - Redundancy thresholds
- `budget_list: [250, 500, 1000, 2000]` - Token budget levels (in CTU)

### scispaCy Concept Extractor
- Model: `en_core_sci_sm`
- `entity_types_kept: ["ALL"]` (mention detector mode)
- `min_entity_len: 3`
- Stop entities: `configs/stop_entities.txt`
- Normalization: lowercase, strip whitespace, canonicalize punctuation

### V3 Variant (Primary Evaluation)
- Calibration: enabled (placeholder for calibration procedure)
- Chunk params: frozen (token_gate settings)
- Time params: frozen (timeout settings)

## Schema Versioning

All data files include schema identification for validation:

```json
{
  "schema_name": "exaid.trace",
  "schema_version": "1.0.0",
  ...
}
```

Schema files use JSON Schema Draft 2020-12.

## Docker Details

### Container Environment
- Python 3.12
- spaCy 3.7.4 with scispaCy 0.5.4
- pyautogen 0.2.32 (for MAC integration)
- en_core_sci_sm model pre-installed
- All dependencies pinned in `requirements-evals.txt`

The evaluation container is pinned to linux/amd64 to ensure compatibility across common reviewer and CI environments.

**Note on scispaCy installation:** We install scispaCy with dependency resolution disabled (`--no-deps`) to ensure compatibility with Python 3.12; concept extraction uses only pretrained NER models and does not rely on SciPy internals beyond array operations.

### Running Commands Manually

```bash
# Enter interactive shell
docker compose -f docker-compose.evals.yml run --rm evals

# Run Python directly
docker compose -f docker-compose.evals.yml run --rm evals -c "python src/validate_logs.py --help"

# Run with environment variables
docker compose -f docker-compose.evals.yml run --rm \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  evals scripts/01_make_traces.sh
```

## Troubleshooting

### Container build fails
```bash
# Rebuild without cache
docker compose -f docker-compose.evals.yml build --no-cache
```

### scispaCy model not found
```bash
# Verify model installation
docker compose -f docker-compose.evals.yml run --rm evals -c "python -c 'import spacy; nlp = spacy.load(\"en_core_sci_sm\"); print(\"OK\")'"
```

### Validation fails
```bash
# Run validation with verbose output
docker compose -f docker-compose.evals.yml run --rm evals -c "python src/validate_logs.py data/traces/*.jsonl.gz --verbose"
```

### MAC submodule not found
```bash
# Initialize submodules
git submodule update --init --recursive
```

## License

MIT - See repository root for full license.
