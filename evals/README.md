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

### Step 1: Generate Traces (STUB)

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/01_make_traces.sh
```

**Note:** This currently generates stub traces. MAC integration will replace the trace generator.

### Step 2: Run Summarizer Variants (STUB)

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/02_run_variants.sh
```

Runs all variants (V0-V4) on the generated traces. To run a specific variant:

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/02_run_variants.sh V3
```

### Step 3: Compute Metrics (STUB)

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/03_compute_metrics.sh
```

Generates:
- `data/metrics/*.jsonl` - Metric records for each variant
- `data/metrics/figures/coverage_vs_budget.pdf` - Main paper figure

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
│   ├── dataset.yaml               # Dataset selection
│   ├── mas_generation.yaml        # MAS trace generation
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
│   ├── make_traces.py             # Trace generation (stub)
│   ├── run_variants.py            # Variant runner (stub)
│   └── compute_metrics.py         # Metrics computation (stub)
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
- `budget_list: [250, 500, 1000, 2000]` - Token budget levels

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

## MAC Integration

The current trace generator (`src/make_traces.py`) is a **stub implementation**. 

When MAC (Multi-Agent Collaboration) is integrated:

1. Add MAC as a git submodule or dependency
2. Update `configs/mas_generation.yaml`:
   ```yaml
   mode: "mac_integration"
   mac:
     enabled: true
     module_path: "path/to/mac"
   ```
3. Modify `src/make_traces.py` to call MAC for trace generation
4. The output format (exaid.trace.schema.json) remains unchanged

## Docker Details

### Container Environment
- Python 3.12
- spaCy 3.7.4 with scispaCy 0.5.4
- en_core_sci_sm model pre-installed
- All dependencies pinned in `requirements-evals.txt`

**Note on scispaCy installation:** We install scispaCy with dependency resolution disabled (`--no-deps`) to ensure compatibility with Python 3.12; concept extraction uses only pretrained NER models and does not rely on SciPy internals beyond array operations.

### Running Commands Manually

```bash
# Enter interactive shell
docker compose -f docker-compose.evals.yml run --rm evals

# Run Python directly
docker compose -f docker-compose.evals.yml run --rm evals -c "python src/validate_logs.py --help"

# Run with environment variables
docker compose -f docker-compose.evals.yml run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  evals scripts/02_run_variants.sh
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

## License

MIT - See repository root for full license.


