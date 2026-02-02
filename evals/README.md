# Evaluation Harness

Performance-only evaluation for EXAIM. Measures throughput, latency, coverage, and faithfulness. No clinical outcome claims are made.

**Note:** For the MedGemma Impact Challenge submission, evaluations are executed in a Kaggle Notebook. See the **"Kaggle & Notebook Execution"** section in the [root README](../README.md#kaggle--notebook-execution).

## Prerequisites

- Docker and Docker Compose (for Docker-based runs)
- Python 3.10+ (for local runs)
- Model API credentials (environment variables)

## Key Directories

- `data/traces/` - Input trace dataset (provided)
- `data/runs_*/` - Model outputs (generated during evaluation)
- `data/metrics_*/` - Computed metrics for each variant (generated during evaluation)

## Docker Evaluation (Recommended)

Docker ensures pinned dependencies and a reproducible environment.

**Build image (one-time):**
```bash
docker compose -f docker-compose.evals.yml build
```

**Generate baseline and compute metrics:**
```bash
# Generate baseline runs
docker compose -f docker-compose.evals.yml run --rm evals \
  python -m evals.cli.run_variants \
  --output data/runs_baseline \
  --eval-run-id baseline_run

# Compute metrics
docker compose -f docker-compose.evals.yml run --rm evals \
  python -m evals.cli.compute_metrics \
  --runs data/runs_baseline \
  --output data/metrics_baseline
```

**Run MedGemma evaluations:**
```bash
# MedGemma 1.5-4b-it
docker compose -f docker-compose.evals.yml run --rm evals \
  python -m evals.cli.run_variants \
  --output data/runs_medgemma15_4b_it \
  --eval-run-id medgemma15_4b_run

docker compose -f docker-compose.evals.yml run --rm evals \
  python -m evals.cli.compute_metrics \
  --runs data/runs_medgemma15_4b_it \
  --output data/metrics_medgemma15_4b_it

# MedGemma 27b-text-it
docker compose -f docker-compose.evals.yml run --rm evals \
  python -m evals.cli.run_variants \
  --output data/runs_medgemma27b_text_it \
  --eval-run-id medgemma27b_text_it_run

docker compose -f docker-compose.evals.yml run --rm evals \
  python -m evals.cli.compute_metrics \
  --runs data/runs_medgemma27b_text_it \
  --output data/metrics_medgemma27b_text_it
```

**Important notes:**
- Container working directory is `evals/` - use `data/...` paths in commands
- Set environment variables before running (see Model Configuration below)

## Local Python Evaluation

**Setup:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Generate runs and compute metrics:**
```powershell
# Generate runs
python -m evals.cli.run_variants \
  --output evals/data/runs_medgemma15_4b_it \
  --eval-run-id medgemma15_4b_run

# Compute metrics
python -m evals.cli.compute_metrics \
  --runs evals/data/runs_medgemma15_4b_it \
  --output evals/data/metrics_medgemma15_4b_it
```

**Note:** When running locally, paths are relative to the repository root (use `evals/data/...`).

## Model Configuration

EXAIM uses role-based environment variables to configure LLM providers:

**Key environment variables:**
- `SUMMARIZER_LLM_PROVIDER` - Provider for SummarizerAgent (e.g., `google`, `openai`, `groq`)
- `BUFFER_AGENT_LLM_PROVIDER` - Provider for BufferAgent
- `SUMMARIZER_LLM_MODEL` - Model identifier override
- `BUFFER_AGENT_LLM_MODEL` - Model identifier override
- `OPENAI_BASE_URL` - Base URL for OpenAI-compatible endpoints
- `OPENAI_API_KEY` - API key for OpenAI provider

**Example (PowerShell):**
```powershell
$env:SUMMARIZER_LLM_PROVIDER = "google"
$env:BUFFER_AGENT_LLM_PROVIDER = "google"
$env:SUMMARIZER_LLM_MODEL = "google/medgemma-1.5-4b-it"
$env:BUFFER_AGENT_LLM_MODEL = "google/medgemma-1.5-4b-it"

docker compose -f docker-compose.evals.yml run --rm evals \
  python -m evals.cli.run_variants \
  --output data/runs_medgemma15_4b_it \
  --eval-run-id medgemma15_4b_run
```

For OpenAI-compatible endpoints:
```powershell
$env:SUMMARIZER_LLM_PROVIDER = "openai"
$env:OPENAI_BASE_URL = "http://localhost:8000/v1"
$env:OPENAI_API_KEY = "EMPTY"
```

## Comparing Results

Inspect these files to compare model performance:
- `data/metrics_*/aggregate.metrics.json` - Summary metrics
- `data/metrics_*/per_case.metrics.jsonl` - Per-case breakdown

**Example comparison:**
```powershell
# Compare baseline vs MedGemma
$baseline = Get-Content evals/data/metrics_baseline/aggregate.metrics.json | ConvertFrom-Json
$medgemma = Get-Content evals/data/metrics_medgemma15_4b_it/aggregate.metrics.json | ConvertFrom-Json

Inspect these files to compare model performance:
- `data/metrics_*/aggregate.metrics.json` - Summary metrics
- `data/metrics_*/per_case.metrics.jsonl` - Per-case breakdown