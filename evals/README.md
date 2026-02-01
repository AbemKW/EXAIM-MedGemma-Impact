Evaluation Harness (Docker-first)

This folder contains the evaluation harness used to run performance-only experiments for this submission. The artifacts and scripts here measure implementation and artifact-level metrics (throughput, latency, per-case metrics, coverage, etc.). This directory and its outputs are for performance evaluation only; no clinical outcome claims are made.

Directory map (only the important paths)
- evals/data/runs_baseline_gemini25flashlite/{V0..V4, run_summaries}
- evals/data/metrics_baseline_gemini25flashlite/{aggregate.metrics.json, per_case.metrics.jsonl, metrics.provenance.json, figures/}
- evals/data/runs_medgemma15_4b_it
- evals/data/runs_medgemma27b_text_it
- evals/data/metrics_medgemma15_4b_it
- evals/data/metrics_medgemma27b_text_it
- Protected/default dirs (do not write): evals/data/runs, evals/data/metrics
- Safe local wrappers: evals/scripts/run_variants_safe.ps1, evals/scripts/compute_metrics_safe.ps1

Docker (recommended)
- Build the evaluation image:

```bash
docker compose -f docker-compose.evals.yml build evals
```

- Validate the environment (runs the repository validation script):

```bash
docker compose -f docker-compose.evals.yml run --rm evals scripts/validate.sh
```

- Recompute baseline metrics into a NEW directory (do NOT overwrite the provided baseline):

```bash
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.compute_metrics \
  --runs data/runs_baseline_gemini25flashlite \
  --output data/metrics_baseline_gemini25flashlite_recomputed
```

Notes:
- Inside the container the working directory is `evals/`; paths referenced above are relative to that (for example `data/...`).
- Always write recomputed outputs to clearly named, new directories (e.g., `*_recomputed`) so provided baseline artifacts remain unchanged.

Local Python (optional)
- For local/host runs on Windows PowerShell use the safe wrappers (they prevent accidental writes to protected locations):

```powershell
.\evals\scripts\run_variants_safe.ps1 -OutputDir evals\data\runs_medgemma15_4b_it_recomputed -EvalRunId medgemma15_4b_test
.\evals\scripts\compute_metrics_safe.ps1 -RunsDir evals\data\runs_medgemma15_4b_it_recomputed -OutputDir evals\data\metrics_medgemma15_4b_it_recomputed
```

- If you run Python directly, be sure to pass explicit `--output`/`--runs` paths that do not target `evals/data/runs` or `evals/data/metrics`.

Where to go next
- Full reproduction, model setup, and environment-variable details live in the repository root README: [README.md](../README.md).

This file is intentionally short and judge-facing. Full internal design notes, replay-engine details, calibration methodology, metric derivations, and development artifacts have been removed from this file and are preserved elsewhere in the repository history or internal documents.
# EXAIM Evaluation Quickstart (Docker-first)

This document is a concise, Docker-first quickstart for running the evaluation pipelines in this repository. The provided Docker Compose configuration (`docker-compose.evals.yml`) and `Dockerfile.evals` are the official, pinned evaluation environment intended for judges and reproducibility.

## Purpose and Claims Boundary

**Performance-only evaluation of EXAIM summarization middleware.**

This evaluation measures:
- Update counts
- Output volume (CTU)
- Concept coverage (trace CUIs recalled in summaries)
- Faithfulness (unsupported content rates)
- Latency and resource usage

**No clinical outcome claims are made.**

**System name:** EXAIM (paper). Legacy evaluation artifacts retain the `exaid.*` namespace (schemas/manifests/IDs) to preserve reproducibility of completed experiments. Run ID format: `eval-<trace_dataset_hash_8>-<exaid_commit_8>` where `exaid_commit_8` is the legacy field name for the EXAIM commit hash.

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- No host Python required (evaluation runs in container)

### MAC Submodule

EXAIM uses a forked MAC repository with delta/chunk-level streaming instrumentation:

- Fork URL: https://github.com/AbemKW/mac-streaming-traces
- Path: `third_party/mac`
- Purpose: Capture per-delta emission timestamps (`t_emitted_ms`) for realistic streaming replay

**Invariant:** This fork only adds transparent delta/chunk-level timing instrumentation. All MAC conversation logic, agent orchestration, speaker selection, and termination conditions remain unchanged from the original implementation.

## Why Docker (recommended)

Docker Compose + the `Dockerfile.evals` are the official evaluation path. Using Docker ensures pinned dependencies, a reproducible Linux runtime, and isolation from host Python environments so judges can run the same image and get consistent results. The repository's `docker-compose.evals.yml` exposes a single `evals` service (the official entrypoint) and uses `Dockerfile.evals` as its build recipe.

Key benefits:
- Deterministic, pinned dependencies (installed in the image)
- Easy, single-file commands for judges (`docker compose -f docker-compose.evals.yml run --rm evals ...`)
- Avoids local setup drift and dependency conflicts on reviewer machines

Note: Docker is recommended; a Local Python fallback is provided below for convenience and debugging.

## Directory map (relevant artifact paths)

- `evals/data/traces`
- `evals/data/runs_baseline_gemini25flashlite`
- `evals/data/metrics_baseline_gemini25flashlite`
- `evals/data/runs_medgemma15_4b_it`
- `evals/data/runs_medgemma27b_text_it`
- `evals/data/metrics_medgemma15_4b_it`
- `evals/data/metrics_medgemma27b_text_it`

These are the only directories judges need to inspect or read. Baseline folders are provided as shipped artifacts.

## Baseline (provided)

Baseline runs and metrics are included in the repository as read-only artifacts. Inspect the primary metric files:

- Aggregate metrics: `evals/data/metrics_baseline_gemini25flashlite/aggregate.metrics.json`
- Per-case metrics: `evals/data/metrics_baseline_gemini25flashlite/per_case.metrics.jsonl`
- Provenance: `evals/data/metrics_baseline_gemini25flashlite/metrics.provenance.json`

If you want to verify the metric computation yourself, recompute metrics with Docker and write results to a NEW directory (do not overwrite baseline):

# From PowerShell (project root)
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.compute_metrics --runs data/runs_baseline_gemini25flashlite --output data/metrics_baseline_recomputed

Or, using the host-safe wrapper (PowerShell):

.\evals\scripts\compute_metrics_safe.ps1 -RunsDir evals\data\runs_baseline_gemini25flashlite -OutputDir evals\data\metrics_baseline_recomputed

The safe wrapper enforces explicit output paths and prevents accidental overwrites of protected locations.

## Run MedGemma evaluation (Docker — primary)

This submission evaluated two MedGemma configurations. The evaluation pipeline is deterministic: the trace replay is fixed and only the internal agents (BufferAgent and SummarizerAgent) are switched to MedGemma variants. Upstream traces are replayed unchanged.

We recommend running the pipeline in two stages: (A) generate runs (variant replay), (B) compute metrics for those runs. The Docker commands below are PowerShell-friendly and use the repository's `docker-compose.evals.yml` and the `evals` service.

Model variant A: `google/medgemma-1.5-4b-it`

# 1) Generate runs (write to a NEW output directory)
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.run_variants --output data/runs_medgemma15_4b_it_recomputed --eval-run-id medgemma15_4b_test

# 2) Compute metrics for those runs (write to a NEW metrics directory)
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.compute_metrics --runs data/runs_medgemma15_4b_it_recomputed --output data/metrics_medgemma15_4b_it_recomputed

Model variant B: `google/medgemma-27b-text-it`

# 1) Generate runs
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.run_variants --output data/runs_medgemma27b_text_it_recomputed --eval-run-id medgemma27b_text_it_test

# 2) Compute metrics
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.compute_metrics --runs data/runs_medgemma27b_text_it_recomputed --output data/metrics_medgemma27b_text_it_recomputed

Important notes:
- The above `run_variants` invocation replays the frozen traces; only internal agents (BufferAgent + SummarizerAgent) are configured to use the MedGemma model. The trace dataset itself is unchanged.
- Prefer `_recomputed` output directories to avoid overwriting provided artifacts. The included PowerShell wrappers (`evals/scripts/run_variants_safe.ps1` and `evals/scripts/compute_metrics_safe.ps1`) enforce safe output directory choices when run on the host.

### Model setup (if you must provide your own model endpoints)

This repository uses role-based environment variables to select providers and model names. The following variables are recognized and can be set in the host environment before running Docker Compose. They will be visible to the container at runtime if exported in your shell.

- `SUMMARIZER_LLM_PROVIDER` (e.g., `google`, `openai`, `groq`)
- `BUFFER_AGENT_LLM_PROVIDER`
- `SUMMARIZER_LLM_MODEL` (model identifier override)
- `BUFFER_AGENT_LLM_MODEL`
- `OPENAI_BASE_URL` (optional base URL for OpenAI-compatible endpoints)

Example (PowerShell) — configure environment then run the generation step:

$env:SUMMARIZER_LLM_PROVIDER = "google"
$env:BUFFER_AGENT_LLM_PROVIDER = "google"
$env:SUMMARIZER_LLM_MODEL = "google/medgemma-1.5-4b-it"
$env:BUFFER_AGENT_LLM_MODEL = "google/medgemma-1.5-4b-it"
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.run_variants --output data/runs_medgemma15_4b_it_recomputed --eval-run-id medgemma15_4b_test

If you use an externally hosted OpenAI-compatible endpoint, set `OPENAI_BASE_URL` appropriately before invoking the container. The code reads that environment variable when constructing OpenAI-compatible clients.

## Local Python fallback (optional)

If Docker is unavailable, the same two-stage workflow can be executed locally using the provided PowerShell safe wrappers. These wrappers enforce explicit output directories and will refuse to write to protected paths.

# 1) Generate runs (host PowerShell)
.\evals\scripts\run_variants_safe.ps1 -OutputDir evals\data\runs_medgemma15_4b_it_recomputed -EvalRunId medgemma15_4b_test

# 2) Compute metrics (host PowerShell)
.\evals\scripts\compute_metrics_safe.ps1 -RunsDir evals\data\runs_medgemma15_4b_it_recomputed -OutputDir evals\data\metrics_medgemma15_4b_it_recomputed

The equivalent direct-Python commands (run inside an activated venv) are:

python -m evals.cli.run_variants --output evals/data/runs_medgemma15_4b_it_recomputed --eval-run-id medgemma15_4b_test
python -m evals.cli.compute_metrics --runs evals/data/runs_medgemma15_4b_it_recomputed --output evals/data/metrics_medgemma15_4b_it_recomputed

## Common pitfalls

- Do NOT write to `evals/data/runs` or `evals/data/metrics` — these default locations are protected and the safe wrappers will block attempts to write there.
- Baseline folders such as `evals/data/runs_baseline_gemini25flashlite` are provided artifacts; judges are not expected to regenerate the Gemini baseline outputs.
- Always specify explicit `--output` / `--runs` paths (or use the safe wrappers) to avoid accidental overwrites.
- If Docker cannot reach a host model server on Windows or macOS, use `host.docker.internal` in the endpoint configuration. On Linux, either configure an extra host mapping when invoking the container (for example: `docker run --add-host=host.docker.internal:host-gateway`) or run the model server in a container on the same Docker network.

If anything here is unclear or you want me to add a tiny verification script that prints the baseline artifact checksums, I can add that next.
