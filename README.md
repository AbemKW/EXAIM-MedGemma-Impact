
# EXAIM – MedGemma Impact Challenge Submission

EXAIM (Explainable AI Middleware) is an experimental system that captures timed multi-agent reasoning traces, buffers them, and produces concise structured summaries via role-based LLMs. This repository contains the EXAIM code, reproducible evaluation tooling, and pre-computed baseline artifacts used for the MedGemma Impact Challenge submission. The evaluation harness is Docker-first to ensure deterministic, pinned environments for judges and reproducibility.

## Submodules

This project uses a git submodule for the MAC (Multi-Agent Conversation) trace generator:

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/AbemKW/ExAIM.git

# Or if already cloned
git submodule update --init --recursive
```

The MAC submodule (`third_party/mac`) is a fork of the original MAC framework:

- Fork: https://github.com/AbemKW/mac-streaming-traces
- Purpose: Enables per-delta `t_emitted_ms` capture for realistic streaming replay

**Invariant:** This fork only adds transparent delta/chunk-level timing instrumentation. All MAC conversation logic, agent orchestration, speaker selection, and termination conditions remain unchanged from the original implementation.

## Evaluation Data

The `evals/` directory contains pre-generated timed traces for reproducible evaluation:

- **Traces**: `evals/data/traces/*.trace.jsonl.gz` - Timed multi-agent conversation traces
- **Manifests**: `evals/data/manifests/*.manifest.jsonl` - Provenance and integrity metadata
- **Case lists**: `evals/data/manifests/*.case_list.jsonl` - Selected case IDs

**Data provenance**: Traces derive from [MAC's public rare-disease dataset](https://github.com/microsoft/MAC) (CC BY 4.0). No PHI - safe to redistribute.

**System name:** EXAIM (paper). Legacy evaluation artifacts retain the `exaid.*` namespace (schemas/manifests/IDs) to preserve reproducibility of completed experiments.

See `evals/README.md` for evaluation quickstarts (Docker-first) and exact evaluation commands.

## Quickstart (Docker – recommended)

This repository ships a dedicated evaluation image and a Docker Compose file tuned for reproducible evaluation runs. Use Docker Compose as the primary evaluation path to ensure pinned dependencies and a deterministic environment.

---
# EXAIM — MedGemma Impact Challenge Submission

EXAIM ingests timed, multi-agent reasoning traces and produces concise, schema-constrained summaries. This repository contains the EXAIM implementation, a reproducible evaluation harness, and committed evaluation artifacts used for the MedGemma Impact Challenge.

Claims boundary
- Research prototype only; no claims about diagnostic correctness, clinical safety, or outcomes.
- Evaluation reports system proxies: output volume, coverage, faithfulness proxies, and latency.

What’s provided (no API keys required)
- Baseline runs (shipped): `evals/data/runs_baseline_gemini25flashlite/{V0..V4, run_summaries}`
- Baseline metrics (shipped): `evals/data/metrics_baseline_gemini25flashlite/{aggregate.metrics.json, per_case.metrics.jsonl, metrics.provenance.json, figures/}`
- Committed MedGemma outputs and metrics:
  - `evals/data/runs_medgemma15_4b_it`
  - `evals/data/runs_medgemma27b_text_it`
  - `evals/data/metrics_medgemma15_4b_it`
  - `evals/data/metrics_medgemma27b_text_it`

Note: baseline artifacts are provided as read-only artifacts; judges are NOT expected to regenerate baseline model outputs.

Quickstart

Option A — Docker (recommended)
- Build the evaluation image (one-time):

```powershell
docker compose -f docker-compose.evals.yml build
```

- Recompute baseline metrics (container workdir is `evals/`; use `data/...` paths):

```powershell
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.compute_metrics --runs data/runs_baseline_gemini25flashlite --output data/metrics_baseline_recomputed
```

- Run MedGemma evaluations (generate runs -> compute metrics). Write to NEW output dirs to avoid overwriting committed artifacts:

```powershell
# MedGemma 1.5-4b-it
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.run_variants --output data/runs_medgemma15_4b_it_recomputed --eval-run-id medgemma15_4b_test
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.compute_metrics --runs data/runs_medgemma15_4b_it_recomputed --output data/metrics_medgemma15_4b_it_recomputed

# MedGemma 27b-text-it
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.run_variants --output data/runs_medgemma27b_text_it_recomputed --eval-run-id medgemma27b_text_it_test
docker compose -f docker-compose.evals.yml run --rm evals python -m evals.cli.compute_metrics --runs data/runs_medgemma27b_text_it_recomputed --output data/metrics_medgemma27b_text_it_recomputed
```

Option B — Local Python (optional)
- Install deps and use the safe PowerShell wrappers (they prevent accidental overwrites):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Generate runs (safe wrapper)
.\evals\scripts\run_variants_safe.ps1 -OutputDir evals\data\runs_medgemma15_4b_it_recomputed -EvalRunId medgemma15_4b_test

# Compute metrics (safe wrapper)
.\evals\scripts\compute_metrics_safe.ps1 -RunsDir evals\data\runs_medgemma15_4b_it_recomputed -OutputDir evals\data\metrics_medgemma15_4b_it_recomputed
```

Paths and safety
- Inside the Docker container the working directory is `evals/`; use `data/...` when running inside the container.
- Locally, use `evals\data\...` (PowerShell) or `evals/data/...` (bash).
- Do NOT write to `evals/data/runs` or `evals/data/metrics`; the safe wrappers block those paths.

How to compare results
- Compare aggregate metrics JSON files:
  - Baseline: `evals/data/metrics_baseline_gemini25flashlite/aggregate.metrics.json`
  - MedGemma recomputed: `evals/data/metrics_medgemma15_4b_it_recomputed/aggregate.metrics.json` (or the committed `evals/data/metrics_medgemma15_4b_it/aggregate.metrics.json`)
- For per-case checks compare the `per_case.metrics.jsonl` files in the same folders.

Model setup (MedGemma)
- EXAIM supports role-based model overrides for internal agents (BufferAgent, SummarizerAgent).
- See `evals/README.md` for exact environment-variable usage and server details.
- To discover available override keys, search the repo for `BUFFER_AGENT_LLM_`, `SUMMARIZER_LLM_`, or `OPENAI_BASE_URL`.

Submodules
- The MAC trace generator is included as a pinned submodule at `third_party/mac`. Initialize with:

```bash
git submodule update --init --recursive
```

Repo structure (high-level)
- `exaim_core/` — core implementation (BufferAgent, SummarizerAgent, TokenGate)
- `evals/` — evaluation harness, CLI, configs, and committed artifacts
- `infra/` — model registry and configuration
- `third_party/mac/` — MAC trace generator submodule
- `demos/` — demo integrations (not required for evaluation)
- `tools/` — small helpers to inspect metrics and provenance

---

If you want an even shorter one-page TL;DR, tell me which sections to collapse.