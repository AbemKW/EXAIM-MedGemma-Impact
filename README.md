
# EXAIM

EXAIM (Explainable AI Middleware) is an experimental Python project for capturing short, live traces from multiple agents, buffering those traces, and producing concise summaries using an LLM. It is designed as a minimal prototype for medical multi-agent reasoning workflows, where specialized agents (e.g., InfectiousDiseaseAgent, HematologyAgent, OncologyAgent) collaborate on clinical cases, and their reasoning traces are captured and condensed into structured summaries optimized for physician understanding.

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

See `evals/README.md` for full evaluation documentation.

## Quick Start

1. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\Activate.ps1
   # On Unix/Mac:
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   
   Create a `.env` file in the project root with your LLM provider settings:
   ```bash
   # Role-based provider selection (mas, summarizer, buffer_agent)
   MAS_LLM_PROVIDER=groq
   SUMMARIZER_LLM_PROVIDER=google
   BUFFER_AGENT_LLM_PROVIDER=google
   
   # Google Gemini configuration
   GOOGLE_API_KEY=your-google-api-key
   GOOGLE_MODEL_NAME=gemini-2.5-flash-lite
   # Override models per role (optional):
   SUMMARIZER_LLM_MODEL=gemini-2.5-pro
   BUFFER_AGENT_LLM_MODEL=gemini-2.5-pro
   
   # Groq configuration (for multi-agent reasoning)
   GROQ_API_KEY=your-groq-api-key
   GROQ_MODEL=llama-3.3-70b-versatile
   # Override model for MAS role (optional):
   MAS_LLM_MODEL=llama-3.3-70b-versatile
   
   # OpenAI configuration (optional, if using OpenAI provider)
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL=gpt-4
   ```
   
   EXAIM supports multiple LLM providers through role-based environment variable configuration. See DOCUMENTATION.md for more details.

4. **Run the CDSS demo:**

   ```bash
   python demos/cdss_example/demo_cdss.py
   ```

5. **Use the EXAIM class in your code:**

   ```python
   import asyncio
   from exaim_core import EXAIM

   async def main():
       exaim = EXAIM()
       # Add traces for any agent (agent_id, text)
       summary = await exaim.received_trace("agent_1", "Some trace text")
       if summary:
           print("Updated summary:", summary)
           # Access summary fields
           print(f"Status/Action: {summary.status_action}")
           print(f"Key Findings: {summary.key_findings}")
           print(f"Differential/Rationale: {summary.differential_rationale}")
           print(f"Uncertainty/Confidence: {summary.uncertainty_confidence}")
           print(f"Recommendation/Next Step: {summary.recommendation_next_step}")
           print(f"Agent Contributions: {summary.agent_contributions}")
   
   asyncio.run(main())
   ```


## High-level Design

The system is organized around a few small modules:

- `exaim_core/exaim.py` — EXAIM orchestrator class
  - Purpose: Collects traces from agents, buffers them, and produces summaries using an LLM. Maintains a list of all summaries.
  - Note: `flush_agent(...)` parks any remaining tokens for later summarization rather than forcing a summary.
  - Key methods:
    - `received_trace(agent_id, text)` — Call this to add a trace for an agent. If a summary is triggered, it returns the new `AgentSummary` object.
    - `on_new_token(agent_id, token)` — Process a single streaming token from an agent using TokenGate for intelligent chunking.
    - `get_all_summaries()` — Returns all summaries as a list of `AgentSummary` objects.
    - `get_summaries_by_agent(agent_id)` — Returns summaries involving a specific agent.
    - `get_agent_trace_count(agent_id)` — Returns the number of traces received from an agent.

- `exaim_core/summarizer_agent/summarizer_agent.py` — Summarization wrapper
  - Purpose: Contains the `SummarizerAgent` class, which wraps calls to the LLM (via `infra/llm_registry.py`) and produces structured `AgentSummary` objects from input text.
  - Features:
    - Uses structured output with Pydantic models for consistent summaries
    - Enforces evidence-based character limits aligned with clinical documentation research
    - Optimized for medical/clinical reasoning with SBAR/SOAP-aligned prompts
    - Returns `AgentSummary` objects with fields: `status_action`, `key_findings`, `differential_rationale`, `uncertainty_confidence`, `recommendation_next_step`, `agent_contributions`

- `exaim_core/buffer_agent/buffer_agent.py` — Intelligent trace buffer
  - Purpose: Implements `BufferAgent`, a buffer that accumulates traces per agent. Uses an LLM-based prompt to decide when to trigger summarization (event-driven, not just a static threshold).
  - Features:
    - LLM-powered trigger logic that evaluates trace content
    - Decides summarization based on completed thoughts, topic changes, or accumulated context
    - Tags traces with agent IDs for multi-agent tracking
    - Tracks trace counts per agent

- `exaim_core/token_gate/token_gate.py` — Token streaming pre-buffer
  - Purpose: A lightweight, syntax-aware pre-buffer that regulates token flow into BufferAgent for streaming scenarios.
  - Features:
    - Configurable word thresholds (min/max words, whitespace-delimited)
    - Boundary cue detection (punctuation, newlines)
    - Silence timer and max wait timeout
    - Per-agent text buffering

- `infra/llm_registry.py` — LLM client configuration
  - Purpose: Holds LLM client instances used for summarization and trigger decisions. Uses environment variables for provider selection with role-based configuration.
  - Supports multiple providers: Google Gemini, Groq, OpenAI, and OpenAI-compatible endpoints.
  - Provider selection via role-based environment variables: `MAS_LLM_PROVIDER`, `SUMMARIZER_LLM_PROVIDER`, `BUFFER_AGENT_LLM_PROVIDER`
  - Provides different LLM instances optimized for different use cases (speed vs. reasoning quality)

- `demos/cdss_example/` — Clinical Decision Support System demo
  - Purpose: Complete demonstration of EXAIM integrated with a multi-agent clinical decision support system using LangGraph.
  - Components:
    - `cdss.py` — CDSS orchestrator class
    - `demo_cdss.py` — Example clinical cases demonstrating the system
    - `agents/` — Specialized medical agents (OrchestratorAgent, CardiologyAgent, LaboratoryAgent)
    - `graph/` — LangGraph workflow definition
    - `schema/` — Clinical case and graph state data models

- `requirements.txt` — Python dependencies
  - Purpose: Lists the project's external Python dependencies (LangChain, LangGraph, Pydantic, python-dotenv, etc.).

## Features

- **LLM-powered event-driven summarization:** The buffer uses an LLM to intelligently decide when to trigger summarization based on trace content, not just a static threshold. Summarization triggers when thoughts complete, topics change, or sufficient context accumulates.
- **Multi-agent support:** Traces are tagged by agent ID and summarized in context, allowing multiple specialized agents to contribute to a single reasoning workflow.
- **Streaming token support:** `TokenGate` provides intelligent chunking of streaming tokens with configurable word thresholds (whitespace-delimited), boundary detection, and timeout mechanisms.
- **Structured summaries:** Summaries are generated as structured `AgentSummary` objects with fields optimized for medical reasoning and aligned with SBAR/SOAP documentation standards:
  - `status_action`: Concise description of system/agent activity (max 150 chars)
  - `key_findings`: Minimal clinical facts driving the reasoning step (max 180 chars)
  - `differential_rationale`: Leading diagnostic hypotheses and rationale (max 210 chars)
  - `uncertainty_confidence`: Model/system uncertainty representation (max 120 chars)
  - `recommendation_next_step`: Specific diagnostic/therapeutic/follow-up step (max 180 chars)
  - `agent_contributions`: List of agents and their contributions (max 150 chars)
- **Character limit enforcement:** Automatic truncation ensures summaries remain concise and physician-friendly.
- **Medical/clinical focus:** Prompts and summaries are optimized for physician understanding of multi-agent clinical reasoning.
- **Simple async API:** Add traces and get summaries with a single async method call.
- **CDSS demo:** Complete clinical decision support system demonstration using LangGraph for workflow orchestration.
- **Environment variable configuration:** LLM settings can be configured via environment variables for easy deployment.

## Development Notes and Suggestions

- The project is a prototype. Expect to iterate on the summarization prompt and LLM configuration.
- Configure LLM settings via environment variables (`.env` file). EXAIM supports Google Gemini, Groq, and OpenAI providers.
- Switch between providers by changing environment variables—no code changes needed.
- The system uses async/await patterns throughout, so ensure you're running within an async context when calling methods.
- For streaming scenarios, use `on_new_token()` which processes tokens one at a time and leverages `TokenGate` for intelligent chunking.
- The CDSS demo showcases integration with LangGraph for complex multi-agent workflows.

## Project Structure

```
ExAIM/
├── exaim_core/             # Core EXAIM package
│   ├── exaim.py            # Main orchestrator class
│   ├── buffer_agent/       # Intelligent trace buffer
│   │   └── buffer_agent.py
│   ├── summarizer_agent/   # Summarization logic
│   │   └── summarizer_agent.py
│   ├── token_gate/         # Token streaming pre-buffer
│   │   └── token_gate.py
│   ├── schema/             # Data models
│   │   └── agent_summary.py
│   └── utils/              # Utility functions
│       └── prompts.py
├── infra/                  # Infrastructure
│   ├── llm_registry.py     # LLM client configuration
│   └── model_configs.yaml  # LLM model configurations
├── demos/                  # Demo applications
│   └── cdss_example/       # Clinical Decision Support System demo
│       ├── cdss.py         # CDSS orchestrator
│       ├── demo_cdss.py    # Example clinical cases
│       ├── agents/         # Specialized medical agents
│       │   ├── orchestrator_agent.py
│       │   ├── cardiology_agent.py
│       │   └── laboratory_agent.py
│       ├── graph/          # LangGraph workflow
│       │   ├── cdss_graph.py
│       │   ├── nodes.py
│       │   └── edges.py
│       └── schema/         # Clinical data models
│           ├── clinical_case.py
│           └── graph_state.py
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── DOCUMENTATION.md        # Comprehensive documentation
```

## Files Summary

- `exaim_core/exaim.py`: Orchestrator class that collects traces, buffers them, and records summaries. Provides methods for trace processing, streaming token handling, and summary retrieval.
- `exaim_core/summarizer_agent/summarizer_agent.py`: Summarization logic with structured output. Defines `SummarizerAgent` class that generates `AgentSummary` objects.
- `exaim_core/buffer_agent/buffer_agent.py`: `BufferAgent` implementation with LLM-based trigger logic for event-driven summarization.
- `exaim_core/token_gate/token_gate.py`: Token streaming pre-buffer that regulates token flow with configurable thresholds and timers.
- `exaim_core/schema/agent_summary.py`: Pydantic model defining the structured `AgentSummary` format.
- `infra/llm_registry.py`: LLM client configuration with role-based setup using LangChain (supports environment variables for configuration).
- `demos/cdss_example/cdss.py`: CDSS orchestrator that integrates EXAIM with LangGraph for clinical decision support workflows.
- `demos/cdss_example/demo_cdss.py`: Example usage demonstrating complete clinical case workflows with multiple specialized agents.
- `requirements.txt`: Project dependencies (LangChain, LangGraph, Pydantic, python-dotenv, etc.).

## License

MIT
