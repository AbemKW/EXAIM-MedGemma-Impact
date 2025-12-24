
# EXAID

EXAID is an experimental Python project for capturing short, live traces from multiple agents, buffering those traces, and producing concise summaries using an LLM. It is designed as a minimal prototype for medical multi-agent reasoning workflows, where specialized agents (e.g., InfectiousDiseaseAgent, HematologyAgent, OncologyAgent) collaborate on clinical cases, and their reasoning traces are captured and condensed into structured summaries optimized for physician understanding.

## Submodules

This project uses a git submodule for the MAC (Multi-Agent Conversation) trace generator:

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/AbemKW/ExAID.git

# Or if already cloned
git submodule update --init --recursive
```

The MAC submodule (`third_party/mac`) is a fork of the original MAC framework:

- Fork: https://github.com/AbemKW/mac-streaming-traces
- Purpose: Enables per-token `t_emitted_ms` capture for realistic streaming replay

**Invariant:** This fork only adds transparent token-level timing instrumentation. All MAC conversation logic, agent orchestration, speaker selection, and termination conditions remain unchanged from the original implementation.

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
   # Choose your provider (google, groq, or openai)
   LLM_PROVIDER=google
   MAS_LLM_PROVIDER=groq
   EXAID_LLM_PROVIDER=google
   
   # Google Gemini configuration
   LLM_API_KEY=your-google-api-key
   LLM_MODEL_NAME=gemini-2.5-flash-lite
   EXAID_LLM_MODEL=gemini-2.5-pro
   
   # Groq configuration (for multi-agent reasoning)
   GROQ_API_KEY=your-groq-api-key
   GROQ_MODEL=llama-3.3-70b-versatile
   
   # OpenAI configuration (optional, if using OpenAI provider)
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL=gpt-4
   ```
   
   EXAID supports multiple LLM providers through environment variable configuration. See DOCUMENTATION.md for more details.

5. **Run the CDSS demo:**

   ```bash
   python cdss_demo/demo_cdss.py
   ```

6. **Use the EXAID class in your code:**

   ```python
   import asyncio
   from exaid_core import EXAID

   async def main():
       exaid = EXAID()
       # Add traces for any agent (agent_id, text)
       summary = await exaid.received_trace("agent_1", "Some trace text")
       if summary:
           print("Updated summary:", summary)
           # Access summary fields
           print(f"Action: {summary.action}")
           print(f"Reasoning: {summary.reasoning}")
           print(f"Agents: {', '.join(summary.agents)}")
   
   asyncio.run(main())
   ```


## High-level Design

The system is organized around a few small modules:

- `exaid.py` — EXAID orchestrator class
  - Purpose: Collects traces from agents, buffers them, and produces summaries using an LLM. Maintains a list of all summaries.
  - Key methods:
    - `received_trace(agent_id, text)` — Call this to add a trace for an agent. If a summary is triggered, it returns the new `AgentSummary` object.
    - `received_streamed_tokens(agent_id, token_generator)` — Process streaming tokens from an agent using TokenGate for intelligent chunking.
    - `get_all_summaries()` — Returns all summaries as a list of `AgentSummary` objects.
    - `get_summaries_by_agent(agent_id)` — Returns summaries involving a specific agent.
    - `get_agent_trace_count(agent_id)` — Returns the number of traces received from an agent.

- `agents/summarizer_agent.py` — Summarization wrapper
  - Purpose: Contains the `SummarizerAgent` class, which wraps calls to the LLM (via `llm.py`) and produces structured `AgentSummary` objects from input text.
  - Features:
    - Uses structured output with Pydantic models for consistent summaries
    - Enforces character limits (action: 100, reasoning: 200, findings: 150, next_steps: 100)
    - Optimized for medical/clinical reasoning with physician-focused prompts
    - Returns `AgentSummary` objects with fields: `agents`, `action`, `reasoning`, `findings`, `next_steps`

- `agents/buffer_agent.py` — Intelligent trace buffer
  - Purpose: Implements `BufferAgent`, a buffer that accumulates traces per agent. Uses an LLM-based prompt to decide when to trigger summarization (event-driven, not just a static threshold).
  - Features:
    - LLM-powered trigger logic that evaluates trace content
    - Decides summarization based on completed thoughts, topic changes, or accumulated context
    - Tags traces with agent IDs for multi-agent tracking
    - Tracks trace counts per agent

- `agents/token_gate.py` — Token streaming pre-buffer
  - Purpose: A lightweight, syntax-aware pre-buffer that regulates token flow into BufferAgent for streaming scenarios.
  - Features:
    - Configurable word thresholds (min/max words, whitespace-delimited)
    - Boundary cue detection (punctuation, newlines)
    - Silence timer and max wait timeout
    - Per-agent text buffering

- `agents/base_agent.py` — Base agent interface
  - Purpose: Abstract base class (`BaseAgent`) defining the interface for agents that can be integrated with EXAID.
  - Key method: `act(input: str) -> str` — Abstract method that agents must implement.

- `llm.py` — LLM client configuration
  - Purpose: Holds LLM client instances used for summarization and trigger decisions. Uses environment variables for provider selection.
  - Supports multiple providers: Google Gemini, Groq, OpenAI, and OpenAI-compatible endpoints.
  - Provider selection via environment variables: `LLM_PROVIDER`, `MAS_LLM_PROVIDER`, `EXAID_LLM_PROVIDER`
  - Provides different LLM instances optimized for different use cases (speed vs. reasoning quality)

- `cdss_demo/` — Clinical Decision Support System demo
  - Purpose: Complete demonstration of EXAID integrated with a multi-agent clinical decision support system using LangGraph.
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
- **Structured summaries:** Summaries are generated as structured `AgentSummary` objects with fields optimized for medical reasoning:
  - `agents`: List of agent IDs involved
  - `action`: Brief action statement (max 100 chars)
  - `reasoning`: Concise reasoning explanation (max 200 chars)
  - `findings`: Key clinical findings or recommendations (max 150 chars, optional)
  - `next_steps`: Suggested next actions (max 100 chars, optional)
- **Character limit enforcement:** Automatic truncation ensures summaries remain concise and physician-friendly.
- **Medical/clinical focus:** Prompts and summaries are optimized for physician understanding of multi-agent clinical reasoning.
- **Simple async API:** Add traces and get summaries with a single async method call.
- **CDSS demo:** Complete clinical decision support system demonstration using LangGraph for workflow orchestration.
- **Environment variable configuration:** LLM settings can be configured via environment variables for easy deployment.

## Development Notes and Suggestions

- The project is a prototype. Expect to iterate on the summarization prompt and LLM configuration.
- Configure LLM settings via environment variables (`.env` file). EXAID supports Google Gemini, Groq, and OpenAI providers.
- Switch between providers by changing environment variables—no code changes needed.
- The system uses async/await patterns throughout, so ensure you're running within an async context when calling methods.
- For streaming scenarios, use `received_streamed_tokens()` which leverages `TokenGate` for intelligent chunking.
- The CDSS demo showcases integration with LangGraph for complex multi-agent workflows.

## Project Structure

```
ExAID/
├── exaid.py                 # Main orchestrator class
├── llm.py                   # LLM client configuration
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── DOCUMENTATION.md         # Comprehensive documentation
├── agents/
│   ├── base_agent.py       # Abstract base class for agents
│   ├── buffer_agent.py     # Intelligent trace buffer with LLM triggers
│   ├── summarizer_agent.py # Summarization logic with structured output
│   └── token_gate.py       # Token streaming pre-buffer
├── schema/
│   └── agent_summary.py    # AgentSummary Pydantic model
└── cdss_demo/              # Clinical Decision Support System demo
    ├── cdss.py             # CDSS orchestrator
    ├── demo_cdss.py        # Example clinical cases
    ├── agents/             # Specialized medical agents
    │   ├── orchestrator_agent.py
    │   ├── cardiology_agent.py
    │   └── laboratory_agent.py
    ├── graph/              # LangGraph workflow
    │   ├── cdss_graph.py
    │   ├── nodes.py
    │   └── edges.py
    └── schema/             # Clinical data models
        ├── clinical_case.py
        └── graph_state.py
```

## Files Summary

- `exaid.py`: Orchestrator class that collects traces, buffers them, and records summaries. Provides methods for trace processing, streaming token handling, and summary retrieval.
- `agents/summarizer_agent.py`: Summarization logic with structured output. Defines `SummarizerAgent` class that generates `AgentSummary` objects.
- `agents/buffer_agent.py`: `BufferAgent` implementation with LLM-based trigger logic for event-driven summarization.
- `agents/token_gate.py`: Token streaming pre-buffer that regulates token flow with configurable thresholds and timers.
- `agents/base_agent.py`: Abstract base class (`BaseAgent`) defining the interface for agents that can integrate with EXAID.
- `schema/agent_summary.py`: Pydantic model defining the structured `AgentSummary` format.
- `llm.py`: LLM client configuration using LangChain's `ChatOpenAI` (supports environment variables for configuration).
- `cdss_demo/cdss.py`: CDSS orchestrator that integrates EXAID with LangGraph for clinical decision support workflows.
- `cdss_demo/demo_cdss.py`: Example usage demonstrating complete clinical case workflows with multiple specialized agents.
- `requirements.txt`: Project dependencies (LangChain, LangGraph, Pydantic, python-dotenv, etc.).

## License

MIT