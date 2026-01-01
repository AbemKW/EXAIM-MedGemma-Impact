# EXAID - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [File-by-File Documentation](#file-by-file-documentation)
5. [Data Structures](#data-structures)
6. [Workflow and Usage](#workflow-and-usage)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Examples](#examples)

---

## Overview

**EXAID** (Explainable AI for Diagnoses) is a Python framework designed for capturing, buffering, and summarizing reasoning traces from multiple AI agents in real-time. Originally designed for medical multi-agent reasoning workflows, EXAID enables specialized agents (e.g., `InfectiousDiseaseAgent`, `HematologyAgent`, `OncologyAgent`) to collaborate on complex cases while their reasoning traces are intelligently captured and condensed into structured summaries optimized for physician understanding.

### Key Features

- **LLM-Powered Event-Driven Summarization**: Uses an LLM to intelligently decide when to trigger summarization based on trace content, not just static thresholds
- **Multi-Agent Support**: Tracks and summarizes traces from multiple agents simultaneously
- **Structured Output**: Generates structured summaries with enforced character limits
- **Medical/Clinical Focus**: Optimized prompts and summaries for physician understanding
- **Async API**: Fully asynchronous implementation for efficient processing

---

## Architecture

EXAID follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                        EXAID (Orchestrator)                  │
│  - Manages agent lifecycle                                  │
│  - Coordinates summarization workflow                       │
│  - Maintains summary history                                │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┴────────┬──────────────┐
       │                │              │
┌──────▼──────┐  ┌──────▼──────┐  ┌───▼──────────┐
│ BufferAgent │  │Summarizer   │  │   LLM        │
│             │  │Agent         │  │  Client      │
│ - Buffers   │  │              │  │              │
│   traces    │  │ - Generates  │  │ - Provides   │
│ - LLM-based │  │   summaries  │  │   AI         │
│   trigger   │  │ - Structured │  │   services   │
│   logic     │  │   output     │  │              │
└─────────────┘  └──────────────┘  └──────────────┘
```

### Data Flow

1. **Trace Reception**: Agents send traces to `EXAID.received_trace()`
2. **Buffering**: `BufferAgent` accumulates traces and uses LLM to determine when to trigger summarization
3. **Summarization**: `SummarizerAgent` generates structured summaries from buffered traces
4. **Storage**: Summaries are stored in EXAID instance

---

## Core Components

### 1. EXAID (Main Orchestrator)

The central class that coordinates all operations. It manages:
- Trace buffering and summarization triggers
- Summary generation
- History of summaries

### 2. BufferAgent

Intelligently buffers traces per agent and uses an LLM to decide when summarization should be triggered. Unlike simple threshold-based systems, it evaluates:
- Completion of thoughts or reasoning steps
- Topic or focus changes
- Accumulated context sufficiency
- Natural pauses or conclusions

### 3. SummarizerAgent

Generates structured summaries from buffered traces. Features:
- Structured output using Pydantic models
- Character limit enforcement
- Medical/clinical optimization

---

## File-by-File Documentation

### `exaid_core/exaid.py` - Main Orchestrator

**Purpose**: The central orchestrator class that coordinates trace collection, buffering, and summarization.

**Key Components**:

```python
class EXAID:
    def __init__(self):
        self.buffer_agent = BufferAgent()
        self.summarizer_agent = SummarizerAgent()
        self.summaries: list[AgentSummary] = []
```

**Core Methods**:

#### `received_trace(agent_id: str, text: str) -> Optional[AgentSummary]`

The main entry point for processing agent traces. This method:

1. Adds the trace to the buffer via `BufferAgent.addsegment()`
2. If summarization is triggered:
   - Retrieves previous summaries for context
   - Generates summary using `SummarizerAgent`
   - Stores the summary
   - Returns the `AgentSummary`

**Code Snippet**:
```python
async def received_trace(self, agent_id: str, text: str) -> Optional[AgentSummary]:
    """Process a trace from an agent. Returns an AgentSummary if summarization 
    was triggered, None otherwise."""
    # Prepare previous summaries for buffer agent evaluation
    all_summaries = self.get_all_summaries()
    previous_summaries = self._format_summaries_history(all_summaries)
    
    trigger = await self.buffer_agent.addsegment(id, text, previous_summaries)
    if trigger:
        agent_buffer = self.buffer_agent.flush()
        buffer_str = "\n".join(agent_buffer)
        
        # Get previous summaries for context
        all_summaries = self.get_all_summaries()
        summary_history_strs = self._format_summaries_history(all_summaries[:-1]) if len(all_summaries) > 1 else []
        latest_summary_str = self._format_summary_for_history(all_summaries[-1]) if all_summaries else "No summaries yet."
        
        # Generate summary
        summary = await self.summarizer_agent.summarize(
            summary_history_strs,
            latest_summary_str,
            buffer_str
        )
        
        # Store summary
        self.summaries.append(summary)
        
        return summary
    return None
```

#### `get_all_summaries() -> list[AgentSummary]`

Returns all summaries as a list of `AgentSummary` objects.

#### `get_summaries_by_agent(agent_id: str) -> list[AgentSummary]`

Filters summaries to return only those involving a specific agent ID.

#### `get_agent_trace_count(agent_id: str) -> int`

Returns the total number of traces received from a specific agent.

#### `async on_new_token(agent_id: str, token: str) -> Optional[AgentSummary]`

Processes a single streaming token from an agent using TokenGate for intelligent chunking. This method:
1. Receives a single token string
2. Uses TokenGate to accumulate tokens into meaningful chunks
3. Processes chunks through BufferAgent when ready
4. Returns the summary generated, if any

**Parameters**:
- `agent_id` (str): Agent identifier
- `token` (str): Single token string (not an async iterator)

**Returns**: `AgentSummary` if summarization was triggered, or `None` otherwise

**Note**: This method processes tokens one at a time. For streaming scenarios, call this method repeatedly for each token in the stream.

**Helper Methods**:

- `_format_summary_for_history(summary: AgentSummary) -> str`: Converts an `AgentSummary` to a string representation for use in prompt history
- `_format_summaries_history(summaries: list[AgentSummary]) -> list[str]`: Converts a list of summaries to string representations for prompt context

---

### `exaid_core/buffer_agent/buffer_agent.py` - Intelligent Trace Buffer

**Purpose**: Buffers traces per agent and uses an LLM to intelligently decide when summarization should be triggered based on trace content rather than simple thresholds.

**Key Components**:

```python
class BufferAnalysis(BaseModel):
    """Structured analysis of stream state for buffer agent decision-making."""
    reasoning: str
    stream_state: Literal["SAME_TOPIC_CONTINUING", "TOPIC_SHIFT", "CRITICAL_ALERT"]
    is_relevant: bool
    is_novel: bool
    final_trigger: bool

class TraceData(BaseModel):
    count: int

class BufferAgent:
    def __init__(self):
        self.buffer: list[str] = []
        self.base_llm = get_llm(LLMRole.BUFFER_AGENT)
        self.llm = self.base_llm.with_structured_output(BufferAnalysis)
        self.flag_prompt = ChatPromptTemplate.from_messages([...])
        self.traces: dict[str, TraceData] = {}
```

**Core Methods**:

#### `addsegment(agent_id: str, segment: str, previous_summaries: list[str]) -> bool`

**Parameters**:
- `agent_id`: Identifier for the agent generating the trace
- `segment`: The trace text segment to add to the buffer
- `previous_summaries`: List of string-formatted previous summaries for novelty comparison

Adds a trace segment to the buffer and determines if summarization should be triggered using a structured state machine approach.

**Process**:
1. Tags the segment with agent ID: `| {agent_id} | {segment}`
2. Records the trace count in `traces` dictionary
3. Adds the tagged segment to the buffer
4. Uses an LLM with structured output to evaluate stream state, relevance, and novelty
5. Returns `True` if summarization should be triggered, `False` otherwise

**Structured Output Model**:
The buffer uses a `BufferAnalysis` Pydantic model that decouples three evaluation dimensions:

- **Stream State** (`stream_state`): Three-state machine based on topic continuity:
  - `SAME_TOPIC_CONTINUING`: Agent is still refining, listing, or explaining the same clinical issue (wait)
  - `TOPIC_SHIFT`: Agent moves to a different organ system, problem, or section (proceed to checks)
  - `CRITICAL_ALERT`: Immediate life-safety notification (proceed immediately)

- **Relevance** (`is_relevant`): Independent evaluation of clinical importance
  - Relevant: New diagnosis, refined differential, specific treatment dose/plan, condition changes
  - Not Relevant: "Thinking out loud", obvious facts without interpretation, formatting tokens

- **Novelty** (`is_novel`): Independent evaluation against previous summaries
  - Novel: New values, new actions, new insights not already covered
  - Not Novel: Continuing statements, reiteration, status quo confirmations

- **Final Trigger** (`final_trigger`): Only `True` when `(stream_state == TOPIC_SHIFT OR CRITICAL_ALERT) AND is_relevant AND is_novel`

**Code Snippet**:
```python
class BufferAnalysis(BaseModel):
    reasoning: str  # Chain-of-thought analysis
    stream_state: Literal["SAME_TOPIC_CONTINUING", "TOPIC_SHIFT", "CRITICAL_ALERT"]
    is_relevant: bool
    is_novel: bool
    final_trigger: bool  # True only if all conditions met

async def addsegment(self, agent_id: str, segment: str, previous_summaries: list[str]) -> bool:
    tagged_segment = f"| {agent_id} | {segment}"
    if agent_id not in self.traces:
        self.traces[agent_id] = TraceData(count=0)
    self.traces[agent_id].count += 1
    
    # Add to buffer first
    self.buffer.append(tagged_segment)
    
    # Prepare context: rest of buffer separate from new segment
    buffer_context = "\n".join(self.buffer[:-1]) if len(self.buffer) > 1 else "(Buffer empty)"
    
    # Invoke LLM with structured output
    chain = self.flag_prompt | self.llm
    analysis: BufferAnalysis = await chain.ainvoke({
        "summaries": previous_summaries,
        "previous_trace": buffer_context,
        "new_trace": tagged_segment
    })
    
    return analysis.final_trigger
```

#### `peek() -> list[str]`

Returns a copy of the current buffer without flushing it. Useful for capturing buffer state before summarization.

#### `flush() -> list[str]`

Returns a copy of the buffer and clears it. Called after summarization is triggered.

#### `get_trace_count(agent_id: str) -> int`

Returns the total number of traces received from a specific agent.

**Prompt Template**:
The buffer agent uses a structured prompt that instructs the LLM to analyze stream state, relevance, and novelty independently. The prompt template is defined in `exaid_core/utils/prompts.py` and includes:

- **Stream State Detection**: Classifies the stream into one of three states based on topic continuity:
  - `SAME_TOPIC_CONTINUING`: Agent is still refining the same clinical issue (wait)
  - `TOPIC_SHIFT`: Agent moves to a different topic or concludes a thought (proceed)
  - `CRITICAL_ALERT`: Immediate life-safety notification (proceed immediately)

- **Relevance Detection**: Evaluates if the content is clinically important

- **Novelty Detection**: Compares against previous summaries to determine if information is new

- **Final Trigger Logic**: Only triggers when `(TOPIC_SHIFT OR CRITICAL_ALERT) AND is_relevant AND is_novel`

The prompt uses structured output via Pydantic to ensure consistent, parseable responses:

```python
self.flag_prompt = ChatPromptTemplate.from_messages([
    ("system", get_buffer_agent_system_prompt()),
    ("user", get_buffer_agent_user_prompt())
])
self.llm = self.base_llm.with_structured_output(BufferAnalysis)
```
    "Be decisive - prefer YES when in doubt, as summaries help track agent progress."),
    ("user", "Previous traces in buffer:\n{previous_trace}\n\nNew trace to evaluate:\n{new_trace}\n\nShould this trigger summarization? Reply with only 'YES' or 'NO'."),
])
```

---

### `exaid_core/summarizer_agent/summarizer_agent.py` - Summary Generator

**Purpose**: Generates structured summaries from buffered traces using an LLM with structured output.

**Key Components**:

```python
class SummarizerAgent:
    def __init__(self):
        self.llm = llm.with_structured_output(schema=AgentSummary)
```

**Core Methods**:

#### `summarize(summary_history: List[str], latest_summary: str, new_buffer: str) -> AgentSummary`

Generates a structured summary from buffered traces.

**Parameters**:
- `summary_history`: List of previous summary strings (excluding the latest)
- `latest_summary`: The most recent summary string
- `new_buffer`: New reasoning buffer content to summarize

**Process**:
1. Formats the prompt with summary history, latest summary, and new buffer
2. Invokes LLM with structured output to generate `AgentSummary`
3. Returns the structured summary object

**Code Snippet**:
```python
async def summarize(
    self, 
    summary_history: List[str], 
    latest_summary: str, 
    new_buffer: str
) -> AgentSummary:
    """Updates the summary given the summary history (as a list), latest summary, 
    and new reasoning buffer."""
    summarize_chain = self.summarize_prompt | self.llm
    
    summary = await summarize_chain.ainvoke({
        "summary_history": ",\n".join(summary_history),
        "latest_summary": latest_summary,
        "new_buffer": new_buffer
    })
    return summary
```

**Prompt Template**:
The summarizer uses prompts from `exaid_core/utils/prompts.py` that align with SBAR/SOAP documentation standards. The system prompt includes:

- **Delta-first summarization**: Prioritizes new, changed, or newly concluded information
- **Controlled continuity**: Allows restating prior information only for sticky context (active interventions, current leading assessment, unresolved critical abnormalities, safety constraints, decision blockers)
- **Non-empty field rule**: All 6 fields must be populated, using explicit placeholders when no supported content exists
- **Strict character limits**: 
  - `status_action`: MAX 150 chars
  - `key_findings`: MAX 180 chars
  - `differential_rationale`: MAX 210 chars
  - `uncertainty_confidence`: MAX 120 chars
  - `recommendation_next_step`: MAX 180 chars
  - `agent_contributions`: MAX 150 chars

The prompt template structure:
```python
self.summarize_prompt = ChatPromptTemplate.from_messages([    
    ("system", get_summarizer_system_prompt()),
    ("user", get_summarizer_user_prompt()),
])
```

**Features**:
- Uses structured output with Pydantic models for consistent formatting
- Enforces evidence-based character limits aligned with clinical documentation research
- Optimized for medical/clinical reasoning with SBAR/SOAP alignment
- Prevents hallucination through explicit placeholder rules

---

---

### `exaid_core/schema/agent_summary.py` - Summary Data Model

**Purpose**: Defines the structured data model for agent summaries using Pydantic.

**Code**:
```python
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional

class AgentSummary(BaseModel):
    """Structured summary for medical multi-agent reasoning, optimized for physician understanding."""
    agents: List[str] = Field(description="List of agent IDs involved in this reasoning step")
    action: str = Field(max_length=100, description="Brief action statement describing what the agents did")
    reasoning: str = Field(max_length=200, description="Concise reasoning explaining why this action was taken")
    findings: Optional[str] = Field(max_length=150, default=None, description="Key clinical findings or recommendations if applicable")
    next_steps: Optional[str] = Field(max_length=100, default=None, description="Suggested next actions if applicable")
    
    @model_validator(mode='before')
    @classmethod
    def truncate_fields(cls, data):
        """Truncate fields to meet length constraints if they exceed limits."""
        if isinstance(data, dict):
            if 'action' in data and len(data.get('action', '')) > 100:
                data['action'] = data['action'][:97] + '...'
            if 'reasoning' in data and len(data.get('reasoning', '')) > 200:
                data['reasoning'] = data['reasoning'][:197] + '...'
            if 'findings' in data and data.get('findings') and len(data['findings']) > 150:
                data['findings'] = data['findings'][:147] + '...'
            if 'next_steps' in data and data.get('next_steps') and len(data['next_steps']) > 100:
                data['next_steps'] = data['next_steps'][:97] + '...'
        return data
```

**Fields**:
- `agents`: List of agent IDs involved in the reasoning step
- `action`: Brief action statement (max 100 characters)
- `reasoning`: Concise reasoning explanation (max 200 characters)
- `findings`: Optional key clinical findings or recommendations (max 150 characters)
- `next_steps`: Optional suggested next actions (max 100 characters)

**Features**:
- Automatic truncation via `model_validator` if fields exceed limits
- Pydantic validation ensures data integrity
- Optimized for physician understanding with concise, action-oriented fields

---

### `exaid_core/token_gate/token_gate.py` - Token Streaming Pre-Buffer

**Purpose**: A lightweight, syntax-aware pre-buffer that regulates token flow into BufferAgent for streaming scenarios. It does not interpret meaning - it only decides when enough structure has accumulated to pass tokens upstream for semantic evaluation.

**Note**: TokenGate counts **whitespace-delimited words** (not model tokenizer tokens) for its min/max thresholds. The class name "TokenGate" refers to its role in gating streaming tokens from the LLM, not to the counting method.

**Key Components**:

```python
class TokenGate:
    def __init__(
        self,
        min_words: int = 60,
        max_words: int = 100,
        silence_timer: float = 1,
        max_wait_timeout: float = 4
    ):
```

**Note**: Evaluation configurations (e.g., `evals/configs/variants/V0.yaml`) use calibrated values (`min_words=35`, `max_words=90`, `silence_timer=15`, `max_wait_timeout=40`) which differ from these runtime defaults. The runtime defaults are optimized for general use, while eval configs are calibrated for specific evaluation scenarios.

Boundary cues are hardcoded to `.?!\n` (period, question mark, exclamation, newline) and cannot be configured.

**Core Methods**:

#### `async add_token(agent_id: str, token: str) -> Optional[str]`

Adds a token to the buffer for the given agent. If flush conditions are met, returns the buffered text and clears the buffer.

**Flush Conditions**:
- Maximum word cap reached (`max_words`)
- Minimum word threshold reached (`min_words`) AND boundary cue detected
- Silence timer expired (no tokens received for `silence_timer` seconds)
- Max wait timeout expired (buffer has existed for `max_wait_timeout` seconds)

**Returns**: Flushed chunk text if flush triggered, `None` otherwise

#### `async flush(agent_id: str) -> Optional[str]`

Force flush the buffer for the given agent.

**Returns**: Flushed buffer text, or `None` if buffer is empty

#### `async check_timers(agent_id: str) -> Optional[str]`

Check if timers have expired and flush if needed. Should be called periodically or after async operations.

**Returns**: Flushed chunk if timer expired, `None` otherwise

**Features**:
- Per-agent text buffering
- Configurable word thresholds (whitespace-delimited)
- Boundary cue detection (punctuation, newlines)
- Timeout mechanisms (silence timer, max wait timeout)
- Word counting using whitespace-based splitting

---

### `infra/llm_registry.py` - LLM Client Configuration

**Purpose**: Centralized LLM management with role-based configuration. Supports multiple LLM providers through environment variable-based configuration with YAML defaults.

**Architecture**:
The module provides role-based LLM configuration using `LLMRole` enum (MAS, SUMMARIZER, BUFFER_AGENT) and a factory function `_create_llm_instance()` that creates LLM instances based on provider type, allowing seamless switching between Google Gemini, Groq, and OpenAI (or OpenAI-compatible) providers without code changes.

**LLM Roles**:
- `LLMRole.MAS`: Multi-Agent System LLM (configurable, defaults to Groq)
- `LLMRole.SUMMARIZER`: Summarizer LLM (configurable, defaults to Gemini Pro)
- `LLMRole.BUFFER_AGENT`: Buffer Agent LLM (configurable, defaults to Gemini Pro)

**Configuration**:
- Role-based provider selection via `get_llm(role)` function
- YAML configuration file (`infra/model_configs.yaml`) with environment variable overrides
- Provider selection via environment variables
- Clean, maintainable configuration
- Each role can use a different provider

**Environment Variables**:

*Role-Based Provider Selection:*
- `MAS_LLM_PROVIDER`: Provider for MAS role (default: "groq")
- `SUMMARIZER_LLM_PROVIDER`: Provider for Summarizer role (default: "google")
- `BUFFER_AGENT_LLM_PROVIDER`: Provider for Buffer Agent role (default: "google")

*Role-Based Model Overrides (optional):*
- `MAS_LLM_MODEL`: Model name for MAS role (overrides default from YAML)
- `SUMMARIZER_LLM_MODEL`: Model name for Summarizer role (overrides default from YAML)
- `BUFFER_AGENT_LLM_MODEL`: Model name for Buffer Agent role (overrides default from YAML)

*Google Gemini Configuration:*
- `GOOGLE_API_KEY`: Google API key (required for Google provider)
- `GOOGLE_MODEL_NAME`: Default model name (default: "gemini-2.5-flash-lite")

*Groq Configuration:*
- `GROQ_API_KEY`: Groq API key (required for Groq provider)
- `GROQ_MODEL`: Default model name (default: "llama-3.3-70b-versatile")

*OpenAI Configuration:*
- `OPENAI_API_KEY`: API key (required for OpenAI provider)
- `OPENAI_BASE_URL`: Base URL for API (optional, for OpenAI-compatible endpoints)
- `OPENAI_MODEL`: Default model name (optional)

**Usage**: Imported by:
- `BufferAgent` uses `get_llm(LLMRole.BUFFER_AGENT)` for trigger decisions
- `SummarizerAgent` uses `get_llm(LLMRole.SUMMARIZER)` for summary generation
- Demo agents use `get_llm(LLMRole.MAS)` for multi-agent reasoning

**Note**: For production use, always use environment variables for sensitive information. Create a `.env` file in the project root.

---

### `demos/cdss_example/` - Clinical Decision Support System Demo

**Purpose**: Complete demonstration of EXAID integrated with a multi-agent clinical decision support system using LangGraph for workflow orchestration.

#### `demos/cdss_example/cdss.py` - CDSS Orchestrator

**Purpose**: Orchestrates the clinical decision support system workflow using LangGraph.

**Key Components**:

```python
class CDSS:
    def __init__(self):
        self.exaid = EXAID()
        self.graph = build_cdss_graph(self.exaid)
```

**Core Methods**:

#### `async process_case(case: Union[ClinicalCase, str], use_streaming: bool = True) -> dict`

Processes a clinical case through the multi-agent system using LangGraph.

**Parameters**:
- `case`: ClinicalCase object or free-text case description
- `use_streaming`: Whether to use streaming (accepted but not actively used in current implementation)

**Returns**: Dictionary containing:
- `case_summary`: Case text summary
- `final_synthesis`: Final synthesis from the workflow
- `running_summary`: Running summary maintained by orchestrator
- `specialists_called`: List of specialist agents that were invoked
- `iteration_count`: Number of workflow iterations
- `agent_summaries`: List of AgentSummary objects from EXAID (for UI display only; workflow does not depend on these)

#### `get_all_summaries() -> list[AgentSummary]`

Get all summaries from EXAID.

#### `get_summaries_by_agent(agent_id: str) -> list[AgentSummary]`

Get summaries for a specific agent.

#### `reset()`

Reset the CDSS system (creates new EXAID instance).

#### `demos/cdss_example/demo_cdss.py` - Example Clinical Cases

**Purpose**: Demonstrates CDSS usage with complete clinical case workflows.

**Features**:
- Multiple clinical case scenarios (chest pain, fever, etc.)
- Complete diagnostic workflow from initial case to treatment planning
- Formats summaries for clean console display
- Shows integration with LangGraph workflow

**Example Output Format**:
```
┌─ Agents: Orchestrator, CardiologyAgent, LaboratoryAgent
├─ Action: Initial diagnostic hypothesis generation
├─ Reasoning: Multiple agents evaluating clinical presentation
├─ Findings: Elevated troponin suggests acute coronary syndrome
└─ Next Steps: Order ECG, cardiac enzymes, chest X-ray
```

#### `demos/cdss_example/agents/` - Specialized Medical Agents

**Purpose**: Specialized agents for clinical decision support.

- **OrchestratorAgent**: Coordinates case analysis and agent invocation
- **CardiologyAgent**: Provides cardiology-specific analysis
- **LaboratoryAgent**: Analyzes laboratory results and findings

#### `demos/cdss_example/graph/` - LangGraph Workflow

**Purpose**: Defines the LangGraph workflow for multi-agent clinical reasoning.

- **cdss_graph.py**: Main graph builder
- **nodes.py**: Graph node implementations
- **edges.py**: Edge conditions and routing logic

---

### `requirements.txt` - Dependencies

**Purpose**: Lists all Python package dependencies required for EXAID.

**Contents**:
```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-groq>=0.2.0
langchain-google-genai>=1.0.0
langgraph>=0.2.0
pydantic>=2.0.0
python-dotenv>=1.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0
tiktoken>=0.5.0
matplotlib>=3.8.0
seaborn>=0.13.0
pyyaml>=6.0.0
```

**Dependencies**:
- **langchain**: Core LangChain framework for LLM integration
- **langchain-community**: Community integrations
- **langchain-core**: Core LangChain abstractions
- **langchain-openai**: OpenAI integration for ChatOpenAI
- **langchain-groq**: Groq integration for LLM providers
- **langchain-google-genai**: Google Gemini integration
- **langgraph**: LangGraph for workflow orchestration (used in CDSS demo)
- **pydantic**: Data validation and structured output
- **python-dotenv**: Environment variable management
- **fastapi**: Web framework (for API endpoints)
- **uvicorn**: ASGI server (for FastAPI)
- **websockets**: WebSocket support
- **tiktoken**: Token counting utilities
- **matplotlib**: Plotting library (for evaluation visualizations)
- **seaborn**: Statistical visualization (for evaluation visualizations)
- **pyyaml**: YAML parsing (for configuration files)
- **python-dotenv**: Environment variable management

---

## Data Structures

### AgentSummary

Structured summary object with the following fields:

| Field | Type | Max Length | Description |
|-------|------|------------|-------------|
| `agents` | `List[str]` | - | List of agent IDs involved |
| `action` | `str` | 100 | Brief action statement |
| `reasoning` | `str` | 200 | Concise reasoning explanation |
| `findings` | `Optional[str]` | 150 | Key clinical findings (optional) |
| `next_steps` | `Optional[str]` | 100 | Suggested next actions (optional) |

### TraceData

Trace metadata stored by BufferAgent:

| Field | Type | Description |
|-------|------|-------------|
| `count` | `int` | Total number of traces from this agent |

---

## Workflow and Usage

### Basic Usage Pattern

```python
import asyncio
from exaid_core import EXAID

async def main():
    # Initialize EXAID
    exaid = EXAID()
    
    # Send traces from agents
    summary = await exaid.received_trace("agent_1", "Some reasoning trace")
    
    # Check if summary was generated
    if summary:
        print("New summary:", summary.status_action)
        
        # Convert summary to JSON using Pydantic
        import json
        json_summary = summary.model_dump_json()
        print(json_summary)
    
    # Retrieve all summaries
    all_summaries = exaid.get_all_summaries()
    print(f"Total summaries: {len(all_summaries)}")
    
    # Get summaries for specific agent
    agent_summaries = exaid.get_summaries_by_agent("agent_1")
    print(f"Agent 1 summaries: {len(agent_summaries)}")

asyncio.run(main())
```

### Complete Workflow

1. **Initialize EXAID**:
   ```python
   exaid = EXAID()
   ```

2. **Send Traces**:
   ```python
   summary = await exaid.received_trace(agent_id, trace_text)
   ```

3. **Process Summary** (if generated):
   ```python
   if summary:
       # Access summary fields
       status_action = summary.status_action
       key_findings = summary.key_findings
       differential_rationale = summary.differential_rationale
       uncertainty_confidence = summary.uncertainty_confidence
       recommendation_next_step = summary.recommendation_next_step
       agent_contributions = summary.agent_contributions
   ```

4. **Export Data**:
   ```python
   # Convert summary to JSON using Pydantic
   import json
   json_summary = summary.model_dump_json()  # Pydantic v2
   # or for Pydantic v1: json_summary = summary.json()
   ```

---

## API Reference

### EXAID Class

#### `__init__()`

Initialize EXAID instance.

#### `async received_trace(agent_id: str, text: str) -> Optional[AgentSummary]`

Process a trace from an agent.

**Parameters**:
- `id` (str): Agent identifier
- `text` (str): Trace text content

**Returns**: `AgentSummary` if summarization was triggered, `None` otherwise

#### `latest_summary() -> Union[AgentSummary, str]`

Get the most recent summary.

**Returns**: `AgentSummary` object or `"No summaries yet."` string

#### `get_all_summaries() -> list[AgentSummary]`

Get all summaries.

**Returns**: List of `AgentSummary` objects

#### `get_summaries_by_agent(agent_id: str) -> list[AgentSummary]`

Get summaries involving a specific agent.

**Parameters**:
- `agent_id` (str): Agent identifier

**Returns**: List of `AgentSummary` objects

#### `get_agent_trace_count(agent_id: str) -> int`

Get the total number of traces received from an agent.

**Parameters**:
- `agent_id` (str): Agent identifier

**Returns**: Trace count (int)

#### `async on_new_token(agent_id: str, token: str) -> Optional[AgentSummary]`

Processes a single streaming token from an agent using TokenGate for intelligent chunking. This method:
1. Receives a single token string
2. Uses TokenGate to accumulate tokens into meaningful chunks
3. Processes chunks through BufferAgent when ready
4. Returns the summary generated, if any

**Parameters**:
- `agent_id` (str): Agent identifier
- `token` (str): Single token string (not an async iterator)

**Returns**: `AgentSummary` if summarization was triggered, or `None` otherwise

**Note**: This method processes tokens one at a time. For streaming scenarios, call this method repeatedly for each token in the stream.

### BufferAgent Class

#### `async addsegment(agent_id: str, segment: str, previous_summaries: list[str]) -> bool`

Add a trace segment and determine if summarization should trigger using structured state machine analysis.

**Parameters**:
- `agent_id` (str): Agent identifier
- `segment` (str): Trace text segment from token gate
- `previous_summaries` (list[str]): List of string-formatted previous summaries for novelty comparison

**Returns**:
- `bool`: `True` if summarization should be triggered, `False` otherwise

**Process**:
Uses structured output (`BufferAnalysis`) to evaluate three independent dimensions:
1. **Stream State**: Classifies as `SAME_TOPIC_CONTINUING`, `TOPIC_SHIFT`, or `CRITICAL_ALERT`
2. **Relevance**: Determines if content is clinically important
3. **Novelty**: Compares against previous summaries to detect new information

Only triggers when `(TOPIC_SHIFT OR CRITICAL_ALERT) AND is_relevant AND is_novel`

**Returns**: `True` if summarization should trigger, `False` otherwise

#### `peek() -> list[str]`

Get a copy of the current buffer without flushing.

**Returns**: List of buffered trace strings

#### `flush() -> list[str]`

Get a copy of the buffer and clear it.

**Returns**: List of buffered trace strings

#### `get_trace_count(agent_id: str) -> int`

Get trace count for an agent.

**Parameters**:
- `agent_id` (str): Agent identifier

**Returns**: Trace count (int)

### TokenGate Class

Note: TokenGate counts whitespace-delimited words (not model tokenizer tokens) for its min/max thresholds.

#### `async add_token(agent_id: str, token: str) -> Optional[str]`

Add a streaming token to the buffer for the given agent. Flushes when word thresholds are met.

**Parameters**:
- `agent_id` (str): Agent identifier
- `token` (str): Token string to add (streaming token from LLM)

**Returns**: Flushed chunk text if flush triggered, `None` otherwise

#### `async flush(agent_id: str) -> Optional[str]`

Force flush the buffer for the given agent.

**Parameters**:
- `agent_id` (str): Agent identifier

**Returns**: Flushed buffer text, or `None` if buffer is empty

#### `async check_timers(agent_id: str) -> Optional[str]`

Check if timers have expired and flush if needed.

**Parameters**:
- `agent_id` (str): Agent identifier

**Returns**: Flushed chunk if timer expired, `None` otherwise

### SummarizerAgent Class

#### `async summarize(summary_history: List[str], latest_summary: str, new_buffer: str) -> AgentSummary`

Generate a structured summary from buffered traces.

**Parameters**:
- `summary_history` (List[str]): Previous summary strings
- `latest_summary` (str): Most recent summary string
- `new_buffer` (str): New buffer content to summarize

**Returns**: `AgentSummary` object

---

## Configuration

### LLM Configuration

EXAID supports multiple LLM providers (Google Gemini, Groq, OpenAI, and OpenAI-compatible endpoints) through environment variable-based configuration. This allows switching between providers without code changes.

**Provider Selection**:

Create a `.env` file in the project root:

```bash
# Role-based provider selection (mas, summarizer, buffer_agent)
MAS_LLM_PROVIDER=groq
SUMMARIZER_LLM_PROVIDER=google
BUFFER_AGENT_LLM_PROVIDER=google

# Google Gemini configuration
GOOGLE_API_KEY=your-google-api-key
GOOGLE_MODEL_NAME=gemini-2.5-flash-lite
# Optional: Override models per role
SUMMARIZER_LLM_MODEL=gemini-2.5-pro
BUFFER_AGENT_LLM_MODEL=gemini-2.5-pro

# Groq configuration (for fast multi-agent reasoning)
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama-3.3-70b-versatile
# Optional: Override model for MAS role
MAS_LLM_MODEL=llama-3.3-70b-versatile

# OpenAI configuration (or OpenAI-compatible endpoints)
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
OPENAI_MODEL=gpt-4  # Optional
```

**Default Configuration** (from `infra/model_configs.yaml`):
- `mas`: Groq with llama-3.3-70b-versatile (optimized for multi-agent speed)
- `summarizer`: Google Gemini Flash Lite (strong reasoning, cost-effective)
- `buffer_agent`: Google Gemini Flash Lite (strong reasoning, cost-effective)

**Example Configurations**:

*All Google Gemini:*
```bash
MAS_LLM_PROVIDER=google
SUMMARIZER_LLM_PROVIDER=google
BUFFER_AGENT_LLM_PROVIDER=google
GOOGLE_API_KEY=your-google-api-key
```

*All Groq:*
```bash
MAS_LLM_PROVIDER=groq
SUMMARIZER_LLM_PROVIDER=groq
BUFFER_AGENT_LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key
```

*Custom OpenAI-compatible endpoint:*
```bash
MAS_LLM_PROVIDER=openai
SUMMARIZER_LLM_PROVIDER=openai
BUFFER_AGENT_LLM_PROVIDER=openai
OPENAI_BASE_URL=https://your-endpoint.com/v1
OPENAI_MODEL=your-model-name
OPENAI_API_KEY=your-api-key
```

**Supported Providers**:
- **google**: Google Gemini models via LangChain's `ChatGoogleGenerativeAI`
- **groq**: Groq models via LangChain's `ChatGroq`
- **openai**: OpenAI or OpenAI-compatible endpoints via LangChain's `ChatOpenAI`


---

## Examples

### Example 1: Simple Single Agent

```python
import asyncio
from exaid_core import EXAID

async def main():
    exaid = EXAID()
    
    # Send multiple traces from one agent
    await exaid.received_trace("DoctorAgent", "Reviewing patient symptoms")
    await exaid.received_trace("DoctorAgent", "Ordering lab tests")
    
    summary = await exaid.received_trace("DoctorAgent", "Lab results received")
    
    if summary:
        print(f"Status/Action: {summary.status_action}")
        print(f"Key Findings: {summary.key_findings}")
        print(f"Differential/Rationale: {summary.differential_rationale}")

asyncio.run(main())
```

### Example 2: Multi-Agent Collaboration

```python
import asyncio
from exaid_core import EXAID

async def main():
    exaid = EXAID()
    
    # Multiple agents collaborating
    await exaid.received_trace("Orchestrator", "Starting case analysis")
    await exaid.received_trace("DiagnosticAgent", "Analyzing symptoms")
    await exaid.received_trace("TreatmentAgent", "Recommending treatment")
    
    summary = await exaid.received_trace("Orchestrator", "Case analysis complete")
    
    if summary:
        print(f"Agent Contributions: {summary.agent_contributions}")
        print(f"Status/Action: {summary.status_action}")
        print(f"Recommendation/Next Step: {summary.recommendation_next_step}")

asyncio.run(main())
```

### Example 3: Streaming Tokens

```python
import asyncio
from exaid_core import EXAID

async def main():
    exaid = EXAID()
    
    # Simulate streaming tokens
    tokens = ["Patient", " presents", " with", " chest", " pain", ".", " ", "History", " of", " hypertension", "."]
    
    # Process each token individually
    for token in tokens:
        summary = await exaid.on_new_token("DoctorAgent", token)
        if summary:
            print(f"Status/Action: {summary.status_action}")
            print(f"Key Findings: {summary.key_findings}")
            print(f"Differential/Rationale: {summary.differential_rationale}")
        await asyncio.sleep(0.1)  # Simulate streaming delay
    
    # Flush any remaining tokens (parks tail content; no summary is produced here)
    await exaid.flush_agent("DoctorAgent")

asyncio.run(main())
```

### Example 4: CDSS Integration

```python
import asyncio
from demos.cdss_example.cdss import CDSS
from demos.cdss_example.schema.clinical_case import ClinicalCase

async def main():
    cdss = CDSS()
    
    # Create a clinical case
    case = ClinicalCase(
        patient_id="PAT-001",
        age=58,
        sex="M",
        chief_complaint="Chest pain and shortness of breath",
        history_of_present_illness="6-hour history of substernal chest pain..."
    )
    
    # Process the case
    result = await cdss.process_case(case)
    
    # Access results
    print(f"Final recommendation: {result['final_recommendation']['action']}")
    print(f"Agents called: {result['agents_called']}")
    print(f"Total summaries: {len(result['agent_summaries'])}")

asyncio.run(main())
```

### Example 5: Querying History

```python
import asyncio
from exaid_core import EXAID

async def main():
    exaid = EXAID()
    
    # Process multiple traces
    await exaid.received_trace("Agent1", "Trace 1")
    await exaid.received_trace("Agent2", "Trace 2")
    await exaid.received_trace("Agent1", "Trace 3")
    
    # Get all summaries
    all_summaries = exaid.get_all_summaries()
    print(f"Total summaries: {len(all_summaries)}")
    
    # Get summaries for specific agent
    agent1_summaries = exaid.get_summaries_by_agent("Agent1")
    print(f"Agent1 summaries: {len(agent1_summaries)}")
    
    # Get trace count
    trace_count = exaid.get_agent_trace_count("Agent1")
    print(f"Agent1 trace count: {trace_count}")

asyncio.run(main())
```

---

## Design Decisions and Rationale

### Why LLM-Based Trigger Logic?

Traditional threshold-based buffering (e.g., "trigger after N traces") doesn't account for:
- Variable trace lengths
- Different reasoning patterns
- Natural completion points
- Topic changes

LLM-based triggers provide intelligent, context-aware summarization timing.

### Why Character Limits?

Physician-focused design requires:
- Quick comprehension
- Concise, actionable information
- Focus on essential details
- Reduced cognitive load

Character limits enforce these requirements.

---

## Future Enhancements

Potential improvements and extensions:

1. **Event Subscription System**: Pub-sub mechanism for summary events
2. **Trace Semantics**: More structured trace formats with metadata
3. **Custom Validation Rules**: User-defined validation criteria
4. **Summary Templates**: Customizable summary formats
5. **Multi-Language Support**: Internationalization for non-English traces
6. **Performance Optimization**: Caching, batching, and parallel processing
7. **Monitoring and Metrics**: Summary quality metrics and dashboards
8. **Integration Hooks**: Webhooks and API endpoints for external integration
9. **Enhanced TokenGate**: More sophisticated tokenization and boundary detection
10. **LangGraph Streaming**: Full streaming support for LangGraph workflows
11. **Summary Export Formats**: Additional export formats (CSV, XML, etc.)
12. **Agent Registry**: Centralized agent registration and management system

---

## License

MIT

---

## Contributing

This is an experimental prototype. Contributions and feedback are welcome!

---

*Last Updated: Generated from codebase analysis*
