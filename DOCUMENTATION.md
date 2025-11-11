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

**EXAID** (Experimental Agent Intelligence Documentation) is a Python framework designed for capturing, buffering, and summarizing reasoning traces from multiple AI agents in real-time. Originally designed for medical multi-agent reasoning workflows, EXAID enables specialized agents (e.g., `InfectiousDiseaseAgent`, `HematologyAgent`, `OncologyAgent`) to collaborate on complex cases while their reasoning traces are intelligently captured and condensed into structured summaries optimized for physician understanding.

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

### `exaid.py` - Main Orchestrator

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

#### `received_trace(id: str, text: str) -> Optional[AgentSummary]`

The main entry point for processing agent traces. This method:

1. Adds the trace to the buffer via `BufferAgent.addchunk()`
2. If summarization is triggered:
   - Retrieves previous summaries for context
   - Generates summary using `SummarizerAgent`
   - Stores the summary
   - Returns the `AgentSummary`

**Code Snippet**:
```python
async def received_trace(self, id: str, text: str) -> Optional[AgentSummary]:
    """Process a trace from an agent. Returns an AgentSummary if summarization 
    was triggered, None otherwise."""
    trigger = await self.buffer_agent.addchunk(id, text)
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

#### `async received_streamed_tokens(agent_id: str, token_generator: AsyncIterator[str]) -> Optional[AgentSummary]`

Processes streaming tokens from an agent using TokenGate for intelligent chunking. This method:
1. Receives tokens from an async iterator
2. Uses TokenGate to accumulate tokens into meaningful chunks
3. Processes chunks through BufferAgent when ready
4. Returns the last summary generated, if any

**Parameters**:
- `agent_id` (str): Agent identifier
- `token_generator` (AsyncIterator[str]): Async iterator yielding token strings

**Returns**: Last `AgentSummary` generated, or `None` if no summary was triggered

**Helper Methods**:

- `_format_summary_for_history(summary: AgentSummary) -> str`: Converts an `AgentSummary` to a string representation for use in prompt history
- `_format_summaries_history(summaries: list[AgentSummary]) -> list[str]`: Converts a list of summaries to string representations for prompt context

---

### `agents/buffer_agent.py` - Intelligent Trace Buffer

**Purpose**: Buffers traces per agent and uses an LLM to intelligently decide when summarization should be triggered based on trace content rather than simple thresholds.

**Key Components**:

```python
class TraceData(BaseModel):
    count: int

class BufferAgent:
    def __init__(self):
        self.buffer: list[str] = []
        self.llm = llm
        self.traces: dict[str, TraceData] = {}
```

**Core Methods**:

#### `addchunk(agent_id: str, chunk: str) -> bool`

Adds a trace chunk to the buffer and determines if summarization should be triggered.

**Process**:
1. Tags the chunk with agent ID: `| {agent_id} | {chunk}`
2. Records the trace count in `traces` dictionary
3. Adds the tagged chunk to the buffer
4. Uses an LLM prompt to evaluate if summarization should trigger (even for first trace)
5. Returns `True` if summarization should be triggered, `False` otherwise

**LLM Trigger Logic**:
The buffer uses a prompt that asks the LLM to reply with "YES" or "NO" based on:
- Completion of thoughts or reasoning steps
- Topic or focus changes
- Sufficient accumulated context
- Natural pauses or conclusions

**Code Snippet**:
```python
async def addchunk(self, agent_id: str, chunk: str) -> bool:
    tagged_chunk = f"| {agent_id} | {chunk}"
    if agent_id not in self.traces:
        self.traces[agent_id] = TraceData(count=0)
    self.traces[agent_id].count += 1
    
    # Check if buffer was empty before adding this chunk
    was_empty = not self.buffer
    
    # Always add the chunk to buffer
    self.buffer.append(tagged_chunk)
    
    # Always call the LLM to decide if summarization should be triggered
    previous_traces = "\n".join(self.buffer[:-1]) if not was_empty else ""
    
    flag_chain = self.flag_prompt | self.llm
    flag_response = await flag_chain.ainvoke({
        "previous_trace": previous_traces if previous_traces else "(No previous traces - this is the first trace)",
        "new_trace": tagged_chunk
    })
    return "YES" in flag_response.content.strip().upper()
```

#### `peek() -> list[str]`

Returns a copy of the current buffer without flushing it. Useful for capturing buffer state before summarization.

#### `flush() -> list[str]`

Returns a copy of the buffer and clears it. Called after summarization is triggered.

#### `get_trace_count(agent_id: str) -> int`

Returns the total number of traces received from a specific agent.

**Prompt Template**:
```python
self.flag_prompt = ChatPromptTemplate.from_messages([
    ("system",
    "You are monitoring the reasoning streams of multiple AI agents. "
    "Your task is to decide if the new reasoning trace should trigger summarization. "
    "Reply with EXACTLY 'YES' (all caps) if ANY of these conditions are met:\n"
    "- The new trace completes a thought or reasoning step\n"
    "- The topic or focus has changed from previous traces\n"
    "- Enough context has accumulated to warrant a summary\n"
    "- The reasoning has reached a natural pause or conclusion\n\n"
    "Reply with EXACTLY 'NO' (all caps) if the new trace is just continuing the same line of reasoning without completion.\n"
    "Be decisive - prefer YES when in doubt, as summaries help track agent progress."),
    ("user", "Previous traces in buffer:\n{previous_trace}\n\nNew trace to evaluate:\n{new_trace}\n\nShould this trigger summarization? Reply with only 'YES' or 'NO'."),
])
```

---

### `agents/summarizer_agent.py` - Summary Generator

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
```python
self.summarize_prompt = ChatPromptTemplate.from_messages([    
    ("system", "You are an expert summarizer for a medical multi-agent reasoning system. "
    "Extract essential information about agent actions and clinical reasoning from the new buffer. "
    "Focus on what agents did and why, using brief, action-oriented language. "
    "Be concise and practical - physicians need to quickly understand agent decisions. "
    "Extract only new information from the buffer, do not repeat previous summaries. "
    "Identify all agents mentioned in the buffer. "
    "CRITICAL: Strictly enforce character limits - action: MAX 100 chars, reasoning: MAX 200 chars, findings: MAX 150 chars, next_steps: MAX 100 chars. "
    "If content exceeds limits, prioritize the most essential information and truncate."),
    ("user", "Summary history:\n[ {summary_history} ]\n\nLatest summary:\n{latest_summary}\n\nNew reasoning buffer:\n{new_buffer}\n\nExtract structured summary of new agent actions and reasoning."),
])
```

**Features**:
- Uses structured output with Pydantic models for consistent formatting
- Enforces character limits (action: 100, reasoning: 200, findings: 150, next_steps: 100)
- Optimized for medical/clinical reasoning

---

### `agents/base_agent.py` - Base Agent Interface

**Purpose**: Abstract base class defining the interface for agents that can be integrated with EXAID.

**Code**:
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    @abstractmethod
    async def act(self, input: str) -> str:
        pass
```

**Usage**: Agents that integrate with EXAID should inherit from `BaseAgent` and implement the `act()` method. This provides a consistent interface for agent integration, though EXAID can also work with agents that don't inherit from this class as long as they can send traces via `received_trace()`.

---

### `schema/agent_summary.py` - Summary Data Model

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

### `agents/token_gate.py` - Token Streaming Pre-Buffer

**Purpose**: A lightweight, syntax-aware pre-buffer that regulates token flow into BufferAgent for streaming scenarios. It does not interpret meaning - it only decides when enough structure has accumulated to pass tokens upstream for semantic evaluation.

**Key Components**:

```python
class TokenGate:
    def __init__(
        self,
        min_tokens: int = 35,
        max_tokens: int = 90,
        boundary_cues: str = ".?!\n",
        silence_timer: float = 15,
        max_wait_timeout: float = 40
    ):
```

**Core Methods**:

#### `async add_token(agent_id: str, token: str) -> Optional[str]`

Adds a token to the buffer for the given agent. If flush conditions are met, returns the buffered text and clears the buffer.

**Flush Conditions**:
- Maximum token cap reached (`max_tokens`)
- Minimum token threshold reached (`min_tokens`) AND boundary cue detected
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
- Per-agent token buffering
- Configurable token thresholds
- Boundary cue detection (punctuation, newlines)
- Timeout mechanisms (silence timer, max wait timeout)
- Approximate token counting using whitespace-based splitting

---

### `llm.py` - LLM Client Configuration

**Purpose**: Centralized configuration for the LLM client used throughout EXAID.

**Code**:
```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    api_key=os.getenv("LLM_API_KEY")
)
```

**Configuration**:
- Uses LangChain's `ChatOpenAI` for OpenAI-compatible API endpoints
- Supports environment variables for configuration (`.env` file)
- Default configuration uses Google Gemini API endpoint
- Can be easily modified to use OpenAI, Anthropic, or other providers

**Environment Variables**:
- `LLM_MODEL`: Model name (default: "gemini-2.5-flash")
- `LLM_BASE_URL`: API endpoint URL
- `LLM_API_KEY`: API key for authentication

**Usage**: Imported by:
- `BufferAgent` for trigger decisions
- `SummarizerAgent` for summary generation

**Note**: For production use, use environment variables for sensitive information. Create a `.env` file in the project root.

---

### `cdss_demo/` - Clinical Decision Support System Demo

**Purpose**: Complete demonstration of EXAID integrated with a multi-agent clinical decision support system using LangGraph for workflow orchestration.

#### `cdss_demo/cdss.py` - CDSS Orchestrator

**Purpose**: Orchestrates the clinical decision support system workflow using LangGraph.

**Key Components**:

```python
class CDSS:
    def __init__(self):
        self.exaid = EXAID()
        self.orchestrator = OrchestratorAgent()
        self.cardiology = CardiologyAgent()
        self.laboratory = LaboratoryAgent()
        self.graph = build_cdss_graph()
```

**Core Methods**:

#### `async process_case(case: Union[ClinicalCase, str], use_streaming: bool = True) -> dict`

Processes a clinical case through the multi-agent system using LangGraph.

**Parameters**:
- `case`: ClinicalCase object or free-text case description
- `use_streaming`: Whether to use streaming token processing (currently ignored as LangGraph doesn't support streaming)

**Returns**: Dictionary containing:
- `case_summary`: Case text summary
- `agent_summaries`: List of all summaries generated
- `final_recommendation`: Final summary with recommendations
- `trace_count`: Trace counts per agent
- `agents_called`: Which agents were invoked

#### `get_all_summaries() -> list[AgentSummary]`

Get all summaries from EXAID.

#### `get_summaries_by_agent(agent_id: str) -> list[AgentSummary]`

Get summaries for a specific agent.

#### `reset()`

Reset the CDSS system (creates new EXAID instance).

#### `cdss_demo/demo_cdss.py` - Example Clinical Cases

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

#### `cdss_demo/agents/` - Specialized Medical Agents

**Purpose**: Specialized agents for clinical decision support.

- **OrchestratorAgent**: Coordinates case analysis and agent invocation
- **CardiologyAgent**: Provides cardiology-specific analysis
- **LaboratoryAgent**: Analyzes laboratory results and findings

#### `cdss_demo/graph/` - LangGraph Workflow

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
langgraph>=0.2.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

**Dependencies**:
- **langchain**: Core LangChain framework for LLM integration
- **langchain-community**: Community integrations
- **langchain-core**: Core LangChain abstractions
- **langchain-openai**: OpenAI integration for ChatOpenAI
- **langgraph**: LangGraph for workflow orchestration (used in CDSS demo)
- **pydantic**: Data validation and structured output
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
from exaid import EXAID

async def main():
    # Initialize EXAID
    exaid = EXAID()
    
    # Send traces from agents
    summary = await exaid.received_trace("agent_1", "Some reasoning trace")
    
    # Check if summary was generated
    if summary:
        print("New summary:", summary.action)
        
        # Get summary as JSON
        json_summary = exaid.get_summary_json(summary)
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
       agents = summary.agents
       action = summary.action
       reasoning = summary.reasoning
       findings = summary.findings
       next_steps = summary.next_steps
   ```

4. **Export Data**:
   ```python
   # Export summary as JSON
   json_summary = exaid.get_summary_json()
   ```

---

## API Reference

### EXAID Class

#### `__init__()`

Initialize EXAID instance.

#### `async received_trace(id: str, text: str) -> Optional[AgentSummary]`

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

#### `async received_streamed_tokens(agent_id: str, token_generator: AsyncIterator[str]) -> Optional[AgentSummary]`

Process streaming tokens from an agent using TokenGate for intelligent chunking.

**Parameters**:
- `agent_id` (str): Agent identifier
- `token_generator` (AsyncIterator[str]): Async iterator yielding token strings

**Returns**: Last `AgentSummary` generated, or `None` if no summary was triggered

### BufferAgent Class

#### `async addchunk(agent_id: str, chunk: str) -> bool`

Add a trace chunk and determine if summarization should trigger.

**Parameters**:
- `agent_id` (str): Agent identifier
- `chunk` (str): Trace text

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

#### `async add_token(agent_id: str, token: str) -> Optional[str]`

Add a token to the buffer for the given agent.

**Parameters**:
- `agent_id` (str): Agent identifier
- `token` (str): Token string to add

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

Configure the LLM client using environment variables (recommended):

Create a `.env` file in the project root:

```bash
LLM_MODEL=your-model-name
LLM_BASE_URL=https://your-api-endpoint.com/v1
LLM_API_KEY=your-api-key
```

Alternatively, configure directly in `llm.py`:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "your-model-name"),
    base_url=os.getenv("LLM_BASE_URL", "https://your-api-endpoint.com/v1"),
    api_key=os.getenv("LLM_API_KEY")
)
```

**Supported Providers**:
- OpenAI (via `base_url="https://api.openai.com/v1"`)
- Google Gemini (via `base_url="https://generativelanguage.googleapis.com/v1beta/openai/"`)
- Anthropic (via `ChatAnthropic` from `langchain_anthropic`)
- Custom OpenAI-compatible endpoints
- Local models via compatible APIs


---

## Examples

### Example 1: Simple Single Agent

```python
import asyncio
from exaid import EXAID

async def main():
    exaid = EXAID()
    
    # Send multiple traces from one agent
    await exaid.received_trace("DoctorAgent", "Reviewing patient symptoms")
    await exaid.received_trace("DoctorAgent", "Ordering lab tests")
    
    summary = await exaid.received_trace("DoctorAgent", "Lab results received")
    
    if summary:
        print(f"Action: {summary.action}")
        print(f"Reasoning: {summary.reasoning}")

asyncio.run(main())
```

### Example 2: Multi-Agent Collaboration

```python
import asyncio
from exaid import EXAID

async def main():
    exaid = EXAID()
    
    # Multiple agents collaborating
    await exaid.received_trace("Orchestrator", "Starting case analysis")
    await exaid.received_trace("DiagnosticAgent", "Analyzing symptoms")
    await exaid.received_trace("TreatmentAgent", "Recommending treatment")
    
    summary = await exaid.received_trace("Orchestrator", "Case analysis complete")
    
    if summary:
        print(f"Agents involved: {', '.join(summary.agents)}")
        print(f"Action: {summary.action}")

asyncio.run(main())
```

### Example 3: Streaming Tokens

```python
import asyncio
from exaid import EXAID

async def token_stream():
    """Simulate a token stream"""
    tokens = ["Patient", " presents", " with", " chest", " pain", ".", " ", "History", " of", " hypertension", "."]
    for token in tokens:
        yield token
        await asyncio.sleep(0.1)  # Simulate streaming delay

async def main():
    exaid = EXAID()
    
    # Process streaming tokens
    summary = await exaid.received_streamed_tokens("DoctorAgent", token_stream())
    
    if summary:
        print(f"Action: {summary.action}")
        print(f"Reasoning: {summary.reasoning}")

asyncio.run(main())
```

### Example 4: CDSS Integration

```python
import asyncio
from cdss_demo.cdss import CDSS
from cdss_demo.schema.clinical_case import ClinicalCase

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
from exaid import EXAID

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

