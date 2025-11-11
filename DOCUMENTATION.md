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

#### `latest_summary() -> Union[AgentSummary, str]`

Returns the most recent summary as an `AgentSummary` object, or a string message if no summaries exist yet.

#### `get_all_summaries() -> list[AgentSummary]`

Returns all summaries as a list of `AgentSummary` objects.

#### `get_summaries_by_agent(agent_id: str) -> list[AgentSummary]`

Filters summaries to return only those involving a specific agent ID.

#### `get_summary_json(summary: Optional[AgentSummary] = None) -> Optional[str]`

Converts a summary to JSON format. If no summary is provided, returns the latest summary as JSON.

**Helper Methods**:

- `_format_summary_for_history(summary: AgentSummary) -> str`: Converts an `AgentSummary` to a string representation for use in prompt history
- `_format_summaries_history(summaries: list[AgentSummary]) -> list[str]`: Converts a list of summaries to string representations for prompt context

---

### `agents/buffer_agent.py` - Intelligent Trace Buffer

**Purpose**: Buffers traces per agent and uses an LLM to intelligently decide when summarization should be triggered based on trace content rather than simple thresholds.

**Key Components**:

```python
class TraceData(BaseModel):
    trace_text: List[str]
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
2. Records the trace with timestamp and metadata in `traces` dictionary
3. If this is the first trace, always triggers summarization
4. Otherwise, uses an LLM prompt to evaluate if summarization should trigger
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

    # Add the new chunk to the traces
    if agent_id not in self.traces:
        self.traces[agent_id] = TraceData(trace_text=[], count=0)
    self.traces[agent_id].count += 1
    trace_text = f"| Timestamp: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} | Trace Length: {len(chunk)} | {chunk}"
    self.traces[agent_id].trace_text.append(trace_text)
    
    # If this is the first trace, always trigger
    if not self.buffer:
        self.buffer.append(tagged_chunk)
        return True
    
    # Get previous traces (before adding the new one)
    previous_traces = "\n".join(self.buffer)
    
    # Add the new chunk to buffer
    self.buffer.append(tagged_chunk)

    flag_chain = self.flag_prompt | llm
    flag_response = await flag_chain.ainvoke({
        "previous_trace": previous_traces,
        "new_trace": tagged_chunk
    })
    # Extract text content from AIMessage object
    response_text = flag_response.content.strip().upper()

    return "YES" in response_text
```

#### `peek() -> list[str]`

Returns a copy of the current buffer without flushing it. Useful for capturing buffer state before summarization.

#### `flush() -> list[str]`

Returns a copy of the buffer and clears it. Called after summarization is triggered.

#### `get_trace_count(agent_id: str) -> int`

Returns the total number of traces received from a specific agent.

#### `get_traces(agent_id: str) -> List[str]`

Returns all trace texts for a specific agent.

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

### `llm.py` - LLM Client Configuration

**Purpose**: Centralized configuration for the LLM client used throughout EXAID.

**Code**:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen/qwen3-4b-2507",
    base_url="https://ed7a5a297b8b.ngrok-free.app/v1",
    api_key="dummykey"
)
```

**Configuration**:
- Uses LangChain's `ChatOpenAI` for OpenAI-compatible API endpoints
- Currently configured for a custom endpoint (ngrok tunnel)
- Can be easily modified to use OpenAI, Anthropic, or other providers

**Usage**: Imported by:
- `BufferAgent` for trigger decisions
- `SummarizerAgent` for summary generation

**Note**: Update this file with your actual API credentials and endpoint. For production use, consider using environment variables for sensitive information.

---

### `demo.py` - Example Usage

**Purpose**: Demonstrates EXAID usage with a complete medical case workflow.

**Features**:
- Simulates traces from multiple specialized agents
- Shows complete diagnostic workflow from initial case to treatment planning
- Formats summaries for clean console display

**Code Snippet**:
```python
async def main():
    exaid = EXAID()
    # Simulate traces from two agents
    traces = [
        ("Orchestrator", "Patient case received: 62F, 6-week history of fatigue..."),
        ("InfectiousDiseaseAgent", "Considering subacute infections..."),
        ("HematologyAgent", "Constitutional B symptoms raise concern..."),
        # ... more traces
    ]

    for agent_id, text in traces:
        summary = await exaid.received_trace(agent_id, text)
        if summary:
            print(f"\n{'='*60}")
            print(f"Summary Update")
            print(f"{'='*60}")
            print(format_summary_display(summary))
            print()
```

**Example Output Format**:
```
┌─ Agents: Orchestrator, InfectiousDiseaseAgent, HematologyAgent
├─ Action: Initial diagnostic hypothesis generation
├─ Reasoning: Multiple agents evaluating constitutional symptoms
├─ Findings: B symptoms suggest lymphoproliferative disorders
└─ Next Steps: Order CBC, LDH, peripheral smear, CT imaging
```

---

### `requirements.txt` - Dependencies

**Purpose**: Lists all Python package dependencies required for EXAID.

**Contents**:
```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
pydantic>=2.0.0
```

**Dependencies**:
- **langchain**: Core LangChain framework for LLM integration
- **langchain-community**: Community integrations
- **langchain-core**: Core LangChain abstractions
- **langchain-openai**: OpenAI integration for ChatOpenAI
- **pydantic**: Data validation and structured output

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
| `trace_text` | `List[str]` | List of trace text strings with timestamps |
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

#### `get_summary_json(summary: Optional[AgentSummary] = None) -> Optional[str]`

Get summary as JSON string.

**Parameters**:
- `summary` (Optional[AgentSummary]): Summary to convert (defaults to latest)

**Returns**: JSON string or `None`

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

#### `get_traces(agent_id: str) -> List[str]`

Get all traces for an agent.

**Parameters**:
- `agent_id` (str): Agent identifier

**Returns**: List of trace strings

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

Configure the LLM client in `llm.py`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="your-model-name",
    base_url="https://your-api-endpoint.com/v1",
    api_key="your-api-key"
)
```

**Supported Providers**:
- OpenAI (via `base_url="https://api.openai.com/v1"`)
- Anthropic (via `ChatAnthropic`)
- Custom OpenAI-compatible endpoints
- Local models via compatible APIs

### Environment Variables (Recommended)

For production use, consider using environment variables:

```python
import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "default-model"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
```


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

### Example 3: Exporting Data

```python
import asyncio
import json
from exaid import EXAID

async def main():
    exaid = EXAID()
    
    # Process traces
    await exaid.received_trace("Agent1", "Trace 1")
    summary = await exaid.received_trace("Agent2", "Trace 2")
    
    if summary:
        # Export summary as JSON
        summary_json = exaid.get_summary_json(summary)
        print("Summary JSON:", summary_json)
        
        # Save to file
        with open("summary.json", "w") as f:
            f.write(summary_json)

asyncio.run(main())
```

### Example 4: Querying History

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

---

## License

MIT

---

## Contributing

This is an experimental prototype. Contributions and feedback are welcome!

---

*Last Updated: Generated from codebase analysis*

