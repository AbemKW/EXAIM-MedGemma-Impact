"""EXAID Core - Explanation-based AI Debugging Framework

This module contains the core EXAID research artifact including:
- TokenGate: Token streaming pre-buffer with syntax awareness
- BufferAgent: Intelligent trace buffer with LLM-based trigger
- SummarizerAgent: LLM-based summarization with structured output
- EXAID: Main orchestrator class
- AgentSummary: Structured summary schema
"""

from exaid_core.exaid import EXAID
from exaid_core.schema.agent_summary import AgentSummary
from exaid_core.token_gate.token_gate import TokenGate
from exaid_core.buffer_agent.buffer_agent import BufferAgent
from exaid_core.summarizer_agent.summarizer_agent import SummarizerAgent

__all__ = ['EXAID', 'AgentSummary', 'TokenGate', 'BufferAgent', 'SummarizerAgent']
