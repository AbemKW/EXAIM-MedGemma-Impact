"""EXAID Core - Explanation-based AI Debugging Framework

This module contains the core EXAID research artifact including:
- TokenGate: Token streaming pre-buffer with syntax awareness
- BufferAgent: Intelligent trace buffer with LLM-based trigger
- SummarizerAgent: LLM-based summarization with structured output
- EXAID: Main orchestrator class
- AgentSummary: Structured summary schema

"""

__all__ = ['EXAID', 'AgentSummary', 'TokenGate', 'BufferAgent', 'SummarizerAgent']

# Lazy imports using PEP 562 __getattr__
# This allows importing exaid_core without pulling in heavy dependencies
# like LangChain unless the specific classes are accessed


def __getattr__(name: str):
    """Lazy import for module-level attributes (PEP 562)."""
    if name == 'TokenGate':
        from exaid_core.token_gate.token_gate import TokenGate
        return TokenGate
    elif name == 'AgentSummary':
        from exaid_core.schema.agent_summary import AgentSummary
        return AgentSummary
    elif name == 'EXAID':
        from exaid_core.exaid import EXAID
        return EXAID
    elif name == 'BufferAgent':
        from exaid_core.buffer_agent.buffer_agent import BufferAgent
        return BufferAgent
    elif name == 'SummarizerAgent':
        from exaid_core.summarizer_agent.summarizer_agent import SummarizerAgent
        return SummarizerAgent
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
