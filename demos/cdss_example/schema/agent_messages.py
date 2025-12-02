"""Agent-to-agent communication message schemas.

This module provides both Pydantic models (for LangChain structured output)
and TypedDict schemas (for state type hints and runtime dict compatibility).

Pydantic models are used by agents with with_structured_output() to ensure
proper formatting. TypedDict schemas are used for state field type hints
and provide structural typing for dicts passed through LangGraph state.
"""

from pydantic import BaseModel, Field
from typing import Optional, TypedDict


# ============================================================================
# Pydantic Models (for agent structured output)
# ============================================================================

class ConsultationRequest(BaseModel):
    """Request for consultation from another specialist agent.
    
    Used by agents to formally request input from other specialists
    with clinical justification.
    """
    
    requested_specialist: str = Field(
        description="The specialist to consult. Must be one of: 'internal_medicine', 'radiology', 'laboratory', 'cardiology'"
    )
    
    reason: str = Field(
        description="Brief clinical justification for why this consultation is needed (1-2 sentences)"
    )


# ============================================================================
# TypedDict Schemas (for state type hints)
# ============================================================================

class ConsultationRequestDict(TypedDict):
    """TypedDict version of ConsultationRequest for state typing.
    
    Used in CDSSGraphState for the consultation_request field.
    Nodes convert Pydantic ConsultationRequest to this dict format.
    """
    requested_specialist: str
    reason: str


class ChallengeRequestDict(TypedDict):
    """Challenge/critique issued by one agent to another.
    
    Used by nodes to surface challenges to the orchestrator.
    The orchestrator converts this to DebateEntryDict before storing.
    """
    from_agent: str
    to_agent: str
    question: str


class DebateEntryDict(TypedDict):
    """Full debate entry stored in state's debate_requests list.
    
    Extends ChallengeRequestDict with timestamp and resolution tracking.
    Orchestrator creates these from ChallengeRequestDict and manages them.
    """
    from_agent: str
    to_agent: str
    question: str
    timestamp: str
    resolved: bool
