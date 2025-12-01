"""Pydantic models for agent-to-agent communication messages.

These schemas are used with LangChain's with_structured_output() to ensure
agents return properly formatted consultation requests, challenges, and other
inter-agent messages.
"""

from pydantic import BaseModel, Field
from typing import Optional


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
