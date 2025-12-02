"""Simplified CDSS graph edges - orchestrator-driven routing"""

from typing import Literal
from demos.cdss_example.schema.graph_state import CDSSGraphState


def route_from_orchestrator(state: CDSSGraphState) -> Literal["laboratory", "cardiology", "internal_medicine", "radiology", "synthesis"]:
    """Route from orchestrator based on next_specialist_to_call
    
    Orchestrator decides which specialist to call next or if synthesis should begin.
    Simple pass-through of orchestrator's decision.
    
    Args:
        state: Current graph state
        
    Returns:
        Specialist name or "synthesis"
    """
    next_specialist = state.get("next_specialist_to_call", "synthesis")
    
    # Validate and return
    valid_options = ["laboratory", "cardiology", "internal_medicine", "radiology", "synthesis"]
    if next_specialist in valid_options:
        return next_specialist
    
    # Default to synthesis if invalid
    return "synthesis"


def route_to_orchestrator(state: CDSSGraphState) -> Literal["orchestrator"]:
    """Route specialist nodes back to orchestrator
    
    All specialists return to orchestrator for compression and routing decisions.
    
    Args:
        state: Current graph state
        
    Returns:
        Always "orchestrator"
    """
    return "orchestrator"


