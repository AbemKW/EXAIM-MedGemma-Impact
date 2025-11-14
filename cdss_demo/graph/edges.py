from typing import Literal
from cdss_demo.schema.graph_state import CDSSGraphState


def route_to_orchestrator(state: CDSSGraphState) -> Literal["orchestrator"]:
    """Route reasoning agents back to orchestrator after analysis"""
    return "orchestrator"


def evaluate_orchestrator_routing(state: CDSSGraphState) -> Literal["laboratory", "cardiology", "synthesis"]:
    """Evaluate orchestrator routing: check consultation requests first, then initial agents_to_call"""
    agents_to_call = state.get("agents_to_call")
    
    # Check if synthesis was explicitly requested
    if agents_to_call and agents_to_call.get("synthesis", False):
        return "synthesis"
    
    # Check consultation request first (handled by orchestrator_node, but check agents_to_call result)
    # The orchestrator_node sets agents_to_call based on consultation_request evaluation
    
    # Route to laboratory if requested
    if agents_to_call and agents_to_call.get("laboratory", False):
        return "laboratory"
    
    # Route to cardiology if requested
    if agents_to_call and agents_to_call.get("cardiology", False):
        return "cardiology"
    
    # Default to synthesis if no agents to call
    return "synthesis"


# Keep old function names for backward compatibility, but use new logic
def should_call_laboratory(state: CDSSGraphState) -> Literal["laboratory", "cardiology", "synthesis"]:
    """Route to laboratory node if needed, otherwise check cardiology or go to synthesis"""
    return evaluate_orchestrator_routing(state)


def should_call_cardiology(state: CDSSGraphState) -> Literal["cardiology", "synthesis"]:
    """Route to cardiology node if needed, otherwise go to synthesis"""
    agents_to_call = state.get("agents_to_call")
    
    if agents_to_call and agents_to_call.get("cardiology", False):
        return "cardiology"
    
    # Cardiology not needed, go to synthesis
    return "synthesis"

