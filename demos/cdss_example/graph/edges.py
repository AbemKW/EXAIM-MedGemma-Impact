from typing import Literal
from demos.cdss_example.schema.graph_state import CDSSGraphState


def route_to_orchestrator(state: CDSSGraphState) -> Literal["orchestrator"]:
    """Route reasoning agents back to orchestrator after analysis"""
    return "orchestrator"


def evaluate_orchestrator_routing(state: CDSSGraphState) -> Literal["laboratory", "cardiology", "synthesis"]:
    """Evaluate orchestrator routing: route based on agents_to_call, consensus, debate requests, and max iterations"""
    agents_to_call = state.get("agents_to_call")
    
    # Check if synthesis was explicitly requested
    if agents_to_call and agents_to_call.get("synthesis", False):
        return "synthesis"
    
    # Check consensus status
    consensus_status = state.get("consensus_status")
    if consensus_status and consensus_status.get("consensus_reached", False):
        return "synthesis"
    
    # Check max iterations
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    if iteration_count >= max_iterations:
        return "synthesis"
    
    # Check for debate requests that need routing
    debate_requests = state.get("debate_requests") or []
    unresolved_debates = [d for d in debate_requests if not d.get("resolved", False)]
    if unresolved_debates:
        # The orchestrator should have already set agents_to_call for the debate target
        # But we can also check here as a fallback
        next_debate = unresolved_debates[0]
        target_agent = next_debate.get("to_agent")
        if target_agent == "laboratory":
            return "laboratory"
        elif target_agent == "cardiology":
            return "cardiology"
    
    # Route based on agents_to_call (set by orchestrator)
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
    
    # Check consensus and max iterations first
    consensus_status = state.get("consensus_status")
    if consensus_status and consensus_status.get("consensus_reached", False):
        return "synthesis"
    
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    if iteration_count >= max_iterations:
        return "synthesis"
    
    if agents_to_call and agents_to_call.get("cardiology", False):
        return "cardiology"
    
    # Cardiology not needed, go to synthesis
    return "synthesis"

