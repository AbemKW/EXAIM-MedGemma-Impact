from typing import Literal
from cdss_demo.schema.graph_state import CDSSGraphState


def should_call_laboratory(state: CDSSGraphState) -> Literal["laboratory", "cardiology", "synthesis"]:
    """Route to laboratory node if needed, otherwise check cardiology or go to synthesis"""
    agents_to_call = state.get("agents_to_call")
    
    if agents_to_call and agents_to_call.get("laboratory", False):
        return "laboratory"
    
    # If laboratory not needed, check cardiology
    if agents_to_call and agents_to_call.get("cardiology", False):
        return "cardiology"
    
    # Neither agent needed, go to synthesis
    return "synthesis"


def should_call_cardiology(state: CDSSGraphState) -> Literal["cardiology", "synthesis"]:
    """Route to cardiology node if needed, otherwise go to synthesis"""
    agents_to_call = state.get("agents_to_call")
    
    if agents_to_call and agents_to_call.get("cardiology", False):
        return "cardiology"
    
    # Cardiology not needed, go to synthesis
    return "synthesis"

