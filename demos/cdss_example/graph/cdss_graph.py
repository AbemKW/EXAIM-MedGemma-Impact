from langgraph.graph import StateGraph, END
from demos.cdss_example.schema.graph_state import CDSSGraphState
from .nodes import (
    orchestrator_node,
    laboratory_node,
    cardiology_node,
    internal_medicine_node,
    radiology_node,
    synthesis_node
)
from .edges import (
    should_call_laboratory,
    route_to_orchestrator
)


def build_cdss_graph():
    """Build and compile the CDSS LangGraph workflow
    
    Phase 3 upgrade: Supports all four specialist agents (laboratory, cardiology, 
    internal_medicine, radiology) with conditional routing for consultations and debates.
    """
    
    # Create the graph
    workflow = StateGraph(CDSSGraphState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("laboratory", laboratory_node)
    workflow.add_node("cardiology", cardiology_node)
    workflow.add_node("internal_medicine", internal_medicine_node)
    workflow.add_node("radiology", radiology_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Add conditional edges from orchestrator to all four specialists and synthesis
    workflow.add_conditional_edges(
        "orchestrator",
        should_call_laboratory,
        {
            "laboratory": "laboratory",
            "cardiology": "cardiology",
            "internal_medicine": "internal_medicine",
            "radiology": "radiology",
            "synthesis": "synthesis"
        }
    )
    
    # Add conditional edges from all specialist agents back to orchestrator
    workflow.add_conditional_edges(
        "laboratory",
        route_to_orchestrator,
        {
            "orchestrator": "orchestrator"
        }
    )
    
    workflow.add_conditional_edges(
        "cardiology",
        route_to_orchestrator,
        {
            "orchestrator": "orchestrator"
        }
    )
    
    workflow.add_conditional_edges(
        "internal_medicine",
        route_to_orchestrator,
        {
            "orchestrator": "orchestrator"
        }
    )
    
    workflow.add_conditional_edges(
        "radiology",
        route_to_orchestrator,
        {
            "orchestrator": "orchestrator"
        }
    )
    
    # Add edge from synthesis to END
    workflow.add_edge("synthesis", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


