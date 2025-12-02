"""Simplified CDSS graph - orchestrator-driven workflow with specialist reasoning"""

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
    route_from_orchestrator,
    route_to_orchestrator
)


def build_cdss_graph():
    """Build and compile the simplified CDSS LangGraph workflow
    
    Workflow:
    - Orchestrator compresses context and decides next specialist
    - Specialists provide domain reasoning based on case + summary + recent_delta
    - All specialists route back to orchestrator
    - Orchestrator eventually routes to synthesis
    - Synthesis ends the workflow
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
    
    # Add conditional edge from orchestrator to specialists or synthesis
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "laboratory": "laboratory",
            "cardiology": "cardiology",
            "internal_medicine": "internal_medicine",
            "radiology": "radiology",
            "synthesis": "synthesis"
        }
    )
    
    # Add edges from all specialists back to orchestrator
    workflow.add_conditional_edges(
        "laboratory",
        route_to_orchestrator,
        {"orchestrator": "orchestrator"}
    )
    
    workflow.add_conditional_edges(
        "cardiology",
        route_to_orchestrator,
        {"orchestrator": "orchestrator"}
    )
    
    workflow.add_conditional_edges(
        "internal_medicine",
        route_to_orchestrator,
        {"orchestrator": "orchestrator"}
    )
    
    workflow.add_conditional_edges(
        "radiology",
        route_to_orchestrator,
        {"orchestrator": "orchestrator"}
    )
    
    # Add edge from synthesis to END
    workflow.add_edge("synthesis", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app



