import sys
from pathlib import Path

# CRITICAL: Add parent directory to path BEFORE any other imports
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph.graph import StateGraph, END
from cdss_demo.schema.graph_state import CDSSGraphState
from cdss_demo.graph.nodes import (
    orchestrator_node,
    laboratory_node,
    cardiology_node,
    synthesis_node
)
from cdss_demo.graph.edges import (
    should_call_laboratory,
    should_call_cardiology
)


def build_cdss_graph():
    """Build and compile the CDSS LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(CDSSGraphState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("laboratory", laboratory_node)
    workflow.add_node("cardiology", cardiology_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Add conditional edges from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        should_call_laboratory,
        {
            "laboratory": "laboratory",
            "cardiology": "cardiology",
            "synthesis": "synthesis"
        }
    )
    
    # Add conditional edges from laboratory
    workflow.add_conditional_edges(
        "laboratory",
        should_call_cardiology,
        {
            "cardiology": "cardiology",
            "synthesis": "synthesis"
        }
    )
    
    # Add edge from cardiology to synthesis
    workflow.add_edge("cardiology", "synthesis")
    
    # Add edge from synthesis to END
    workflow.add_edge("synthesis", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

