from typing import TypedDict, Optional
from exaid import EXAID


class CDSSGraphState(TypedDict):
    """State schema for the CDSS LangGraph workflow"""
    
    case_text: str
    """The clinical case input text"""
    
    orchestrator_analysis: Optional[str]
    """Orchestrator's initial analysis of the case"""
    
    agents_to_call: Optional[dict]
    """Dictionary indicating which agents should be called.
    Format: {"laboratory": bool, "cardiology": bool}
    """
    
    consultation_request: Optional[str]
    """Agent name requested for consultation (e.g., "cardiology", "laboratory").
    Only reasoning agents can set this field.
    """
    
    consulted_agents: Optional[set[str]]
    """Set of agents that have been consulted. Used for loop prevention."""
    
    laboratory_findings: Optional[str]
    """Laboratory agent's findings and recommendations"""
    
    cardiology_findings: Optional[str]
    """Cardiology agent's findings and recommendations"""
    
    final_synthesis: Optional[str]
    """Final synthesis from orchestrator combining all findings"""
    
    exaid: EXAID
    """EXAID instance for trace capture and summarization"""

