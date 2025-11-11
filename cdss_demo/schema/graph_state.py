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
    
    laboratory_findings: Optional[str]
    """Laboratory agent's findings and recommendations"""
    
    cardiology_findings: Optional[str]
    """Cardiology agent's findings and recommendations"""
    
    final_synthesis: Optional[str]
    """Final synthesis from orchestrator combining all findings"""
    
    exaid: EXAID
    """EXAID instance for trace capture and summarization"""

