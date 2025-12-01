from typing import TypedDict, Optional, List, Dict, Any
from demos.cdss_example.agents.cardiology_agent import CardiologyAgent
from demos.cdss_example.agents.laboratory_agent import LaboratoryAgent
from demos.cdss_example.agents.orchestrator_agent import OrchestratorAgent
from exaid_core.exaid import EXAID


class CDSSGraphState(TypedDict):
    """State schema for the CDSS LangGraph workflow"""
    
    case_text: str
    """The clinical case input text"""
    
    # Agent instances
    orchestrator_agent: Optional[OrchestratorAgent]
    """Orchestrator agent instance"""
    
    laboratory_agent: Optional[LaboratoryAgent]
    """Laboratory agent instance"""
    
    cardiology_agent: Optional[CardiologyAgent]
    """Cardiology agent instance"""
    
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
    
    consulted_agents: Optional[list[str]]
    """List of agents that have been consulted. Used for loop prevention."""
    
    laboratory_findings: Optional[str]
    """Laboratory agent's findings and recommendations"""
    
    cardiology_findings: Optional[str]
    """Cardiology agent's findings and recommendations"""
    
    final_synthesis: Optional[str]
    """Final synthesis from orchestrator combining all findings"""
    
    exaid: EXAID
    """EXAID instance for trace capture and summarization"""
    
    # New fields for collaborative workflow
    agent_turn_history: Optional[List[Dict[str, Any]]]
    """Track each agent's turn with timestamp, agent_id, and findings"""
    
    new_findings_since_last_turn: Optional[Dict[str, str]]
    """Track what's new for each agent since their last turn. Key: agent_id, Value: findings"""
    
    agent_awareness: Optional[Dict[str, List[str]]]
    """Track which findings each agent has seen. Key: agent_id, Value: list of finding IDs or timestamps"""
    
    debate_requests: Optional[List[Dict[str, Any]]]
    """Track when agents want to challenge/question other agents. Each dict contains:
    - from_agent: agent_id making the request
    - to_agent: agent_id being challenged
    - question: the question or challenge
    - timestamp: when the request was made
    """
    
    consensus_status: Optional[Dict[str, Any]]
    """Track if agents agree and no new findings. Contains:
    - consensus_reached: bool
    - reason: str explaining why consensus was reached or not
    - last_change_turn: int indicating when last change occurred
    """
    
    iteration_count: Optional[int]
    """Track number of turns to prevent infinite loops"""
    
    max_iterations: Optional[int]
    """Maximum turns before forcing synthesis"""

