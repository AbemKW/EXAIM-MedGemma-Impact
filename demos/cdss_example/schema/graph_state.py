from typing import TypedDict, Optional, List, Dict, Any
from demos.cdss_example.agents.cardiology_agent import CardiologyAgent
from demos.cdss_example.agents.laboratory_agent import LaboratoryAgent
from demos.cdss_example.agents.orchestrator_agent import OrchestratorAgent
from demos.cdss_example.agents.internal_medicine_agent import InternalMedicineAgent
from demos.cdss_example.agents.radiology_agent import RadiologyAgent
from demos.cdss_example.schema.agent_messages import ConsultationRequestDict, ChallengeRequestDict, DebateEntryDict
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
    
    internal_medicine_agent: Optional[InternalMedicineAgent]
    """Internal Medicine agent instance"""
    
    radiology_agent: Optional[RadiologyAgent]
    """Radiology agent instance"""
    
    orchestrator_analysis: Optional[str]
    """Orchestrator's initial analysis of the case"""
    
    agents_to_call: Optional[dict]
    """Dictionary indicating which agents should be called.
    Format: {"laboratory": bool, "cardiology": bool, "internal_medicine": bool, "radiology": bool}
    """
    
    consultation_request: Optional[ConsultationRequestDict]
    """Structured consultation request from an agent.
    Contains requested_specialist and reason fields.
    Replaced legacy string-based consultation_request in Phase 3.
    """
    
    challenge: Optional[ChallengeRequestDict]
    """Challenge issued by a specialist node to be ingested by orchestrator.
    Orchestrator converts this to DebateEntryDict and appends to debate_requests.
    Cleared after consumption.
    """
    
    consulted_agents: Optional[list[str]]
    """List of agents that have been consulted. Used for loop prevention."""
    
    laboratory_findings: Optional[str]
    """Laboratory agent's findings and recommendations"""
    
    cardiology_findings: Optional[str]
    """Cardiology agent's findings and recommendations"""
    
    internal_medicine_findings: Optional[str]
    """Internal Medicine agent's findings and recommendations"""
    
    radiology_findings: Optional[str]
    """Radiology agent's findings and recommendations"""
    
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
    
    debate_requests: Optional[list[DebateEntryDict]]
    """Track when agents want to challenge/question other agents. Each entry contains:
    - from_agent: agent_id making the request
    - to_agent: agent_id being challenged
    - question: the question or challenge
    - timestamp: when the request was made
    - resolved: whether the challenge has been addressed (Phase 4)
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
    """Maximum turns before forcing synthesis (default: 20 to accommodate four-agent collaboration)"""

