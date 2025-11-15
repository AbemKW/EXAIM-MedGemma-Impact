import sys
from pathlib import Path

# CRITICAL: Add parent directory to path BEFORE any other imports
# This must happen first so that exaid.py can find its dependencies
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can safely import modules that depend on the project root being in sys.path
from typing import Union
from exaid import EXAID
from schema.agent_summary import AgentSummary
from cdss_demo.agents.orchestrator_agent import OrchestratorAgent
from cdss_demo.agents.cardiology_agent import CardiologyAgent
from cdss_demo.agents.laboratory_agent import LaboratoryAgent
from cdss_demo.schema.clinical_case import ClinicalCase
from cdss_demo.graph.cdss_graph import build_cdss_graph
from cdss_demo.schema.graph_state import CDSSGraphState


class CDSS:
    """Clinical Decision Support System orchestrator"""
    
    def __init__(self):
        """Initialize CDSS with EXAID and LangGraph workflow"""
        self.exaid = EXAID()
        self.orchestrator = OrchestratorAgent()
        self.cardiology = CardiologyAgent()
        self.laboratory = LaboratoryAgent()
        self.graph = build_cdss_graph()
    
    async def process_case(
        self, 
        case: Union[ClinicalCase, str],
        use_streaming: bool = True
    ) -> dict:
        """Process a clinical case through the multi-agent system using LangGraph
        
        Args:
            case: ClinicalCase object or free-text case description
            use_streaming: Whether to use streaming token processing (default: True)
                Note: Currently LangGraph doesn't support streaming, so this is ignored
            
        Returns:
            Dictionary containing agent findings and final recommendation
        """
        # Convert case to clinical summary if it's a ClinicalCase object
        if isinstance(case, ClinicalCase):
            case_text = case.to_clinical_summary()
        else:
            case_text = str(case)
        
        # Initialize graph state
        initial_state: CDSSGraphState = {
            "case_text": case_text,
            "orchestrator_analysis": None,
            "agents_to_call": None,
            "laboratory_findings": None,
            "cardiology_findings": None,
            "final_synthesis": None,
            "exaid": self.exaid
        }
        
        # Run the graph workflow
        # Note: LangGraph doesn't currently support streaming in the same way,
        # so we use the standard invoke method
        final_state = await self.graph.ainvoke(initial_state)
        
        # Get all summaries
        all_summaries = self.exaid.get_all_summaries()
        
        # Get the final summary (last one from synthesis)
        # Try to find orchestrator summary first, fallback to last summary
        final_summary = None
        if all_summaries:
            orchestrator_summaries = [
                s for s in all_summaries 
                if self.orchestrator.agent_id.lower() in s.agent_contributions.lower()
            ]
            if orchestrator_summaries:
                final_summary = orchestrator_summaries[-1]
            else:
                # Fallback to last summary if no orchestrator summary found
                final_summary = all_summaries[-1]
        
        # Compile results
        result = {
            "case_summary": case_text,
            "agent_summaries": [
                {
                    "status_action": s.status_action,
                    "key_findings": s.key_findings,
                    "differential_rationale": s.differential_rationale,
                    "uncertainty_confidence": s.uncertainty_confidence,
                    "recommendation_next_step": s.recommendation_next_step,
                    "agent_contributions": s.agent_contributions
                }
                for s in all_summaries
            ],
            "final_recommendation": {
                "status_action": final_summary.status_action if final_summary else "",
                "key_findings": final_summary.key_findings if final_summary else "",
                "differential_rationale": final_summary.differential_rationale if final_summary else "",
                "uncertainty_confidence": final_summary.uncertainty_confidence if final_summary else "",
                "recommendation_next_step": final_summary.recommendation_next_step if final_summary else "",
                "agent_contributions": final_summary.agent_contributions if final_summary else ""
            },
            "trace_count": {
                "orchestrator": self.exaid.get_agent_trace_count(self.orchestrator.agent_id),
                "cardiology": self.exaid.get_agent_trace_count(self.cardiology.agent_id),
                "laboratory": self.exaid.get_agent_trace_count(self.laboratory.agent_id)
            },
            "agents_called": {
                "laboratory": (final_state.get("agents_to_call") or {}).get("laboratory", False),
                "cardiology": (final_state.get("agents_to_call") or {}).get("cardiology", False)
            }
        }
        
        return result
    
    def get_all_summaries(self) -> list[AgentSummary]:
        """Get all summaries from EXAID"""
        return self.exaid.get_all_summaries()
    
    def get_summaries_by_agent(self, agent_id: str) -> list[AgentSummary]:
        """Get summaries for a specific agent"""
        return self.exaid.get_summaries_by_agent(agent_id)
    
    def reset(self):
        """Reset the CDSS system (create new EXAID instance)"""
        self.exaid = EXAID()

