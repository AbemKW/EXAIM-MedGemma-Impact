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
    """Clinical Decision Support System orchestrator using LangGraph workflow"""
    
    def __init__(self):
        """Initialize CDSS with EXAID and graph-based workflow"""
        self.exaid = EXAID()
        self.graph = build_cdss_graph()
        # Keep agent references for trace count queries
        self.orchestrator = OrchestratorAgent()
        self.cardiology = CardiologyAgent()
        self.laboratory = LaboratoryAgent()
    
    async def process_case(
        self, 
        case: Union[ClinicalCase, str],
        use_streaming: bool = True
    ) -> dict:
        """Process a clinical case through the graph-based multi-agent system
        
        Args:
            case: ClinicalCase object or free-text case description
            use_streaming: Whether to use streaming token processing (default: True)
                          Note: Graph implementation always uses streaming internally
            
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
        final_state = await self.graph.ainvoke(initial_state)
        
        # Get all summaries after graph execution
        all_summaries = self.exaid.get_all_summaries()
        
        # Get the final synthesis summary (last one from orchestrator)
        final_summary = None
        if all_summaries:
            # Try to find orchestrator summary with synthesis/final recommendation action
            orchestrator_summaries = [
                s for s in all_summaries 
                if self.orchestrator.agent_id in s.agents
            ]
            synthesis_actions = {"synthesis", "final recommendation", "final_synthesis"}
            synthesis_summaries = [
                s for s in orchestrator_summaries
                if s.action and s.action.strip().lower().replace(" ", "_") in synthesis_actions
            ]
            if synthesis_summaries:
                final_summary = synthesis_summaries[-1]
            elif orchestrator_summaries:
                # Fallback to last orchestrator summary if no synthesis-specific summary found
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
            "graph_state": {
                "orchestrator_analysis": final_state.get("orchestrator_analysis"),
                "agents_called": final_state.get("agents_to_call"),
                "laboratory_findings": final_state.get("laboratory_findings"),
                "cardiology_findings": final_state.get("cardiology_findings"),
                "final_synthesis": final_state.get("final_synthesis")
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

