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
        final_summary = all_summaries[-1] if all_summaries else None
        
        # Compile results
        result = {
            "case_summary": case_text,
            "agent_summaries": [
                {
                    "agents": s.agents,
                    "action": s.action,
                    "reasoning": s.reasoning,
                    "findings": s.findings,
                    "next_steps": s.next_steps
                }
                for s in all_summaries
            ],
            "final_recommendation": {
                "agents": final_summary.agents if final_summary else [],
                "action": final_summary.action if final_summary else "",
                "reasoning": final_summary.reasoning if final_summary else "",
                "findings": final_summary.findings if final_summary else None,
                "next_steps": final_summary.next_steps if final_summary else None
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

