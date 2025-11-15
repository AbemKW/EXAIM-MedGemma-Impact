import sys
from pathlib import Path

# CRITICAL: Add parent directory to path BEFORE any other imports
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any
from cdss_demo.schema.graph_state import CDSSGraphState
from cdss_demo.agents.orchestrator_agent import OrchestratorAgent, AgentDecision
from cdss_demo.agents.laboratory_agent import LaboratoryAgent
from cdss_demo.agents.cardiology_agent import CardiologyAgent


async def orchestrator_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Orchestrator node: analyzes case and decides which agents to call"""
    orchestrator = OrchestratorAgent()
    exaid = state["exaid"]
    case_text = state["case_text"]
    
    # Get decision on which agents to call
    decision: AgentDecision = await orchestrator.analyze_and_decide(case_text)
    
    # Capture orchestrator's analysis/decision as trace
    decision_text = (
        f"Analyzed clinical case and decided which agents to consult.\n"
        f"Reasoning: {decision.reasoning}\n"
        f"Call Laboratory Agent: {decision.call_laboratory}\n"
        f"Call Cardiology Agent: {decision.call_cardiology}"
    )
    await exaid.received_trace(orchestrator.agent_id, decision_text)
    
    return {
        "orchestrator_analysis": decision_text,
        "agents_to_call": {
            "laboratory": decision.call_laboratory,
            "cardiology": decision.call_cardiology
        }
    }


async def laboratory_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Laboratory node: analyzes laboratory results"""
    laboratory = LaboratoryAgent()
    exaid = state["exaid"]
    case_text = state["case_text"]
    
    lab_input = (
        f"Clinical Case:\n{case_text}\n\n"
        "Analyze the laboratory results and provide interpretation. "
        "Identify any abnormal values, critical findings, or patterns that suggest specific diagnoses. "
        "Recommend additional tests if needed."
    )
    
    # Get laboratory agent's analysis stream
    token_stream = laboratory.act_stream(lab_input)
    
    # Collect tokens while streaming to EXAID
    collected = []
    async def wrapper():
        async for token in token_stream:
            collected.append(token)
            yield token
    
    # Process streamed tokens through EXAID
    await exaid.received_streamed_tokens(laboratory.agent_id, wrapper())
    
    # Build full findings from collected tokens
    findings = "".join(collected)
    
    return {
        "laboratory_findings": findings
    }


async def cardiology_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Cardiology node: assesses cardiac aspects"""
    cardiology = CardiologyAgent()
    exaid = state["exaid"]
    case_text = state["case_text"]
    
    cardio_input = (
        f"Clinical Case:\n{case_text}\n\n"
        "Assess the cardiac aspects of this case. Consider:\n"
        "- Cardiovascular risk factors\n"
        "- Cardiac symptoms and signs\n"
        "- Cardiac biomarkers and tests\n"
        "- ECG or imaging findings if available\n"
        "- Cardiac medication considerations\n"
        "Provide cardiac assessment and recommendations."
    )
    
    # Get cardiology agent's analysis stream
    token_stream = cardiology.act_stream(cardio_input)
    
    # Collect tokens while streaming to EXAID
    collected = []
    async def wrapper():
        async for token in token_stream:
            collected.append(token)
            yield token
    
    # Process streamed tokens through EXAID
    await exaid.received_streamed_tokens(cardiology.agent_id, wrapper())
    
    # Build full findings from collected tokens
    findings = "".join(collected)
    
    return {
        "cardiology_findings": findings
    }


async def synthesis_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Synthesis node: orchestrator synthesizes all findings"""
    orchestrator = OrchestratorAgent()
    exaid = state["exaid"]
    case_text = state["case_text"]
    
    # Collect findings from called agents
    findings_parts = []
    
    if state.get("laboratory_findings"):
        findings_parts.append(f"Laboratory Agent Findings:\n{state['laboratory_findings']}")
    
    if state.get("cardiology_findings"):
        findings_parts.append(f"Cardiology Agent Findings:\n{state['cardiology_findings']}")
    
    # Get all summaries for context
    all_summaries = exaid.get_all_summaries()
    summary_context = "\n\n".join([
        f"Status/Action: {s.status_action}\n"
        f"Key Findings: {s.key_findings}\n"
        f"Differential/Rationale: {s.differential_rationale}\n"
        f"Uncertainty/Confidence: {s.uncertainty_confidence}\n"
        f"Recommendation/Next Step: {s.recommendation_next_step}\n"
        f"Agent Contributions: {s.agent_contributions}"
        for s in all_summaries
    ])
    
    synthesis_input = (
        f"Original Clinical Case:\n{case_text}\n\n"
    )
    
    if findings_parts:
        synthesis_input += f"Specialist Agent Findings:\n\n" + "\n\n".join(findings_parts) + "\n\n"
    
    synthesis_input += (
        f"Agent Summaries:\n{summary_context}\n\n"
        "Synthesize all findings from the specialist agents into a comprehensive "
        "clinical assessment and recommendation. Provide:\n"
        "- Overall clinical assessment\n"
        "- Key findings from each specialist\n"
        "- Integrated diagnosis or differential diagnosis\n"
        "- Prioritized recommendations\n"
        "- Follow-up plan"
    )
    
    # Get synthesis stream
    token_stream = orchestrator.act_stream(synthesis_input)
    
    # Collect tokens while streaming to EXAID
    collected = []
    async def wrapper():
        async for token in token_stream:
            collected.append(token)
            yield token
    
    # Process streamed tokens through EXAID
    await exaid.received_streamed_tokens(orchestrator.agent_id, wrapper())
    
    # Build full synthesis from collected tokens
    synthesis = "".join(collected)
    
    return {
        "final_synthesis": synthesis
    }

