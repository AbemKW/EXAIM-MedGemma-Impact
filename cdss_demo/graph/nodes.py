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
from cdss_demo.constants import LABORATORY_AGENT, CARDIOLOGY_AGENT, SYNTHESIS_ACTION


async def orchestrator_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Orchestrator node: analyzes case and decides which agents to call, or evaluates consultation requests"""
    orchestrator = OrchestratorAgent()
    exaid = state["exaid"]
    case_text = state["case_text"]
    
    # Initialize consulted_agents if not present (reasoning agents will add themselves upon completion of their analysis)
    consulted_agents = state.get("consulted_agents")
    if consulted_agents is None:
        consulted_agents = []
    
    consultation_request = state.get("consultation_request")
    
    # Consultation evaluation mode: Check if a reasoning agent requested consultation
    if consultation_request:
        # Check if requested agent has already been consulted (loop prevention)
        if consultation_request in consulted_agents:
            # Already consulted, route to synthesis
            decision_text = (
                f"Consultation request for {consultation_request} received, but this agent "
                f"has already been consulted. Routing to synthesis to prevent loop."
            )
            await exaid.received_trace(orchestrator.agent_id, decision_text)
            return {
                "consultation_request": None,  # Clear the request
                "agents_to_call": {SYNTHESIS_ACTION: True}
            }
        else:
            # Honor the consultation request
            decision_text = (
                f"Honoring consultation request for {consultation_request} agent. "
                f"This agent will be consulted to provide additional expertise."
            )
            await exaid.received_trace(orchestrator.agent_id, decision_text)
            # Add the requested agent to consulted_agents
            updated_consulted_agents = consulted_agents + [consultation_request]
            return {
                "consultation_request": None,  # Clear the request after honoring
                "agents_to_call": {consultation_request: True},
                "consulted_agents": updated_consulted_agents
            }
    
    # Check if agent findings are already present; if so, avoid re-analysis and preserve agents_to_call
    laboratory_done = state.get("laboratory_findings") is not None
    cardiology_done = state.get("cardiology_findings") is not None

    # If either agent has already completed, only call agents that have not yet been consulted
    if laboratory_done or cardiology_done:
        # Determine which agents still need to be called
        agents_to_call = {}
        if not laboratory_done and (LABORATORY_AGENT not in consulted_agents):
            agents_to_call[LABORATORY_AGENT] = True
        if not cardiology_done and (CARDIOLOGY_AGENT not in consulted_agents):
            agents_to_call[CARDIOLOGY_AGENT] = True
        # If no agents left to call, route to synthesis
        if not agents_to_call:
            return {
                "agents_to_call": {SYNTHESIS_ACTION: True},
                "consulted_agents": consulted_agents
            }
        else:
            return {
                "agents_to_call": agents_to_call,
                "consulted_agents": consulted_agents
            }
    
    # Initial analysis mode: Perform initial case analysis
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
            LABORATORY_AGENT: decision.call_laboratory,
            CARDIOLOGY_AGENT: decision.call_cardiology
        },
        "consulted_agents": consulted_agents
    }


async def laboratory_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Laboratory node: analyzes laboratory results and decides if consultation is needed"""
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
    
    # Decide if consultation is needed
    consulted_agents = state.get("consulted_agents") or []
    consultation_request = await laboratory.decide_consultation(findings, consulted_agents)
    
    # Update consulted_agents to include laboratory
    updated_consulted_agents = consulted_agents + [LABORATORY_AGENT]
    
    return {
        "laboratory_findings": findings,
        "consultation_request": consultation_request,
        "consulted_agents": updated_consulted_agents
    }


async def cardiology_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Cardiology node: assesses cardiac aspects and decides if consultation is needed"""
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
    
    # Decide if consultation is needed
    consulted_agents = state.get("consulted_agents") or []
    consultation_request = await cardiology.decide_consultation(findings, consulted_agents)
    
    # Update consulted_agents to include cardiology
    updated_consulted_agents = consulted_agents + [CARDIOLOGY_AGENT]
    
    return {
        "cardiology_findings": findings,
        "consultation_request": consultation_request,
        "consulted_agents": updated_consulted_agents
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
        f"Agent: {', '.join(s.agents)}\n"
        f"Action: {s.action}\n"
        f"Reasoning: {s.reasoning}\n"
        f"Findings: {s.findings or 'N/A'}\n"
        f"Next Steps: {s.next_steps or 'N/A'}"
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

