import sys
from pathlib import Path
from datetime import datetime

# CRITICAL: Add parent directory to path BEFORE any other imports
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any
from cdss_demo.schema.graph_state import CDSSGraphState
from cdss_demo.agents.orchestrator_agent import OrchestratorAgent, AgentDecision
from cdss_demo.agents.laboratory_agent import LaboratoryAgent
from cdss_demo.agents.cardiology_agent import CardiologyAgent
from cdss_demo.graph.agent_context import build_agent_context, get_new_findings_for_agent
from cdss_demo.graph.consensus import check_consensus

def _get_or_create_orchestrator(state: CDSSGraphState) -> OrchestratorAgent:
    """Get orchestrator agent from state or create if not exists"""
    if state.get("orchestrator_agent") is None:
        state["orchestrator_agent"] = OrchestratorAgent()
    return state["orchestrator_agent"]


def _get_or_create_laboratory(state: CDSSGraphState) -> LaboratoryAgent:
    """Get laboratory agent from state or create if not exists"""
    if state.get("laboratory_agent") is None:
        state["laboratory_agent"] = LaboratoryAgent()
    return state["laboratory_agent"]


def _get_or_create_cardiology(state: CDSSGraphState) -> CardiologyAgent:
    """Get cardiology agent from state or create if not exists"""
    if state.get("cardiology_agent") is None:
        state["cardiology_agent"] = CardiologyAgent()
    return state["cardiology_agent"]

async def orchestrator_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Orchestrator node: manages collaborative workflow, tracks agent awareness, handles debates, checks consensus"""
    orchestrator = _get_or_create_orchestrator(state)
    exaid = state["exaid"]
    case_text = state["case_text"]
    
    # Initialize state fields if not present
    consulted_agents = state.get("consulted_agents") or []
    agent_turn_history = state.get("agent_turn_history") or []
    agent_awareness = state.get("agent_awareness") or {}
    debate_requests = state.get("debate_requests") or []
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    
    # Initialize agent_awareness for each agent if not present
    if "laboratory" not in agent_awareness:
        agent_awareness["laboratory"] = []
    if "cardiology" not in agent_awareness:
        agent_awareness["cardiology"] = []
    
    # Increment iteration count
    iteration_count += 1
    
    # Check max iterations
    if iteration_count >= max_iterations:
        decision_text = f"Maximum iterations ({max_iterations}) reached. Routing to synthesis."
        await exaid.received_trace(orchestrator.agent_id, decision_text)
        orchestrator.add_to_history("assistant", decision_text)
        return {
            "iteration_count": iteration_count,
            "agents_to_call": {"synthesis": True}
        }
    
    # PRIORITY 1: Check for consensus
    consensus_status = check_consensus(state)
    if consensus_status.get("consensus_reached", False):
        decision_text = f"Consensus reached: {consensus_status.get('reason', 'All agents agree')}. Routing to synthesis."
        await exaid.received_trace(orchestrator.agent_id, decision_text)
        orchestrator.add_to_history("assistant", decision_text)
        return {
            "consensus_status": consensus_status,
            "iteration_count": iteration_count,
            "agents_to_call": {"synthesis": True}
        }
    
    # PRIORITY 2: Handle consultation requests
    consultation_request = state.get("consultation_request")
    if consultation_request:
        # Check if requested agent has already been consulted (loop prevention)
        if consultation_request in consulted_agents:
            decision_text = (
                f"Consultation request for {consultation_request} received, but this agent "
                f"has already been consulted. Routing to synthesis to prevent loop."
            )
            await exaid.received_trace(orchestrator.agent_id, decision_text)
            orchestrator.add_to_history("assistant", decision_text)
            return {
                "consultation_request": None,
                "iteration_count": iteration_count,
                "agents_to_call": {"synthesis": True}
            }
        else:
            # Honor the consultation request
            decision_text = (
                f"Honoring consultation request for {consultation_request} agent. "
                f"This agent will be consulted to provide additional expertise."
            )
            await exaid.received_trace(orchestrator.agent_id, decision_text)
            orchestrator.add_to_history("assistant", decision_text)
            return {
                "consultation_request": None,
                "iteration_count": iteration_count,
                "agents_to_call": {consultation_request: True}
            }
    
    # PRIORITY 3: Handle debate requests
    unresolved_debates = [d for d in debate_requests if not d.get("resolved", False)]
    if unresolved_debates:
        # Route to the agent being challenged
        next_debate = unresolved_debates[0]
        target_agent = next_debate.get("to_agent")
        decision_text = f"Routing debate request to {target_agent} agent: {next_debate.get('question', '')}"
        await exaid.received_trace(orchestrator.agent_id, decision_text)
        orchestrator.add_to_history("assistant", decision_text)
        return {
            "iteration_count": iteration_count,
            "agents_to_call": {target_agent: True}
        }
    
    # PRIORITY 4: Check if agents need to respond to new findings
    # Identify which agents have new findings to review
    lab_turns = [t for t in agent_turn_history if t.get("agent_id") == "laboratory"]
    cardio_turns = [t for t in agent_turn_history if t.get("agent_id") == "cardiology"]
    
    # Check if cardiology has new findings that lab hasn't seen
    if state.get("cardiology_findings") and cardio_turns:
        last_cardio_turn = cardio_turns[-1]
        finding_id = f"cardiology_{len(agent_turn_history)}"
        if finding_id not in agent_awareness.get("laboratory", []):
            # Lab should see cardiology's findings
            if "laboratory" in consulted_agents:
                decision_text = f"Laboratory agent has new findings from Cardiology to review. Routing to Laboratory."
                await exaid.received_trace(orchestrator.agent_id, decision_text)
                orchestrator.add_to_history("assistant", decision_text)
                return {
                    "iteration_count": iteration_count,
                    "agents_to_call": {"laboratory": True}
                }
    
    # Check if laboratory has new findings that cardiology hasn't seen
    if state.get("laboratory_findings") and lab_turns:
        last_lab_turn = lab_turns[-1]
        finding_id = f"laboratory_{len(agent_turn_history)}"
        if finding_id not in agent_awareness.get("cardiology", []):
            # Cardiology should see lab's findings
            if "cardiology" in consulted_agents:
                decision_text = f"Cardiology agent has new findings from Laboratory to review. Routing to Cardiology."
                await exaid.received_trace(orchestrator.agent_id, decision_text)
                orchestrator.add_to_history("assistant", decision_text)
                return {
                    "iteration_count": iteration_count,
                    "agents_to_call": {"cardiology": True}
                }
    
    # PRIORITY 5: Initial analysis mode - determine which agents to call initially
    if not agent_turn_history:
        # First time - perform initial case analysis
        orchestrator.add_to_history("user", f"Clinical Case:\n{case_text}")
        decision: AgentDecision = await orchestrator.analyze_and_decide(case_text)
        
        decision_text = (
            f"Analyzed clinical case and decided which agents to consult.\n"
            f"Reasoning: {decision.reasoning}\n"
            f"Call Laboratory Agent: {decision.call_laboratory}\n"
            f"Call Cardiology Agent: {decision.call_cardiology}"
        )
        await exaid.received_trace(orchestrator.agent_id, decision_text)
        orchestrator.add_to_history("assistant", decision_text)
        
        agents_to_call = {}
        if decision.call_laboratory:
            agents_to_call["laboratory"] = True
        if decision.call_cardiology:
            agents_to_call["cardiology"] = True
        
        if not agents_to_call:
            agents_to_call = {"synthesis": True}
        
        return {
            "orchestrator_analysis": decision_text,
            "iteration_count": iteration_count,
            "agents_to_call": agents_to_call,
            "consulted_agents": consulted_agents,
            "agent_awareness": agent_awareness
        }
    
    # PRIORITY 6: If all agents have been consulted and no new activity, route to synthesis
    if len(consulted_agents) >= 2 and not unresolved_debates:
        decision_text = "All agents have been consulted. Routing to synthesis."
        await exaid.received_trace(orchestrator.agent_id, decision_text)
        orchestrator.add_to_history("assistant", decision_text)
        return {
            "iteration_count": iteration_count,
            "agents_to_call": {"synthesis": True}
        }
    
    # Default: route to synthesis if unclear
    decision_text = "No clear next step. Routing to synthesis."
    await exaid.received_trace(orchestrator.agent_id, decision_text)
    orchestrator.add_to_history("assistant", decision_text)
    return {
        "iteration_count": iteration_count,
        "agents_to_call": {"synthesis": True}
    }


async def laboratory_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Laboratory node: analyzes laboratory results incrementally, building on previous work"""
    laboratory = _get_or_create_laboratory(state)
    exaid = state["exaid"]
    
    # Initialize state fields
    agent_turn_history = state.get("agent_turn_history") or []
    agent_awareness = state.get("agent_awareness") or {}
    iteration_count = state.get("iteration_count", 0)
    consulted_agents = state.get("consulted_agents") or []
    
    # Build incremental context for this agent
    context = build_agent_context("laboratory", state)
    
    # Determine if this is first turn or continuation
    lab_turns = [t for t in agent_turn_history if t.get("agent_id") == "laboratory"]
    is_first_turn = len(lab_turns) == 0
    
    # Build the input prompt
    if is_first_turn:
        lab_input = (
            f"{context}\n\n"
            "Analyze the laboratory results and provide interpretation. "
            "Identify any abnormal values, critical findings, or patterns that suggest specific diagnoses. "
            "Recommend additional tests if needed."
        )
    else:
        # Continue from previous analysis
        lab_input = (
            f"{context}\n\n"
            "Based on the new information provided above, continue your analysis. "
            "If there are new findings from other agents, incorporate them into your assessment. "
            "If there are questions or challenges directed at you, address them. "
            "Update or refine your previous findings as needed."
        )
    
    # Add to conversation history
    laboratory.add_to_history("user", lab_input)
    
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
    
    # Add response to conversation history
    laboratory.add_to_history("assistant", findings)
    
    # Update agent awareness - mark that lab has seen current findings
    if "laboratory" not in agent_awareness:
        agent_awareness["laboratory"] = []
    
    # Mark findings from other agents as seen
    new_findings = get_new_findings_for_agent("laboratory", state)
    for other_agent_id, other_findings in new_findings.items():
        finding_id = f"{other_agent_id}_{len(agent_turn_history)}"
        if finding_id not in agent_awareness["laboratory"]:
            agent_awareness["laboratory"].append(finding_id)
    
    # Add turn to history
    turn_entry = {
        "agent_id": "laboratory",
        "turn_number": iteration_count,
        "timestamp": datetime.now().isoformat(),
        "findings": findings
    }
    updated_turn_history = agent_turn_history + [turn_entry]
    
    # Decide if consultation is needed
    consultation_request = await laboratory.decide_consultation(findings, consulted_agents)
    
    # Update consulted_agents if not already present
    updated_consulted_agents = consulted_agents.copy()
    if "laboratory" not in updated_consulted_agents:
        updated_consulted_agents.append("laboratory")
    
    # Check for debate requests (if agent wants to challenge other findings)
    # This would be implemented in the agent's decide_if_needs_response method
    # For now, we'll just return the findings
    
    return {
        "laboratory_findings": findings,
        "consultation_request": consultation_request,
        "consulted_agents": updated_consulted_agents,
        "agent_turn_history": updated_turn_history,
        "agent_awareness": agent_awareness,
        "new_findings_since_last_turn": {}  # Will be updated by orchestrator
    }


async def cardiology_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Cardiology node: assesses cardiac aspects incrementally, building on previous work"""
    cardiology = _get_or_create_cardiology(state)
    exaid = state["exaid"]
    
    # Initialize state fields
    agent_turn_history = state.get("agent_turn_history") or []
    agent_awareness = state.get("agent_awareness") or {}
    iteration_count = state.get("iteration_count", 0)
    consulted_agents = state.get("consulted_agents") or []
    
    # Build incremental context for this agent
    context = build_agent_context("cardiology", state)
    
    # Determine if this is first turn or continuation
    cardio_turns = [t for t in agent_turn_history if t.get("agent_id") == "cardiology"]
    is_first_turn = len(cardio_turns) == 0
    
    # Build the input prompt
    if is_first_turn:
        cardio_input = (
            f"{context}\n\n"
            "Assess the cardiac aspects of this case. Consider:\n"
            "- Cardiovascular risk factors\n"
            "- Cardiac symptoms and signs\n"
            "- Cardiac biomarkers and tests\n"
            "- ECG or imaging findings if available\n"
            "- Cardiac medication considerations\n"
            "Provide cardiac assessment and recommendations."
        )
    else:
        # Continue from previous analysis
        cardio_input = (
            f"{context}\n\n"
            "Based on the new information provided above, continue your analysis. "
            "If there are new findings from other agents, incorporate them into your assessment. "
            "If there are questions or challenges directed at you, address them. "
            "Update or refine your previous findings as needed."
        )
    
    # Add to conversation history
    cardiology.add_to_history("user", cardio_input)
    
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
    
    # Add response to conversation history
    cardiology.add_to_history("assistant", findings)
    
    # Update agent awareness - mark that cardiology has seen current findings
    if "cardiology" not in agent_awareness:
        agent_awareness["cardiology"] = []
    
    # Mark findings from other agents as seen
    new_findings = get_new_findings_for_agent("cardiology", state)
    for other_agent_id, other_findings in new_findings.items():
        finding_id = f"{other_agent_id}_{len(agent_turn_history)}"
        if finding_id not in agent_awareness["cardiology"]:
            agent_awareness["cardiology"].append(finding_id)
    
    # Add turn to history
    turn_entry = {
        "agent_id": "cardiology",
        "turn_number": iteration_count,
        "timestamp": datetime.now().isoformat(),
        "findings": findings
    }
    updated_turn_history = agent_turn_history + [turn_entry]
    
    # Decide if consultation is needed
    consultation_request = await cardiology.decide_consultation(findings, consulted_agents)
    
    # Update consulted_agents if not already present
    updated_consulted_agents = consulted_agents.copy()
    if "cardiology" not in updated_consulted_agents:
        updated_consulted_agents.append("cardiology")
    
    return {
        "cardiology_findings": findings,
        "consultation_request": consultation_request,
        "consulted_agents": updated_consulted_agents,
        "agent_turn_history": updated_turn_history,
        "agent_awareness": agent_awareness,
        "new_findings_since_last_turn": {}  # Will be updated by orchestrator
    }


async def synthesis_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Synthesis node: orchestrator synthesizes all findings"""
    orchestrator = _get_or_create_orchestrator(state)
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
    
    # Add to conversation history
    orchestrator.add_to_history("user", synthesis_input)
    
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
    
    # Add response to conversation history
    orchestrator.add_to_history("assistant", synthesis)
    
    return {
        "final_synthesis": synthesis
    }
