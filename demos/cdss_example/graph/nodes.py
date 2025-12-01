from datetime import datetime

from typing import Dict, Any, Optional
from demos.cdss_example.schema.graph_state import CDSSGraphState
from demos.cdss_example.agents.orchestrator_agent import OrchestratorAgent, AgentDecision
from demos.cdss_example.agents.laboratory_agent import LaboratoryAgent
from demos.cdss_example.agents.cardiology_agent import CardiologyAgent
from demos.cdss_example.agents.internal_medicine_agent import InternalMedicineAgent
from demos.cdss_example.agents.radiology_agent import RadiologyAgent
from demos.cdss_example.schema.agent_messages import ConsultationRequest
from .agent_context import build_agent_context
from .consensus import check_consensus
from .utils import (
    ensure_agent_awareness,
    get_new_findings_for_agent,
    create_agent_turn_entry,
    update_consulted_agents,
    create_challenge_dict,
    create_consultation_request_dict
)


def _generate_finding_id(agent_id: str, turns: list) -> str:
    """Generate a stable finding_id based on the latest turn number.
    
    Args:
        agent_id: The agent identifier (e.g., 'cardiology', 'laboratory')
        turns: List of turn entries for this agent
        
    Returns:
        A stable finding_id string like 'cardiology_3' or 'laboratory_2'
    """
    latest_turn = turns[-1] if turns else None
    if latest_turn:
        turn_number = latest_turn.get("turn_number", len(turns))
    else:
        turn_number = len(turns)
    return f"{agent_id}_{turn_number}"


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


def _get_or_create_internal_medicine(state: CDSSGraphState) -> InternalMedicineAgent:
    """Get internal medicine agent from state or create if not exists"""
    if state.get("internal_medicine_agent") is None:
        state["internal_medicine_agent"] = InternalMedicineAgent()
    return state["internal_medicine_agent"]


def _get_or_create_radiology(state: CDSSGraphState) -> RadiologyAgent:
    """Get radiology agent from state or create if not exists"""
    if state.get("radiology_agent") is None:
        state["radiology_agent"] = RadiologyAgent()
    return state["radiology_agent"]

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
    # Uses the agent_awareness tracking to determine if findings have been reviewed
    if state.get("cardiology_findings") and cardio_turns:
        finding_id = _generate_finding_id("cardiology", cardio_turns)
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
    # Uses the agent_awareness tracking to determine if findings have been reviewed
    if state.get("laboratory_findings") and lab_turns:
        finding_id = _generate_finding_id("laboratory", lab_turns)
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
    
    # Ensure agent awareness is initialized
    awareness_update = ensure_agent_awareness("laboratory", state)
    updated_agent_awareness = agent_awareness.copy()
    updated_agent_awareness.update(awareness_update)
    
    # Get new findings from other agents
    new_findings_dict, finding_ids = get_new_findings_for_agent("laboratory", state)
    
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
    
    # Update agent awareness with new findings seen
    for finding_id in finding_ids:
        if finding_id not in updated_agent_awareness["laboratory"]:
            updated_agent_awareness["laboratory"].append(finding_id)
    
    # Evaluate other agents' findings and potentially issue challenges
    challenge: Optional[Dict[str, str]] = None
    for other_agent_id, other_findings in new_findings_dict.items():
        critique = await laboratory.evaluate_other_agent_findings(other_agent_id, other_findings)
        if critique:
            challenge = create_challenge_dict("laboratory", other_agent_id, critique)
            break  # Only issue one challenge per turn
    
    # Create turn entry
    turn_entry = create_agent_turn_entry("laboratory", findings, iteration_count)
    updated_turn_history = agent_turn_history + [turn_entry]
    
    # Decide if consultation is needed
    consultation_request_obj: Optional[ConsultationRequest] = await laboratory.decide_consultation(findings, consulted_agents)
    
    # Convert ConsultationRequest to dict format for node output
    consultation_request_dict: Optional[Dict[str, str]] = None
    consultation_request_str: Optional[str] = None  # For backward compatibility with orchestrator
    if consultation_request_obj:
        consultation_request_dict = create_consultation_request_dict(
            consultation_request_obj.requested_specialist,
            consultation_request_obj.reason
        )
        consultation_request_str = consultation_request_obj.requested_specialist
    
    # Update consulted_agents
    updated_consulted_agents = update_consulted_agents("laboratory", consulted_agents)
    
    return {
        "laboratory_findings": findings,
        "consultation_request": consultation_request_str,  # For orchestrator (backward compatible)
        "consulted_agents": updated_consulted_agents,
        "agent_turn_history": updated_turn_history,
        "agent_awareness": updated_agent_awareness,
        "new_findings_since_last_turn": {},  # Placeholder for Phase 3
        # Phase 2 additions (stored in node output only, not in state):
        "_laboratory_consultation_request": consultation_request_dict,  # Full dict with reason
        "_laboratory_challenge": challenge  # Challenge dict if issued
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
    
    # Ensure agent awareness is initialized
    awareness_update = ensure_agent_awareness("cardiology", state)
    updated_agent_awareness = agent_awareness.copy()
    updated_agent_awareness.update(awareness_update)
    
    # Get new findings from other agents
    new_findings_dict, finding_ids = get_new_findings_for_agent("cardiology", state)
    
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
    
    # Update agent awareness with new findings seen
    for finding_id in finding_ids:
        if finding_id not in updated_agent_awareness["cardiology"]:
            updated_agent_awareness["cardiology"].append(finding_id)
    
    # Evaluate other agents' findings and potentially issue challenges
    challenge: Optional[Dict[str, str]] = None
    for other_agent_id, other_findings in new_findings_dict.items():
        critique = await cardiology.evaluate_other_agent_findings(other_agent_id, other_findings)
        if critique:
            challenge = create_challenge_dict("cardiology", other_agent_id, critique)
            break  # Only issue one challenge per turn
    
    # Create turn entry
    turn_entry = create_agent_turn_entry("cardiology", findings, iteration_count)
    updated_turn_history = agent_turn_history + [turn_entry]
    
    # Decide if consultation is needed
    consultation_request_obj: Optional[ConsultationRequest] = await cardiology.decide_consultation(findings, consulted_agents)
    
    # Convert ConsultationRequest to dict format for node output
    consultation_request_dict: Optional[Dict[str, str]] = None
    consultation_request_str: Optional[str] = None  # For backward compatibility with orchestrator
    if consultation_request_obj:
        consultation_request_dict = create_consultation_request_dict(
            consultation_request_obj.requested_specialist,
            consultation_request_obj.reason
        )
        consultation_request_str = consultation_request_obj.requested_specialist
    
    # Update consulted_agents
    updated_consulted_agents = update_consulted_agents("cardiology", consulted_agents)
    
    return {
        "cardiology_findings": findings,
        "consultation_request": consultation_request_str,  # For orchestrator (backward compatible)
        "consulted_agents": updated_consulted_agents,
        "agent_turn_history": updated_turn_history,
        "agent_awareness": updated_agent_awareness,
        "new_findings_since_last_turn": {},  # Placeholder for Phase 3
        # Phase 2 additions (stored in node output only, not in state):
        "_cardiology_consultation_request": consultation_request_dict,  # Full dict with reason
        "_cardiology_challenge": challenge  # Challenge dict if issued
    }


async def internal_medicine_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Internal Medicine node: provides comprehensive diagnostic reasoning and clinical integration"""
    internal_med = _get_or_create_internal_medicine(state)
    exaid = state["exaid"]
    
    # Initialize state fields
    agent_turn_history = state.get("agent_turn_history") or []
    agent_awareness = state.get("agent_awareness") or {}
    iteration_count = state.get("iteration_count", 0)
    consulted_agents = state.get("consulted_agents") or []
    
    # Ensure agent awareness is initialized
    awareness_update = ensure_agent_awareness("internal_medicine", state)
    updated_agent_awareness = agent_awareness.copy()
    updated_agent_awareness.update(awareness_update)
    
    # Get new findings from other agents
    new_findings_dict, finding_ids = get_new_findings_for_agent("internal_medicine", state)
    
    # Build incremental context for this agent
    context = build_agent_context("internal_medicine", state)
    
    # Determine if this is first turn or continuation
    im_turns = [t for t in agent_turn_history if t.get("agent_id") == "internal_medicine"]
    is_first_turn = len(im_turns) == 0
    
    # Build the input prompt
    if is_first_turn:
        # First turn: receive orchestrator's initial analysis and case
        im_input = (
            f"{context}\n\n"
            "As the Internal Medicine specialist, provide a comprehensive diagnostic assessment. "
            "Build a broad differential diagnosis considering all body systems. "
            "Integrate all available clinical information (history, symptoms, vital signs, findings from other specialists). "
            "Identify the most likely diagnoses and recommend appropriate workup and management. "
            "Show your step-by-step reasoning process."
        )
    else:
        # Continue from previous analysis
        im_input = (
            f"{context}\n\n"
            "Based on the new information provided above, continue your analysis. "
            "If there are new findings from other specialists, integrate them into your assessment. "
            "If there are questions or challenges directed at you, address them. "
            "Update or refine your previous differential diagnosis and recommendations as needed."
        )
    
    # Add to conversation history
    internal_med.add_to_history("user", im_input)
    
    # Get internal medicine agent's analysis stream
    token_stream = internal_med.act_stream(im_input)
    
    # Collect tokens while streaming to EXAID
    collected = []
    async def wrapper():
        async for token in token_stream:
            collected.append(token)
            yield token
    
    # Process streamed tokens through EXAID
    await exaid.received_streamed_tokens(internal_med.agent_id, wrapper())
    
    # Build full findings from collected tokens
    findings = "".join(collected)
    
    # Add response to conversation history
    internal_med.add_to_history("assistant", findings)
    
    # Update agent awareness with new findings seen
    for finding_id in finding_ids:
        if finding_id not in updated_agent_awareness["internal_medicine"]:
            updated_agent_awareness["internal_medicine"].append(finding_id)
    
    # Evaluate other agents' findings and potentially issue challenges
    challenge: Optional[Dict[str, str]] = None
    for other_agent_id, other_findings in new_findings_dict.items():
        critique = await internal_med.evaluate_other_agent_findings(other_agent_id, other_findings)
        if critique:
            challenge = create_challenge_dict("internal_medicine", other_agent_id, critique)
            break  # Only issue one challenge per turn
    
    # Create turn entry
    turn_entry = create_agent_turn_entry("internal_medicine", findings, iteration_count)
    updated_turn_history = agent_turn_history + [turn_entry]
    
    # Decide if consultation is needed (now wired up for Phase 2)
    consultation_request_obj: Optional[ConsultationRequest] = await internal_med.decide_consultation(findings, consulted_agents)
    
    # Convert ConsultationRequest to dict format for node output
    consultation_request_dict: Optional[Dict[str, str]] = None
    consultation_request_str: Optional[str] = None  # For backward compatibility with orchestrator
    if consultation_request_obj:
        consultation_request_dict = create_consultation_request_dict(
            consultation_request_obj.requested_specialist,
            consultation_request_obj.reason
        )
        consultation_request_str = consultation_request_obj.requested_specialist
    
    # Update consulted_agents
    updated_consulted_agents = update_consulted_agents("internal_medicine", consulted_agents)
    
    return {
        "internal_medicine_findings": findings,
        "consultation_request": consultation_request_str,  # For orchestrator (backward compatible)
        "consulted_agents": updated_consulted_agents,
        "agent_turn_history": updated_turn_history,
        "agent_awareness": updated_agent_awareness,
        "new_findings_since_last_turn": {},  # Placeholder for Phase 3
        # Phase 2 additions (stored in node output only, not in state):
        "_internal_medicine_consultation_request": consultation_request_dict,  # Full dict with reason
        "_internal_medicine_challenge": challenge  # Challenge dict if issued
    }


async def radiology_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Radiology node: interprets imaging studies and correlates with clinical presentation"""
    radiology = _get_or_create_radiology(state)
    exaid = state["exaid"]
    
    # Initialize state fields
    agent_turn_history = state.get("agent_turn_history") or []
    agent_awareness = state.get("agent_awareness") or {}
    iteration_count = state.get("iteration_count", 0)
    consulted_agents = state.get("consulted_agents") or []
    
    # Ensure agent awareness is initialized
    awareness_update = ensure_agent_awareness("radiology", state)
    updated_agent_awareness = agent_awareness.copy()
    updated_agent_awareness.update(awareness_update)
    
    # Get new findings from other agents
    new_findings_dict, finding_ids = get_new_findings_for_agent("radiology", state)
    
    # Build incremental context for this agent
    context = build_agent_context("radiology", state)
    
    # Determine if this is first turn or continuation
    rad_turns = [t for t in agent_turn_history if t.get("agent_id") == "radiology"]
    is_first_turn = len(rad_turns) == 0
    
    # Build the input prompt
    if is_first_turn:
        rad_input = (
            f"{context}\n\n"
            "Interpret any available imaging studies (X-ray, CT, MRI, ultrasound). "
            "Systematically describe the imaging findings using standard radiological terminology. "
            "Build a radiological differential diagnosis and correlate findings with the clinical presentation. "
            "Recommend additional imaging if needed. Show your step-by-step reasoning process."
        )
    else:
        # Continue from previous analysis
        rad_input = (
            f"{context}\n\n"
            "Based on the new information provided above, continue your analysis. "
            "If there are new findings from other specialists, incorporate them into your imaging interpretation. "
            "If there are questions or challenges directed at you, address them. "
            "Update or refine your previous radiological assessment as needed."
        )
    
    # Add to conversation history
    radiology.add_to_history("user", rad_input)
    
    # Get radiology agent's analysis stream
    token_stream = radiology.act_stream(rad_input)
    
    # Collect tokens while streaming to EXAID
    collected = []
    async def wrapper():
        async for token in token_stream:
            collected.append(token)
            yield token
    
    # Process streamed tokens through EXAID
    await exaid.received_streamed_tokens(radiology.agent_id, wrapper())
    
    # Build full findings from collected tokens
    findings = "".join(collected)
    
    # Add response to conversation history
    radiology.add_to_history("assistant", findings)
    
    # Update agent awareness with new findings seen
    for finding_id in finding_ids:
        if finding_id not in updated_agent_awareness["radiology"]:
            updated_agent_awareness["radiology"].append(finding_id)
    
    # Evaluate other agents' findings and potentially issue challenges
    challenge: Optional[Dict[str, str]] = None
    for other_agent_id, other_findings in new_findings_dict.items():
        critique = await radiology.evaluate_other_agent_findings(other_agent_id, other_findings)
        if critique:
            challenge = create_challenge_dict("radiology", other_agent_id, critique)
            break  # Only issue one challenge per turn
    
    # Create turn entry
    turn_entry = create_agent_turn_entry("radiology", findings, iteration_count)
    updated_turn_history = agent_turn_history + [turn_entry]
    
    # Decide if consultation is needed (now wired up for Phase 2)
    consultation_request_obj: Optional[ConsultationRequest] = await radiology.decide_consultation(findings, consulted_agents)
    
    # Convert ConsultationRequest to dict format for node output
    consultation_request_dict: Optional[Dict[str, str]] = None
    consultation_request_str: Optional[str] = None  # For backward compatibility with orchestrator
    if consultation_request_obj:
        consultation_request_dict = create_consultation_request_dict(
            consultation_request_obj.requested_specialist,
            consultation_request_obj.reason
        )
        consultation_request_str = consultation_request_obj.requested_specialist
    
    # Update consulted_agents
    updated_consulted_agents = update_consulted_agents("radiology", consulted_agents)
    
    return {
        "radiology_findings": findings,
        "consultation_request": consultation_request_str,  # For orchestrator (backward compatible)
        "consulted_agents": updated_consulted_agents,
        "agent_turn_history": updated_turn_history,
        "agent_awareness": updated_agent_awareness,
        "new_findings_since_last_turn": {},  # Placeholder for Phase 3
        # Phase 2 additions (stored in node output only, not in state):
        "_radiology_consultation_request": consultation_request_dict,  # Full dict with reason
        "_radiology_challenge": challenge  # Challenge dict if issued
    }


async def synthesis_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Synthesis node: orchestrator synthesizes all findings"""
    orchestrator = _get_or_create_orchestrator(state)
    exaid = state["exaid"]
    case_text = state["case_text"]
    
    # Collect findings from called agents
    findings_parts = []
    
    if state.get("internal_medicine_findings"):
        findings_parts.append(f"Internal Medicine Agent Findings:\n{state['internal_medicine_findings']}")
    
    if state.get("laboratory_findings"):
        findings_parts.append(f"Laboratory Agent Findings:\n{state['laboratory_findings']}")
    
    if state.get("cardiology_findings"):
        findings_parts.append(f"Cardiology Agent Findings:\n{state['cardiology_findings']}")
    
    if state.get("radiology_findings"):
        findings_parts.append(f"Radiology Agent Findings:\n{state['radiology_findings']}")
    
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
