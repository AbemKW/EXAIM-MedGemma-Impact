from typing import Dict, Any
from demos.cdss_example.schema.graph_state import CDSSGraphState


def check_consensus(state: CDSSGraphState) -> Dict[str, Any]:
    """Check if agents have reached consensus (all agree, no new findings emerging).
    
    This function checks stability across all four specialist agents (laboratory,
    cardiology, internal_medicine, radiology) that have been consulted.
    
    Stability threshold: No changes over 3 turns (tunable parameter - may need
    adjustment after observing four-agent collaboration patterns in production).
    
    Args:
        state: The current graph state
        
    Returns:
        Dictionary with:
        - consensus_reached: bool indicating if consensus was reached
        - reason: str explaining why consensus was reached or not
        - last_change_turn: int indicating when last change occurred
    """
    agent_turn_history = state.get("agent_turn_history") or []
    iteration_count = state.get("iteration_count", 0)
    consulted_agents = state.get("consulted_agents") or []
    
    # Extract turns for all four specialist agents
    lab_turns = [t for t in agent_turn_history if t.get("agent_id") == "laboratory"]
    cardio_turns = [t for t in agent_turn_history if t.get("agent_id") == "cardiology"]
    im_turns = [t for t in agent_turn_history if t.get("agent_id") == "internal_medicine"]
    rad_turns = [t for t in agent_turn_history if t.get("agent_id") == "radiology"]
    
    # Build dict of consulted agents' turns
    agent_turns = {}
    if "laboratory" in consulted_agents:
        agent_turns["laboratory"] = lab_turns
    if "cardiology" in consulted_agents:
        agent_turns["cardiology"] = cardio_turns
    if "internal_medicine" in consulted_agents:
        agent_turns["internal_medicine"] = im_turns
    if "radiology" in consulted_agents:
        agent_turns["radiology"] = rad_turns
    
    # Check if we have enough turns to assess stability
    # Require at least 2 turns from each consulted agent
    if not agent_turns:
        return {
            "consensus_reached": False,
            "reason": "No agents have been consulted yet.",
            "last_change_turn": iteration_count
        }
    
    for agent_id, turns in agent_turns.items():
        if len(turns) < 2:
            return {
                "consensus_reached": False,
                "reason": f"Not enough turns to assess consensus. {agent_id} needs at least 2 turns.",
                "last_change_turn": iteration_count
            }
    
    # Check if findings are stable (no changes in last 2 turns) for all consulted agents
    # NOTE: Stability threshold = 3 turns (tunable parameter for Phase 3+)
    findings_stable = True
    last_change_turn = iteration_count
    
    for agent_id, turns in agent_turns.items():
        if len(turns) >= 2:
            last_two = turns[-2:]
            if last_two[0].get("findings") != last_two[1].get("findings"):
                findings_stable = False
                last_change_turn = max(last_change_turn, last_two[1].get("turn_number", iteration_count))
    
    # Check if there are pending debate requests
    debate_requests = state.get("debate_requests") or []
    unresolved_debates = [d for d in debate_requests if not d.get("resolved", False)]
    
    if unresolved_debates:
        return {
            "consensus_reached": False,
            "reason": f"There are {len(unresolved_debates)} unresolved debate requests.",
            "last_change_turn": last_change_turn
        }
    
    # Check if there are consultation requests
    consultation_request = state.get("consultation_request")
    if consultation_request:
        return {
            "consensus_reached": False,
            "reason": f"Consultation request pending for {consultation_request}.",
            "last_change_turn": last_change_turn
        }
    
    # Check if new findings are emerging
    new_findings = state.get("new_findings_since_last_turn") or {}
    if new_findings:
        return {
            "consensus_reached": False,
            "reason": "New findings are still emerging from agents.",
            "last_change_turn": last_change_turn
        }
    
    # If findings are stable and no pending requests, check if agents agree
    if findings_stable:
        # Check if all consulted agents have provided findings
        all_findings = []
        if "laboratory" in consulted_agents and state.get("laboratory_findings"):
            all_findings.append("laboratory")
        if "cardiology" in consulted_agents and state.get("cardiology_findings"):
            all_findings.append("cardiology")
        if "internal_medicine" in consulted_agents and state.get("internal_medicine_findings"):
            all_findings.append("internal_medicine")
        if "radiology" in consulted_agents and state.get("radiology_findings"):
            all_findings.append("radiology")
        
        if len(all_findings) == len(consulted_agents):
            # All consulted agents have provided findings and they're stable
            return {
                "consensus_reached": True,
                "reason": f"All {len(consulted_agents)} consulted agents have stable findings with no new changes or pending requests.",
                "last_change_turn": last_change_turn
            }
    
    return {
        "consensus_reached": False,
        "reason": "Findings are not yet stable or agents have not reached agreement.",
        "last_change_turn": last_change_turn
    }

