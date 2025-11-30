import sys
from pathlib import Path

# CRITICAL: Add parent directory to path BEFORE any other imports
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any
from cdss_demo.schema.graph_state import CDSSGraphState


def check_consensus(state: CDSSGraphState) -> Dict[str, Any]:
    """Check if agents have reached consensus (all agree, no new findings emerging).
    
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
    
    # Need at least 2 turns from each agent to check for stability
    lab_turns = [t for t in agent_turn_history if t.get("agent_id") == "laboratory"]
    cardio_turns = [t for t in agent_turn_history if t.get("agent_id") == "cardiology"]
    
    # Check if we have enough turns to assess stability
    if len(lab_turns) < 2 and len(cardio_turns) < 2:
        return {
            "consensus_reached": False,
            "reason": "Not enough turns to assess consensus. Need at least 2 turns from each agent.",
            "last_change_turn": iteration_count
        }
    
    # Check if findings are stable (no changes in last 2 turns)
    findings_stable = True
    last_change_turn = iteration_count
    
    # Check laboratory agent stability
    if len(lab_turns) >= 2:
        last_two_lab = lab_turns[-2:]
        if last_two_lab[0].get("findings") != last_two_lab[1].get("findings"):
            findings_stable = False
            last_change_turn = max(last_change_turn, last_two_lab[1].get("turn_number", iteration_count))
    
    # Check cardiology agent stability
    if len(cardio_turns) >= 2:
        last_two_cardio = cardio_turns[-2:]
        if last_two_cardio[0].get("findings") != last_two_cardio[1].get("findings"):
            findings_stable = False
            last_change_turn = max(last_change_turn, last_two_cardio[1].get("turn_number", iteration_count))
    
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
        # Simple check: if both agents have findings and they're stable, consider it consensus
        lab_findings = state.get("laboratory_findings")
        cardio_findings = state.get("cardiology_findings")
        
        if lab_findings and cardio_findings:
            # Both agents have provided findings and they're stable
            return {
                "consensus_reached": True,
                "reason": "All agents have stable findings with no new changes or pending requests.",
                "last_change_turn": last_change_turn
            }
        elif lab_findings or cardio_findings:
            # Only one agent has findings - check if the other was consulted and declined
            consulted_agents = state.get("consulted_agents") or []
            if "laboratory" in consulted_agents and "cardiology" in consulted_agents:
                return {
                    "consensus_reached": True,
                    "reason": "All consulted agents have provided stable findings.",
                    "last_change_turn": last_change_turn
                }
    
    return {
        "consensus_reached": False,
        "reason": "Findings are not yet stable or agents have not reached agreement.",
        "last_change_turn": last_change_turn
    }

