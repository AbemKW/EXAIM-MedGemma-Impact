"""Utility functions for managing agent state in the CDSS graph.

All functions follow immutable state principles:
- They do NOT modify the state parameter
- They return partial state updates as dicts
- Nodes merge these updates using .update() and return them

This aligns with LangGraph's state semantics and MAS multi-agent behavior.
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple
from demos.cdss_example.schema.graph_state import CDSSGraphState


def ensure_agent_awareness(agent_id: str, state: CDSSGraphState) -> Dict[str, List]:
    """Ensure agent awareness entry exists for the given agent.
    
    Args:
        agent_id: The agent identifier (e.g., 'laboratory', 'cardiology', 'internal_medicine', 'radiology')
        state: The current graph state
        
    Returns:
        Dict with single key: {agent_id: []}
        Nodes should merge this into the full agent_awareness dict
    """
    agent_awareness = state.get("agent_awareness") or {}
    
    if agent_id not in agent_awareness:
        return {agent_id: []}
    
    return {agent_id: agent_awareness[agent_id]}


def get_new_findings_for_agent(agent_id: str, state: CDSSGraphState) -> Tuple[Dict[str, str], List[str]]:
    """Get new findings from other agents that this agent hasn't seen yet.
    
    Args:
        agent_id: The requesting agent's ID
        state: The current graph state
        
    Returns:
        Tuple of:
        - Dict mapping other agent IDs to their latest findings (only new ones)
        - List of finding_ids that should be marked as seen by this agent
    """
    agent_awareness = state.get("agent_awareness") or {}
    agent_turn_history = state.get("agent_turn_history") or []
    seen_findings = agent_awareness.get(agent_id, [])
    
    new_findings = {}
    finding_ids_to_mark = []
    
    # Define all possible agents
    all_agents = ["laboratory", "cardiology", "internal_medicine", "radiology"]
    
    # Check each other agent for new findings
    for other_agent_id in all_agents:
        if other_agent_id == agent_id:
            continue
        
        # Get the other agent's turns
        other_agent_turns = [t for t in agent_turn_history if t.get("agent_id") == other_agent_id]
        
        if not other_agent_turns:
            continue
        
        # Get the latest findings from this agent
        latest_turn = other_agent_turns[-1]
        latest_findings = latest_turn.get("findings")
        
        if not latest_findings:
            continue
        
        # Generate finding_id for this finding
        turn_number = latest_turn.get("turn_number", len(other_agent_turns))
        finding_id = f"{other_agent_id}_{turn_number}"
        
        # Check if this agent has already seen this finding
        if finding_id not in seen_findings:
            new_findings[other_agent_id] = latest_findings
            finding_ids_to_mark.append(finding_id)
    
    return new_findings, finding_ids_to_mark


def create_agent_turn_entry(agent_id: str, findings: str, iteration_count: int) -> Dict[str, Any]:
    """Create a turn entry dict for the agent_turn_history.
    
    DEPRECATED: Use add_turn_history() instead for Phase 3+ consistency.
    This function kept temporarily for backward compatibility.
    
    Args:
        agent_id: The agent identifier
        findings: The agent's findings/analysis text
        iteration_count: Current iteration number
        
    Returns:
        Dict with turn entry structure
    """
    return {
        "agent_id": agent_id,
        "turn_number": iteration_count,
        "timestamp": datetime.now().isoformat(),
        "findings": findings
    }


def add_turn_history(agent_id: str, findings: str, state: CDSSGraphState) -> Dict[str, List[Dict[str, Any]]]:
    """Add a turn entry to agent_turn_history following immutable pattern.
    
    Reads current turn history from state, creates new entry with turn_number = iteration_count,
    and returns complete updated history list.
    
    Args:
        agent_id: The agent identifier
        findings: The agent's findings/analysis text
        state: The current graph state
        
    Returns:
        Dict with key "agent_turn_history" containing full updated list.
        LangGraph will replace the existing agent_turn_history with this list.
    """
    agent_turn_history = state.get("agent_turn_history") or []
    iteration_count = state.get("iteration_count", 0)
    
    new_entry = {
        "agent_id": agent_id,
        "turn_number": iteration_count,
        "timestamp": datetime.now().isoformat(),
        "findings": findings
    }
    
    return {"agent_turn_history": agent_turn_history + [new_entry]}


def update_consulted_agents(agent_id: str, state: CDSSGraphState) -> Dict[str, List[str]]:
    """Add agent to consulted agents list following immutable pattern.
    
    Reads current consulted_agents from state, adds agent_id if not present,
    and returns complete updated list.
    
    Args:
        agent_id: The agent identifier to add
        state: The current graph state
        
    Returns:
        Dict with key "consulted_agents" containing full updated list.
        LangGraph will replace the existing consulted_agents with this list.
    """
    consulted_agents = state.get("consulted_agents") or []
    updated_list = consulted_agents.copy()
    
    if agent_id not in updated_list:
        updated_list.append(agent_id)
    
    return {"consulted_agents": updated_list}


def create_challenge_dict(from_agent: str, to_agent: str, question: str) -> Dict[str, str]:
    """Create a standardized challenge/critique dict.
    
    Args:
        from_agent: The agent issuing the challenge
        to_agent: The agent being challenged
        question: The challenge/critique text
        
    Returns:
        Dict with standardized challenge structure
    """
    return {
        "from_agent": from_agent,
        "to_agent": to_agent,
        "question": question
    }


def create_consultation_request_dict(requested_specialist: str, reason: str) -> Dict[str, str]:
    """Create a standardized consultation request dict from ConsultationRequest object.
    
    Args:
        requested_specialist: The specialist being requested
        reason: Clinical justification for the consultation
        
    Returns:
        Dict with standardized consultation request structure
    """
    return {
        "requested_specialist": requested_specialist,
        "reason": reason
    }
