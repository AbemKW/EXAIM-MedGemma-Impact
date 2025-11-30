import sys
from pathlib import Path

# CRITICAL: Add parent directory to path BEFORE any other imports
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict
from cdss_demo.schema.graph_state import CDSSGraphState


def build_agent_context(agent_id: str, state: CDSSGraphState) -> str:
    """Build incremental context for an agent based on what they've seen and what's new.
    
    Args:
        agent_id: The ID of the agent ("laboratory" or "cardiology")
        state: The current graph state
        
    Returns:
        A formatted context string containing only relevant new information
    """
    context_parts = []
    
    # Initialize state fields if not present
    agent_awareness = state.get("agent_awareness") or {}
    agent_turn_history = state.get("agent_turn_history") or []
    debate_requests = state.get("debate_requests") or []
    
    # Check if this is the agent's first turn
    agent_turns = [turn for turn in agent_turn_history if turn.get("agent_id") == agent_id]
    is_first_turn = len(agent_turns) == 0
    
    # 1. Case text (only on first turn)
    if is_first_turn:
        context_parts.append("=== CLINICAL CASE ===")
        context_parts.append(state.get("case_text", ""))
        context_parts.append("")
    
    # 2. Agent's own previous findings
    if not is_first_turn:
        # Get the agent's most recent findings
        last_turn = agent_turns[-1] if agent_turns else None
        if last_turn and last_turn.get("findings"):
            context_parts.append(f"=== YOUR PREVIOUS ANALYSIS ===")
            context_parts.append(last_turn["findings"])
            context_parts.append("")
    
    # 3. New findings from other agents since last turn
    new_findings = state.get("new_findings_since_last_turn") or {}
    seen_findings = agent_awareness.get(agent_id, [])
    
    # Determine which agent is the "other" agent
    other_agent_id = "cardiology" if agent_id == "laboratory" else "laboratory"
    
    # Check for new findings from the other agent
    other_agent_findings = None
    if agent_id == "laboratory" and state.get("cardiology_findings"):
        other_agent_findings = state["cardiology_findings"]
    elif agent_id == "cardiology" and state.get("laboratory_findings"):
        other_agent_findings = state["laboratory_findings"]
    
    # Check if this finding is new (not in agent's awareness)
    if other_agent_findings:
        # Create a simple hash/identifier for this finding
        finding_id = f"{other_agent_id}_{len(agent_turn_history)}"
        
        # Check if agent has seen this finding
        if finding_id not in seen_findings:
            context_parts.append(f"=== NEW FINDINGS FROM {other_agent_id.upper()} AGENT ===")
            context_parts.append(other_agent_findings)
            context_parts.append("")
            
            # Mark this finding as seen (will be updated in state by the node)
            if agent_id not in agent_awareness:
                agent_awareness[agent_id] = []
            agent_awareness[agent_id].append(finding_id)
    
    # 4. Debate/challenge requests directed at this agent
    relevant_debates = [
        d for d in debate_requests 
        if d.get("to_agent") == agent_id and not d.get("resolved", False)
    ]
    
    if relevant_debates:
        context_parts.append("=== QUESTIONS/CHALLENGES FOR YOU ===")
        for debate in relevant_debates:
            from_agent = debate.get("from_agent", "Unknown")
            question = debate.get("question", "")
            context_parts.append(f"{from_agent.upper()} Agent asks: {question}")
        context_parts.append("")
    
    # 5. Orchestrator guidance/updates
    orchestrator_analysis = state.get("orchestrator_analysis")
    if orchestrator_analysis and is_first_turn:
        context_parts.append("=== ORCHESTRATOR GUIDANCE ===")
        context_parts.append(orchestrator_analysis)
        context_parts.append("")
    
    # Build the final context
    context = "\n".join(context_parts)
    
    # If no new context, provide a minimal prompt
    if not context.strip():
        context = "Continue your analysis based on the information you already have."
    
    return context


def get_agent_previous_findings(agent_id: str, state: CDSSGraphState) -> str:
    """Get the agent's previous findings from their last turn.
    
    Args:
        agent_id: The ID of the agent
        state: The current graph state
        
    Returns:
        The agent's previous findings, or empty string if none
    """
    agent_turn_history = state.get("agent_turn_history") or []
    agent_turns = [turn for turn in agent_turn_history if turn.get("agent_id") == agent_id]
    
    if agent_turns:
        last_turn = agent_turns[-1]
        return last_turn.get("findings", "")
    
    return ""


def get_new_findings_for_agent(agent_id: str, state: CDSSGraphState) -> Dict[str, str]:
    """Get new findings from other agents that this agent hasn't seen yet.
    
    Args:
        agent_id: The ID of the agent
        state: The current graph state
        
    Returns:
        Dictionary mapping other agent IDs to their findings
    """
    new_findings = {}
    agent_awareness = state.get("agent_awareness") or {}
    seen_findings = agent_awareness.get(agent_id, [])
    
    # Determine the other agent
    other_agent_id = "cardiology" if agent_id == "laboratory" else "laboratory"
    
    # Get other agent's findings
    if agent_id == "laboratory" and state.get("cardiology_findings"):
        finding_id = f"{other_agent_id}_{len(state.get('agent_turn_history', []))}"
        if finding_id not in seen_findings:
            new_findings[other_agent_id] = state["cardiology_findings"]
    elif agent_id == "cardiology" and state.get("laboratory_findings"):
        finding_id = f"{other_agent_id}_{len(state.get('agent_turn_history', []))}"
        if finding_id not in seen_findings:
            new_findings[other_agent_id] = state["laboratory_findings"]
    
    return new_findings

