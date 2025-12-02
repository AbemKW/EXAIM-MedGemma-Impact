"""Simplified CDSS graph nodes - orchestrator-driven workflow with specialist reasoning"""

from typing import Dict, Any
from demos.cdss_example.schema.graph_state import CDSSGraphState
from demos.cdss_example.agents.orchestrator_agent import OrchestratorAgent
from demos.cdss_example.agents.laboratory_agent import LaboratoryAgent
from demos.cdss_example.agents.cardiology_agent import CardiologyAgent
from demos.cdss_example.agents.internal_medicine_agent import InternalMedicineAgent
from demos.cdss_example.agents.radiology_agent import RadiologyAgent


async def orchestrator_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Orchestrator node: compresses context, decides next specialist, generates task instruction
    
    Workflow:
    1. If recent_delta exists, compress it into running_summary
    2. Decide next specialist or synthesis
    3. If not synthesis, generate task instruction for next specialist
    4. Update state and return
    """
    orchestrator = OrchestratorAgent()
    exaid = state["exaid"]
    case_text = state["case_text"]
    running_summary = state.get("running_summary", "")
    recent_delta = state.get("recent_delta", "")
    recent_agent = state.get("recent_agent", "none")
    specialists_called = state.get("specialists_called", [])
    iteration_count = state.get("iteration_count", 0)
    
    # Step 1: Compress recent specialist output into running_summary (if exists)
    if recent_delta and recent_agent != "none":
        # Compress using orchestrator method (streams to EXAID via act_stream internally)
        # The AgentStreamingCallback in orchestrator.act_stream handles message bus streaming
        running_summary = await orchestrator.compress_to_summary(
            running_summary, recent_delta, recent_agent
        )
    
    # Step 2: Decide next specialist or synthesis
    next_specialist = await orchestrator.decide_next_specialist(
        case_text, running_summary, recent_agent, specialists_called
    )
    
    # Step 3: Generate task instruction if not synthesis
    task_instruction = ""
    if next_specialist != "synthesis":
        task_instruction = await orchestrator.generate_task_instruction(
            case_text, running_summary, recent_delta, recent_agent, next_specialist
        )
    
    return {
        "running_summary": running_summary,
        "next_specialist_to_call": next_specialist,
        "task_instruction": task_instruction,
        "iteration_count": iteration_count + 1
    }


async def laboratory_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Laboratory specialist node: analyze labs and provide domain reasoning"""
    agent = LaboratoryAgent()
    exaid = state["exaid"]
    
    # Build context for specialist
    context = _build_specialist_context(state, "laboratory")
    
    # Stream specialist reasoning
    # The AgentStreamingCallback in agent.act_stream handles message bus streaming
    collected_tokens = []
    async def token_stream():
        async for token in agent.act_stream(context):
            collected_tokens.append(token)
            yield token
    
    # Stream to EXAID (non-blocking)
    await exaid.received_streamed_tokens(agent.agent_id, token_stream())
    
    # Collect raw output as recent_delta
    recent_delta = "".join(collected_tokens)
    
    # Update specialists_called
    specialists_called = state.get("specialists_called", [])
    if "laboratory" not in specialists_called:
        specialists_called = specialists_called + ["laboratory"]
    
    return {
        "recent_delta": recent_delta,
        "recent_agent": "laboratory",
        "specialists_called": specialists_called
    }


async def cardiology_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Cardiology specialist node: assess cardiovascular findings and provide domain reasoning"""
    agent = CardiologyAgent()
    exaid = state["exaid"]
    
    # Build context for specialist
    context = _build_specialist_context(state, "cardiology")
    
    # Stream specialist reasoning
    # The AgentStreamingCallback in agent.act_stream handles message bus streaming
    collected_tokens = []
    async def token_stream():
        async for token in agent.act_stream(context):
            collected_tokens.append(token)
            yield token
    
    # Stream to EXAID (non-blocking)
    await exaid.received_streamed_tokens(agent.agent_id, token_stream())
    
    # Collect raw output as recent_delta
    recent_delta = "".join(collected_tokens)
    
    # Update specialists_called
    specialists_called = state.get("specialists_called", [])
    if "cardiology" not in specialists_called:
        specialists_called = specialists_called + ["cardiology"]
    
    return {
        "recent_delta": recent_delta,
        "recent_agent": "cardiology",
        "specialists_called": specialists_called
    }


async def internal_medicine_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Internal Medicine specialist node: comprehensive clinical assessment"""
    agent = InternalMedicineAgent()
    exaid = state["exaid"]
    
    # Build context for specialist
    context = _build_specialist_context(state, "internal_medicine")
    
    # Stream specialist reasoning
    # The AgentStreamingCallback in agent.act_stream handles message bus streaming
    collected_tokens = []
    async def token_stream():
        async for token in agent.act_stream(context):
            collected_tokens.append(token)
            yield token
    
    # Stream to EXAID (non-blocking)
    await exaid.received_streamed_tokens(agent.agent_id, token_stream())
    
    # Collect raw output as recent_delta
    recent_delta = "".join(collected_tokens)
    
    # Update specialists_called
    specialists_called = state.get("specialists_called", [])
    if "internal_medicine" not in specialists_called:
        specialists_called = specialists_called + ["internal_medicine"]
    
    return {
        "recent_delta": recent_delta,
        "recent_agent": "internal_medicine",
        "specialists_called": specialists_called
    }


async def radiology_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Radiology specialist node: interpret imaging findings and provide domain reasoning"""
    agent = RadiologyAgent()
    exaid = state["exaid"]
    
    # Build context for specialist
    context = _build_specialist_context(state, "radiology")
    
    # Stream specialist reasoning
    # The AgentStreamingCallback in agent.act_stream handles message bus streaming
    collected_tokens = []
    async def token_stream():
        async for token in agent.act_stream(context):
            collected_tokens.append(token)
            yield token
    
    # Stream to EXAID (non-blocking)
    await exaid.received_streamed_tokens(agent.agent_id, token_stream())
    
    # Collect raw output as recent_delta
    recent_delta = "".join(collected_tokens)
    
    # Update specialists_called
    specialists_called = state.get("specialists_called", [])
    if "radiology" not in specialists_called:
        specialists_called = specialists_called + ["radiology"]
    
    return {
        "recent_delta": recent_delta,
        "recent_agent": "radiology",
        "specialists_called": specialists_called
    }


async def synthesis_node(state: CDSSGraphState) -> Dict[str, Any]:
    """Synthesis node: orchestrator generates final clinical recommendations"""
    orchestrator = OrchestratorAgent()
    exaid = state["exaid"]
    case_text = state["case_text"]
    running_summary = state.get("running_summary", "")
    
    # Build synthesis prompt
    synthesis_prompt = (
        f"CLINICAL CASE:\n{case_text}\n\n"
        f"RUNNING SUMMARY OF SPECIALIST FINDINGS:\n{running_summary}\n\n"
        f"As the coordinating physician, synthesize all findings into a comprehensive final assessment. Provide:\n"
        f"1. Clinical Summary: Brief overview of the patient and key findings\n"
        f"2. Differential Diagnosis: Most likely diagnoses ranked by probability\n"
        f"3. Critical Findings: Any urgent concerns requiring immediate attention\n"
        f"4. Recommended Management: Specific treatment recommendations and interventions\n"
        f"5. Follow-up Plan: Monitoring and additional testing recommendations\n\n"
        f"Provide clear, actionable clinical recommendations."
    )
    
    # Stream synthesis
    # The AgentStreamingCallback in orchestrator.act_stream handles message bus streaming
    collected_tokens = []
    async def token_stream():
        async for token in orchestrator.act_stream(synthesis_prompt):
            collected_tokens.append(token)
            yield token
    
    # Stream to EXAID
    await exaid.received_streamed_tokens(f"{orchestrator.agent_id}_synthesis", token_stream())
    
    # Collect final synthesis
    final_synthesis = "".join(collected_tokens)
    
    return {
        "final_synthesis": final_synthesis
    }


def _build_specialist_context(state: CDSSGraphState, specialist_name: str) -> str:
    """Build context string for specialist
    
    Args:
        state: Current graph state
        specialist_name: Name of the specialist receiving this context
        
    Returns:
        Formatted context string with case, summary, recent update, and task
    """
    case_text = state["case_text"]
    running_summary = state.get("running_summary", "")
    recent_delta = state.get("recent_delta", "")
    recent_agent = state.get("recent_agent", "none")
    task_instruction = state.get("task_instruction", "")
    
    # Build context
    context_parts = [f"CLINICAL CASE:\n{case_text}"]
    
    if running_summary:
        context_parts.append(f"\n\nSUMMARY SO FAR:\n{running_summary}")
    
    if recent_delta and recent_agent != "none":
        context_parts.append(f"\n\nRECENT UPDATE FROM {recent_agent.upper()}:\n{recent_delta}")
    
    if task_instruction:
        context_parts.append(f"\n\nYOUR TASK:\n{task_instruction}")
    
    return "".join(context_parts)
