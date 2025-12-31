from pydantic import BaseModel, Field
from typing import Literal


class BufferAnalysis(BaseModel):
    """Structured analysis of stream state for buffer agent decision-making.

    Uses a three-state machine to classify stream completeness based on topic continuity,
    then independently evaluates clinical relevance and novelty to determine if summarization
    should be triggered.
    """
    reasoning: str = Field(
        description="Chain-of-thought analysis of the stream structure. Analyze completeness (whether "
        "this is a self-contained reasoning unit), whether the agent is still refining the same topic, "
        "has shifted to a new topic, or has issued a critical alert. Consider list markers, transition "
        "words, rationale gaps, topic boundaries, and whether coherent ideas are finished."
    )
    stream_state: Literal["SAME_TOPIC_CONTINUING", "TOPIC_SHIFT", "CRITICAL_ALERT"] = Field(
        description="State machine for stream completeness based on topic continuity. "
        "SAME_TOPIC_CONTINUING: Agent is still refining, listing, or explaining the same specific clinical issue "
        "(e.g., mid-list with markers like '1.', 'First,', reasoning loops, adding detail). WAIT - do not trigger. "
        "TOPIC_SHIFT: Agent explicitly moves to a distinctly different organ system, problem, or section "
        "(explicit transitions, implicit shifts, conclusions, new problem list items). PROCEED to relevance/novelty checks. "
        "CRITICAL_ALERT: Immediate life-safety notification (e.g., 'V-Fib detected', 'Code Blue'). PROCEED immediately."
    )
    is_relevant: bool = Field(
        description="Is this clinically important? Does it add medical reasoning or context that would help "
        "the clinician understand the case? Relevant: new diagnosis, refined differential, specific treatment "
        "dose/plan, condition changes. Not Relevant: 'thinking out loud', obvious facts without interpretation, "
        "formatting tokens."
    )
    is_novel: bool = Field(
        description="STRICT: Is this TRULY new vs previous summaries? Does it introduce something substantively different "
        "not already covered in prior summaries? Novel: NEW values/changes (e.g., 'Creatinine rose to 2.2' when previously 1.8), "
        "NEW actions not previously mentioned (e.g., 'Start Amiodarone' when not in prior summaries), "
        "NEW insights with changed reasoning (e.g., 'Diagnosis upgraded from possible to likely'). "
        "NOT Novel: Rephrasing same findings, restating same differentials, confirming existing plans, "
        "adding minor details to already-summarized content, or continuing the same line of reasoning without new conclusions."
    )
    is_complete: bool = Field(
        description="Is this a fully formed, self-contained reasoning unit with clear closure or an actionable conclusion? "
        "Complete: A substantial coherent thought with explicit interpretation or conclusion. "
        "Examples: A diagnostic interpretation with rationale ('This suggests prerenal AKI due to volume depletion'), "
        "a finalized treatment recommendation with reasoning ('Start furosemide for volume overload'), "
        "a clinical conclusion ('At this point, the likely diagnosis is X based on Y and Z'), "
        "or explicit closure signals ('Therefore...', 'In summary...', 'The diagnosis is...'). "
        "Incomplete: Partial thoughts, observations without interpretation, mid-reasoning statements, "
        "lists without closure, or thoughts that feel like they're building toward a conclusion."
    )
    final_trigger: bool = Field(
        description="True if ANY of these conditions are met: "
        "1) (is_complete AND is_relevant AND is_novel) OR "
        "2) (stream_state == TOPIC_SHIFT AND is_relevant AND is_novel) OR "
        "3) (stream_state == CRITICAL_ALERT). "
        "This dual-path approach allows triggering on completed thoughts even when the topic hasn't shifted, preventing the 'wait too long' failure mode while preserving smart pacing."
    )


class BufferAnalysisNoNovelty(BaseModel):
    """Structured analysis of stream state for buffer agent decision-making without novelty check.

    Uses a three-state machine to classify stream completeness based on topic continuity,
    then independently evaluates clinical relevance to determine if summarization should be triggered.
    """
    reasoning: str = Field(
        description="Chain-of-thought analysis of the stream structure. Analyze completeness (whether "
        "this is a self-contained reasoning unit), whether the agent is still refining the same topic, "
        "has shifted to a new topic, or has issued a critical alert. Consider list markers, transition "
        "words, rationale gaps, topic boundaries, and whether coherent ideas are finished."
    )
    stream_state: Literal["SAME_TOPIC_CONTINUING", "TOPIC_SHIFT", "CRITICAL_ALERT"] = Field(
        description="State machine for stream completeness based on topic continuity. "
        "SAME_TOPIC_CONTINUING: Agent is still refining, listing, or explaining the same specific clinical issue "
        "(e.g., mid-list with markers like '1.', 'First,', reasoning loops, adding detail). WAIT - do not trigger. "
        "TOPIC_SHIFT: Agent explicitly moves to a distinctly different organ system, problem, or section "
        "(explicit transitions, implicit shifts, conclusions, new problem list items). PROCEED to relevance checks. "
        "CRITICAL_ALERT: Immediate life-safety notification (e.g., 'V-Fib detected', 'Code Blue'). PROCEED immediately."
    )
    is_relevant: bool = Field(
        description="Is this clinically important? Does it add medical reasoning or context that would help "
        "the clinician understand the case? Relevant: new diagnosis, refined differential, specific treatment "
        "dose/plan, condition changes. Not Relevant: 'thinking out loud', obvious facts without interpretation, "
        "formatting tokens."
    )
    is_complete: bool = Field(
        description="Is this a fully formed, self-contained reasoning unit with clear closure or an actionable conclusion? "
        "Complete: A substantial coherent thought with explicit interpretation or conclusion. "
        "Examples: A diagnostic interpretation with rationale ('This suggests prerenal AKI due to volume depletion'), "
        "a finalized treatment recommendation with reasoning ('Start furosemide for volume overload'), "
        "a clinical conclusion ('At this point, the likely diagnosis is X based on Y and Z'), "
        "or explicit closure signals ('Therefore...', 'In summary...', 'The diagnosis is...'). "
        "Incomplete: Partial thoughts, observations without interpretation, mid-reasoning statements, "
        "lists without closure, or thoughts that feel like they're building toward a conclusion."
    )
    final_trigger: bool = Field(
        description="True if ANY of these conditions are met: "
        "1) (is_complete AND is_relevant) OR "
        "2) (stream_state == TOPIC_SHIFT AND is_relevant) OR "
        "3) (stream_state == CRITICAL_ALERT). "
        "This dual-path approach allows triggering on completed thoughts even when the topic hasn't shifted, preventing the 'wait too long' failure mode while preserving smart pacing."
    )
