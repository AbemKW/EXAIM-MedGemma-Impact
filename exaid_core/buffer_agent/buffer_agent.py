from langchain_core.prompts import ChatPromptTemplate
from infra import get_llm, LLMRole
from pydantic import BaseModel, Field
from typing import Literal, Union
from exaid_core.schema.agent_segment import AgentSegment
from exaid_core.utils.prompts import (
    get_buffer_agent_system_prompt,
    get_buffer_agent_system_prompt_no_novelty,
    get_buffer_agent_user_prompt
)

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
        "(explicit transitions, implicit shifts, conclusions). PROCEED to relevance/novelty checks. "
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
        description="Is this a fully formed, self-contained reasoning unit with clear closure? "
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
        "(explicit transitions, implicit shifts, conclusions). PROCEED to relevance checks. "
        "CRITICAL_ALERT: Immediate life-safety notification (e.g., 'V-Fib detected', 'Code Blue'). PROCEED immediately."
    )
    is_relevant: bool = Field(
        description="Is this clinically important? Does it add medical reasoning or context that would help "
        "the clinician understand the case? Relevant: new diagnosis, refined differential, specific treatment "
        "dose/plan, condition changes. Not Relevant: 'thinking out loud', obvious facts without interpretation, "
        "formatting tokens."
    )
    is_complete: bool = Field(
        description="Is this a fully formed, self-contained reasoning unit with clear closure? "
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


class TraceData(BaseModel):
    count: int

class BufferAgent:
    def __init__(self, disable_novelty: bool = False):
        self.buffer: list[AgentSegment] = []
        self.tail_segments: list[AgentSegment] = []  # Deferred segments from non-trigger flushes
        # Use a smarter model if available, or the same base model
        self.base_llm = get_llm(LLMRole.BUFFER_AGENT)
        # Conditionally initialize LLM and prompt based on novelty check
        if disable_novelty:
            self.llm = self.base_llm.with_structured_output(BufferAnalysisNoNovelty)
            self.flag_prompt = ChatPromptTemplate.from_messages([
                ("system", get_buffer_agent_system_prompt_no_novelty()),
                ("user", get_buffer_agent_user_prompt())
            ])
        else:
            self.llm = self.base_llm.with_structured_output(BufferAnalysis)
            self.flag_prompt = ChatPromptTemplate.from_messages([
                ("system", get_buffer_agent_system_prompt()),
                ("user", get_buffer_agent_user_prompt())
            ])
        self.traces: dict[str, TraceData] = {}
        self.last_analysis: dict[str, Union[BufferAnalysis, BufferAnalysisNoNovelty, None]] = {}

    @staticmethod
    def format_segments_for_prompt(segments: list[AgentSegment]) -> str:
        if not segments:
            return "(Buffer empty)"

        agent_segments_map: dict[str, list[str]] = {}
        for item in segments:
            agent_segments_map.setdefault(item.agent_id, []).append(item.segment)

        formatted_parts = ["BEGIN AGENT SEGMENTS"]
        for agent_id, grouped_segments in agent_segments_map.items():
            formatted_parts.append(f"[{agent_id}] " + " ".join(grouped_segments))
        formatted_parts.append("END AGENT SEGMENTS")
        return "\n".join(formatted_parts)

    async def addsegment(self, agent_id: str, segment: str, previous_summaries: list[str]) -> bool:
        new_text = segment
        if agent_id not in self.traces:
            self.traces[agent_id] = TraceData(count=0)
        self.traces[agent_id].count += 1
        
        # Add to buffer and track agent_id
        self.buffer.append(AgentSegment(agent_id=agent_id, segment=new_text))
        
        # Prepare context
        # We pass the *rest* of the buffer separate from the *new* segment so the LLM sees the flow.
        # Include deferred tail content so trigger decisions account for previously parked segments.
        # Group by agent to keep the prompt compact.
        prior_segments = self.tail_segments + self.buffer[:-1]
        buffer_context = self.format_segments_for_prompt(prior_segments)
        new_trace_block = self.format_segments_for_prompt([self.buffer[-1]])
        
        chain = self.flag_prompt | self.llm
        
        try:
            analysis: Union[BufferAnalysis, BufferAnalysisNoNovelty] = await chain.ainvoke({
                "summaries": previous_summaries,
                "previous_trace": buffer_context,
                "new_trace": new_trace_block
            })
            self.last_analysis[agent_id] = analysis
            
            should_trigger = analysis.final_trigger
            
            # DEBUG: Print the LLM's analysis to verify it's working
            novelty_str = f" | Novel={analysis.is_novel}" if hasattr(analysis, "is_novel") else ""
            print(f"DEBUG [{agent_id}]: State={analysis.stream_state} | Complete={analysis.is_complete} | Relevant={analysis.is_relevant}{novelty_str} | Trigger={analysis.final_trigger}")
            
            return should_trigger

        except Exception as e:
            # Fallback if structured output fails (fail closed/safe to avoid spam)
            print(f"Buffer decision failed: {e}")
            self.last_analysis[agent_id] = None
            return False
    
    def flush(self) -> list[AgentSegment]:
        """Flush tail + live buffer and return segments with their corresponding agent IDs.

        Tail segments are deferred content parked by `park_tail()` when a forced
        flush occurs without a BufferAgent trigger. They are included in the
        `buffer_context` passed to `addsegment()` decisions so trigger logic can
        consider them, and they are prepended here so the next summarization
        includes them with their original agent IDs.
        
        Returns:
            List of AgentSegment items, preserving original agent attribution
        """
        flushed_segments = self.tail_segments + self.buffer
        self.tail_segments.clear()
        self.buffer.clear()
        return flushed_segments

    def park_tail(self, segments: list[AgentSegment]) -> None:
        """Append leftover segments to the tail buffer without summarizing."""
        if not segments:
            return
        self.tail_segments.extend(segments)
        
    def get_trace_count(self, agent_id: str) -> int:
        return self.traces.get(agent_id, TraceData(count=0)).count

    def get_last_analysis(self, agent_id: str) -> Union[BufferAnalysis, BufferAnalysisNoNovelty, None]:
        return self.last_analysis.get(agent_id)
