from langchain_core.prompts import ChatPromptTemplate
from infra import get_llm, LLMRole
from pydantic import BaseModel, Field
from typing import Literal, Union
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
        description="Is this new vs previous summaries? Does it introduce something not already covered in "
        "prior summaries? Novel: new values (e.g., 'Creatinine rose to 2.2'), new actions (e.g., 'Start Amiodarone'), "
        "new insights (e.g., 'Diagnosis upgraded from possible to likely'). Not Novel: continuing statements, "
        "reiteration of already-summarized findings, status quo confirmations."
    )
    is_complete: bool = Field(
        description="STRICT: Is this a fully self-contained reasoning unit with clear closure? "
        "A trace is complete ONLY if it finishes a coherent idea with EXPLICIT closure signals. "
        "Complete (RARE): explicit conclusions ('Therefore...', 'In summary...', 'The diagnosis is...'), "
        "finalized recommendations with rationale ('Recommend X because Y'), clear topic boundaries "
        "with transition signals. Incomplete (DEFAULT): single observations, partial interpretations, "
        "mid-sentence thoughts, statements that could continue, lists without closure, "
        "reasoning that feels like it's building toward something, any uncertainty about continuation."
    )
    final_trigger: bool = Field(
        description="True if ANY of these conditions are met: "
        "1) (is_complete AND is_relevant AND is_novel) - STRICT: is_complete requires explicit closure signals OR "
        "2) (stream_state == TOPIC_SHIFT AND is_relevant AND is_novel) OR "
        "3) (stream_state == CRITICAL_ALERT). "
        "This allows triggering on completed thoughts even without topic shift, but completeness must be strict."
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
        description="STRICT: Is this a fully self-contained reasoning unit with clear closure? "
        "A trace is complete ONLY if it finishes a coherent idea with EXPLICIT closure signals. "
        "Complete (RARE): explicit conclusions ('Therefore...', 'In summary...', 'The diagnosis is...'), "
        "finalized recommendations with rationale ('Recommend X because Y'), clear topic boundaries "
        "with transition signals. Incomplete (DEFAULT): single observations, partial interpretations, "
        "mid-sentence thoughts, statements that could continue, lists without closure, "
        "reasoning that feels like it's building toward something, any uncertainty about continuation."
    )
    final_trigger: bool = Field(
        description="True if ANY of these conditions are met: "
        "1) (is_complete AND is_relevant) - STRICT: is_complete requires explicit closure signals OR "
        "2) (stream_state == TOPIC_SHIFT AND is_relevant) OR "
        "3) (stream_state == CRITICAL_ALERT). "
        "This allows triggering on completed thoughts even without topic shift, but completeness must be strict."
    )


class TraceData(BaseModel):
    count: int

class BufferAgent:
    def __init__(self, disable_novelty: bool = False):
        self.buffer: list[str] = []
        self.buffer_agent_ids: list[str] = []  # Track which agent each segment came from
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

    async def addsegment(self, agent_id: str, segment: str, previous_summaries: list[str]) -> bool:
        new_text = segment
        if agent_id not in self.traces:
            self.traces[agent_id] = TraceData(count=0)
        self.traces[agent_id].count += 1
        
        # Add to buffer and track agent_id
        self.buffer.append(new_text)
        self.buffer_agent_ids.append(agent_id)
        
        # Prepare context
        # We pass the *rest* of the buffer separate from the *new* segment so the LLM sees the flow
        buffer_context = "\n".join(self.buffer[:-1]) if len(self.buffer) > 1 else "(Buffer empty)"
        
        chain = self.flag_prompt | self.llm
        
        try:
            analysis: Union[BufferAnalysis, BufferAnalysisNoNovelty] = await chain.ainvoke({
                "agent_id": agent_id,
                "summaries": previous_summaries,
                "previous_trace": buffer_context,
                "new_trace": new_text
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
    
    def flush(self) -> tuple[list[str], list[str]]:
        """Flush buffer and return both segments and their corresponding agent IDs.
        
        Returns:
            Tuple of (segments, agent_ids) where agent_ids[i] corresponds to segments[i]
        """
        flushed_segments = self.buffer.copy()
        flushed_agent_ids = self.buffer_agent_ids.copy()
        self.buffer.clear()
        self.buffer_agent_ids.clear()
        return flushed_segments, flushed_agent_ids
        
    def get_trace_count(self, agent_id: str) -> int:
        return self.traces.get(agent_id, TraceData(count=0)).count

    def get_last_analysis(self, agent_id: str) -> Union[BufferAnalysis, BufferAnalysisNoNovelty, None]:
        return self.last_analysis.get(agent_id)
