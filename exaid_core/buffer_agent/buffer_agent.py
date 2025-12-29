from langchain_core.prompts import ChatPromptTemplate
from infra import get_llm, LLMRole
from pydantic import BaseModel, Field
from typing import Literal
from exaid_core.utils.prompts import get_buffer_agent_system_prompt, get_buffer_agent_user_prompt

class BufferAnalysis(BaseModel):
    """Structured analysis of stream state for buffer agent decision-making.
    
    Uses a three-state machine to classify stream completeness based on topic continuity,
    then independently evaluates clinical relevance and novelty to determine if summarization
    should be triggered.
    """
    reasoning: str = Field(
        description="Chain-of-thought analysis of the stream structure. Analyze whether the agent "
        "is still refining the same topic, has shifted to a new topic, or has issued a critical alert. "
        "Consider list markers, transition words, rationale gaps, and topic boundaries."
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
    final_trigger: bool = Field(
        description="True ONLY if (stream_state is TOPIC_SHIFT OR CRITICAL_ALERT) AND is_relevant is True "
        "AND is_novel is True. This is the final gate for triggering summarization."
    )

class TraceData(BaseModel):
    count: int

class BufferAgent:
    def __init__(self):
        self.buffer: list[str] = []
        # Use a smarter model if available, or the same base model
        self.base_llm = get_llm(LLMRole.BUFFER_AGENT)
        # We bind the structured output to force the reasoning step
        self.llm = self.base_llm.with_structured_output(BufferAnalysis)
        self.flag_prompt = ChatPromptTemplate.from_messages([
            ("system", get_buffer_agent_system_prompt()),
            ("user", get_buffer_agent_user_prompt())
        ])
        self.traces: dict[str, TraceData] = {}

    async def addsegment(self, agent_id: str, segment: str, previous_summaries: list[str]) -> bool:
        tagged_segment = f"| {agent_id} | {segment}"
        if agent_id not in self.traces:
            self.traces[agent_id] = TraceData(count=0)
        self.traces[agent_id].count += 1
        
        # Add to buffer first
        self.buffer.append(new_text)
        
        # Prepare context
        # We pass the *rest* of the buffer separate from the *new* segment so the LLM sees the flow
        buffer_context = "\n".join(self.buffer[:-1]) if len(self.buffer) > 1 else "(Buffer empty)"
        
        chain = self.flag_prompt | self.llm
        
        try:
            analysis: BufferAnalysis = await chain.ainvoke({
                "summaries": previous_summaries,
                "previous_trace": buffer_context,
                "new_trace": new_text
            })
            
            should_trigger = analysis.final_trigger
            
            # DEBUG: Print the LLM's analysis to verify it's working
            print(f"DEBUG [{agent_id}]: State={analysis.stream_state} | Trigger={analysis.final_trigger}")
            
            return should_trigger

        except Exception as e:
            # Fallback if structured output fails (fail closed/safe to avoid spam)
            print(f"Buffer decision failed: {e}")
            return False
    
    def flush(self) -> list[str]:
        flushed = self.buffer.copy()
        self.buffer.clear()
        return flushed
        
    def get_trace_count(self, agent_id: str) -> int:
        return self.traces.get(agent_id, TraceData(count=0)).count