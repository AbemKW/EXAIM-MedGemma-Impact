from langchain_core.prompts import ChatPromptTemplate
from infra import get_llm, LLMRole
from pydantic import BaseModel
from typing import Union
from exaid_core.schema.agent_segment import AgentSegment
from exaid_core.schema.buffer_analysis import BufferAnalysis, BufferAnalysisNoNovelty
from exaid_core.utils.prompts import (
    get_buffer_agent_system_prompt,
    get_buffer_agent_system_prompt_no_novelty,
    get_buffer_agent_user_prompt
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

        lines = ["BEGIN TRACE"]
        last_agent = None
        acc = []

        def flush():
            nonlocal acc, last_agent
            if acc and last_agent is not None:
                lines.append(f"[{last_agent}] " + " ".join(acc))
            acc = []

        for s in segments:
            if s.agent_id != last_agent:
                flush()
                last_agent = s.agent_id
            acc.append(s.segment)

        flush()
        lines.append("END TRACE")
        return "\n".join(lines)

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
            
            # DEBUG: Print the LLM's analysis to verify it's working
            novelty_str = f" | Novel={analysis.is_novel}" if hasattr(analysis, "is_novel") else ""
            print(f"DEBUG [{agent_id}]: State={analysis.stream_state} | Complete={analysis.is_complete} | Relevant={analysis.is_relevant}{novelty_str} | Trigger={analysis.final_trigger}")
            
            return analysis.final_trigger

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
