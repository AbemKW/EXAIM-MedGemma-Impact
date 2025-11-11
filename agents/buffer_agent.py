from langchain_core.prompts import ChatPromptTemplate
from llm import llm
from pydantic import BaseModel

class TraceData(BaseModel):
    count: int

class BufferAgent:
    def __init__(self):
        self.buffer: list[str] = []
        self.llm = llm
        self.flag_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are monitoring the reasoning streams of multiple AI agents. "
            "Your task is to decide if the new reasoning trace should trigger summarization. "
            "Reply with EXACTLY 'YES' (all caps) if ANY of these conditions are met:\n"
            "- The new trace completes a thought or reasoning step\n"
            "- The topic or focus has changed from previous traces\n"
            "- Enough context has accumulated to warrant a summary\n"
            "- The reasoning has reached a natural pause or conclusion\n\n"
            "Reply with EXACTLY 'NO' (all caps) if the new trace is just continuing the same line of reasoning without completion.\n"
            "Be decisive - prefer YES when in doubt, as summaries help track agent progress."),
            ("user", "Previous traces in buffer:\n{previous_trace}\n\nNew trace to evaluate:\n{new_trace}\n\nShould this trigger summarization? Reply with only 'YES' or 'NO'."),
        ])
        self.traces: dict[str, TraceData] = {}

    async def addchunk(self, agent_id: str, chunk: str) -> bool:
        tagged_chunk = f"| {agent_id} | {chunk}"
        if agent_id not in self.traces:
            self.traces[agent_id] = TraceData(count=0)
        self.traces[agent_id].count += 1
        
        # Check if buffer was empty before adding this chunk
        was_empty = not self.buffer
        
        # Always add the chunk to buffer
        self.buffer.append(tagged_chunk)
        
        # Always call the LLM to decide if summarization should be triggered
        # Use empty string for previous_trace if buffer was empty before adding this chunk
        previous_traces = "\n".join(self.buffer[:-1]) if not was_empty else ""
        
        flag_chain = self.flag_prompt | self.llm
        flag_response = await flag_chain.ainvoke({
            "previous_trace": previous_traces if previous_traces else "(No previous traces - this is the first trace)",
            "new_trace": tagged_chunk
        })
        return "YES" in flag_response.content.strip().upper()
    
    def flush(self) -> list[str]:
        flushed = self.buffer.copy()
        self.buffer.clear()
        return flushed
        
    def get_trace_count(self, agent_id: str) -> int:
        return self.traces.get(agent_id, TraceData(count=0)).count