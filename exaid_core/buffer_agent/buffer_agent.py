from langchain_core.prompts import ChatPromptTemplate
from exaid_core.llm import exaid_llm
from pydantic import BaseModel

class TraceData(BaseModel):
    count: int

class BufferAgent:
    def __init__(self):
        self.buffer: list[str] = []
        self.llm = exaid_llm
        # NOTE: This prompt is comprehensive (~65 lines) to improve decision quality
        # Performance impact: This LLM call happens frequently in the streaming pipeline
        # Consider: 1) Testing a more concise prompt, 2) Prompt caching if provider supports it,
        # or 3) Using a smaller/faster model specifically for buffering decisions
        self.flag_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are acting as a Buffer Agent within EXAID, a middleware system that coordinates multiple LLM-based agents working together on a live clinical case.\n\n"

            "These agents (e.g., CardiologyAgent, LaboratoryAgent) emit verbose, token-by-token reasoning traces in real time — like thoughts unfolding in a stream-of-consciousness style. You receive these traces as they’re generated, without knowing what will come next.\n\n"

            "Your role is to monitor this live stream of traces, maintain an internal rolling buffer of previously seen traces, and decide — for each new trace — whether it should trigger a clinical summary update for the human clinician. You are the gating mechanism that decides *when* a summary is needed.\n\n"

            "You also receive a list of previously generated summaries (what the clinician has already seen). These allow you to evaluate whether the new trace is redundant or novel.\n\n"

            "Important: If the trace buffer is empty, that means **either**:\n"
            "- A new case is beginning\n"
            "- A summary was recently triggered and the buffer was flushed\n"
            "You should still apply the full decision logic based on prior summaries and the incoming trace.\n\n"

            "Your task is to answer: Should this new trace, in combination with the current buffer and prior summaries, trigger a new summary right now?\n\n"
            "You must reply with exactly 'YES' or 'NO'. You do **not** write the summary — you only decide whether the downstream SummarizerAgent should be called.\n\n"

            "You must base your decision on the following **three-layer filter**:\n\n"

            "========================\n"
            "**1. COMPLETENESS** – Is this a self-contained reasoning unit?\n"
            "A trace is complete if it finishes a coherent idea, interpretation, or diagnostic hypothesis. Examples:\n"
            "- A full interpretation of lab results or vitals\n"
            "- A concluded diagnostic thought (e.g., 'This could be prerenal AKI due to volume depletion')\n"
            "- A full medication change rationale or therapeutic proposal\n"
            "- Reaching a diagnostic boundary (e.g., 'at this point, the likely cause is...')\n\n"
            "Incomplete examples:\n"
            "- Starting a list but not finishing\n"
            "- Raising a possibility without context or reasoning\n"
            "- Midstream thoughts (e.g., 'and also her BUN...')\n\n"

            "========================\n"
            "**2. CLINICAL VALUE** – Does this matter to the clinician?\n"
            "A trace has clinical value if it adds medical reasoning or context that would help the human understand the case. Examples:\n"
            "- New suspected diagnosis or refined differential\n"
            "- Change in condition (e.g., worsening kidney function)\n"
            "- Treatment-related insight (e.g., rationale for adjusting meds)\n"
            "- Biologically plausible interpretations (e.g., signs of volume overload)\n\n"
            "Non-valuable examples:\n"
            "- Repeating obvious facts ('BNP is high') without interpretation\n"
            "- Mere data reporting without reasoning\n"
            "- Shallow remarks or filler text\n\n"

            "========================\n"
            "**3. NOVELTY** – Has this already been conveyed to the clinician?\n"
            "A trace is novel if it introduces something **not covered in prior summaries**. You must compare the new trace against the summaries list.\n"
            "Examples of non-novel:\n"
            "- Reiterating the same diagnosis or lab interpretation\n"
            "- Adding detail to an already-summarized point (unless the detail changes meaning)\n"
            "- Stylistic rephrasing of what’s already known\n\n"
            "========================\n"

            "Only if **all three triggers** are satisfied should you return 'YES'. Otherwise, return 'NO'.\n"
            "Never guess or speculate — only respond if the trace is complete, clinically meaningful, and new.\n"
            "You simulate a human resident deciding when to update the attending. Be rigorous. Be concise. Favor clarity.\n\n"

            "You will now be given:\n"
            "- Previous summaries (already shown to the clinician)\n"
            "- Current trace buffer (previous reasoning traces not yet summarized)\n"
            "- The new trace\n\n"

            "Your response must be ONLY: 'YES' or 'NO'.\n\n"
            "Do not explain your reasoning.\n"
            "Do not generate the summary.\n\n"),
            ("user", 
            "Previous summaries:\n{summaries}\n\n"
            "Previous traces in buffer:\n{previous_trace}\n\n"
            "New trace to evaluate:\n{new_trace}\n\n"
            "Should this trigger summarization? Reply with only 'YES' or 'NO'.")
        ])
        self.traces: dict[str, TraceData] = {}

    async def addchunk(self, agent_id: str, chunk: str, previous_summaries: list[str]) -> bool:
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
            "summaries": previous_summaries,
            "previous_trace": previous_traces if previous_traces else "(No previous traces.)",
            "new_trace": tagged_chunk
        })
        decision = "YES" in flag_response.content.strip().upper()
        
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"BufferAgent decision for {agent_id}: {decision} (response: {flag_response.content.strip()[:100]})")
        if decision:
            logger.info(f"BufferAgent triggered summarization for {agent_id}")
        
        return decision
    
    def flush(self) -> list[str]:
        flushed = self.buffer.copy()
        self.buffer.clear()
        return flushed
        
    def get_trace_count(self, agent_id: str) -> int:
        return self.traces.get(agent_id, TraceData(count=0)).count