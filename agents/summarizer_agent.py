from langchain_core.prompts import ChatPromptTemplate
from schema.agent_summary import AgentSummary
from typing import List
from llm import groq_llm

class SummarizerAgent:
    def __init__(self):
        self.llm = groq_llm.with_structured_output(schema=AgentSummary)
        self.summarize_prompt = ChatPromptTemplate.from_messages([    
            ("system", "You are an expert summarizer for a medical multi-agent reasoning system. "
            "Extract essential information about agent actions and clinical reasoning from the new buffer. "
            "Focus on what agents did and why, using brief, action-oriented language. "
            "Be concise and practical - physicians need to quickly understand agent decisions. "
            "Extract only new information from the buffer, do not repeat previous summaries. "
            "Identify all agents mentioned in the buffer. "
            "CRITICAL: Strictly enforce character limits - action: MAX 100 chars, reasoning: MAX 200 chars, findings: MAX 150 chars, next_steps: MAX 100 chars. "
            "If content exceeds limits, prioritize the most essential information and truncate."),
            ("user", "Summary history:\n[ {summary_history} ]\n\nLatest summary:\n{latest_summary}\n\nNew reasoning buffer:\n{new_buffer}\n\nExtract structured summary of new agent actions and reasoning."),
        ])

    async def summarize(self, summary_history: List[str], latest_summary: str, new_buffer: str) -> AgentSummary:
        summarize_chain = self.summarize_prompt | self.llm
        summary = await summarize_chain.ainvoke({
            "summary_history": ",\n".join(summary_history),
            "latest_summary": latest_summary,
            "new_buffer": new_buffer
        })
        return summary