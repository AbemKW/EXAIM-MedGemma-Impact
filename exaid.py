from typing import Optional, AsyncIterator
from agents.buffer_agent import BufferAgent
from agents.summarizer_agent import SummarizerAgent
from agents.token_gate import TokenGate
from schema.agent_summary import AgentSummary

class EXAID:
    def __init__(self):
        self.buffer_agent = BufferAgent()
        self.summarizer_agent = SummarizerAgent()
        self.token_gate = TokenGate()
        self.summaries: list[AgentSummary] = []
    
    def _print_summary(self, summary: AgentSummary):
        agents_str = ", ".join(summary.agents)
        print(f"\n{'='*60}")
        print(f"Summary Update - Agents: {agents_str}")
        print(f"{'='*60}")
        print(f"Action: {summary.action}")
        print(f"Reasoning: {summary.reasoning}")
        if summary.findings:
            print(f"Findings: {summary.findings}")
        if summary.next_steps:
            print(f"Next Steps: {summary.next_steps}")
        print()
    
    def get_all_summaries(self) -> list[AgentSummary]:
        """Returns all summaries as AgentSummary objects."""
        return self.summaries

    def get_summaries_by_agent(self, agent_id: str) -> list[AgentSummary]:
        """Get all summaries involving a specific agent."""
        return [s for s in self.summaries if agent_id in s.agents]

    def get_agent_trace_count(self, agent_id: str) -> int:
        return self.buffer_agent.get_trace_count(agent_id)

    def _format_summary_for_history(self, summary: AgentSummary) -> str:
        agents_str = ", ".join(summary.agents)
        parts = [f"Agents: {agents_str}", f"Action: {summary.action}", f"Reasoning: {summary.reasoning}"]
        if summary.findings:
            parts.append(f"Findings: {summary.findings}")
        if summary.next_steps:
            parts.append(f"Next: {summary.next_steps}")
        return " | ".join(parts)
    
    def _format_summaries_history(self, summaries: list[AgentSummary]) -> list[str]:
        return [self._format_summary_for_history(s) for s in summaries]

    async def received_trace(self, id: str, text: str) -> Optional[AgentSummary]:
        trigger = await self.buffer_agent.addchunk(id, text)
        if trigger:
            agent_buffer = self.buffer_agent.flush()
            buffer_str = "\n".join(agent_buffer)
            all_summaries = self.get_all_summaries()
            summary_history_strs = self._format_summaries_history(all_summaries[:-1]) if len(all_summaries) > 1 else []
            latest_summary_str = self._format_summary_for_history(all_summaries[-1]) if all_summaries else "No summaries yet."
            summary = await self.summarizer_agent.summarize(
                summary_history_strs,
                latest_summary_str,
                buffer_str
            )
            self.summaries.append(summary)
            self._print_summary(summary)
            return summary
        return None
    
    async def received_streamed_tokens(self, agent_id: str, token_generator: AsyncIterator[str]) -> Optional[AgentSummary]:
        last_summary = None
        
        async def process_chunk(chunk: str) -> Optional[AgentSummary]:
            trigger = await self.buffer_agent.addchunk(agent_id, chunk)
            if trigger:
                agent_buffer = self.buffer_agent.flush()
                buffer_str = "\n".join(agent_buffer)
                all_summaries = self.get_all_summaries()
                summary_history_strs = self._format_summaries_history(all_summaries[:-1]) if len(all_summaries) > 1 else []
                latest_summary_str = self._format_summary_for_history(all_summaries[-1]) if all_summaries else "No summaries yet."
                summary = await self.summarizer_agent.summarize(
                    summary_history_strs,
                    latest_summary_str,
                    buffer_str
                )
                self.summaries.append(summary)
                self._print_summary(summary)
                return summary
            return None
        
        async for token in token_generator:
            chunk = await self.token_gate.add_token(agent_id, token)
            if chunk:
                summary = await process_chunk(chunk)
                if summary:
                    last_summary = summary
            timer_chunk = await self.token_gate.check_timers(agent_id)
            if timer_chunk:
                summary = await process_chunk(timer_chunk)
                if summary:
                    last_summary = summary
        
        remaining = await self.token_gate.flush(agent_id)
        if remaining:
            summary = await process_chunk(remaining)
            if summary:
                last_summary = summary
        
        return last_summary
