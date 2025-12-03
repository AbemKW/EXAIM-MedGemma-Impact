from typing import Optional, AsyncIterator, Callable, List
from exaid_core.buffer_agent.buffer_agent import BufferAgent
from exaid_core.summarizer_agent.summarizer_agent import SummarizerAgent
from exaid_core.token_gate.token_gate import TokenGate
from exaid_core.schema.agent_summary import AgentSummary

class EXAID:
    def __init__(self):
        self.buffer_agent = BufferAgent()
        self.summarizer_agent = SummarizerAgent()
        self.token_gate = TokenGate()
        self.summaries: list[AgentSummary] = []
        self.trace_callbacks: List[Callable[[str, str], None]] = []
        self.summary_callbacks: List[Callable[[AgentSummary], None]] = []
    
    def register_trace_callback(self, callback: Callable[[str, str], None]):
        """Register a callback function to be called when trace tokens are received.
        
        Args:
            callback: Function that takes (agent_id: str, token: str) as arguments
        """
        self.trace_callbacks.append(callback)
    
    def register_summary_callback(self, callback: Callable[[AgentSummary], None]):
        """Register a callback function to be called when summaries are created.
        
        Args:
            callback: Function that takes (summary: AgentSummary) as argument
        """
        self.summary_callbacks.append(callback)
    
    def _print_summary(self, summary: AgentSummary):
        if summary is None:
            print(f"\n{'='*60}")
            print(f"Summary Update")
            print(f"{'='*60}")
            print(f"Warning: Received None summary")
            print()
            return
        
        print(f"\n{'='*60}")
        print(f"Summary Update")
        print(f"{'='*60}")
        print(f"Status / Action: {summary.status_action}")
        print(f"Key Findings: {summary.key_findings}")
        print(f"Differential & Rationale: {summary.differential_rationale}")
        print(f"Uncertainty / Confidence: {summary.uncertainty_confidence}")
        print(f"Recommendation / Next Step: {summary.recommendation_next_step}")
        print(f"Agent Contributions: {summary.agent_contributions}")
        print()
    
    def get_all_summaries(self) -> list[AgentSummary]:
        """Returns all summaries as AgentSummary objects."""
        return self.summaries

    def get_summaries_by_agent(self, agent_id: str) -> list[AgentSummary]:
        """Get all summaries involving a specific agent."""
        return [s for s in self.summaries if agent_id.lower() in s.agent_contributions.lower()]

    def get_agent_trace_count(self, agent_id: str) -> int:
        return self.buffer_agent.get_trace_count(agent_id)

    def _format_summary_for_history(self, summary: AgentSummary) -> str:
        """Converts an AgentSummary to a string representation for use in history."""
        parts = [
            f"Status/Action: {summary.status_action}",
            f"Key Findings: {summary.key_findings}",
            f"Differential/Rationale: {summary.differential_rationale}",
            f"Uncertainty/Confidence: {summary.uncertainty_confidence}",
            f"Recommendation/Next: {summary.recommendation_next_step}",
            f"Agent Contributions: {summary.agent_contributions}"
        ]
        return " | ".join(parts)
    
    def _format_summaries_history(self, summaries: list[AgentSummary]) -> list[str]:
        """Converts a list of AgentSummary objects to string representations for prompt history."""
        return [self._format_summary_for_history(s) for s in summaries]

    async def received_trace(self, id: str, text: str) -> Optional[AgentSummary]:
        """
        Processes a trace for the given agent ID and text, triggering summarization if appropriate.

        Returns:
            AgentSummary: if summarization was triggered.
            None: otherwise.
        """
        # Emit trace text as tokens to callbacks (for non-streaming traces)
        for callback in self.trace_callbacks:
            try:
                # Send the entire text as a single "token" for non-streaming traces
                callback(id, text)
            except Exception as e:
                print(f"Error in trace callback: {e}")
        
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
            if summary is not None:
                self.summaries.append(summary)
                self._print_summary(summary)
            
            # Emit summary event to callbacks
            for callback in self.summary_callbacks:
                try:
                    callback(summary)
                except Exception as e:
                    print(f"Error in summary callback: {e}")
            
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
                if summary is not None:
                    self.summaries.append(summary)
                    self._print_summary(summary)
                
                # Emit summary event to callbacks
                for callback in self.summary_callbacks:
                    try:
                        callback(summary)
                    except Exception as e:
                        print(f"Error in summary callback: {e}")
                
                return summary
            return None
        
        async for token in token_generator:
            # Emit token event to callbacks
            for callback in self.trace_callbacks:
                try:
                    callback(agent_id, token)
                except Exception as e:
                    # Don't let callback errors break trace processing
                    print(f"Error in trace callback: {e}")
            
            
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
        
        # Note: Trace completion is handled via token-by-token callbacks above
        # No separate completion event needed since tokens are streamed in real-time
        
        return last_summary
