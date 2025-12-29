from typing import Optional, Callable, List
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

    async def received_trace(self, agent_id: str, text: str) -> Optional[AgentSummary]:
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
                callback(agent_id, text)
            except Exception as e:
                print(f"Error in trace callback: {e}")
        
        # Prepare previous summaries for buffer agent evaluation
        all_summaries = self.get_all_summaries()
        previous_summaries = self._format_summaries_history(all_summaries)
        
        trigger = await self.buffer_agent.addsegment(agent_id, text, previous_summaries)
        if trigger:
            agent_buffer = self.buffer_agent.flush()
            buffer_str = "\n".join(agent_buffer)
            all_summaries = self.get_all_summaries()
            summary_history_strs = self._format_summaries_history(all_summaries[:-1]) if len(all_summaries) > 1 else []
            latest_summary_str = self._format_summary_for_history(all_summaries[-1]) if all_summaries else "No summaries yet."
            summary = await self.summarizer_agent.summarize(
                agent_id,
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

    async def on_new_token(self, agent_id: str, token: str) -> Optional[AgentSummary]:
        """Process a single token for the given agent."""
        # Emit trace callbacks
        for callback in self.trace_callbacks:
            try:
                callback(agent_id, token)
            except Exception as e:
                print(f"Error in trace callback: {e}")

        # Pass token through TokenGate
        chunk = await self.token_gate.add_token(agent_id, token)

        # Process chunk if complete
        if chunk:
            summaries = self.get_all_summaries()
            previous_summaries = self._format_summaries_history(summaries)
            return await self._process_chunk(agent_id, chunk, previous_summaries, summaries)

        # Check TokenGate timers
        timer_chunk = await self.token_gate.check_timers(agent_id)
        if timer_chunk:
            summaries = self.get_all_summaries()
            previous_summaries = self._format_summaries_history(summaries)
            return await self._process_chunk(agent_id, timer_chunk, previous_summaries, summaries)

        return None

    async def _process_chunk(self, agent_id: str, chunk: str, previous_summaries: list[str], summaries: list[AgentSummary]) -> Optional[AgentSummary]:
        """Process a chunk of text for summarization."""
        trigger = await self.buffer_agent.addsegment(
            agent_id,
            chunk,
            previous_summaries
        )
        if trigger:
            agent_buffer = self.buffer_agent.flush()
            buffer_str = "\n".join(agent_buffer)
            summary_history_strs = self._format_summaries_history(summaries[:-1]) if len(summaries) > 1 else []
            latest_summary_str = self._format_summary_for_history(summaries[-1]) if summaries else "No summaries yet."
            summary = await self.summarizer_agent.summarize(
                agent_id,
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

    async def flush_agent(self, agent_id: str) -> Optional[AgentSummary]:
        """Flush remaining tokens for the given agent."""
        remaining = await self.token_gate.flush(agent_id)
        if remaining:
            summaries = self.get_all_summaries()
            previous_summaries = self._format_summaries_history(summaries)
            summary = await self._process_chunk(agent_id, remaining, previous_summaries, summaries)
            
            # If no summary was triggered but buffer has content, force a summary
            if summary is None and self.buffer_agent.buffer:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Forcing final summary for {agent_id} with remaining buffer content")
                agent_buffer = self.buffer_agent.flush()
                buffer_str = "\n".join(agent_buffer)
                summary_history_strs = self._format_summaries_history(summaries[:-1]) if len(summaries) > 1 else []
                latest_summary_str = self._format_summary_for_history(summaries[-1]) if summaries else "No summaries yet."
                summary = await self.summarizer_agent.summarize(
                    agent_id,
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