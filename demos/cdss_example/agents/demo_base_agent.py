"""Minimal base class for CDSS agents - provides agent_id consistency only"""

from exaid_core.exaid import EXAID


class DemoBaseAgent:
    """Minimal base class for all CDSS agents
    
    Provides agent_id and EXAID reference storage.
    No conversation history, no abstract methods, no complexity.
    """
    
    def __init__(self, agent_id: str, exaid: EXAID):
        self.agent_id = agent_id
        self.exaid = exaid
    
    @staticmethod
    def _extract_token(chunk):
        """Universal token extractor for LangChain streaming chunks"""
        if hasattr(chunk, "text") and chunk.text:
            return chunk.text
        if hasattr(chunk, "content") and chunk.content:
            return chunk.content
        if isinstance(chunk, str) and chunk:
            return chunk
        return None