"""Minimal base class for CDSS agents - provides agent_id consistency only"""

import logging
from typing import Optional
from exaid_core.exaid import EXAID

logger = logging.getLogger(__name__)


class DemoBaseAgent:
    """Minimal base class for all CDSS agents
    
    Provides agent_id and EXAID reference storage.
    No conversation history, no abstract methods, no complexity.
    
    Note: EXAID is required for full agent functionality (token streaming to UI).
    If exaid is None, agent will function but EXAID integration will be disabled.
    """
    
    def __init__(self, agent_id: str, exaid: Optional[EXAID] = None):
        if exaid is None:
            logger.warning(
                f"EXAID instance is None for agent '{agent_id}'. "
                "Agent will function but EXAID integration (token streaming) will be disabled."
            )
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