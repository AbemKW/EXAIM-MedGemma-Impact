"""Minimal base class for CDSS agents - provides agent_id consistency only"""


class DemoBaseAgent:
    """Minimal base class for all CDSS agents
    
    Provides only agent_id storage for consistency.
    No conversation history, no abstract methods, no complexity.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        pass