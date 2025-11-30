from abc import ABC, abstractmethod
from typing import List, Dict

class DemoBaseAgent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    @abstractmethod
    async def act(self, input: str) -> str:
        pass