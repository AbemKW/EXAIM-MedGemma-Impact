import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from exaid_core.llm import mas_llm
from exaid_core.exaid import EXAID
from demos.cdss_example.callbacks.agent_streaming_callback import AgentStreamingCallback

logger = logging.getLogger(__name__)


class OrchestratorAgent(DemoBaseAgent):
    """Orchestrator agent that maintains running summary and coordinates specialist workflow
    
    Responsibilities:
    - Compress specialist outputs into running_summary (via nodes.py orchestrator_node)
    - Decide next specialist to call (via nodes.py orchestrator_node)
    - Generate task instructions for specialists (via nodes.py orchestrator_node)
    - Synthesize final recommendations (via nodes.py synthesis_node)
    
    Note: The orchestration logic (compression, decision, task generation) is implemented
    in nodes.py orchestrator_node, which uses this agent's stream() method for LLM interaction.
    """
    
    def __init__(self, agent_id: str = "OrchestratorAgent", exaid: EXAID = None):
        super().__init__(agent_id, exaid)
        self.llm = mas_llm
        
        # System prompt for orchestrator (used in all tasks)
        self.system_prompt = (
            "You are the OrchestratorAgent in a multi-agent clinical decision support system.\n\n"
            "CLINICAL ROLE:\n"
            "- You think and reason like an experienced attending physician.\n"
            "- You understand pathophysiology, diagnostic reasoning, risk stratification, and guideline-based care.\n"
            "- You coordinate work between four specialist agents:\n"
            "  - InternalMedicineAgent: broad, general diagnostic integration.\n"
            "  - CardiologyAgent: cardiovascular symptoms, ECGs, troponins, chest pain, etc.\n"
            "  - RadiologyAgent: imaging findings (X-ray, CT, MRI, US) and their interpretation.\n"
            "  - LaboratoryAgent: lab tests, reference ranges, and pathophysiology of abnormalities.\n\n"
            "SYSTEM RESPONSIBILITIES:\n"
            "- Maintain a concise running summary of the evolving case.\n"
            "- Decide which specialist should contribute next, or when it is time to synthesize.\n"
            "- Generate focused, clinically meaningful task instructions for specialists.\n"
            "- When requested, synthesize the specialists' outputs into a coherent clinical recommendation.\n\n"
            "STYLE AND REASONING:\n"
            "- Use rigorous clinical reasoning, as a good attending would.\n"
            "- Write as if you are thinking aloud. Use explicit chain-of-thought.\n"
            "- Track the overall diagnostic picture and make sure important problems are not ignored.\n"
            "- When updating the running summary, keep only key clinical findings, differential diagnoses, "
            "major uncertainties, and important recommendations; remove redundant or outdated details.\n"
            "- When asked to DECIDE THE NEXT SPECIALIST, strictly follow the user instructions and respond "
            "with ONLY the required token: 'laboratory', 'cardiology', 'internal_medicine', 'radiology', or 'synthesis'.\n"
            "- When generating task instructions or synthesis, you may be more verbose and explanatory.\n\n"
            "CONTEXT:\n"
            "- This system is used in a simulated / research environment for exploring multi-agent reasoning.\n"
            "- Your outputs may inform clinicians but must not be treated as definitive medical advice.\n"
        )
    
    async def stream(self, input: str) -> AsyncIterator[str]:
        """Unified streaming method that handles LLM streaming, UI callbacks, and EXAID token delivery
        
        Ephemeral prompt building for single turn - no conversation history.
        
        This method:
        1. Attaches AgentStreamingCallback for UI token delivery via message_queue
        2. Streams tokens from LLM
        3. Collects tokens while yielding them to the caller
        4. Sends collected tokens to EXAID for summarization pipeline
        
        Args:
            input: Task-specific input (compression, decision, task generation, or synthesis)
            
        Yields:
            Tokens as strings as they are generated
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", input)
        ])
        
        chain = prompt | self.llm
        callback = AgentStreamingCallback(agent_id=self.agent_id)
        
        # Collect tokens for EXAID while yielding to caller
        collected_tokens = []
        
        async def token_generator():
            """Internal generator that yields collected tokens to EXAID"""
            for token in collected_tokens:
                yield token
        
        try:
            async for chunk in chain.astream({}, callbacks=[callback]):
                token = self._extract_token(chunk)
                if token:
                    collected_tokens.append(token)
                    yield token
        except ValueError as e:
            if "No generation chunks were returned" in str(e):
                logger.warning(f"Streaming failed for {self.agent_id}, falling back to non-streaming mode")
                # Fallback: use ainvoke instead
                response = await chain.ainvoke({})
                for char in response.content:
                    collected_tokens.append(char)
                    yield char
            else:
                raise
        
        # Send collected tokens to EXAID for summarization pipeline (with error handling)
        if self.exaid and collected_tokens:
            try:
                await self.exaid.received_streamed_tokens(self.agent_id, token_generator())
            except Exception as e:
                logger.error(f"EXAID streaming failed for {self.agent_id}: {e}")


