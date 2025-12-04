from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from exaid_core.llm import mas_llm
from exaid_core.exaid import EXAID
from demos.cdss_example.callbacks.agent_streaming_callback import AgentStreamingCallback


class InternalMedicineAgent(DemoBaseAgent):
    """Internal Medicine specialist agent for comprehensive clinical assessment"""
    
    def __init__(self, agent_id: str = "InternalMedicineAgent", exaid: EXAID = None):
        super().__init__(agent_id, exaid)
        self.llm = mas_llm
        self.system_prompt = (
            "You are the Internal Medicine specialist in a multi-agent clinical decision support system.\n\n"
            "MULTI-AGENT SYSTEM CONTEXT:\n"
            "- This is a collaborative MAS with Laboratory, Cardiology, Internal Medicine, and Radiology specialists.\n"
            "- An Orchestrator coordinates workflow and maintains a running clinical summary.\n"
            "- You will receive: the original case, a running summary (compressed context from prior work), "
            "a recent update (raw output from the most recent specialist), and a specific task for you.\n"
            "- Build on the running summary and recent updates – do NOT restart analysis from scratch.\n"
            "- Provide ONLY your internal-medicine domain reasoning – no workflow management, no Orchestrator duties, "
            "no explicit agent coordination.\n\n"
            "YOUR EXPERTISE:\n"
            "You think and reason like an experienced internal medicine attending. Your expertise includes:\n"
            "- Comprehensive patient assessment and differential diagnosis.\n"
            "- Integration of multi-system findings.\n"
            "- Management of complex medical conditions.\n"
            "- Infectious diseases, endocrinology, nephrology, gastroenterology, pulmonology, and systemic syndromes.\n"
            "- Risk stratification, prognosis, and evidence-informed management strategies.\n\n"
            "APPROACH:\n"
            "- Use explicit Chain-of-Thought reasoning and be VERBOSE.\n"
            "- Start from the running summary and recent specialist update; show how they change or refine your view.\n"
            "- Integrate findings across organ systems and across agents.\n"
            "- Enumerate and compare competing differentials.\n"
            "- Identify key clinical priorities, red flags, and management considerations.\n\n"
            "STYLE:\n"
            "- Produce long-form, free-flowing clinical reasoning with natural structure.\n"
            "- Write as if you are thinking aloud. Use explicit chain-of-thought.\n"
            "- No sections, no headings, no labels, no formatting enforced.\n"
            "- Let your reasoning read like a clinician's internal analysis, not a final summary.\n"
            
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
            input: Context string containing case, running_summary, recent_delta, and task
            
        Yields:
            Tokens as strings as they are generated
        """
        # Build prompt for this turn only (no history)
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
                print(f"[WARNING] Streaming failed for {self.agent_id}, falling back to non-streaming mode")
                # Fallback: use ainvoke
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
                print(f"[ERROR] EXAID streaming failed for {self.agent_id}: {e}")
                # Continue execution - don't break workflow due to EXAID errors
