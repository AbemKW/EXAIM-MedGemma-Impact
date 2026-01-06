import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from infra import get_llm, LLMRole
from exaim_core.exaim import EXAIM

logger = logging.getLogger(__name__)


class InternalMedicineAgent(DemoBaseAgent):
    """Internal Medicine specialist agent for comprehensive clinical assessment"""
    
    def __init__(self, agent_id: str = "Internal Medicine Agent", exaim: EXAIM = None):
        super().__init__(agent_id, exaim)
        self.llm = get_llm(LLMRole.MAS)
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
        """Stream LLM output while sending tokens live to EXAIM and UI."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", input)
        ])

        chain = prompt | self.llm

        try:
            # LIVE token streaming loop
            async for chunk in chain.astream({}):
                token = self._extract_token(chunk)
                if not token:
                    continue

                # 1. Send to EXAIM in real-time
                if self.exaim:
                    await self.exaim.on_new_token(self.agent_id, token)

                # 2. Yield token to MAS graph
                yield token

        except ValueError as e:
            # Handle fallback (rare, but needed)
            if "No generation chunks were returned" in str(e):
                response = await chain.ainvoke({})
                for char in response.content:
                    if self.exaim:
                        await self.exaim.on_new_token(self.agent_id, char)
                    yield char
            else:
                raise

        # 3. After stream ends: flush remaining TokenGate content (parks tail content for later)
        if self.exaim:
            await self.exaim.flush_agent(self.agent_id)
