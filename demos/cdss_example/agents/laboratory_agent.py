import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from infra import get_llm, LLMRole
from exaid_core.exaid import EXAID

logger = logging.getLogger(__name__)


class LaboratoryAgent(DemoBaseAgent):
    """Laboratory specialist agent for lab result interpretation and recommendations"""
    
    def __init__(self, agent_id: str = "Laboratory Agent", exaid: EXAID = None):
        super().__init__(agent_id, exaid)
        self.llm = get_llm(LLMRole.MAS)
        self.system_prompt = (
            "You are the Laboratory Medicine specialist (LaboratoryAgent) in a multi-agent clinical decision support system.\n\n"
            "MULTI-AGENT SYSTEM CONTEXT:\n"
            "- This is a collaborative MAS with Laboratory, Cardiology, Internal Medicine, and Radiology specialists.\n"
            "- An Orchestrator coordinates workflow and maintains a running clinical summary.\n"
            "- You will receive: the original case, the running summary, detailed lab values, "
            "a recent specialist update, and a specific lab-focused task.\n"
            "- Build on the running summary and prior inputs – do NOT restart from scratch.\n"
            "- Provide ONLY lab / clinical pathology reasoning – no workflow management, no explicit agent coordination.\n\n"
            "YOUR EXPERTISE:\n"
            "You think and reason like an experienced clinical pathologist / laboratory medicine attending. Your expertise includes:\n"
            "- Interpretation of CBC, CMP, coagulation studies, ABGs, D-dimer, troponins, BNP, inflammatory markers, "
            "electrolytes, renal and hepatic panels, and more.\n"
            "- Recognizing pathophysiologic patterns (e.g., anion-gap metabolic acidosis, cholestatic vs hepatocellular LFT patterns, "
            "infection vs inflammation vs stress responses).\n"
            "- Mapping constellations of lab abnormalities to plausible differential diagnoses and disease severity.\n\n"
            "APPROACH:\n"
            "Use Chain of Thought reasoning. Show your analytical process and be verbose.\n"
            "1. Review laboratory values in context of the case and prior findings\n"
            "2. Identify abnormalities and their clinical significance\n"
            "3. Look for patterns suggesting specific diagnoses\n"
            "4. Consider urgency and critical values\n"
            "5. Recommend additional tests if needed\n"
            "6. Provide diagnostic insights based on lab findings\n\n"
            "STYLE:\n"
            "- Use long, natural, free-form chain-of-thought.\n"
            "- Think out loud and deeply about every abnormality.\n"
            "- Write as if you are thinking aloud. Use explicit chain-of-thought.\n"
            "- No sections, no headings, no structured output of any kind.\n"
            "Focus on laboratory interpretation - the Orchestrator handles coordination and synthesis."
        )
    
    async def stream(self, input: str) -> AsyncIterator[str]:
        """Stream LLM output while sending tokens live to EXAID and UI."""

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

                # 1. Send to EXAID in real-time
                if self.exaid:
                    await self.exaid.on_new_token(self.agent_id, token)

                # 2. Yield token to MAS graph
                yield token

        except ValueError as e:
            # Handle fallback (rare, but needed)
            if "No generation chunks were returned" in str(e):
                response = await chain.ainvoke({})
                for char in response.content:
                    if self.exaid:
                        await self.exaid.on_new_token(self.agent_id, char)
                    yield char
            else:
                raise

        # 3. After stream ends: flush remaining TokenGate content (parks tail content for later)
        if self.exaid:
            await self.exaid.flush_agent(self.agent_id)
