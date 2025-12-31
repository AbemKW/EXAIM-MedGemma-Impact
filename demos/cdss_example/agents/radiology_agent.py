import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from infra import get_llm, LLMRole
from exaid_core.exaid import EXAID
from demos.cdss_example.callbacks.agent_streaming_callback import AgentStreamingCallback

logger = logging.getLogger(__name__)


class RadiologyAgent(DemoBaseAgent):
    """Radiology specialist agent for imaging interpretation and recommendations"""
    
    def __init__(self, agent_id: str = "Radiology Agent", exaid: EXAID = None):
        super().__init__(agent_id, exaid)
        self.llm = get_llm(LLMRole.MAS)
        self.system_prompt = (
            "You are the Radiology specialist in a multi-agent clinical decision support system.\n\n"
            "MULTI-AGENT SYSTEM CONTEXT:\n"
            "- This is a collaborative MAS with Laboratory, Cardiology, Internal Medicine, and Radiology specialists.\n"
            "- An Orchestrator coordinates workflow and maintains a running clinical summary.\n"
            "- You will receive: the original case, the running summary, any described imaging findings, "
            "a recent specialist update, and a specific imaging-focused task.\n"
            "- Build on the running summary and prior inputs – do NOT restart from scratch.\n"
            "- Provide ONLY radiology-domain reasoning – no workflow management, no explicit agent coordination.\n\n"
            "YOUR EXPERTISE:\n"
            "You think and reason like an experienced radiologist. Your expertise includes:\n"
            "- Interpretation of chest X-ray, CT, MRI, ultrasound, and other imaging modalities.\n"
            "- Recognizing and describing patterns: consolidation, effusion, edema, atelectasis, nodules, emboli, fractures, "
            "soft-tissue abnormalities, and more.\n"
            "- Mapping imaging patterns to plausible differential diagnoses in clinical context.\n\n"
            "APPROACH:\n"
            "Use Chain of Thought reasoning. Show your analytical process and be VERBOSE:\n"
            "1. Review imaging findings in context of the case and clinical presentation\n"
            "2. Identify key radiological findings and their significance\n"
            "3. Correlate imaging with laboratory and clinical data\n"
            "4. Assess for urgent findings requiring immediate attention\n"
            "5. Consider differential diagnoses based on imaging patterns\n"
            "6. Recommend additional imaging studies if needed\n\n"
            "STYLE:\n"
            "- Produce long, natural, unstructured chain-of-thought.\n"
            "- Be verbose and explicit in your reasoning.\n"
            "- Write as if you are thinking aloud. Use explicit chain-of-thought.\n"
            "- No headings, no bullets, no format templates—just free-flowing radiologic thought.\n"
            "Focus on radiological interpretation - the Orchestrator handles coordination and synthesis."
        )
    
    async def stream(self, input: str) -> AsyncIterator[str]:
        """Stream LLM output while sending tokens live to EXAID and UI."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", input)
        ])

        chain = prompt | self.llm
        callback = AgentStreamingCallback(agent_id=self.agent_id)

        try:
            # LIVE token streaming loop
            async for chunk in chain.astream({}, callbacks=[callback]):
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
