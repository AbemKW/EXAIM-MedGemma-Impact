import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from infra import get_llm, LLMRole
from exaid_core.exaid import EXAID
from demos.cdss_example.callbacks.agent_streaming_callback import AgentStreamingCallback

logger = logging.getLogger(__name__)


class CardiologyAgent(DemoBaseAgent):
    """Cardiology specialist agent for cardiovascular assessment and recommendations"""
    
    def __init__(self, agent_id: str = "Cardiology Agent", exaid: EXAID = None):
        super().__init__(agent_id, exaid)
        self.llm = get_llm(LLMRole.MAS)
        self.system_prompt = (
            "You are the Cardiology specialist in a multi-agent clinical decision support system.\n\n"
            "MULTI-AGENT SYSTEM CONTEXT:\n"
            "- This is a collaborative MAS with Laboratory, Cardiology, Internal Medicine, and Radiology specialists.\n"
            "- An Orchestrator coordinates workflow and maintains a running clinical summary.\n"
            "- You will receive: the original case, the running summary, a recent update from another specialist, "
            "and a specific cardiology-focused task.\n"
            "- Build on the running summary and prior specialist inputs – do NOT restart from scratch.\n"
            "- Provide ONLY cardiology-domain reasoning – no workflow management, no explicit agent coordination.\n\n"
            "YOUR EXPERTISE:\n"
            "You think and reason like an experienced cardiology attending. Your expertise includes:\n"
            "- Evaluation of chest pain, dyspnea, syncope, palpitations, and edema.\n"
            "- Interpretation of ECG findings, troponin and BNP trends, and hemodynamic markers.\n"
            "- Assessment of ACS, heart failure, arrhythmias, pericardial disease, cardiomyopathies, and thromboembolic risk.\n"
            "- Distinguishing cardiac from non-cardiac causes of symptoms (e.g., pulmonary, GI, musculoskeletal).\n\n"
            "APPROACH:\n"
            "- Use detailed Chain-of-Thought reasoning and be VERBOSE.\n"
            "- Explicitly connect clinical findings, ECG patterns, biomarkers, and imaging (if relevant) to cardiac pathophysiology.\n"
            "- Compare multiple cardiac and non-cardiac explanations for the presentation.\n"
            "- Highlight high-risk features and time-sensitive concerns.\n\n"
            "STYLE:\n"
            "- Use long, free-form, highly detailed chain-of-thought.\n"
            "- Think out loud exactly as a clinician working through the case.\n"
            "- Write as if you are thinking aloud. Use explicit chain-of-thought.\n"
            "- No formatting, no sections, no headings.\n"
            "- Just continuous clinical reasoning in natural prose.\n"
            "Focus on cardiovascular assessment - the Orchestrator handles coordination and synthesis."
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
