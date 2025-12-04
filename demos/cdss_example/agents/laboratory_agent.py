from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from exaid_core.llm import mas_llm
from exaid_core.exaid import EXAID
from demos.cdss_example.callbacks.agent_streaming_callback import AgentStreamingCallback


class LaboratoryAgent(DemoBaseAgent):
    """Laboratory specialist agent for lab result interpretation and recommendations"""
    
    def __init__(self, agent_id: str = "LaboratoryAgent", exaid: EXAID = None):
        super().__init__(agent_id, exaid)
        self.llm = mas_llm
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
            "Use Chain of Thought reasoning. Show your analytical process and be VERBOSE.:\n"
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
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if content:
                        collected_tokens.append(content)
                        yield content
                elif isinstance(chunk, str) and chunk:
                    collected_tokens.append(chunk)
                    yield chunk
                elif isinstance(chunk, dict) and 'content' in chunk:
                    if chunk['content']:
                        collected_tokens.append(chunk['content'])
                        yield chunk['content']
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

