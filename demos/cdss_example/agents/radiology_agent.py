from typing import AsyncIterator, Optional
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from exaid_core.llm import mas_llm
from exaid_core.callbacks.agent_streaming_callback import AgentStreamingCallback


class RadiologyAgent(DemoBaseAgent):
    """Radiology specialist agent for medical imaging interpretation and diagnostic insights"""
    
    def __init__(self, agent_id: str = "RadiologyAgent"):
        super().__init__(agent_id)
        self.llm = mas_llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert radiologist in a clinical decision support system. "
             "You specialize in interpreting medical imaging studies and providing diagnostic insights "
             "based on radiological findings.\n\n"
             "IMPORTANT: Use Chain of Thought reasoning. Show your thinking process step-by-step:\n"
             "1. First, identify what imaging modalities are available (X-ray, CT, MRI, ultrasound)\n"
             "2. Systematically describe the imaging findings (normal and abnormal)\n"
             "3. Analyze patterns and their clinical significance\n"
             "4. Correlate imaging findings with the clinical presentation\n"
             "5. Build a radiological differential diagnosis\n"
             "6. Provide recommendations for additional imaging or clinical correlation\n\n"
             "Always show your reasoning process explicitly. Use phrases like:\n"
             "- 'Let me analyze the imaging studies systematically...'\n"
             "- 'On the [X-ray/CT/MRI], I observe...'\n"
             "- 'This pattern is consistent with...'\n"
             "- 'The combination of findings suggests...'\n"
             "- 'Differential considerations include...'\n"
             "- 'I recommend the following additional imaging...'\n\n"
             "Your expertise includes:\n"
             "- Plain radiography (X-ray) interpretation\n"
             "- Computed tomography (CT) analysis\n"
             "- Magnetic resonance imaging (MRI) interpretation\n"
             "- Ultrasound assessment\n"
             "- Chest imaging (pneumonia, heart failure, pneumothorax, masses)\n"
             "- Cardiovascular imaging (cardiomegaly, aortic abnormalities, pulmonary vasculature)\n"
             "- Abdominal and pelvic imaging\n"
             "- Musculoskeletal imaging\n"
             "- Neuroimaging (brain, spine)\n"
             "- Interventional radiology considerations\n\n"
             "Guidelines:\n"
             "- Systematically evaluate all available imaging studies\n"
             "- Describe findings using standard radiological terminology\n"
             "- Correlate imaging findings with clinical history and other diagnostic data\n"
             "- Identify critical findings that require immediate attention\n"
             "- Build differential diagnoses based on imaging patterns\n"
             "- Recommend additional imaging when initial studies are insufficient\n"
             "- Consider prior imaging for comparison when available\n"
             "- Recognize limitations of each imaging modality\n"
             "- Suggest appropriate follow-up imaging intervals\n\n"
             "When you need input from other specialists (Internal Medicine, Cardiology, Laboratory), "
             "you may request consultation by mentioning their expertise in your findings. "
             "Always show your step-by-step diagnostic reasoning process."),
            ("user", "{input}")
        ])
    
    def _build_prompt_with_history(self, input: str) -> ChatPromptTemplate:
        """Build prompt including conversation history"""
        # Extract system message content from the prompt template
        system_message = self.prompt.messages[0]
        if hasattr(system_message, 'prompt') and hasattr(system_message.prompt, 'template'):
            system_content = system_message.prompt.template
        elif hasattr(system_message, 'content'):
            system_content = system_message.content
        else:
            # Fallback: get from original prompt definition
            system_content = self.prompt.messages[0].prompt.template if hasattr(self.prompt.messages[0], 'prompt') else str(self.prompt.messages[0])
        
        messages = [
            ("system", system_content)
        ]
        
        # Add conversation history
        for msg in self.conversation_history:
            if msg["role"] == "user":
                messages.append(("user", msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(("assistant", msg["content"]))
        
        # Add current input
        messages.append(("user", input))
        
        return ChatPromptTemplate.from_messages(messages)
    
    async def act(self, input: str) -> str:
        """Interpret imaging studies and provide radiological assessment"""
        prompt = self._build_prompt_with_history(input)
        chain = prompt | self.llm
        response = await chain.ainvoke({"input": input})
        return response.content
    
    async def act_stream(self, input: str) -> AsyncIterator[str]:
        """Stream tokens as they are generated by the LLM
        
        Args:
            input: Input text for the agent
            
        Yields:
            Tokens as strings as they are generated
        """
        prompt = self._build_prompt_with_history(input)
        chain = prompt | self.llm
        callback = AgentStreamingCallback(agent_id=self.agent_id)
        try:
            async for chunk in chain.astream({"input": input}, callbacks=[callback]):
                # Handle different chunk formats from LangChain
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if content:
                        yield content
                elif isinstance(chunk, str) and chunk:
                    yield chunk
                elif isinstance(chunk, dict) and 'content' in chunk:
                    if chunk['content']:
                        yield chunk['content']
        except ValueError as e:
            # If streaming fails, fall back to non-streaming and yield the full response
            if "No generation chunks were returned" in str(e):
                print(f"[WARNING] Streaming failed for {self.agent_id}, falling back to non-streaming mode")
                response = await self.act(input)
                # Yield the response character by character to simulate streaming
                for char in response:
                    yield char
            else:
                raise
    
    async def decide_consultation(self, findings: str, consulted_agents: list[str]) -> Optional[str]:
        """Decide if specialist consultation is needed based on imaging findings
        
        Args:
            findings: The radiology agent's findings and analysis
            consulted_agents: List of agents that have already been consulted
            
        Returns:
            Agent name if consultation is needed (e.g., "internal_medicine", "cardiology", "laboratory"), None otherwise
        """
        # Check which specialists haven't been consulted yet
        available_specialists = []
        if "internal_medicine" not in consulted_agents:
            available_specialists.append("internal_medicine")
        if "cardiology" not in consulted_agents:
            available_specialists.append("cardiology")
        if "laboratory" not in consulted_agents:
            available_specialists.append("laboratory")
        
        if not available_specialists:
            return None
        
        consultation_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a radiologist analyzing your imaging findings to determine "
             "if clinical or specialist consultation is needed.\n\n"
             "Available specialists for consultation:\n"
             f"- Internal Medicine: For clinical correlation and overall patient management\n"
             f"- Cardiology: For cardiac imaging correlation and cardiac-specific assessment\n"
             f"- Laboratory: For lab correlation with imaging findings\n\n"
             "Request consultation if:\n"
             "- Your imaging findings require clinical correlation\n"
             "- You identified abnormalities that need specialist interpretation\n"
             "- The case requires integration with clinical or laboratory data\n"
             "- You need specialist input to narrow the differential\n\n"
             "Do NOT request consultation if:\n"
             "- Your radiological interpretation is complete and clear\n"
             "- The imaging findings are definitive without additional input\n"
             "- No clinical correlation is needed\n\n"
             f"Available specialists: {', '.join(available_specialists)}\n\n"
             "Respond with ONLY the specialist name if consultation is needed (e.g., 'internal_medicine', 'cardiology', 'laboratory'), "
             "or 'none' if no consultation is needed."),
            ("user", 
             "Radiology Findings:\n{findings}\n\n"
             "Based on these imaging findings, do you need specialist consultation? "
             f"Respond with one of: {', '.join(available_specialists)}, or 'none'.")
        ])
        
        chain = consultation_prompt | self.llm
        response = await chain.ainvoke({"findings": findings})
        response_text = response.content.strip().lower()
        
        # Check if any specialist name is mentioned
        for specialist in available_specialists:
            if specialist in response_text:
                return specialist
        
        return None
    
    async def evaluate_other_agent_findings(self, other_agent_id: str, findings: str) -> Optional[str]:
        """Review and potentially challenge other agents' findings
        
        Args:
            other_agent_id: ID of the agent whose findings are being reviewed
            findings: The findings to review
            
        Returns:
            Question or challenge if one is needed, None otherwise
        """
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a radiologist reviewing findings from another specialist. "
             "Your role is to evaluate if their findings are consistent with imaging data, "
             "if there are any concerns, or if you need clarification.\n\n"
             "You should raise a question or challenge if:\n"
             "- The findings seem inconsistent with imaging findings\n"
             "- Important imaging features appear to be overlooked or misinterpreted\n"
             "- You need clarification on how they integrated imaging data\n"
             "- There are contradictions between their assessment and radiological evidence\n"
             "- Their interpretation doesn't align with imaging patterns\n\n"
             "If everything looks reasonable, respond with 'none'.\n\n"
             "If you have a question or concern, state it clearly and concisely."),
            ("user",
             f"Findings from {other_agent_id.upper()} Specialist:\n{findings}\n\n"
             "Do you have any questions, concerns, or need clarification? "
             "Respond with your question/concern or 'none' if everything looks good.")
        ])
        
        chain = evaluation_prompt | self.llm
        response = await chain.ainvoke({})
        response_text = response.content.strip()
        
        if response_text.lower() == "none" or not response_text:
            return None
        
        return response_text
    
    async def decide_if_needs_response(self, new_findings: dict) -> bool:
        """Determine if new findings require a response or update to your imaging interpretation
        
        Args:
            new_findings: Dictionary of new findings from other agents
            
        Returns:
            True if a response/update is needed, False otherwise
        """
        if not new_findings:
            return False
        
        # Radiology may need to re-interpret imaging in light of new clinical or lab data
        # This is a simple heuristic - in practice, the agent will be called
        # by the orchestrator when new findings are available
        return True
