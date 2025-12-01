from typing import AsyncIterator, Optional
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from exaid_core.llm import mas_llm
from exaid_core.callbacks.agent_streaming_callback import AgentStreamingCallback
from demos.cdss_example.schema.agent_messages import ConsultationRequest


class InternalMedicineAgent(DemoBaseAgent):
    """Internal Medicine specialist agent for comprehensive diagnostic reasoning and clinical integration"""
    
    def __init__(self, agent_id: str = "InternalMedicineAgent"):
        super().__init__(agent_id)
        self.llm = mas_llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert Internal Medicine physician in a clinical decision support system. "
             "You serve as the general diagnostician who integrates all clinical data to build "
             "comprehensive differential diagnoses and guide overall patient management.\n\n"
             "IMPORTANT: Use Chain of Thought reasoning. Show your thinking process step-by-step:\n"
             "1. First, synthesize all available clinical information (history, symptoms, vital signs)\n"
             "2. Identify key clinical features and their significance\n"
             "3. Build a broad differential diagnosis considering all body systems\n"
             "4. Integrate findings from specialists (lab, cardiology, radiology) into your assessment\n"
             "5. Prioritize diagnoses based on likelihood and severity\n"
             "6. Formulate a comprehensive management plan with clear recommendations\n\n"
             "Always show your reasoning process explicitly. Use phrases like:\n"
             "- 'Let me synthesize the clinical presentation...'\n"
             "- 'The key features of this case are...'\n"
             "- 'Considering these findings together...'\n"
             "- 'My differential diagnosis includes...'\n"
             "- 'The most likely diagnosis is... because...'\n"
             "- 'I recommend the following workup...'\n\n"
             "Your expertise includes:\n"
             "- Comprehensive history and physical examination interpretation\n"
             "- Building broad differential diagnoses across all systems\n"
             "- Integrating multi-specialty input into cohesive assessments\n"
             "- Recognizing complex multi-system diseases\n"
             "- Infectious disease management\n"
             "- Endocrine and metabolic disorders\n"
             "- Rheumatologic and autoimmune conditions\n"
             "- Hematologic disorders\n"
             "- Nephrology and fluid/electrolyte management\n"
             "- Gastroenterology and hepatology\n"
             "- General medical management and risk stratification\n\n"
             "Guidelines:\n"
             "- Always consider the full clinical context\n"
             "- Build comprehensive differential diagnoses (typically 3-5 possibilities)\n"
             "- Integrate findings from all specialists into a unified assessment\n"
             "- Identify which diagnoses are most likely and why\n"
             "- Recognize when specialist consultation is needed\n"
             "- Consider both common and serious/life-threatening conditions\n"
             "- Recommend appropriate diagnostic workup and management\n"
             "- Address immediate concerns and long-term management\n"
             "- Consider social, functional, and prognostic factors\n\n"
             "When you need input from specialists (Laboratory, Cardiology, Radiology), you may request "
             "consultation by mentioning their expertise in your findings. Always show your step-by-step "
             "diagnostic reasoning process."),
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
        """Provide comprehensive internal medicine analysis and recommendations"""
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
    
    async def decide_consultation(self, findings: str, consulted_agents: list[str]) -> Optional[ConsultationRequest]:
        """Decide if specialist consultation is needed based on findings
        
        Args:
            findings: The internal medicine agent's findings and analysis
            consulted_agents: List of agents that have already been consulted
            
        Returns:
            ConsultationRequest object if consultation is needed, None otherwise
        """
        # Check which specialists haven't been consulted yet
        available_specialists = []
        if "laboratory" not in consulted_agents:
            available_specialists.append("laboratory")
        if "cardiology" not in consulted_agents:
            available_specialists.append("cardiology")
        if "radiology" not in consulted_agents:
            available_specialists.append("radiology")
        
        if not available_specialists:
            return None
        
        consultation_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an Internal Medicine physician analyzing your findings to determine "
             "if specialist consultation is needed.\n\n"
             "Available specialists for consultation:\n"
             "- Laboratory: For lab test interpretation (CBC, metabolic panels, biomarkers, etc.)\n"
             "- Cardiology: For cardiac assessment, ECG interpretation, cardiac biomarkers\n"
             "- Radiology: For imaging interpretation (X-ray, CT, MRI findings)\n\n"
             "Request consultation if:\n"
             "- You need specialist expertise to interpret specific findings\n"
             "- The case requires detailed analysis from that specialty\n"
             "- You've identified concerns that require specialist input\n"
             "- Additional diagnostic information from that specialty would be valuable\n\n"
             "Do NOT request consultation if:\n"
             "- Your findings are complete without specialist input\n"
             "- The case has no concerns requiring that specialty\n"
             "- You can provide complete assessment without additional consultation\n\n"
             f"Available specialists: {', '.join(available_specialists)}\n\n"
             "If consultation is needed, respond with the specialist name and a brief clinical reason (1-2 sentences). "
             "If no consultation is needed, respond with requested_specialist as empty string."),
            ("user", 
             "Internal Medicine Findings:\n{findings}\n\n"
             "Based on these findings, do you need specialist consultation? "
             f"Available options: {', '.join(available_specialists)}")
        ])
        
        structured_llm = self.llm.with_structured_output(ConsultationRequest)
        chain = consultation_prompt | structured_llm
        
        try:
            response = await chain.ainvoke({"findings": findings})
            
            # Validate the requested specialist is in available list
            if response.requested_specialist and response.requested_specialist in available_specialists:
                return response
            else:
                return None
        except Exception as e:
            # If structured output fails, return None
            print(f"[WARNING] Structured output failed in internal_medicine decide_consultation: {e}")
            return None
    
    async def analyze_with_context(self, context: str, previous_findings: str, new_findings: dict) -> str:
        """Analyze with incremental context, building on previous findings and incorporating new information
        
        Args:
            context: The incremental context built by the context builder
            previous_findings: The agent's own previous findings
            new_findings: Dictionary of new findings from other agents (agent_id -> findings)
            
        Returns:
            Updated analysis incorporating new information
        """
        # Build comprehensive input
        input_text = context
        if previous_findings:
            input_text += f"\n\nYour Previous Analysis:\n{previous_findings}\n"
        if new_findings:
            input_text += "\n\nNew Findings from Other Specialists:\n"
            for agent_id, findings in new_findings.items():
                input_text += f"\n{agent_id.upper()} Specialist:\n{findings}\n"
        
        input_text += "\n\nBased on this information, provide your updated analysis integrating all specialist input."
        
        return await self.act(input_text)
    
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
             "You are an Internal Medicine physician reviewing findings from a specialist. "
             "Your role is to evaluate if their findings are consistent with the overall clinical picture, "
             "if there are any concerns, or if you need clarification.\n\n"
             "You should raise a question or challenge if:\n"
             "- The findings seem inconsistent with the clinical presentation\n"
             "- Important clinical data appears to be overlooked\n"
             "- You need clarification on their interpretation\n"
             "- There are contradictions with other available information\n"
             "- Their recommendations may not fit the broader clinical context\n\n"
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
        """Determine if new findings require a response or update to your analysis
        
        Args:
            new_findings: Dictionary of new findings from other agents
            
        Returns:
            True if a response/update is needed, False otherwise
        """
        if not new_findings:
            return False
        
        # Internal Medicine typically needs to integrate all specialist findings
        # This is a simple heuristic - in practice, the agent will be called
        # by the orchestrator when new findings are available
        return True
