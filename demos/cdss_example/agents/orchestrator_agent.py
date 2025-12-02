from typing import AsyncIterator, List
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from exaid_core.llm import mas_llm
from exaid_core.callbacks.agent_streaming_callback import AgentStreamingCallback


class OrchestratorAgent(DemoBaseAgent):
    """Orchestrator agent that maintains running summary and coordinates specialist workflow
    
    Responsibilities:
    - Compress specialist outputs into running_summary
    - Decide next specialist to call
    - Generate task instructions for specialists
    - Synthesize final recommendations
    """
    
    def __init__(self, agent_id: str = "OrchestratorAgent"):
        super().__init__(agent_id)
        self.llm = mas_llm
        
        # System prompt for orchestrator (used in all tasks)
        self.system_prompt = (
            "You are the Orchestrator in a clinical decision support multi-agent system. "
            "You coordinate Laboratory, Cardiology, Internal Medicine, and Radiology specialists. "
            "Your role is to:\n"
            "- Maintain a concise running summary of clinical findings\n"
            "- Decide which specialist should contribute next\n"
            "- Generate specific task instructions for specialists\n"
            "- Synthesize final recommendations\n\n"
            "You work with compressed context (running_summary) and raw specialist outputs. "
            "Keep the running_summary focused on key clinical findings, differential diagnoses, "
            "and actionable information. Discard redundant details."
        )
    
    async def compress_to_summary(
        self, 
        previous_summary: str, 
        new_raw_output: str, 
        recent_agent: str
    ) -> str:
        """Compress previous summary + new specialist output into updated summary
        
        Streams tokens to EXAID for UI transparency while maintaining bounded context.
        
        Args:
            previous_summary: The current running summary
            new_raw_output: Raw output from the most recent specialist
            recent_agent: Name of the specialist who produced new_raw_output
            
        Returns:
            Updated compressed summary string
        """
        if not previous_summary:
            prompt_text = (
                f"A specialist ({recent_agent}) has provided analysis of a clinical case.\n\n"
                f"Specialist Output:\n{new_raw_output}\n\n"
                f"Create a concise summary capturing:\n"
                f"- Key clinical findings\n"
                f"- Differential diagnoses\n"
                f"- Recommended tests or interventions\n"
                f"- Critical concerns or urgent issues\n\n"
                f"Keep it focused and actionable."
            )
        else:
            prompt_text = (
                f"Previous Summary:\n{previous_summary}\n\n"
                f"New Findings from {recent_agent.upper()}:\n{new_raw_output}\n\n"
                f"Generate an updated concise summary that:\n"
                f"- Integrates the new findings\n"
                f"- Maintains key information from previous summary\n"
                f"- Removes redundant or superseded information\n"
                f"- Keeps focus on differential diagnosis and clinical decision-making\n"
                f"- Stays within ~300-500 tokens\n\n"
                f"Provide only the updated summary."
            )
        
        # Build prompt and stream
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", prompt_text)
        ])
        
        chain = prompt | self.llm
        
        # Collect tokens while streaming
        collected = []
        async for chunk in chain.astream({}):
            if hasattr(chunk, 'content'):
                content = chunk.content
                if content:
                    collected.append(content)
        
        return "".join(collected)
    
    async def decide_next_specialist(
        self,
        case_text: str,
        running_summary: str,
        recent_agent: str,
        specialists_called: List[str]
    ) -> str:
        """Decide which specialist should contribute next or if synthesis should begin
        
        Args:
            case_text: Original clinical case
            running_summary: Current compressed summary
            recent_agent: Last specialist who contributed
            specialists_called: List of specialists already called
            
        Returns:
            Specialist name ('laboratory', 'cardiology', 'internal_medicine', 'radiology') 
            or 'synthesis' to end workflow
        """
        available_specialists = ['laboratory', 'cardiology', 'internal_medicine', 'radiology']
        not_called = [s for s in available_specialists if s not in specialists_called]
        
        prompt_text = (
            f"Clinical Case:\n{case_text}\n\n"
            f"Running Summary:\n{running_summary}\n\n"
            f"Recent Contributor: {recent_agent}\n"
            f"Specialists Called: {', '.join(specialists_called) if specialists_called else 'none yet'}\n"
            f"Available Specialists: {', '.join(not_called) if not_called else 'all have contributed'}\n\n"
            f"Decide the next action:\n"
            f"- If critical questions remain that a specific specialist should address, "
            f"respond with ONLY the specialist name: 'laboratory' OR 'cardiology' OR 'internal_medicine' OR 'radiology'\n"
            f"- If enough information has been gathered and a final synthesis should be generated, "
            f"respond with ONLY: 'synthesis'\n\n"
            f"Consider:\n"
            f"- What clinical questions remain unanswered?\n"
            f"- What specialist expertise would help clarify the diagnosis or treatment?\n"
            f"- Have the key specialists for this case contributed?\n"
            f"- Is there sufficient information for a final recommendation?\n\n"
            f"Respond with ONLY ONE WORD: the specialist name or 'synthesis'"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", prompt_text)
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        
        # Extract decision and clean it
        decision = response.content.strip().lower()
        
        # Validate decision
        valid_options = available_specialists + ['synthesis']
        if decision not in valid_options:
            # Try to extract valid option from response
            for option in valid_options:
                if option in decision:
                    decision = option
                    break
            else:
                # Default to synthesis if unclear
                decision = 'synthesis'
        
        return decision
    
    async def generate_task_instruction(
        self,
        case_text: str,
        running_summary: str,
        recent_delta: str,
        recent_agent: str,
        next_specialist: str
    ) -> str:
        """Generate specific task instruction for the next specialist
        
        Args:
            case_text: Original clinical case
            running_summary: Current compressed summary
            recent_delta: Raw output from most recent specialist
            recent_agent: Name of most recent specialist
            next_specialist: Name of specialist who will receive this task
            
        Returns:
            Task instruction string
        """
        prompt_text = (
            f"Clinical Case:\n{case_text}\n\n"
            f"Running Summary:\n{running_summary}\n\n"
            f"Recent Update from {recent_agent.upper()}:\n{recent_delta}\n\n"
            f"You are preparing a task for the {next_specialist.upper()} specialist.\n\n"
            f"Generate a specific, focused task instruction that tells them:\n"
            f"- What aspect of the case they should focus on\n"
            f"- What questions they should address\n"
            f"- What prior findings they should consider or verify\n"
            f"- What their analysis should contribute to the diagnosis/treatment plan\n\n"
            f"Keep it concise (2-4 sentences) and actionable."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", prompt_text)
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        
        return response.content.strip()
    
    async def act_stream(self, input: str) -> AsyncIterator[str]:
        """Stream tokens for synthesis or compression (used by EXAID for monitoring)
        
        Args:
            input: Input text for the agent
            
        Yields:
            Tokens as strings as they are generated
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", input)
        ])
        
        chain = prompt | self.llm
        callback = AgentStreamingCallback(agent_id=self.agent_id)
        
        try:
            async for chunk in chain.astream({}, callbacks=[callback]):
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
            if "No generation chunks were returned" in str(e):
                print(f"[WARNING] Streaming failed for {self.agent_id}, falling back to non-streaming mode")
                # Fallback: use ainvoke instead
                response = await chain.ainvoke({})
                for char in response.content:
                    yield char
            else:
                raise


