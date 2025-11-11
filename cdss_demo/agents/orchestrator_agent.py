from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from agents.base_agent import BaseAgent
from llm import llm


class AgentDecision(BaseModel):
    """Structured output for orchestrator's decision on which agents to call"""
    call_laboratory: bool = Field(description="Whether to consult the Laboratory agent")
    call_cardiology: bool = Field(description="Whether to consult the Cardiology agent")
    reasoning: str = Field(description="Brief reasoning for the decision")


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that coordinates clinical decision support workflow"""
    
    def __init__(self, agent_id: str = "OrchestratorAgent"):
        super().__init__(agent_id)
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert clinical decision support orchestrator in a multi-agent healthcare system. "
             "Your role is to coordinate specialized agents (Cardiology, Laboratory) to provide comprehensive "
             "clinical assessments. You analyze patient cases, delegate tasks to appropriate specialists, "
             "synthesize their findings, and provide final clinical recommendations.\n\n"
             "IMPORTANT: Use Chain of Thought reasoning. Show your thinking process step-by-step:\n"
             "1. First, identify and analyze the key clinical information\n"
             "2. Break down the problem into components\n"
             "3. Consider what specialist expertise is needed\n"
             "4. Think through the clinical questions that need answering\n"
             "5. Synthesize findings systematically\n"
             "6. Formulate recommendations based on your reasoning\n\n"
             "Always show your reasoning process explicitly. Use phrases like:\n"
             "- 'Let me analyze this step by step...'\n"
             "- 'First, I need to consider...'\n"
             "- 'This suggests that...'\n"
             "- 'Therefore, I conclude...'\n\n"
             "Guidelines:\n"
             "- Analyze the clinical case comprehensively\n"
             "- Identify which specialist agents need to be consulted\n"
             "- Coordinate information gathering from multiple sources\n"
             "- Synthesize findings into coherent clinical recommendations\n"
             "- Consider evidence-based medicine and clinical guidelines\n"
             "- Prioritize patient safety and appropriate care pathways\n"
             "- Provide clear, actionable recommendations\n\n"
             "Your reasoning should be thorough, evidence-based, and clinically sound. "
             "Always show your thought process explicitly."),
            ("user", "{input}")
        ])
        self.decision_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert clinical decision support orchestrator. Your task is to analyze a clinical case "
             "and determine which specialist agents should be consulted.\n\n"
             "Available agents:\n"
             "- Laboratory Agent: Consult when the case involves laboratory results, abnormal lab values, "
             "or when lab interpretation is needed for diagnosis.\n"
             "- Cardiology Agent: Consult when the case involves cardiac symptoms, cardiac risk factors, "
             "cardiac biomarkers, ECG findings, or cardiovascular concerns.\n\n"
             "Analyze the case and decide which agents are needed. Only call agents that are relevant to the case. "
             "If a case has no lab results or cardiac concerns, you may choose not to call those agents.\n\n"
             "Provide your decision in the structured format with reasoning."),
            ("user", "Clinical Case:\n{case_text}\n\n"
             "Analyze this case and determine which specialist agents should be consulted. "
             "Consider:\n"
             "- Are there laboratory results that need interpretation?\n"
             "- Are there cardiac symptoms, risk factors, or cardiac biomarkers?\n"
             "- What clinical questions need to be answered?\n\n"
             "Respond with your decision on which agents to call.")
        ])
    
    async def analyze_and_decide(self, case_text: str) -> AgentDecision:
        """Analyze case and decide which agents to call using structured output
        
        Args:
            case_text: The clinical case text to analyze
            
        Returns:
            AgentDecision with call_laboratory, call_cardiology, and reasoning
        """
        # Use structured output to get the decision
        structured_llm = self.llm.with_structured_output(AgentDecision)
        chain = self.decision_prompt | structured_llm
        decision = await chain.ainvoke({"case_text": case_text})
        return decision
    
    async def act(self, input: str) -> str:
        """Process clinical case and coordinate agent workflow"""
        chain = self.prompt | self.llm
        response = await chain.ainvoke({"input": input})
        return response.content
    
    async def act_stream(self, input: str) -> AsyncIterator[str]:
        """Stream tokens as they are generated by the LLM
        
        Args:
            input: Input text for the agent
            
        Yields:
            Tokens as strings as they are generated
        """
        chain = self.prompt | self.llm
        try:
            async for chunk in chain.astream({"input": input}):
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

