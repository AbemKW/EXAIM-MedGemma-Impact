from langchain_core.prompts import ChatPromptTemplate
from exaid_core.schema.agent_summary import AgentSummary
from typing import List
from infra import get_llm, LLMRole
from exaid_core.utils.prompts import get_summarizer_system_prompt, get_summarizer_user_prompt

class SummarizerAgent:
    def __init__(self):
        self.base_llm = get_llm(LLMRole.SUMMARIZER)
        self.llm = self.base_llm.with_structured_output(schema=AgentSummary)
        self.summarize_prompt = ChatPromptTemplate.from_messages([    
            ("system", get_summarizer_system_prompt()),
            ("user", get_summarizer_user_prompt()),
        ])

    async def summarize(self, summary_history: List[str], latest_summary: str, new_buffer: str) -> AgentSummary:
        summarize_chain = self.summarize_prompt | self.llm
        
        try:
            summary = await summarize_chain.ainvoke({
                "summary_history": ",\n".join(summary_history),
                "latest_summary": latest_summary,
                "new_buffer": new_buffer
            })
            return summary
        except Exception as e:
            # If validation fails, try to get raw output and truncate
            # Check for validation errors more broadly:
            # 1. Pydantic ValidationError
            # 2. Common validation error message patterns
            # 3. Any error that might be related to structured output validation
            from pydantic import ValidationError
            
            is_validation_error = (
                isinstance(e, ValidationError) or
                "length must be <=" in str(e).lower() or
                "tool call validation failed" in str(e).lower() or
                "validation" in str(e).lower() or
                "constraint" in str(e).lower() or
                "max_length" in str(e).lower() or
                "field required" in str(e).lower() or
                "value error" in str(e).lower()
            )
            
            if is_validation_error:
                # Fallback: get raw response and manually truncate
                try:
                    # Use regular LLM without structured output
                    fallback_chain = self.summarize_prompt | self.base_llm
                    raw_response = await fallback_chain.ainvoke({
                        "summary_history": ",\n".join(summary_history),
                        "latest_summary": latest_summary,
                        "new_buffer": new_buffer
                    })
                    
                    # Parse JSON from response if possible
                    import json
                    import re
                    content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                    
                    # Try to extract JSON from the response using balanced brace matching
                    def find_json_object(text):
                        """Find the first complete JSON object with balanced braces."""
                        start_idx = text.find('{')
                        if start_idx == -1:
                            return None
                        
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        
                        for i in range(start_idx, len(text)):
                            char = text[i]
                            
                            if escape_next:
                                escape_next = False
                                continue
                            
                            if char == '\\':
                                escape_next = True
                                continue
                            
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        # Found balanced JSON object
                                        return text[start_idx:i+1]
                        
                        return None
                    
                    json_str = find_json_object(content)
                    data = None
                    
                    if json_str:
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, try non-greedy regex as fallback
                            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                            if json_match:
                                try:
                                    data = json.loads(json_match.group())
                                except json.JSONDecodeError:
                                    # If fallback JSON parsing fails, proceed to create a minimal summary below.
                                    pass
                    
                    if not data:
                        # If no valid JSON found, create a minimal summary
                        data = {
                            "status_action": content[:147] + "..." if len(content) > 150 else content,
                            "key_findings": content[:177] + "..." if len(content) > 180 else content,
                            "differential_rationale": content[:207] + "..." if len(content) > 210 else content,
                            "uncertainty_confidence": "Moderate confidence",
                            "recommendation_next_step": "Continue processing",
                            "agent_contributions": "Multiple agents"
                        }
                    
                    # Truncate fields to meet limits
                    field_limits = {
                        'status_action': 150,
                        'key_findings': 180,
                        'differential_rationale': 210,
                        'uncertainty_confidence': 120,
                        'recommendation_next_step': 180,
                        'agent_contributions': 150
                    }
                    
                    for field, max_len in field_limits.items():
                        if field in data and len(str(data[field])) > max_len:
                            truncate_len = max_len - 3
                            data[field] = str(data[field])[:truncate_len] + '...'
                    
                    # Create AgentSummary with truncated data
                    return AgentSummary(**data)
                except Exception as fallback_error:
                    print(f"Fallback summarization also failed: {fallback_error}")
                    # Return a minimal valid summary
                    return AgentSummary(
                        status_action="Summary generation encountered validation error",
                        key_findings="Unable to extract findings due to length constraints",
                        differential_rationale="Analysis in progress",
                        uncertainty_confidence="High uncertainty due to processing error",
                        recommendation_next_step="Review raw agent traces",
                        agent_contributions="Multiple agents"
                    )
            else:
                # For non-validation errors (network, API, etc.), re-raise
                # This allows upstream error handling for critical failures
                raise