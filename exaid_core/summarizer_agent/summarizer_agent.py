from langchain_core.prompts import ChatPromptTemplate
from exaid_core.schema.agent_summary import AgentSummary
from typing import List, Dict, Any
from pydantic import ValidationError
import json
from infra import get_llm, LLMRole
from exaid_core.utils.prompts import get_summarizer_system_prompt, get_summarizer_user_prompt

class SummarizerAgent:
    def __init__(self):
        self.base_llm = get_llm(LLMRole.SUMMARIZER)
        self.llm = self.base_llm.with_structured_output(
                schema=AgentSummary,
                method="json_schema",
                strict=True
        )

        self.summarize_prompt = ChatPromptTemplate.from_messages([    
            ("system", get_summarizer_system_prompt()),
            ("user", get_summarizer_user_prompt()),
        ])
        
        # Field limits for rewrite prompts
        self.field_limits = {
            'status_action': 150,
            'key_findings': 180,
            'differential_rationale': 210,
            'uncertainty_confidence': 120,
            'recommendation_next_step': 180,
            'agent_contributions': 150
        }
    
    def _extract_max_length_violations(self, validation_error: ValidationError) -> Dict[str, int]:
        """Extract fields that violated max_length constraints from ValidationError.
        
        Returns:
            Dict mapping field names to their max_length limits
        """
        violations = {}
        for error in validation_error.errors():
            if error['type'] == 'string_too_long':
                field_path = error.get('loc', ())
                if field_path:
                    field_name = field_path[-1]
                    # Extract max_length from ctx if available
                    ctx = error.get('ctx', {})
                    max_length = ctx.get('max_length')
                    if max_length:
                        violations[field_name] = max_length
                    elif field_name in self.field_limits:
                        violations[field_name] = self.field_limits[field_name]
        return violations
    
    def _create_rewrite_prompt(self, previous_output: Dict[str, Any], violations: Dict[str, int]) -> str:
        """Create a targeted rewrite prompt for fields that exceeded max_length.
        
        Args:
            previous_output: The previous output that failed validation (as dict)
            violations: Dict mapping field names to their max_length limits
            
        Returns:
            Rewrite prompt string
        """
        violation_list = []
        for field, max_len in violations.items():
            current_value = previous_output.get(field, '')
            current_len = len(str(current_value))
            violation_list.append(
                f"- {field}: currently {current_len} characters, must be ≤ {max_len} characters"
            )
        
        violations_text = '\n'.join(violation_list)
        
        prompt = f"""REWRITE REQUEST: Your previous output exceeded character limits. Please shorten ONLY the following fields while preserving their semantic meaning and clinical accuracy:

{violations_text}

Your previous output:
{json.dumps(previous_output, indent=2)}

Instructions:
- Shorten ONLY the fields listed above to meet their character limits
- Preserve all semantic meaning and clinical information
- Keep all other fields exactly as they are
- Return the complete output with shortened fields"""
        
        return prompt

    def _extract_validation_error_from_exception(self, e: Exception) -> tuple[ValidationError | None, dict | None]:
        """Extract ValidationError and parsed JSON from LangChain exception.
        
        LangChain wraps ValidationError in OutputParserException. This method:
        1. Checks if exception is ValidationError directly
        2. Checks if ValidationError is in __cause__ or args
        3. Tries to extract JSON from error message string
        
        Returns:
            Tuple of (ValidationError or None, parsed JSON dict or None)
        """
        # Check if it's a ValidationError directly
        if isinstance(e, ValidationError):
            return e, None
        
        # Check __cause__ (common pattern for wrapped exceptions)
        if hasattr(e, '__cause__') and isinstance(e.__cause__, ValidationError):
            validation_error = e.__cause__
            # Try to extract JSON from error message
            error_str = str(e)
            json_dict = self._extract_json_from_error_message(error_str)
            return validation_error, json_dict
        
        # Check args for ValidationError
        for arg in getattr(e, 'args', []):
            if isinstance(arg, ValidationError):
                error_str = str(e)
                json_dict = self._extract_json_from_error_message(error_str)
                return arg, json_dict
        
        # Try to extract JSON from error message and create ValidationError
        error_str = str(e)
        json_dict = self._extract_json_from_error_message(error_str)
        if json_dict:
            try:
                # Try to create the object to get the real ValidationError
                AgentSummary(**json_dict)
            except ValidationError as ve:
                return ve, json_dict
        
        return None, json_dict if json_dict else None
    
    def _extract_json_from_error_message(self, error_str: str) -> dict | None:
        """Extract JSON dict from error message string.
        
        LangChain error messages often include the JSON that failed validation.
        """
        import re
        # Look for JSON object in the error message
        # Pattern: {"key": "value", ...}
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', error_str)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None

    async def summarize(self, agent_id: str, summary_history: List[str], latest_summary: str, new_buffer: str) -> AgentSummary:
        summarize_chain = self.summarize_prompt | self.llm
        
        try:
            summary = await summarize_chain.ainvoke({
                "agent_id": agent_id,
                "summary_history": ",\n".join(summary_history),
                "latest_summary": latest_summary,
                "new_buffer": new_buffer
            })
            return summary
        except Exception as e:
            # Extract ValidationError from LangChain exception wrapper
            validation_error, previous_output = self._extract_validation_error_from_exception(e)
            
            if validation_error:
                # Check if this is a max_length violation
                violations = self._extract_max_length_violations(validation_error)
                
                if violations:
                    # Rewrite-and-retry loop (max 1 retry)
                    try:
                        # Use extracted JSON if available, otherwise get raw output
                        if previous_output is None:
                            # Get raw output to use in rewrite prompt
                            raw_chain = self.summarize_prompt | self.base_llm
                            raw_response = await raw_chain.ainvoke({
                                "agent_id": agent_id,
                                "summary_history": ",\n".join(summary_history),
                                "latest_summary": latest_summary,
                                "new_buffer": new_buffer
                            })
                            
                            # Extract JSON from raw response
                            content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                            
                            # Try to parse JSON from response
                            try:
                                # Look for JSON object in the response
                                start_idx = content.find('{')
                                if start_idx != -1:
                                    # Find matching closing brace
                                    brace_count = 0
                                    in_string = False
                                    escape_next = False
                                    
                                    for i in range(start_idx, len(content)):
                                        char = content[i]
                                        
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
                                                    json_str = content[start_idx:i+1]
                                                    previous_output = json.loads(json_str)
                                                    break
                            except (json.JSONDecodeError, ValueError):
                                # If we can't parse JSON, we can't create a rewrite prompt
                                # Re-raise the original validation error
                                raise validation_error
                        
                        if previous_output is None:
                            # Couldn't extract previous output, re-raise
                            raise validation_error
                        
                        # Create rewrite prompt
                        rewrite_prompt_text = self._create_rewrite_prompt(previous_output, violations)
                        
                        # Create rewrite prompt template
                        rewrite_prompt = ChatPromptTemplate.from_messages([
                            ("system", get_summarizer_system_prompt()),
                            ("user", rewrite_prompt_text),
                        ])
                        
                        # Retry with rewrite prompt
                        rewrite_chain = rewrite_prompt | self.llm
                        summary = await rewrite_chain.ainvoke({})
                        return summary
                    
                    except Exception as retry_error:
                        # If retry also fails, re-raise the original validation error
                        raise validation_error
                else:
                    # Not a max_length violation, re-raise
                    raise validation_error
            else:
                # Not a ValidationError, re-raise the original exception
                raise