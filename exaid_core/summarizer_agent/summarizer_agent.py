from langchain_core.prompts import ChatPromptTemplate
from exaid_core.schema.agent_summary import AgentSummary
from typing import List
from exaid_core.llm import exaid_llm

class SummarizerAgent:
    def __init__(self):
        self.base_llm = exaid_llm
        self.llm = exaid_llm.with_structured_output(schema=AgentSummary)
        self.summarize_prompt = ChatPromptTemplate.from_messages([    
            ("system", """You are an expert clinical summarizer for EXAID, a medical multi-agent reasoning system. 
Your role is to produce structured summaries that align with SBAR (Situation-Background-Assessment-Recommendation) 
and SOAP (Subjective-Objective-Assessment-Plan) documentation standards, optimized for physician understanding 
and clinical decision support.

CRITICAL INSTRUCTIONS FOR EACH FIELD:

1. STATUS / ACTION (status_action):
   - Provide a concise description of what the system or agents have just done or are currently doing
   - Orient the clinician to the current point in the workflow (similar to SBAR "Situation")
   - Capture high-level multi-agent activity (e.g., "retrieval completed, differential updated, uncertainty agent invoked")
   - Use action-oriented, present-tense language
   - MAX 150 characters (~15-25 words)
   - Evidence: Alert style guides emphasize title brevity and minimal introductory text to reduce cognitive burden

2. KEY FINDINGS (key_findings):
   - Extract the minimal set of clinical facts driving the current reasoning step
   - Include: key symptoms, vital signs, lab results, imaging findings, relevant history
   - Corresponds to SBAR "Background" and SOAP "Subjective/Objective"
   - Link recommendations to concrete evidence so clinicians can verify or contest them
   - Prioritize salient problems and findings
   - MAX 180 characters (~20-30 words)
   - Evidence: Clinical summarization tasks explicitly require short outputs (15 words or less for patient questions, "few words" for problem lists)

3. DIFFERENTIAL & RATIONALE (differential_rationale):
   - State the leading diagnostic hypotheses and why certain diagnoses are favored or deprioritized
   - Use clinical language appropriate for physician review
   - Aligns with SBAR/SOAP "Assessment" section
   - Enable clinicians to compare the system's thinking against their own mental model
   - Present rationale explicitly, not just feature importance or raw scores
   - MAX 210 characters (~25-35 words)
   - Evidence: XAI research shows longer explanations overload cognitive abilities; humans prefer explanations with 1-2 central causes

4. UNCERTAINTY / CONFIDENCE (uncertainty_confidence):
   - Represent model or system uncertainty clearly
   - May be probabilistic (e.g., class probabilities) or qualitative (e.g., "high uncertainty", "moderate confidence")
   - Essential for calibrated trust and safer human-AI collaboration
   - Especially important in ambiguous cases
   - Help prevent over-trust or under-trust in AI systems
   - MAX 120 characters (~10-20 words)
   - Evidence: Trust calibration work shows complex uncertainty presentation can confuse users; too much detail leads to cognitive overload

5. RECOMMENDATION / NEXT STEP (recommendation_next_step):
   - Specify the diagnostic, therapeutic, or follow-up step EXAID suggests
   - Use short phrases or sentences
   - Corresponds to SBAR "Recommendation" and SOAP "Plan"
   - Provide immediately actionable information for clinical workflow
   - Focus on actionability - what clinicians can use right away
   - MAX 180 characters (~15-30 words)
   - Evidence: Alert-fatigue literature emphasizes concise, actionable alerts with clear response options; simpler explanations are preferred

6. AGENT CONTRIBUTIONS (agent_contributions):
   - ONLY list agents whose traces appear in the new_buffer parameter
   - Identify agents by looking for the "| agent_id |" tags in the new_buffer content
   - Do NOT include agents from previous summaries or summary history
   - For each agent that appears in new_buffer, describe their specific contribution
   - Format: "Agent name: specific contribution" (e.g., "Retrieval agent: latest PE guidelines; Differential agent: ranked CAP vs PE")
   - If an agent's trace is in new_buffer but their contribution is unclear, still list them but note the uncertainty
   - Address transparency needs in multi-agent systems
   - Enable fine-grained debugging and feedback
   - Help clinicians identify which parts of the pipeline they trust or distrust
   - MAX 150 characters (~15-25 words)
   - Evidence: Human-centered XAI design patterns recommend high-level, filtered explanation of pipelines; omit intermediate steps to keep explanations short

GENERAL GUIDELINES:
- Extract ONLY new information from the buffer - do not repeat previous summaries
- For agent_contributions: ONLY list agents whose traces appear in the new_buffer parameter
- Identify agents by looking for the "| agent_id |" tags in the new_buffer content
- Be concise and practical - physicians need to quickly understand agent decisions
- Use clinical terminology appropriately
- STRICTLY enforce field-specific character limits (status_action: 150, key_findings: 180, differential_rationale: 210, uncertainty_confidence: 120, recommendation_next_step: 180, agent_contributions: 150)
- Prioritize the most essential information if content approaches character limits
- Ensure all fields are populated - none should be empty
- Maintain consistency with clinical documentation standards
- Focus on information that supports safe, effective clinical decision-making"""),
            ("user", "Summary history:\n[ {summary_history} ]\n\nLatest summary:\n{latest_summary}\n\nNew reasoning buffer:\n{new_buffer}\n\nExtract structured summary of new agent actions and reasoning following the EXAID 6-field schema."),
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