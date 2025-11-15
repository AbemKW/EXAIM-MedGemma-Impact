from pydantic import BaseModel, Field, model_validator

class AgentSummary(BaseModel):
    """Structured summary for medical multi-agent reasoning, optimized for physician understanding.
    
    Aligned with SBAR/SOAP documentation standards and XAI-CDSS principles for clinical decision support.
    """
    status_action: str = Field(
        max_length=300,
        description="Concise description of what the system or agents have just done or are currently doing in the reasoning process. "
        "Plays a role similar to SBAR 'Situation', orienting the clinician to the current point in the workflow. "
        "Captures high-level multi-agent activity (e.g., 'retrieval completed, differential updated, uncertainty agent invoked')."
    )
    key_findings: str = Field(
        max_length=300,
        description="The minimal set of clinical facts that are driving the current reasoning step, such as key symptoms, "
        "vital signs, lab results, imaging findings, or relevant history. Corresponds to SBAR 'Background' and SOAP 'Subjective/Objective'. "
        "Must link recommendations to concrete evidence so clinicians can verify or contest them."
    )
    differential_rationale: str = Field(
        max_length=300,
        description="A brief statement of the leading diagnostic hypotheses and why certain diagnoses are favored or deprioritized, "
        "expressed in clinical language. Aligns with the 'Assessment' section in SBAR and SOAP, which captures clinical interpretation. "
        "Gives clinicians a way to compare the system's thinking against their own mental model of the case."
    )
    uncertainty_confidence: str = Field(
        max_length=300,
        description="A concise representation of model or system uncertainty, which may be probabilistic (e.g., class probabilities) "
        "or qualitative (e.g., 'high uncertainty', 'moderate confidence'). Essential for calibrated trust and safer human-AI collaboration, "
        "especially in ambiguous cases."
    )
    recommendation_next_step: str = Field(
        max_length=300,
        description="The specific diagnostic, therapeutic, or follow-up step that EXAID suggests at this point, usually a short phrase or sentence. "
        "Corresponds to SBAR 'Recommendation' and SOAP 'Plan'. Provides clinicians with immediately actionable information in their workflow."
    )
    agent_contributions: str = Field(
        max_length=300,
        description="A short list of which agents contributed to this step and how their outputs were used "
        "(e.g., 'Retrieval agent: latest PE guidelines; Differential agent: ranked CAP vs PE; Uncertainty agent: confidence estimates'). "
        "Addresses transparency in multi-agent systems, enabling fine-grained debugging and feedback."
    )
    
    @model_validator(mode='before')
    @classmethod
    def truncate_fields(cls, data):
        """Truncate fields to meet length constraints if they exceed limits."""
        if isinstance(data, dict):
            max_len = 297  # Leave room for '...'
            for field in ['status_action', 'key_findings', 'differential_rationale', 
                         'uncertainty_confidence', 'recommendation_next_step', 'agent_contributions']:
                if field in data and len(data.get(field, '')) > 300:
                    data[field] = data[field][:max_len] + '...'
        return data