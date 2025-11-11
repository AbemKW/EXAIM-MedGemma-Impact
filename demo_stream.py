import asyncio
from exaid import EXAID

def format_summary_display(summary) -> str:
    """Format an AgentSummary for clean console display optimized for physician review."""
    agents_str = ", ".join(summary.agents)
    lines = [
        f"┌─ Agents: {agents_str}",
        f"├─ Action: {summary.action}",
        f"├─ Reasoning: {summary.reasoning}",
    ]
    if summary.findings:
        lines.append(f"├─ Findings: {summary.findings}")
    if summary.next_steps:
        lines.append(f"└─ Next Steps: {summary.next_steps}")
    else:
        # Replace last ├ with └ if no next_steps
        lines[-1] = lines[-1].replace("├─", "└─")
    return "\n".join(lines)

async def token_stream_generator(text: str, delay: float = 0.01):
    """Simulates a token stream by yielding characters one at a time with a small delay."""
    for char in text:
        yield char
        await asyncio.sleep(delay)  # Small delay to simulate real streaming

async def word_stream_generator(text: str, delay: float = 0.05):
    """Simulates a token stream by yielding words one at a time with a small delay."""
    words = text.split()
    for i, word in enumerate(words):
        # Add space before word (except first word)
        if i > 0:
            yield " "
        yield word
        await asyncio.sleep(delay)

async def main():
    exaid = EXAID()
    
    print("="*70)
    print("EXAID Streaming Token Demo")
    print("="*70)
    print("\nThis demo simulates streamed tokens from a long reasoning stream.")
    print("Tokens are accumulated until complete sentences/paragraphs are detected,")
    print("then appended to the trace buffer incrementally.\n")
    
    # Test 1: Character-by-character streaming with multiple sentences
    print("\n" + "="*70)
    print("Test 1: Character-by-character streaming (ReasoningAgent)")
    print("="*70)
    
    reasoning_text = (
        "I need to analyze this patient's symptoms carefully. "
        "The combination of fever, night sweats, and weight loss suggests several possibilities. "
        "Given the patient's history of rheumatoid arthritis and methotrexate use, "
        "I should consider both infectious and autoimmune causes. "
        "The elevated inflammatory markers point toward a systemic process."
    )
    
    print(f"\nStreaming text ({len(reasoning_text)} characters):")
    print(f'"{reasoning_text[:80]}..."\n')
    
    summary = await exaid.received_streamed_tokens(
        "ReasoningAgent",
        token_stream_generator(reasoning_text, delay=0.005)
    )
    
    if summary:
        print("\n" + "="*70)
        print("Summary Generated from Stream")
        print("="*70)
        print(format_summary_display(summary))
        print()
    
    # Test 2: Word-by-word streaming with paragraphs
    print("\n" + "="*70)
    print("Test 2: Word-by-word streaming with paragraphs (ClinicalReasoningAgent)")
    print("="*70)
    
    paragraph_text = (
        "The patient presents with constitutional symptoms that warrant a systematic approach. "
        "B symptoms (fever, night sweats, weight loss) are classic markers of lymphoproliferative disorders. "
        "However, we must also consider infectious etiologies, especially given immunosuppression.\n\n"
        "Initial laboratory workup reveals anemia and elevated inflammatory markers. "
        "The normocytic anemia suggests chronic disease rather than iron deficiency. "
        "Elevated ESR and CRP indicate active inflammation, but are non-specific findings.\n\n"
        "Imaging studies show mediastinal lymphadenopathy, which significantly narrows the differential. "
        "The location and size of the nodes are concerning for lymphoma, particularly in this age group."
    )
    
    print(f"\nStreaming text with paragraphs ({len(paragraph_text.split())} words):")
    print(f'"{paragraph_text[:100]}..."\n')
    
    summary = await exaid.received_streamed_tokens(
        "ClinicalReasoningAgent",
        word_stream_generator(paragraph_text, delay=0.03)
    )
    
    if summary:
        print("\n" + "="*70)
        print("Summary Generated from Stream")
        print("="*70)
        print(format_summary_display(summary))
        print()
    
    # Test 3: Long reasoning stream with incomplete sentence at end
    print("\n" + "="*70)
    print("Test 3: Long reasoning stream with incomplete sentence (DiagnosticAgent)")
    print("="*70)
    
    long_reasoning = (
        "After reviewing all available data, I believe we need to proceed with tissue diagnosis. "
        "The imaging findings are highly suggestive, but histopathology remains the gold standard. "
        "I recommend scheduling a CT-guided biopsy of the largest mediastinal node. "
        "This will provide definitive diagnosis and guide treatment planning. "
        "We should also consider staging studies including PET-CT once diagnosis is confirmed. "
        "The patient's overall prognosis depends on accurate staging and appropriate treatment selection."
    )
    
    # Add incomplete sentence at the end
    incomplete_end = " Additionally, we need to evaluate"
    
    print(f"\nStreaming long reasoning text with incomplete ending:")
    print(f'"{long_reasoning[:80]}..."\n')
    print("Note: Stream ends with incomplete sentence (should be appended immediately)\n")
    
    summary = await exaid.received_streamed_tokens(
        "DiagnosticAgent",
        token_stream_generator(long_reasoning + incomplete_end, delay=0.005)
    )
    
    if summary:
        print("\n" + "="*70)
        print("Summary Generated from Stream")
        print("="*70)
        print(format_summary_display(summary))
        print()
    
    # Test 4: Multiple agents streaming concurrently (simulated sequentially)
    print("\n" + "="*70)
    print("Test 4: Multiple agents with different reasoning streams")
    print("="*70)
    
    agent_streams = [
        ("PathologyAgent", "The biopsy specimen shows characteristic Reed-Sternberg cells. "
         "Immunohistochemistry confirms CD30 and CD15 positivity. "
         "This is consistent with classical Hodgkin lymphoma, nodular sclerosis subtype."),
        
        ("TreatmentAgent", "Based on the diagnosis, I recommend ABVD chemotherapy regimen. "
         "This is the standard first-line treatment for Stage II Hodgkin lymphoma. "
         "We should start with 4-6 cycles and assess response with interim imaging."),
        
        ("SupportAgent", "Patient counseling is essential given the diagnosis. "
         "We need to discuss treatment options, side effects, and prognosis. "
         "Fertility preservation should be addressed before starting chemotherapy.")
    ]
    
    for agent_id, text in agent_streams:
        print(f"\n{agent_id} streaming...")
        summary = await exaid.received_streamed_tokens(
            agent_id,
            word_stream_generator(text, delay=0.02)
        )
        
        if summary:
            print(f"\n{agent_id} Summary:")
            print(format_summary_display(summary))
            print()
    
    # Final summary
    print("\n" + "="*70)
    print("Final Summary Statistics")
    print("="*70)
    all_summaries = exaid.get_all_summaries()
    print(f"Total summaries generated: {len(all_summaries)}")
    
    for i, summary in enumerate(all_summaries, 1):
        print(f"\nSummary {i}:")
        print(f"  Agents: {', '.join(summary.agents)}")
        print(f"  Action: {summary.action}")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())


