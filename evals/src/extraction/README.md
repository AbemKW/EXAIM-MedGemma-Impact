# Extraction Module

Concept extraction logic based on scispaCy and UMLS linking.

## Files
- `concept_extractor.py`: Core `ConceptExtractor` implementation, stoplist handling, and provenance info.
- `provenance.py`: Extractor version/provenance helpers for run logging.

## Notes
- Extraction order (filter → sort → topK → stoplist) is invariant and covered by tests.
