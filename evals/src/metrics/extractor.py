"""Concept extractor wrapper with caching."""

from typing import Dict, Set


class ConceptExtractorWrapper:
    """
    Wrapper for concept extraction with caching.

    Handles cases where scispaCy/UMLS linker may not be available.
    """

    def __init__(self, config: dict, no_linking: bool = False):
        self.config = config
        self.cache: Dict[int, Set[str]] = {}
        self._extractor = None
        self.no_linking = no_linking

        try:
            from ..extraction.concept_extractor import ConceptExtractor

            self._extractor = ConceptExtractor(config, no_linking=no_linking)
        except Exception as e:
            print(f"WARNING: Concept extractor unavailable: {e}")
            print("Using stub extraction (empty sets)")

    def extract(self, text: str) -> Set[str]:
        """Extract concepts with caching."""
        if not text:
            return set()

        cache_key = hash(text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self._extractor:
            concepts = self._extractor.extract(text)
        else:
            concepts = set()

        self.cache[cache_key] = concepts
        return concepts

    def get_version_info(self) -> dict:
        """Get version info for logging."""
        if self._extractor:
            return self._extractor.get_version_info()
        return {"mode": "stub", "reason": "extractor_unavailable"}
