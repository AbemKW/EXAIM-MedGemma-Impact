#!/usr/bin/env python3
"""
CUI-based concept extraction using scispaCy + UMLS EntityLinker.

Paper hook: "CUIs extracted with filter→sort→topK→stoplist ordering
and uppercase normalization (Section 6.1)"

EXTRACTION ORDER (per entity mention):
    1. Retrieve candidates from ent._.kb_ents
    2. Filter by score >= threshold
    3. Sort descending by score
    4. Take top K candidates
    5. Apply stop_cuis filter with normalized matching

CUI NORMALIZATION:
    - CUIs stored in canonical uppercase form (C1234567)
    - Stoplist matching is case-insensitive
    - Stoplist file may contain any case; normalized before use

Dependencies:
    - spacy
    - scispacy
    - scispacy UMLS linker (optional, for CUI linking)
"""

import warnings
from pathlib import Path
from typing import Optional, Set


class ConceptExtractor:
    """
    Extract UMLS CUIs with exact filter→sort→topK→stoplist ordering.
    
    Paper hook: "Concept extraction uses scispaCy with UMLS EntityLinker,
    applying filter→sort→topK→stoplist ordering for deterministic results (Section 6.1)"
    """
    
    def __init__(self, config: dict, no_linking: bool = False):
        """
        Initialize extractor.
        
        Args:
            config: Configuration dict with:
                - scispacy_model: str (e.g., "en_core_sci_sm")
                - cui_score_threshold: float (default 0.7)
                - max_k: int (default 10)
                - min_entity_len: int (default 3)
                - stop_entities_file: Optional[str]
                - stop_cuis_file: Optional[str]
                - linker_resolve_abbreviations: bool (default True)
            no_linking: If True, use NER-only fallback (no UMLS linking)
        """
        import spacy
        
        self.config = config
        self.no_linking = no_linking
        self.min_entity_len = config.get("min_entity_len", 3)
        self.cui_score_threshold = config.get("cui_score_threshold", 0.7)
        self.max_k = config.get("max_k", 10)
        
        # Load spaCy model
        model_name = config.get("scispacy_model", "en_core_sci_sm")
        try:
            self.nlp = spacy.load(model_name)
        except OSError as e:
            raise RuntimeError(
                f"Failed to load scispaCy model '{model_name}'. "
                f"Install with: pip install scispacy && "
                f"pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{model_name}-0.5.4.tar.gz"
            ) from e
        
        # Add UMLS linker if not in fallback mode
        self.linker_config = None
        if not no_linking:
            try:
                # Check if linker already in pipeline
                if "scispacy_linker" not in self.nlp.pipe_names:
                    self.linker_config = {
                        "resolve_abbreviations": config.get("linker_resolve_abbreviations", True),
                        "linker_name": config.get("linker_name", "umls"),
                        "max_entities_per_mention": config.get("linker_max_entities_per_mention", self.max_k),
                        "threshold": config.get("linker_threshold", self.cui_score_threshold)
                    }
                    self.nlp.add_pipe("scispacy_linker", config=self.linker_config)
                
                self.concept_representation = "cui"
            except Exception as e:
                warnings.warn(
                    f"Failed to add UMLS linker: {e}. "
                    "Falling back to surface-only extraction."
                )
                self.no_linking = True
                self.concept_representation = "surface"
        else:
            self.concept_representation = "surface"
        
        # Load stoplists with normalization
        self.stop_entities = self._load_stoplist_normalized(
            config.get("stop_entities_file"),
            normalize_fn=str.lower
        )
        self.stop_cuis = self._load_stoplist_normalized(
            config.get("stop_cuis_file"),
            normalize_fn=str.upper  # CUIs normalized to uppercase
        )
        
        # Version tracking
        self.versions = self._get_versions()
    
    def _load_stoplist_normalized(
        self,
        filepath: Optional[str],
        normalize_fn
    ) -> Set[str]:
        """
        Load stoplist with normalization function.
        
        Args:
            filepath: Path to stoplist file
            normalize_fn: Function to normalize entries
            
        Returns:
            Set of normalized stoplist entries
        """
        if not filepath:
            return set()
        
        path = Path(filepath)
        if not path.exists():
            warnings.warn(f"Stoplist file not found: {filepath}")
            return set()
        
        items = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    items.add(normalize_fn(line))
        return items
    
    def _get_versions(self) -> dict:
        """Get version info for provenance tracking."""
        import spacy
        
        versions = {
            "spacy_version": spacy.__version__,
            "scispacy_model": self.config.get("scispacy_model", "en_core_sci_sm"),
            "concept_representation": self.concept_representation,
            "cui_normalization": "uppercase"
        }
        
        try:
            import scispacy
            versions["scispacy_version"] = scispacy.__version__
        except ImportError:
            versions["scispacy_version"] = "unknown"
        
        if not self.no_linking:
            versions["linker_name"] = self.config.get("linker_name", "umls")
            versions["linker_kb_version"] = self._get_linker_kb_version()
            versions["linker_resolve_abbreviations"] = self.config.get(
                "linker_resolve_abbreviations", True
            )
        else:
            versions["linker_name"] = "none"
            versions["linker_kb_version"] = None
            versions["linker_resolve_abbreviations"] = False
        
        return versions
    
    def _get_linker_kb_version(self) -> Optional[str]:
        """
        Get UMLS KB version from linker if available.
        
        Returns:
            KB version string or default
        """
        try:
            if "scispacy_linker" in self.nlp.pipe_names:
                linker = self.nlp.get_pipe("scispacy_linker")
                # Try different attribute paths
                if hasattr(linker, "kb"):
                    kb = linker.kb
                    if hasattr(kb, "version"):
                        return kb.version
                    if hasattr(kb, "_version"):
                        return kb._version
                # Check linker attributes
                if hasattr(linker, "linker_version"):
                    return linker.linker_version
            return "2023AB"  # Default if not extractable
        except Exception:
            return "2023AB"
    
    def extract(self, text: str) -> Set[str]:
        """
        Extract UMLS CUIs with EXACT ordering.
        
        ORDER: filter → sort → topK → stoplist
        
        Paper hook: "CUI candidates score-filtered (>=0.7), sorted DESC,
        top-K selected, then stoplist filtered (Section 6.1)"
        
        Args:
            text: Input text to extract concepts from
            
        Returns:
            Set of extracted CUIs (uppercase normalized)
        """
        if self.no_linking:
            return self._extract_surface_as_fallback(text)
        
        if not text or not text.strip():
            return set()
        
        doc = self.nlp(text)
        cuis = set()
        
        for ent in doc.ents:
            # Length filter
            entity_text = ent.text.strip()
            if len(entity_text) < self.min_entity_len:
                continue
            
            # Stage 1: Surface filter (reduces false positives and semantic noise)
            if entity_text.lower() in self.stop_entities:
                continue
            
            # Get linker candidates via canonical API
            if not hasattr(ent._, 'kb_ents') or not ent._.kb_ents:
                continue
            
            # EXACT ORDER: filter → sort → topK → stoplist
            
            # Step 1: Retrieve all candidates
            candidates = list(ent._.kb_ents)
            
            # Step 2: Filter by score threshold
            valid_candidates = [
                (cui, score) for cui, score in candidates
                if score >= self.cui_score_threshold
            ]
            
            # Step 3: Sort descending by score
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Step 4: Take top K
            top_k_candidates = valid_candidates[:self.max_k]
            
            # Step 5: Apply stop_cuis filter with normalization
            for cui, score in top_k_candidates:
                cui_normalized = cui.upper()  # Canonical uppercase form
                if cui_normalized not in self.stop_cuis:
                    cuis.add(cui_normalized)
        
        return cuis
    
    def extract_surface(self, text: str) -> Set[str]:
        """
        Extract surface strings (NER-only, no linking).
        
        Used for Stage 1 stoplist generation and diagnostic baseline.
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted surface strings (lowercase normalized)
        """
        if not text or not text.strip():
            return set()
        
        doc = self.nlp(text)
        surfaces = set()
        
        for ent in doc.ents:
            normalized = ent.text.lower().strip()
            if len(normalized) >= self.min_entity_len:
                if normalized not in self.stop_entities:
                    surfaces.add(normalized)
        
        return surfaces
    
    def _extract_surface_as_fallback(self, text: str) -> Set[str]:
        """Fallback when linker unavailable."""
        return self.extract_surface(text)
    
    def get_version_info(self) -> dict:
        """
        Return version info for run_meta logging.
        
        Paper hook: "Extractor provenance includes library versions,
        linker KB version, and configuration for reproducibility (Section 6.1)"
        
        Returns:
            Dict with version and configuration info
        """
        return {
            **self.versions,
            "cui_score_threshold": self.cui_score_threshold,
            "max_k": self.max_k,
            "min_entity_len": self.min_entity_len,
            "stop_entities_count": len(self.stop_entities),
            "stop_cuis_count": len(self.stop_cuis)
        }


def create_extractor_from_config(
    config_path: Path,
    no_linking: bool = False
) -> ConceptExtractor:
    """
    Create ConceptExtractor from YAML config file.
    
    Args:
        config_path: Path to extractor.yaml
        no_linking: Force NER-only mode
        
    Returns:
        Configured ConceptExtractor instance
    """
    import yaml
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    extractor_config = config.get("concept_extractor", {})
    return ConceptExtractor(extractor_config, no_linking=no_linking)


if __name__ == "__main__":
    # Simple test when run directly
    import sys
    
    print("ConceptExtractor Test")
    print("=" * 40)
    
    # Test config
    test_config = {
        "scispacy_model": "en_core_sci_sm",
        "cui_score_threshold": 0.7,
        "max_k": 10,
        "min_entity_len": 3,
        "linker_resolve_abbreviations": True,
    }
    
    # Try with linking first, fall back to surface
    try:
        print("Attempting CUI extraction with UMLS linker...")
        extractor = ConceptExtractor(test_config, no_linking=False)
    except Exception as e:
        print(f"Linker not available ({e}), using surface extraction")
        extractor = ConceptExtractor(test_config, no_linking=True)
    
    # Test text
    test_text = """
    The patient presents with acute myocardial infarction. 
    Troponin levels are elevated at 2.5 ng/mL. 
    ECG shows ST-segment elevation in leads V1-V4.
    Recommend urgent cardiac catheterization.
    """
    
    print(f"\nExtractor mode: {extractor.concept_representation}")
    print(f"Version info: {extractor.get_version_info()}")
    print()
    
    concepts = extractor.extract(test_text)
    print(f"Extracted {len(concepts)} concepts:")
    for c in sorted(concepts):
        print(f"  - {c}")




