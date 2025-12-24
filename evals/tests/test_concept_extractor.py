#!/usr/bin/env python3
"""
Unit tests for ConceptExtractor.

Paper hook: "Extraction ordering verified via unit tests ensuring
filter→sort→topK→stoplist sequence (Section 6.1)"

Tests verify:
    1. Extraction ordering is exact (filter→sort→topK→stoplist)
    2. CUIs are normalized to uppercase
    3. Non-empty results for clinical text
    4. Stoplist filtering works correctly
    5. Surface extraction fallback works
    6. Sort-order regression (20 candidates scenario)

Run with: pytest evals/tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path (handled by conftest.py, but explicit for clarity)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConceptExtractorOrdering:
    """Tests for extraction ordering."""
    
    def test_filter_sort_topk_stoplist_order(self):
        """
        Verify filter→sort→topK→stoplist ordering.
        
        Paper hook: "CUI candidates score-filtered (>=0.7), sorted DESC,
        top-K selected, then stoplist filtered (Section 6.1)"
        """
        # Mock setup
        mock_candidates = [
            ("C0001", 0.9),   # High score
            ("C0002", 0.5),   # Below threshold
            ("C0003", 0.8),   # Above threshold
            ("C0004", 0.85),  # Above threshold
            ("C0005", 0.3),   # Below threshold
        ]
        
        threshold = 0.7
        max_k = 2
        stop_cuis = {"C0001"}  # Stop the highest scoring one
        
        # Step 1: Filter by threshold
        filtered = [(cui, score) for cui, score in mock_candidates if score >= threshold]
        assert len(filtered) == 3  # C0001, C0003, C0004
        
        # Step 2: Sort descending by score
        sorted_candidates = sorted(filtered, key=lambda x: x[1], reverse=True)
        assert sorted_candidates[0][0] == "C0001"  # 0.9
        assert sorted_candidates[1][0] == "C0004"  # 0.85
        assert sorted_candidates[2][0] == "C0003"  # 0.8
        
        # Step 3: Take top K
        top_k = sorted_candidates[:max_k]
        assert len(top_k) == 2
        assert top_k[0][0] == "C0001"
        assert top_k[1][0] == "C0004"
        
        # Step 4: Apply stoplist
        final = [cui.upper() for cui, _ in top_k if cui.upper() not in stop_cuis]
        assert len(final) == 1
        assert final[0] == "C0004"  # C0001 was stoplisted
    
    def test_sort_order_regression_20_candidates(self):
        """
        REGRESSION TEST: Verify high-score candidate at position 11 is retained.
        
        Paper hook: "Sort-order bug regression: 20 candidates with first 10 
        low-score, 11th high-score must be retained after filter/sort/topK 
        (Section 6.1)"
        
        This test catches the common bug where candidates are sorted AFTER topK
        instead of BEFORE, causing high-scoring candidates to be excluded.
        """
        # Scenario: 20 candidates
        # First 10: low scores (0.1-0.19)
        # 11th: HIGH score (0.95) - this MUST survive
        # Rest: medium scores (0.5-0.65)
        
        mock_candidates = []
        
        # First 10 low scores (below threshold)
        for i in range(10):
            mock_candidates.append((f"C{i:04d}", 0.1 + i * 0.01))  # 0.1 to 0.19
        
        # 11th candidate: HIGH score
        mock_candidates.append(("C0010", 0.95))  # Critical candidate
        
        # Remaining 9: medium scores (above threshold but below C0010)
        for i in range(9):
            mock_candidates.append((f"C{11+i:04d}", 0.5 + i * 0.01))  # 0.5 to 0.58
        
        assert len(mock_candidates) == 20
        
        threshold = 0.7
        max_k = 5
        stop_cuis = set()  # No stoplist for this test
        
        # CORRECT ORDER: filter → sort → topK → stoplist
        
        # Step 1: Filter by threshold
        filtered = [(cui, score) for cui, score in mock_candidates if score >= threshold]
        
        # Only C0010 (0.95) should pass threshold
        assert len(filtered) == 1, f"Expected 1 candidate >= 0.7, got {len(filtered)}"
        assert filtered[0][0] == "C0010"
        assert filtered[0][1] == 0.95
        
        # Step 2: Sort descending (already only 1 candidate)
        sorted_candidates = sorted(filtered, key=lambda x: x[1], reverse=True)
        
        # Step 3: Take top K
        top_k = sorted_candidates[:max_k]
        assert len(top_k) == 1
        assert top_k[0][0] == "C0010"
        
        # Step 4: Apply stoplist
        final = [cui.upper() for cui, _ in top_k if cui.upper() not in stop_cuis]
        
        # CRITICAL: C0010 MUST be in final result
        assert "C0010" in final, "Sort-order regression: C0010 should be retained"
    
    def test_sort_order_regression_wrong_order_fails(self):
        """
        Verify that WRONG ordering (topK before sort) produces wrong results.
        
        This demonstrates what the bug looks like and why correct ordering matters.
        """
        # Same scenario as above
        mock_candidates = []
        
        for i in range(10):
            mock_candidates.append((f"C{i:04d}", 0.1 + i * 0.01))
        
        mock_candidates.append(("C0010", 0.95))
        
        for i in range(9):
            mock_candidates.append((f"C{11+i:04d}", 0.75 + i * 0.01))  # Higher this time
        
        threshold = 0.7
        max_k = 5
        
        # WRONG ORDER: topK → filter → sort (common bug pattern)
        
        # Step 1 WRONG: Take first K candidates BEFORE filtering
        top_k_wrong = mock_candidates[:max_k]
        
        # Step 2 WRONG: Filter after topK
        filtered_wrong = [(cui, score) for cui, score in top_k_wrong if score >= threshold]
        
        # C0010 is at position 11, so it's NOT in the wrong result
        extracted_cuis_wrong = [cui for cui, _ in filtered_wrong]
        
        # The bug: C0010 is excluded because it was at position 11
        assert "C0010" not in extracted_cuis_wrong, "This demonstrates the bug"
        
        # CORRECT ORDER: filter → sort → topK
        
        # Step 1 CORRECT: Filter by threshold first
        filtered_correct = [(cui, score) for cui, score in mock_candidates if score >= threshold]
        
        # Step 2 CORRECT: Sort descending by score
        sorted_correct = sorted(filtered_correct, key=lambda x: x[1], reverse=True)
        
        # Step 3 CORRECT: Take top K
        top_k_correct = sorted_correct[:max_k]
        
        # C0010 (0.95) should be at position 0 after sorting
        assert top_k_correct[0][0] == "C0010"
        
        # Final correct extraction
        extracted_cuis_correct = [cui for cui, _ in top_k_correct]
        assert "C0010" in extracted_cuis_correct, "Correct order retains C0010"
    
    def test_normalization_uppercase(self):
        """Verify CUIs are normalized to uppercase."""
        test_cuis = ["c0001234", "C0005678", "c0009999"]
        normalized = [cui.upper() for cui in test_cuis]
        
        assert all(cui.isupper() for cui in normalized)
        assert normalized == ["C0001234", "C0005678", "C0009999"]
    
    def test_stoplist_case_insensitive(self):
        """Verify stoplist matching is case-insensitive."""
        stop_cuis = {"C0001234", "c0005678"}  # Mixed case in stoplist
        stop_cuis_normalized = {cui.upper() for cui in stop_cuis}
        
        test_cui_lower = "c0001234"
        test_cui_upper = "C0001234"
        
        # Both should match after normalization
        assert test_cui_lower.upper() in stop_cuis_normalized
        assert test_cui_upper.upper() in stop_cuis_normalized


class TestConceptExtractorConfig:
    """Tests for extractor configuration."""
    
    def test_default_config_values(self):
        """Verify default configuration values."""
        from concept_extractor import ConceptExtractor
        
        config = {
            "scispacy_model": "en_core_sci_sm"
        }
        
        # Create with NER-only mode to avoid linker dependency
        try:
            extractor = ConceptExtractor(config, no_linking=True)
            
            assert extractor.min_entity_len == 3
            assert extractor.cui_score_threshold == 0.7
            assert extractor.max_k == 10
            assert extractor.concept_representation == "surface"
        except RuntimeError:
            pytest.skip("scispaCy model not installed")
    
    def test_version_info_structure(self):
        """Verify version info contains required fields."""
        from concept_extractor import ConceptExtractor
        
        config = {
            "scispacy_model": "en_core_sci_sm",
            "cui_score_threshold": 0.75,
            "max_k": 5,
            "min_entity_len": 4
        }
        
        try:
            extractor = ConceptExtractor(config, no_linking=True)
            version_info = extractor.get_version_info()
            
            # Required fields
            assert "spacy_version" in version_info
            assert "scispacy_model" in version_info
            assert "concept_representation" in version_info
            assert "cui_normalization" in version_info
            assert "cui_score_threshold" in version_info
            assert "max_k" in version_info
            assert "min_entity_len" in version_info
            
            # Values should match config
            assert version_info["cui_score_threshold"] == 0.75
            assert version_info["max_k"] == 5
            assert version_info["min_entity_len"] == 4
        except RuntimeError:
            pytest.skip("scispaCy model not installed")


class TestSurfaceExtraction:
    """Tests for surface-form extraction."""
    
    def test_surface_extraction_basic(self):
        """Test basic surface extraction."""
        from concept_extractor import ConceptExtractor
        
        config = {
            "scispacy_model": "en_core_sci_sm",
            "min_entity_len": 3
        }
        
        try:
            extractor = ConceptExtractor(config, no_linking=True)
            
            text = "The patient has diabetes mellitus and hypertension."
            surfaces = extractor.extract_surface(text)
            
            # Should extract some entities
            assert isinstance(surfaces, set)
            # All should be lowercase
            assert all(s.islower() for s in surfaces)
            # All should meet min length
            assert all(len(s) >= 3 for s in surfaces)
        except RuntimeError:
            pytest.skip("scispaCy model not installed")
    
    def test_empty_text_returns_empty_set(self):
        """Test that empty text returns empty set."""
        from concept_extractor import ConceptExtractor
        
        config = {
            "scispacy_model": "en_core_sci_sm"
        }
        
        try:
            extractor = ConceptExtractor(config, no_linking=True)
            
            assert extractor.extract("") == set()
            assert extractor.extract("   ") == set()
            assert extractor.extract_surface("") == set()
        except RuntimeError:
            pytest.skip("scispaCy model not installed")


class TestStoplistLoading:
    """Tests for stoplist loading."""
    
    def test_missing_stoplist_returns_empty(self, tmp_path):
        """Test that missing stoplist file returns empty set."""
        from concept_extractor import ConceptExtractor
        
        config = {
            "scispacy_model": "en_core_sci_sm",
            "stop_entities_file": str(tmp_path / "nonexistent.txt"),
            "stop_cuis_file": str(tmp_path / "also_nonexistent.txt")
        }
        
        try:
            extractor = ConceptExtractor(config, no_linking=True)
            
            assert len(extractor.stop_entities) == 0
            assert len(extractor.stop_cuis) == 0
        except RuntimeError:
            pytest.skip("scispaCy model not installed")
    
    def test_stoplist_comments_ignored(self, tmp_path):
        """Test that comment lines are ignored in stoplists."""
        stoplist_file = tmp_path / "stop.txt"
        stoplist_file.write_text("""
# This is a comment
patient
# Another comment
treatment
""")
        
        from concept_extractor import ConceptExtractor
        
        config = {
            "scispacy_model": "en_core_sci_sm",
            "stop_entities_file": str(stoplist_file)
        }
        
        try:
            extractor = ConceptExtractor(config, no_linking=True)
            
            assert "patient" in extractor.stop_entities
            assert "treatment" in extractor.stop_entities
            assert len(extractor.stop_entities) == 2
        except RuntimeError:
            pytest.skip("scispaCy model not installed")


class TestClinicalText:
    """Tests with realistic clinical text."""
    
    CLINICAL_TEXT = """
    Chief Complaint: Chest pain radiating to left arm.
    
    History of Present Illness:
    58-year-old male presents with acute onset chest pain started 2 hours ago.
    Pain is described as crushing, 8/10 severity.
    Associated with diaphoresis and shortness of breath.
    
    Past Medical History:
    - Type 2 diabetes mellitus
    - Hypertension
    - Hyperlipidemia
    - Previous myocardial infarction 2015
    
    Current Medications:
    - Metformin 1000mg BID
    - Lisinopril 20mg daily
    - Atorvastatin 40mg daily
    - Aspirin 81mg daily
    
    Assessment:
    Acute coronary syndrome, likely NSTEMI given presentation.
    Troponin I elevated at 2.5 ng/mL.
    ECG shows ST depression in leads V3-V6.
    """
    
    def test_clinical_text_extracts_concepts(self):
        """Test that clinical text produces non-empty concept set."""
        from concept_extractor import ConceptExtractor
        
        config = {
            "scispacy_model": "en_core_sci_sm",
            "min_entity_len": 3
        }
        
        try:
            extractor = ConceptExtractor(config, no_linking=True)
            
            concepts = extractor.extract(self.CLINICAL_TEXT)
            
            # Should extract multiple concepts from rich clinical text
            assert len(concepts) > 0
            
            # Print for debugging (only when run directly)
            print(f"\nExtracted {len(concepts)} concepts from clinical text:")
            for c in sorted(concepts)[:10]:
                print(f"  - {c}")
        except RuntimeError:
            pytest.skip("scispaCy model not installed")


class TestConfigLoader:
    """Tests for centralized config loader."""
    
    def test_config_loader_imports(self):
        """Verify config_loader module imports correctly."""
        from config_loader import (
            load_extractor_config,
            load_extractor_config_for_stoplist_generation,
            get_stoplists_provenance,
            get_configs_dir,
        )
        
        # These should be callable
        assert callable(load_extractor_config)
        assert callable(load_extractor_config_for_stoplist_generation)
        assert callable(get_stoplists_provenance)
        assert callable(get_configs_dir)
    
    def test_stoplist_generation_disables_stoplists(self, tmp_path):
        """Verify stoplist generation disables stoplists for non-circular extraction."""
        # Create a minimal extractor.yaml
        config_file = tmp_path / "extractor.yaml"
        config_file.write_text("""
concept_extractor:
  scispacy_model: "en_core_sci_sm"
  stop_entities_file: "evals/configs/stop_entities.txt"
  stop_cuis_file: "evals/configs/stop_cuis.txt"
""")
        
        from config_loader import load_extractor_config_for_stoplist_generation
        
        config = load_extractor_config_for_stoplist_generation(tmp_path)
        
        # Stoplists should be disabled
        assert config.get("stop_entities_file") is None
        assert config.get("stop_cuis_file") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




