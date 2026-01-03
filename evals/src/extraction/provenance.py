"""Provenance helpers for extraction workflows."""

from __future__ import annotations

from .concept_extractor import ConceptExtractor


def get_extractor_version_info(extractor_config: dict) -> dict:
    """
    Get real version info from ConceptExtractor for provenance tracking.

    Paper hook: "Extractor provenance includes actual library versions
    collected at runtime to ensure reproducibility (Section 6.1)"

    This function instantiates ConceptExtractor to collect actual installed
    versions of spacy and scispacy, ensuring logged versions match what
    would actually be used during extraction.

    Args:
        extractor_config: Extractor configuration dict

    Returns:
        Dict with version info including spacy_version, scispacy_version,
        and other extractor metadata
    """
    try:
        # Try with linking first to get full version info including KB version
        extractor = ConceptExtractor(extractor_config, no_linking=False)
        version_info = extractor.get_version_info()
    except (RuntimeError, ImportError, Exception):
        # Fall back to no_linking if linker unavailable
        # This still gives us spacy/scispacy versions
        try:
            extractor = ConceptExtractor(extractor_config, no_linking=True)
            version_info = extractor.get_version_info()
        except Exception:
            # Last resort: collect versions directly without loading model
            import spacy

            version_info = {
                "spacy_version": spacy.__version__,
                "scispacy_model": extractor_config.get(
                    "scispacy_model", "en_core_sci_sm"
                ),
                "scispacy_version": "unknown",
                "concept_representation": extractor_config.get(
                    "concept_representation", "cui"
                ),
                "cui_normalization": extractor_config.get(
                    "cui_normalization", "uppercase"
                ),
            }
            import importlib.util

            if importlib.util.find_spec("scispacy") is not None:
                import scispacy

                version_info["scispacy_version"] = scispacy.__version__

    # Merge with config values to ensure all fields are present
    return {
        "spacy_version": version_info.get("spacy_version", "unknown"),
        "scispacy_version": version_info.get("scispacy_version", "unknown"),
        "scispacy_model": extractor_config.get("scispacy_model", "en_core_sci_sm"),
        "linker_name": version_info.get(
            "linker_name", extractor_config.get("linker_name", "umls")
        ),
        "linker_kb_version": version_info.get("linker_kb_version")
        or extractor_config.get("linker_kb_version", "2023AB"),
        "linker_resolve_abbreviations": extractor_config.get(
            "linker_resolve_abbreviations", True
        ),
        "linker_max_entities_per_mention": extractor_config.get(
            "linker_max_entities_per_mention", 10
        ),
        "linker_threshold": extractor_config.get("linker_threshold", 0.7),
        "cui_score_threshold": extractor_config.get("cui_score_threshold", 0.7),
        "max_k": extractor_config.get("max_k", 10),
        "min_entity_len": extractor_config.get("min_entity_len", 3),
        "concept_representation": version_info.get(
            "concept_representation",
            extractor_config.get("concept_representation", "cui"),
        ),
        "cui_normalization": extractor_config.get("cui_normalization", "uppercase"),
        "entity_types_kept": extractor_config.get("entity_types_kept", ["ALL"]),
        "stop_entities_count": version_info.get("stop_entities_count", 0),
        "stop_cuis_count": version_info.get("stop_cuis_count", 0),
    }
