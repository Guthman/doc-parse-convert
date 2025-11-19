"""
Security-related tests for the doc_parse_convert library.
"""

import pytest
import time
import re

from doc_parse_convert.models.document import DocumentSection
from doc_parse_convert.extraction.structure import DocumentStructureExtractor
from unittest.mock import MagicMock

def test_redos_protection():
    """Test that regex doesn't cause catastrophic backtracking (ReDoS)."""
    # Pathological input that would cause exponential backtracking with vulnerable patterns
    malicious_input = "chapter " + "a" * 50000 + " "

    # Test the safe regex pattern
    pattern = r"^(chapter|section|part|appendix|\d+\.)\s+\S+"

    start_time = time.time()
    result = re.match(pattern, malicious_input.lower())
    elapsed = time.time() - start_time

    # Should complete in less than 1 second
    assert elapsed < 1.0, f"Regex took {elapsed:.3f}s - potential ReDoS vulnerability"
    # Pattern should match (chapter + space + long string of 'a's)
    assert result is not None

def test_regex_still_matches_valid():
    """Test that fix doesn't break valid matches."""
    pattern = r"^(chapter|section|part|appendix|\d+\.)\s+\S+"

    valid_inputs = [
        "chapter 1",
        "section 2.1",
        "appendix a",
        "1. introduction"
    ]

    for input_str in valid_inputs:
        assert re.match(pattern, input_str.lower()) is not None

def test_xml_injection_protection():
    """Test that XML special characters are properly escaped."""
    # Create a mock processor and document
    mock_doc = MagicMock()
    mock_doc.page_count = 20
    mock_processor = MagicMock()
    mock_processor.doc = mock_doc

    extractor = DocumentStructureExtractor(mock_processor)

    # Create a mock structure with malicious content
    root_section = DocumentSection(title="Root", start_page=0, end_page=19, level=0)
    malicious_section = DocumentSection(
        title='Chapter <script>alert("XSS")</script>',
        start_page=0,
        end_page=10,
        level=1,
        identifier='Test&"\'<>',
        section_type='chapter<evil>'
    )
    root_section.add_child(malicious_section)

    # Mock the extract_structure to return our malicious structure
    extractor.extract_structure = MagicMock(return_value=root_section)

    # Export to XML
    xml_output = extractor.export_structure(output_format="xml")

    # Verify escaping - ElementTree escapes < > & and " in attribute values
    assert '&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;' in xml_output
    assert '<script>alert("XSS")</script>' not in xml_output
    assert 'identifier="Test&amp;&quot;\'&lt;&gt;"' in xml_output
    assert 'section_type="chapter&lt;evil&gt;"' in xml_output
