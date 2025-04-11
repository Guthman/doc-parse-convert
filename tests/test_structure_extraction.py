"""
Tests for document structure extraction functionality.
"""

import pytest
import json
import os
import xml.etree.ElementTree as ET

from doc_parse_convert.extraction.structure import DocumentStructureExtractor
from doc_parse_convert.models.document import DocumentSection
from doc_parse_convert.config import ExtractionStrategy, ProcessingConfig
from doc_parse_convert.utils.factory import ProcessorFactory


def test_extract_structure(pdf_processor):
    """Test extracting document structure."""
    extractor = DocumentStructureExtractor(pdf_processor)
    structure = extractor.extract_structure()

    # Verify the structure object
    assert structure is not None
    assert isinstance(structure, DocumentSection)
    assert structure.title is not None
    assert structure.start_page == 0
    assert structure.end_page == pdf_processor.doc.page_count - 1

    # Check if there are any children
    if structure.children:
        # Verify the first child
        first_child = structure.children[0]
        assert isinstance(first_child, DocumentSection)
        assert first_child.title is not None
        assert first_child.start_page is not None
        assert first_child.level > 0


def test_export_structure_as_dict(pdf_processor):
    """Test exporting document structure as a dictionary."""
    extractor = DocumentStructureExtractor(pdf_processor)
    structure = extractor.extract_structure()

    # Export as dictionary
    structure_dict = structure.to_dict()

    # Verify the dictionary
    assert structure_dict is not None
    assert isinstance(structure_dict, dict)
    assert 'title' in structure_dict
    assert 'start_page' in structure_dict
    assert 'end_page' in structure_dict
    assert 'level' in structure_dict


def test_export_structure_as_json(pdf_processor):
    """Test exporting document structure as JSON."""
    extractor = DocumentStructureExtractor(pdf_processor)

    # Export as JSON
    json_structure = extractor.export_structure(output_format="json")

    # Verify the JSON
    assert json_structure is not None

    # Attempt to parse it as JSON to confirm it's valid
    try:
        parsed = json.loads(json_structure)
        assert isinstance(parsed, dict)
        assert 'title' in parsed
    except json.JSONDecodeError:
        pytest.fail("Failed to parse exported JSON structure")


def test_export_structure_as_xml(pdf_processor):
    """Test exporting document structure as XML."""
    extractor = DocumentStructureExtractor(pdf_processor)

    # Export as XML
    xml_structure = extractor.export_structure(output_format="xml")

    # Verify the XML
    assert xml_structure is not None
    assert xml_structure.startswith("<?xml")

    # Attempt to parse it as XML to confirm it's valid
    try:
        root = ET.fromstring(xml_structure)
        assert root.tag == "document"
    except ET.ParseError:
        pytest.fail("Failed to parse exported XML structure")


def test_extract_text_by_section(pdf_processor, temp_output_dir):
    """Test extracting text by document section."""
    extractor = DocumentStructureExtractor(pdf_processor)

    # Extract text by section
    section_texts = extractor.extract_text_by_section(str(temp_output_dir))

    # Verify the result
    assert section_texts is not None
    assert isinstance(section_texts, dict)
    assert len(section_texts) > 0

    # Check that files were created in the output directory
    if temp_output_dir:
        files = list(temp_output_dir.glob("*.txt"))
        assert len(files) > 0


def test_native_structure_extraction(pdf_sample_path):
    """Test structure extraction using NATIVE strategy only."""
    # Create a config with NATIVE extraction strategy
    config = ProcessingConfig(
        toc_extraction_strategy=ExtractionStrategy.NATIVE,
        content_extraction_strategy=ExtractionStrategy.NATIVE
    )

    # Create processor with this config
    processor = ProcessorFactory.create_processor(pdf_sample_path, config)

    try:
        # Create the extractor
        structure_extractor = DocumentStructureExtractor(processor)

        # Extract structure
        structure = structure_extractor.extract_structure()

        # Basic structure validation
        assert structure is not None
        assert structure.title is not None
        assert structure.start_page == 0
        assert structure.end_page == processor.doc.page_count - 1

        # Check if there are children
        if structure.children:
            # Verify first child
            first_section = structure.children[0]
            assert isinstance(first_section, DocumentSection)
            assert first_section.title is not None
            assert first_section.start_page is not None
            assert first_section.level > 0

            # Check end page is set
            assert first_section.end_page is not None
            assert first_section.end_page < processor.doc.page_count

            # Check section properties like the end_page actually makes sense
            if len(structure.children) > 1:
                second_section = structure.children[1]
                assert first_section.end_page + 1 == second_section.start_page
    finally:
        processor.close()


@pytest.mark.skipif(not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                    reason="Requires Vertex AI credentials")
def test_ai_structure_extraction(pdf_sample_path, processing_config):
    """Test structure extraction using AI strategy only."""
    # Set extraction strategy explicitly to AI
    processing_config.toc_extraction_strategy = ExtractionStrategy.AI

    # Create processor with this config
    processor = ProcessorFactory.create_processor(pdf_sample_path, processing_config)

    try:
        # Create the extractor
        structure_extractor = DocumentStructureExtractor(processor)

        # Extract structure
        try:
            # Enable debug mode for the test
            original_debug_dir = os.environ.get("AI_DEBUG_DIR")
            temp_debug_dir = os.path.join(os.path.dirname(pdf_sample_path), "ai_debug_test")
            os.environ["AI_DEBUG_DIR"] = temp_debug_dir
            os.makedirs(temp_debug_dir, exist_ok=True)

            # Attempt extraction with more informative error handling
            structure = None
            try:
                structure = structure_extractor.extract_structure()
            except Exception as e:
                error_msg = f"AI extraction failed with error: {e.__class__.__name__}: {str(e)}"
                if "InvalidArgument" in e.__class__.__name__:
                    error_msg += "\nPossible causes:"
                    error_msg += "\n- Image size too large or API limits reached"
                    error_msg += "\n- Too many images in single request"
                    error_msg += "\n- Response schema too complex for API"
                    error_msg += "\nSuggested fixes:"
                    error_msg += "\n- Reduce number of pages processed (max_images parameter)"
                    error_msg += "\n- Simplify schema further (fewer properties, shorter names)"
                    error_msg += "\n- Make fewer properties required in the schema"
                    error_msg += "\n- Remove nested structures like 'children' arrays"
                    error_msg += "\n- Use a flattened schema with hierarchy reconstruction"
                    error_msg += f"\nCheck debug directory: {temp_debug_dir}"
                pytest.fail(error_msg)

            # Restore original debug directory
            if original_debug_dir:
                os.environ["AI_DEBUG_DIR"] = original_debug_dir
            else:
                os.environ.pop("AI_DEBUG_DIR", None)

            # Basic structure validation
            assert structure is not None, "Structure should not be None"
            assert structure.title is not None, "Document title should not be None"
            assert structure.start_page == 0, "Document should start at page 0"
            assert structure.end_page == processor.doc.page_count - 1, "Document should end at last page"

            # Check that at least some sections were extracted
            assert len(structure.children) > 0, "Document should have at least one section"

            # Validate structure format (not exact content)
            first_section = structure.children[0]
            assert first_section.title is not None, "Section title should not be None"
            assert first_section.start_page >= 0, "Section start page should be valid"
            assert first_section.end_page is not None, "Section end page should not be None"
            assert first_section.level > 0, "Section level should be greater than 0"

            # Verify JSON export works with AI-extracted structure
            json_structure = structure_extractor.export_structure(output_format="json")
            assert json_structure is not None, "JSON export should not be None"
            parsed = json.loads(json_structure)
            assert isinstance(parsed, dict), "Parsed JSON should be a dictionary"

        except Exception as e:
            error_class = e.__class__.__name__
            # Clean up test debug directory
            import shutil
            if os.path.exists(temp_debug_dir):
                shutil.rmtree(temp_debug_dir)

            pytest.fail(f"AI extraction test failed: {error_class}: {str(e)}")
    finally:
        processor.close()
