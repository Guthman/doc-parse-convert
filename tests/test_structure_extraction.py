"""
Tests for document structure extraction functionality.
"""

import pytest
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from doc_parse_convert.content_extraction import (
    DocumentStructureExtractor,
    DocumentSection,
    ExtractionStrategy,
    ProcessingConfig
)


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
    from doc_parse_convert.content_extraction import ProcessorFactory
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
    from doc_parse_convert.content_extraction import ProcessorFactory
    processor = ProcessorFactory.create_processor(pdf_sample_path, processing_config)
    
    try:
        # Create the extractor
        structure_extractor = DocumentStructureExtractor(processor)
        
        # Extract structure
        try:
            structure = structure_extractor.extract_structure()
            
            # Basic structure validation
            assert structure is not None
            assert structure.title is not None
            assert structure.start_page == 0
            assert structure.end_page == processor.doc.page_count - 1
            
            # Check that at least some sections were extracted
            assert len(structure.children) > 0
            
            # Validate structure format (not exact content)
            first_section = structure.children[0]
            assert first_section.title is not None
            assert first_section.start_page >= 0
            assert first_section.end_page is not None
            assert first_section.level > 0
            
            # Verify JSON export works with AI-extracted structure
            json_structure = structure_extractor.export_structure(output_format="json")
            assert json_structure is not None
            parsed = json.loads(json_structure)
            assert isinstance(parsed, dict)
            
        except Exception as e:
            pytest.fail(f"AI extraction failed with error: {str(e)}")
    finally:
        processor.close()
