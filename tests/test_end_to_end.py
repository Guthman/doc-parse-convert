"""
End-to-end tests for the document processing pipeline.
"""

import pytest
import os
from doc_parse_convert.config import ProcessingConfig, ExtractionStrategy
from doc_parse_convert.utils.factory import ProcessorFactory
from doc_parse_convert.extraction.structure import DocumentStructureExtractor
from doc_parse_convert.models.document import Chapter
from doc_parse_convert.utils.image import ImageConverter


def test_complete_pdf_processing_pipeline(pdf_sample_path, temp_output_dir):
    """Test the complete PDF processing pipeline with native extraction."""
    # Create config with NATIVE extraction strategies to avoid requiring AI credentials
    config = ProcessingConfig(
        toc_extraction_strategy=ExtractionStrategy.NATIVE,
        content_extraction_strategy=ExtractionStrategy.NATIVE
    )

    # 1. Create processor
    processor = None
    try:
        processor = ProcessorFactory.create_processor(pdf_sample_path, config)

        # 2. Extract document structure
        structure_extractor = DocumentStructureExtractor(processor)
        structure = structure_extractor.extract_structure()

        # Verify structure
        assert structure is not None
        assert structure.title is not None

        # 3. Create a synthetic chapter for testing
        synthetic_chapter = Chapter(
            title="Test Chapter",
            start_page=0,
            end_page=2,  # First two pages
            level=1
        )

        # 4. Extract chapter content
        chapter_content = processor.extract_chapter_text(synthetic_chapter)

        # Verify chapter content
        assert chapter_content is not None
        assert chapter_content.title == synthetic_chapter.title
        assert len(chapter_content.pages) > 0

        # Check page content
        first_page = chapter_content.pages[0]
        assert first_page.chapter_text is not None
        assert len(first_page.chapter_text) > 0

        # 5. Export structure as JSON
        json_structure = structure_extractor.export_structure(output_format="json")
        assert json_structure is not None

        # 6. Extract text by section
        section_texts = structure_extractor.extract_text_by_section(str(temp_output_dir))
        assert section_texts is not None
        assert len(section_texts) > 0

        # Verify files were created
        text_files = list(temp_output_dir.glob("*.txt"))
        assert len(text_files) > 0

    finally:
        if processor:
            processor.close()


def test_document_to_images_conversion(pdf_sample_path, temp_output_dir):
    """Test converting a document to images."""
    # Already imported at the top

    # Create output directories
    png_dir = temp_output_dir / "png"
    png_dir.mkdir()

    # Convert first 3 pages to PNG
    with ImageConverter(pdf_sample_path, format="png") as converter:
        # Only convert the first 3 pages to keep the test fast
        for i, (page_number, page_data) in enumerate(converter):
            if i >= 3:
                break

            # Save the image
            output_path = png_dir / f"page_{page_number + 1:03d}.png"
            with open(output_path, 'wb') as f:
                f.write(page_data)

            # Verify the file exists
            assert output_path.exists()
            assert output_path.stat().st_size > 0


@pytest.mark.skipif(not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                    reason="Requires Vertex AI credentials")
def test_ai_toc_extraction_with_real_api(pdf_sample_path, processing_config):
    """Test AI-based TOC extraction with the real Vertex API."""
    # Set only TOC extraction strategy to AI
    processing_config.toc_extraction_strategy = ExtractionStrategy.AI
    processing_config.content_extraction_strategy = ExtractionStrategy.NATIVE  # Keep content extraction as NATIVE

    # Create processor
    processor = None
    try:
        processor = ProcessorFactory.create_processor(pdf_sample_path, processing_config)

        # Extract TOC with AI
        try:
            chapters = processor.get_table_of_contents()

            # Validation assertions
            assert chapters is not None
            assert isinstance(chapters, list)
            assert len(chapters) > 0

            # Validate chapter properties
            for chapter in chapters:
                assert chapter.title is not None
                assert chapter.start_page >= 0
                assert chapter.level > 0

            # Special check for consecutive chapters
            for i in range(1, len(chapters)):
                assert chapters[i].start_page >= chapters[i-1].start_page, "Chapters should have non-descending page numbers"

            # Log number of chapters found
            print(f"Successfully extracted {len(chapters)} chapters using AI method")

        except Exception as e:
            pytest.fail(f"AI TOC extraction failed with error: {str(e)}")
    finally:
        if processor:
            processor.close()


@pytest.mark.skipif(not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                    reason="Requires Vertex AI credentials")
def test_ai_content_extraction_with_real_api(pdf_sample_path, processing_config):
    """Test AI-based content extraction with the real Vertex API."""
    # Set only content extraction strategy to AI
    processing_config.toc_extraction_strategy = ExtractionStrategy.NATIVE  # Use NATIVE for TOC
    processing_config.content_extraction_strategy = ExtractionStrategy.AI

    # Create processor
    processor = None
    try:
        processor = ProcessorFactory.create_processor(pdf_sample_path, processing_config)

        # Get chapters using NATIVE method
        chapters = processor.get_table_of_contents()
        assert chapters is not None
        assert len(chapters) > 0, "No chapters found in document"

        # Test content extraction on first chapter
        try:
            # Only process a small part (first 2 pages) to keep test duration reasonable
            small_chapter = Chapter(
                title=chapters[0].title,
                start_page=chapters[0].start_page,
                end_page=min(chapters[0].start_page + 1, processor.doc.page_count - 1),
                level=chapters[0].level
            )

            # Extract content with AI
            chapter_content = processor.extract_chapter_text(small_chapter)

            # Validate extracted content
            assert chapter_content is not None
            assert chapter_content.title == small_chapter.title
            assert len(chapter_content.pages) > 0

            # Check for rich content features
            first_page = chapter_content.pages[0]
            assert first_page.chapter_text is not None
            assert len(first_page.chapter_text) > 0

            # AI extraction should recognize structured content
            assert hasattr(first_page, 'text_boxes')
            assert hasattr(first_page, 'tables')
            assert hasattr(first_page, 'figures')

            print(f"Successfully extracted chapter text with AI method (found {len(chapter_content.pages)} pages)")

        except Exception as e:
            pytest.fail(f"AI content extraction failed with error: {str(e)}")
    finally:
        if processor:
            processor.close()
