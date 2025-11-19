"""
Tests for content extraction functionality.
"""

import pytest
import os
import fitz

from doc_parse_convert.ai.client import AIClient
from doc_parse_convert.utils.image import ImageConverter
from doc_parse_convert.models.document import Chapter
from doc_parse_convert.config import ExtractionStrategy


def test_ai_client_initialization(processing_config):
    """Test initializing the AI client."""
    client = AIClient(processing_config)
    assert client is not None
    assert client.model is not None


def test_extract_structure_from_images(ai_client, pdf_sample_path):
    """Test extracting structure from document images."""
    # Convert some pages of the PDF to images
    doc = None
    try:
        doc = fitz.open(pdf_sample_path)
        # Convert first few pages to images for testing
        max_pages = min(5, doc.page_count)
        images = ImageConverter.convert_to_images(doc, num_pages=max_pages, start_page=0)

        # Extract structure from these images
        chapters = ai_client.extract_structure_from_images(images)

        # Verify the result
        assert chapters is not None
        assert isinstance(chapters, list)

        # If chapters were found, check their structure
        if chapters:
            for chapter in chapters:
                assert isinstance(chapter, Chapter)
                assert hasattr(chapter, 'title')
                assert hasattr(chapter, 'start_page')
    finally:
        if doc:
            doc.close()


def test_content_extraction_with_native(pdf_sample_path):
    """Test content extraction with NATIVE strategy."""
    # Create config with explicit NATIVE strategy
    from doc_parse_convert.config import ProcessingConfig
    from doc_parse_convert.utils.factory import ProcessorFactory
    config = ProcessingConfig(
        toc_extraction_strategy=ExtractionStrategy.NATIVE,
        content_extraction_strategy=ExtractionStrategy.NATIVE
    )

    # Create processor with this config
    processor = ProcessorFactory.create_processor(pdf_sample_path, config)

    try:
        # Try to get TOC, but we may get an exception if no native TOC is found
        try:
            chapters = processor.get_table_of_contents()
            # If we get here, we have chapters, so test extraction
            if chapters:
                # Extract content from the first chapter
                chapter_content = processor.extract_chapter_text(chapters[0])

                # Verify the result
                assert chapter_content is not None
                assert chapter_content.title == chapters[0].title
                assert hasattr(chapter_content, 'pages')
                assert isinstance(chapter_content.pages, list)
                assert len(chapter_content.pages) > 0

                # Check that we have pages with content
                page = chapter_content.pages[0]
                assert hasattr(page, 'chapter_text')
                assert isinstance(page.chapter_text, str)
                assert len(page.chapter_text) > 0

                # Verify page numbering makes sense
                assert chapter_content.start_page == chapters[0].start_page
                assert chapter_content.end_page > chapter_content.start_page
        except ValueError as e:
            # Expected for test files without native TOC
            if "No native TOC found in document" in str(e):
                # Create a synthetic chapter for testing content extraction
                synthetic_chapter = Chapter(
                    title="Test Chapter",
                    start_page=0,
                    end_page=1,  # Only extract first page for speed
                    level=1
                )

                # Test content extraction with this synthetic chapter
                chapter_content = processor.extract_chapter_text(synthetic_chapter)

                # Verify the result
                assert chapter_content is not None
                assert hasattr(chapter_content, 'pages')
                assert isinstance(chapter_content.pages, list)
                assert len(chapter_content.pages) > 0

                # Check that we have pages with content
                page = chapter_content.pages[0]
                assert hasattr(page, 'chapter_text')
                assert isinstance(page.chapter_text, str)
                assert len(page.chapter_text) > 0
            else:
                # If it's a different error, re-raise it
                raise
    finally:
        processor.close()


@pytest.mark.skipif(not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                    reason="Requires Vertex AI credentials")
def test_content_extraction_with_ai(pdf_sample_path, processing_config):
    """Test content extraction with AI strategy."""
    # Set extraction strategies explicitly to AI
    processing_config.toc_extraction_strategy = ExtractionStrategy.NATIVE  # Use native for TOC to isolate content extraction
    processing_config.content_extraction_strategy = ExtractionStrategy.AI

    # Create processor with this config
    from doc_parse_convert.utils.factory import ProcessorFactory
    processor = ProcessorFactory.create_processor(pdf_sample_path, processing_config)

    try:
        # Get the first chapter
        try:
            chapters = processor.get_table_of_contents()
            if not chapters:
                pytest.skip("No chapters found in the PDF")

            # Extract a small chapter (first 3 pages max) to keep test duration reasonable
            small_chapter = Chapter(
                title=chapters[0].title,
                start_page=chapters[0].start_page,
                end_page=min(chapters[0].start_page + 2, processor.doc.page_count - 1),
                level=chapters[0].level
            )
        except ValueError as e:
            # Expected for test files without native TOC
            if "No native TOC found in document" in str(e):
                # Create a synthetic chapter for testing content extraction
                small_chapter = Chapter(
                    title="Test Chapter",
                    start_page=0,
                    end_page=2,  # Only extract first two pages for speed
                    level=1
                )
            else:
                # If it's a different error, re-raise it
                raise

        try:
            # Extract content
            chapter_content = processor.extract_chapter_text(small_chapter)

            # Verify the result
            assert chapter_content is not None
            assert chapter_content.title == small_chapter.title
            assert hasattr(chapter_content, 'pages')
            assert isinstance(chapter_content.pages, list)
            assert len(chapter_content.pages) > 0

            # Check that we have pages with content
            page = chapter_content.pages[0]
            assert hasattr(page, 'chapter_text')
            assert isinstance(page.chapter_text, str)
            assert len(page.chapter_text) > 0

            # Verify AI-specific features (may have text boxes, tables, etc.)
            assert hasattr(page, 'text_boxes')
            assert hasattr(page, 'tables')
            assert hasattr(page, 'figures')

        except Exception as e:
            pytest.fail(f"AI content extraction failed with error: {str(e)}")
    finally:
        processor.close()


@pytest.mark.skipif(not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                    reason="Requires Vertex AI credentials")
def test_content_extraction_with_ai_toc(pdf_sample_path, processing_config):
    """Test content extraction with AI strategy for both TOC and content."""
    # Set extraction strategies explicitly to AI for both TOC and content
    processing_config.toc_extraction_strategy = ExtractionStrategy.AI
    processing_config.content_extraction_strategy = ExtractionStrategy.AI

    # Create processor with this config
    from doc_parse_convert.utils.factory import ProcessorFactory
    processor = ProcessorFactory.create_processor(pdf_sample_path, processing_config)

    try:
        # Get the chapters using AI extraction
        chapters = processor.get_table_of_contents()
        if not chapters:
            pytest.skip("AI extraction failed to find chapters in the PDF")

        # Extract a small chapter (first 3 pages max) to keep test duration reasonable
        small_chapter = Chapter(
            title=chapters[0].title,
            start_page=chapters[0].start_page,
            end_page=min(chapters[0].start_page + 2, processor.doc.page_count - 1),
            level=chapters[0].level
        )

        # Extract content using AI
        chapter_content = processor.extract_chapter_text(small_chapter)

        # Verify the result
        assert chapter_content is not None
        assert chapter_content.title == small_chapter.title
        assert hasattr(chapter_content, 'pages')
        assert isinstance(chapter_content.pages, list)
        assert len(chapter_content.pages) > 0

        # Check that we have pages with content
        page = chapter_content.pages[0]
        assert hasattr(page, 'chapter_text')
        assert isinstance(page.chapter_text, str)
        assert len(page.chapter_text) > 0

        # Verify AI-specific features (may have text boxes, tables, etc.)
        assert hasattr(page, 'text_boxes')
        assert hasattr(page, 'tables')
        assert hasattr(page, 'figures')
    finally:
        processor.close()


def test_convert_to_images_static_method(pdf_sample_path):
    """Test the static convert_to_images method."""
    doc = None
    try:
        doc = fitz.open(pdf_sample_path)
        # Convert 3 pages to images (returns a generator)
        images_gen = ImageConverter.convert_to_images(doc, num_pages=3, start_page=0)

        # Convert generator to list for testing
        images = list(images_gen)

        # Verify the result
        assert images is not None
        assert isinstance(images, list)
        assert len(images) == 3

        # Check each image
        for img in images:
            assert 'data' in img
            assert '_mime_type' in img
            assert img['_mime_type'] == 'image/png'
            assert len(img['data']) > 0
    finally:
        if doc:
            doc.close()
