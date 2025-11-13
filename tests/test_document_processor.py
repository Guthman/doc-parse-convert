"""
Tests for document processing functionality.
"""

import pytest
from doc_parse_convert.utils.factory import ProcessorFactory
from doc_parse_convert.extraction.pdf import PDFProcessor


def test_processor_factory_with_pdf(pdf_sample_path, processing_config):
    """Test creating a processor using the factory with a PDF file."""
    processor = ProcessorFactory.create_processor(pdf_sample_path, processing_config)
    try:
        assert processor is not None
        assert isinstance(processor, PDFProcessor)
        assert processor.file_path == pdf_sample_path
        assert processor.doc is not None
    finally:
        processor.close()


def test_pdf_processor_load(pdf_sample_path, processing_config):
    """Test loading a PDF file with PDFProcessor."""
    processor = PDFProcessor(processing_config)
    try:
        processor.load(pdf_sample_path)
        assert processor.file_path == pdf_sample_path
        assert processor.doc is not None
        assert processor.doc.page_count > 0  # Ensure the document has pages
    finally:
        processor.close()


def test_get_table_of_contents(pdf_processor):
    """Test extracting table of contents from a PDF."""
    chapters = pdf_processor.get_table_of_contents()

    # We don't check exact content as it depends on the AI, but we can check structure
    assert chapters is not None
    assert isinstance(chapters, list)

    # If chapters were found, check their structure
    if chapters:
        for chapter in chapters:
            assert hasattr(chapter, 'title')
            assert hasattr(chapter, 'start_page')
            assert isinstance(chapter.start_page, int)

            # End page might be None for the last chapter
            if chapter.end_page is not None:
                assert isinstance(chapter.end_page, int)
                assert chapter.end_page >= chapter.start_page


def test_extract_chapter_text(pdf_processor):
    """Test extracting text from a chapter."""
    # First get the table of contents
    chapters = pdf_processor.get_table_of_contents()

    # Skip if no chapters were found
    if not chapters:
        pytest.skip("No chapters found in the PDF")

    # Extract content from the first chapter
    chapter_content = pdf_processor.extract_chapter_text(chapters[0])

    # Check the structure of the extracted content
    assert chapter_content is not None
    assert hasattr(chapter_content, 'title')
    assert hasattr(chapter_content, 'pages')
    assert isinstance(chapter_content.pages, list)

    # Check that pages contain content
    if chapter_content.pages:
        page = chapter_content.pages[0]
        assert hasattr(page, 'chapter_text')
        assert isinstance(page.chapter_text, str)


def test_extract_chapters(pdf_processor):
    """Test extracting multiple chapters."""
    # Extract the first chapter
    chapters = pdf_processor.extract_chapters([0])

    assert chapters is not None
    assert isinstance(chapters, list)

    if chapters:
        assert len(chapters) == 1
        chapter = chapters[0]
        assert hasattr(chapter, 'content')
        assert chapter.content is not None


def test_split_by_chapters(pdf_processor, temp_output_dir):
    """Test splitting a PDF by chapters."""
    # This test will be skipped if the TOC extraction fails
    chapters = pdf_processor.get_table_of_contents()
    if not chapters:
        pytest.skip("No chapters found in the PDF")

    # Split the PDF by chapters
    pdf_processor.split_by_chapters(str(temp_output_dir))

    # Check that files were created
    files = list(temp_output_dir.glob("*.pdf"))
    assert len(files) > 0
