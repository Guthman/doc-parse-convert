"""
Tests for content conversion functionality.
"""

import io
from pathlib import Path

from doc_parse_convert.conversion.epub import (
    convert_epub_to_html,
    convert_epub_to_txt
)


def test_convert_epub_to_html(epub_sample_path):
    """Test converting EPUB to HTML."""
    # Convert EPUB to HTML
    html_content = convert_epub_to_html(epub_sample_path)

    # Verify the result
    assert html_content is not None
    assert isinstance(html_content, list)
    assert len(html_content) > 0

    # Check the content of the first HTML
    first_html = html_content[0]
    assert isinstance(first_html, str)
    assert len(first_html) > 0
    assert "<html" in first_html.lower() or "<body" in first_html.lower()


def test_convert_epub_to_txt_string(epub_sample_path, temp_output_dir):
    """Test converting EPUB to plain text with file output."""
    # Convert EPUB to text and save to file
    content_str = convert_epub_to_txt(epub_sample_path, temp_output_dir)

    # Verify the returned string
    assert content_str is not None
    assert isinstance(content_str, str)
    assert len(content_str) > 0

    # Check that a file was created
    filename = Path(epub_sample_path).stem
    text_file = Path(temp_output_dir) / f"{filename}.txt"
    assert text_file.exists()

    # Verify file content
    with open(text_file, 'r', encoding='utf-8') as f:
        file_content = f.read()
    assert file_content == content_str


def test_convert_epub_to_txt_stringio(epub_sample_path):
    """Test converting EPUB to plain text with StringIO output."""
    # Convert EPUB to text as StringIO
    stringio_content = convert_epub_to_txt(epub_sample_path)

    # Verify it's a StringIO object
    assert stringio_content is not None
    assert isinstance(stringio_content, io.StringIO)

    # Get the content as a string
    content_str = stringio_content.getvalue()
    assert isinstance(content_str, str)
    assert len(content_str) > 0
