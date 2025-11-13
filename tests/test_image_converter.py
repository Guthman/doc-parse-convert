"""
Tests for the ImageConverter class.
"""

import io
import pytest
from PIL import Image
import fitz  # PyMuPDF

from doc_parse_convert.utils.image import ImageConverter


@pytest.fixture
def temp_pdf_path(tmp_path):
    """Create a temporary test PDF."""
    pdf_path = tmp_path / "test.pdf"

    # Create a simple PDF with 3 pages
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=500, height=700)
        # Add some text to the page
        rect = fitz.Rect(100, 100, 400, 200)
        page.insert_text(rect.tl, f"Test Page {i+1}")

    doc.save(pdf_path)
    doc.close()

    return pdf_path


def test_constructor(temp_pdf_path):
    """Test that the constructor properly initializes the converter."""
    converter = ImageConverter(temp_pdf_path)
    assert converter.file_path == temp_pdf_path
    assert converter.format == "png"  # Default format
    assert converter.dpi == 300  # Default DPI
    assert converter.doc is not None  # Document should be loaded
    converter.close()


def test_format_setter(temp_pdf_path):
    """Test that the format can be set."""
    converter = ImageConverter(temp_pdf_path)
    try:
        # Test setting to jpg
        converter.set_format("jpg")
        assert converter.format == "jpg"

        # Test setting to png
        converter.set_format("png")
        assert converter.format == "png"

        # Test invalid format
        with pytest.raises(ValueError):
            converter.set_format("invalid")
    finally:
        converter.close()


def test_iteration(temp_pdf_path):
    """Test that the iteration works properly."""
    converter = ImageConverter(temp_pdf_path)
    try:
        pages = []
        # Collect all pages
        for page_number, page_data in converter:
            pages.append((page_number, page_data))

        # Check that we got 3 pages
        assert len(pages) == 3

        # Check page numbers
        assert pages[0][0] == 0
        assert pages[1][0] == 1
        assert pages[2][0] == 2

        # Check that each page has data
        for _, page_data in pages:
            assert len(page_data) > 0

            # Verify it's a valid image by loading it with PIL
            img = Image.open(io.BytesIO(page_data))
            assert img is not None
    finally:
        converter.close()


def test_png_format(temp_pdf_path):
    """Test PNG format."""
    converter = ImageConverter(temp_pdf_path, format="png")
    try:
        _, page_data = next(converter)

        # Verify it's a PNG by checking the magic bytes
        assert page_data[:8] == b'\x89PNG\r\n\x1a\n'

        # Load with PIL to verify it's a valid PNG
        img = Image.open(io.BytesIO(page_data))
        assert img.format == "PNG"
    finally:
        converter.close()


def test_jpg_format(temp_pdf_path):
    """Test JPG format."""
    converter = ImageConverter(temp_pdf_path, format="jpg")
    try:
        _, page_data = next(converter)

        # Verify it's a JPEG by checking the magic bytes
        assert page_data[:2] == b'\xff\xd8'

        # Load with PIL to verify it's a valid JPEG
        img = Image.open(io.BytesIO(page_data))
        assert img.format == "JPEG"
    finally:
        converter.close()


def test_context_manager(temp_pdf_path):
    """Test context manager functionality."""
    # Document should be automatically closed after the with block
    with ImageConverter(temp_pdf_path) as converter:
        assert converter.doc is not None
        _, page_data = next(converter)
        assert len(page_data) > 0

    # Document should be closed after exiting the with block
    assert converter.doc is None


def test_real_pdf_conversion(pdf_sample_path):
    """Test conversion with a real PDF file."""
    # This test uses the actual sample PDF
    with ImageConverter(pdf_sample_path) as converter:
        # Just test the first page to keep the test fast
        page_number, page_data = next(converter)

        # Verify basic properties
        assert page_number == 0
        assert len(page_data) > 0

        # Load the image to verify it's valid
        img = Image.open(io.BytesIO(page_data))
        assert img is not None
        assert img.format == "PNG"  # Default format is PNG
