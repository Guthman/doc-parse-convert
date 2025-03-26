"""
Unit tests for the ImageConverter class.
"""

import os
import unittest
from pathlib import Path
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import io

from doc_parse_convert.content_extraction import ImageConverter

class TestImageConverter(unittest.TestCase):
    """Tests for the ImageConverter class."""
    
    def setUp(self):
        """Set up test fixture."""
        # Create a simple test PDF
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_pdf_path = os.path.join(self.temp_dir.name, "test.pdf")
        
        # Create a simple PDF with 3 pages
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page(width=500, height=700)
            # Add some text to the page
            rect = fitz.Rect(100, 100, 400, 200)
            page.insert_text(rect.tl, f"Test Page {i+1}")
        
        doc.save(self.test_pdf_path)
        doc.close()
    
    def tearDown(self):
        """Tear down test fixture."""
        self.temp_dir.cleanup()
    
    def test_constructor(self):
        """Test that the constructor properly initializes the converter."""
        converter = ImageConverter(self.test_pdf_path)
        self.assertEqual(converter.file_path, self.test_pdf_path)
        self.assertEqual(converter.format, "png")  # Default format
        self.assertEqual(converter.dpi, 300)  # Default DPI
        self.assertIsNotNone(converter.doc)  # Document should be loaded
        converter.close()
    
    def test_format_setter(self):
        """Test that the format can be set."""
        converter = ImageConverter(self.test_pdf_path)
        try:
            # Test setting to jpg
            converter.set_format("jpg")
            self.assertEqual(converter.format, "jpg")
            
            # Test setting to png
            converter.set_format("png")
            self.assertEqual(converter.format, "png")
            
            # Test invalid format
            with self.assertRaises(ValueError):
                converter.set_format("invalid")
        finally:
            converter.close()
    
    def test_iteration(self):
        """Test that the iteration works properly."""
        converter = ImageConverter(self.test_pdf_path)
        try:
            pages = []
            # Collect all pages
            for page_number, page_data in converter:
                pages.append((page_number, page_data))
            
            # Check that we got 3 pages
            self.assertEqual(len(pages), 3)
            
            # Check page numbers
            self.assertEqual(pages[0][0], 0)
            self.assertEqual(pages[1][0], 1)
            self.assertEqual(pages[2][0], 2)
            
            # Check that each page has data
            for _, page_data in pages:
                self.assertGreater(len(page_data), 0)
                
                # Verify it's a valid image by loading it with PIL
                img = Image.open(io.BytesIO(page_data))
                self.assertIsNotNone(img)
        finally:
            converter.close()
    
    def test_png_format(self):
        """Test PNG format."""
        converter = ImageConverter(self.test_pdf_path, format="png")
        try:
            _, page_data = next(converter)
            
            # Verify it's a PNG by checking the magic bytes
            self.assertEqual(page_data[:8], b'\x89PNG\r\n\x1a\n')
            
            # Load with PIL to verify it's a valid PNG
            img = Image.open(io.BytesIO(page_data))
            self.assertEqual(img.format, "PNG")
        finally:
            converter.close()
    
    def test_jpg_format(self):
        """Test JPG format."""
        converter = ImageConverter(self.test_pdf_path, format="jpg")
        try:
            _, page_data = next(converter)
            
            # Verify it's a JPEG by checking the magic bytes
            self.assertEqual(page_data[:2], b'\xff\xd8')
            
            # Load with PIL to verify it's a valid JPEG
            img = Image.open(io.BytesIO(page_data))
            self.assertEqual(img.format, "JPEG")
        finally:
            converter.close()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        # Document should be automatically closed after the with block
        with ImageConverter(self.test_pdf_path) as converter:
            self.assertIsNotNone(converter.doc)
            _, page_data = next(converter)
            self.assertGreater(len(page_data), 0)
        
        # Document should be closed after exiting the with block
        self.assertIsNone(converter.doc)

if __name__ == "__main__":
    unittest.main()
