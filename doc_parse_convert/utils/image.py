"""
Utilities for converting document pages to images.
"""

import io
import os
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from PIL import Image

from doc_parse_convert.config import logger


class ImageConverter:
    """Utility class for converting document pages to images.
    
    This class can be used in two ways:
    1. As an iterator to get raw image data:
       ```
       converter = ImageConverter('document.pdf')
       for page_number, page in converter:
           with open(f'document_{page_number}.png', 'wb') as f:
               f.write(page)
       ```
       
    2. Using the static method to get Vertex AI compatible image objects:
       ```
       images = ImageConverter.convert_to_images(document)
       ```
    """
    
    def __init__(self, file_path: str | Path, format: str = "png", dpi: int = 300):
        """Initialize the image converter.
        
        Args:
            file_path: Path to the document file to convert
            format: Image format to use ('png' or 'jpg')
            dpi: DPI for image conversion
        """
        self.file_path = file_path
        self.format = format.lower()
        if self.format not in ['png', 'jpg']:
            raise ValueError("Format must be either 'png' or 'jpg'")
        self.dpi = dpi
        self.doc = None
        self.current_page = 0
        
        # Open the document
        self._open_document()
    
    def _open_document(self):
        """Open the document for conversion."""
        if self.doc is not None:
            self.close()
        
        try:
            self.doc = fitz.open(self.file_path)
            self.current_page = 0
        except Exception as e:
            raise ValueError(f"Failed to open document: {str(e)}")
    
    def close(self):
        """Close the document and free resources."""
        if self.doc:
            self.doc.close()
            self.doc = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __iter__(self):
        """Return iterator for the converter."""
        self.current_page = 0
        return self
    
    def __next__(self):
        """Get the next page as an image.
        
        Returns:
            Tuple of (page_number, image_data_bytes)
        """
        if self.doc is None:
            raise ValueError("Document not open")
        
        if self.current_page >= self.doc.page_count:
            raise StopIteration
        
        page_number = self.current_page
        page = self.doc.load_page(page_number)
        pix = page.get_pixmap(dpi=self.dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG' if self.format == 'png' else 'JPEG')
        img_data = img_byte_arr.getvalue()
        
        self.current_page += 1
        return page_number, img_data
    
    def set_format(self, format: str):
        """Set the image format.
        
        Args:
            format: Image format to use ('png' or 'jpg')
        """
        format = format.lower()
        if format not in ['png', 'jpg']:
            raise ValueError("Format must be either 'png' or 'jpg'")
        self.format = format
        return self
    
    @staticmethod
    def convert_to_images(document: Any, num_pages: int = 20, start_page: int = 0, image_quality: int = 300) -> List[Dict[str, Any]]:
        """Convert document pages to images.
        
        Args:
            document: Document to convert (e.g., PDF)
            num_pages: Number of pages to convert
            start_page: Starting page number (0-based index)
            image_quality: DPI for image conversion (default: 300)
        
        Returns:
            List of dictionaries containing image data in format expected by Vertex AI
            
        Raises:
            ValueError: If document type is not supported or conversion fails
            NotImplementedError: If conversion is not implemented for the document type
        """
        if isinstance(document, fitz.Document):
            images = []
            # Calculate the end page, ensuring we don't exceed document bounds
            end_page = min(start_page + num_pages, document.page_count)
            
            # Estimate reasonable quality based on page count to prevent oversized requests
            # Adjust quality down if we have many pages to process
            adjusted_quality = image_quality
            # if num_pages > 50:
            #     adjusted_quality = min(image_quality, 200)  # Medium quality for many pages
            # if num_pages > 100:
            #     adjusted_quality = min(image_quality, 150)  # Lower quality for very many pages
            
            if adjusted_quality != image_quality:
                logger.info(f"Adjusting image quality from {image_quality} to {adjusted_quality} DPI due to large page count ({num_pages})")
            
            for i in range(start_page, end_page):
                try:
                    page = document.load_page(i)
                    
                    # Calculate matrix for the specified DPI
                    # 72 is the base DPI for PDF
                    zoom = adjusted_quality / 72
                    matrix = fitz.Matrix(zoom, zoom)
                    
                    pix = page.get_pixmap(matrix=matrix)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    
                    # Convert to bytes
                    img_byte_arr = io.BytesIO()
                    
                    # Use compression to reduce file size (quality=85 provides good balance)
                    img.save(img_byte_arr, format='PNG', optimize=True)
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Check image size and log warning if it's large
                    img_size_mb = len(img_byte_arr) / (1024 * 1024)
                    if img_size_mb > 10:
                        logger.warning(f"Page {i+1} image is very large: {img_size_mb:.2f}MB")
                    elif img_size_mb > 5:
                        logger.debug(f"Page {i+1} image is large: {img_size_mb:.2f}MB")
                    
                    # Create image object in format expected by Vertex AI
                    image_obj = {
                        "data": img_byte_arr,
                        "_mime_type": "image/png"
                    }
                    images.append(image_obj)
                    
                except Exception as e:
                    logger.error(f"Error converting page {i+1}: {str(e)}")
                    # Continue with other pages if possible
            
            if not images:
                raise ValueError(f"Failed to convert any pages from document")
            
            return images
        
        raise NotImplementedError(f"Conversion not implemented for {type(document)}")
