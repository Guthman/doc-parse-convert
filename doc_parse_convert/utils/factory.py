"""
Factory classes for creating document processors.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Type

from doc_parse_convert.config import ProcessingConfig, logger
from doc_parse_convert.extraction.base import DocumentProcessor

# Avoid circular import
if TYPE_CHECKING:
    from doc_parse_convert.extraction.pdf import PDFProcessor


class ProcessorFactory:
    """Factory for creating document processors."""

    @staticmethod
    def create_processor(file_path: str, config: ProcessingConfig) -> DocumentProcessor:
        """Create and initialize appropriate processor based on file extension.
        
        Args:
            file_path: Path to the document file
            config: Processing configuration
            
        Returns:
            Initialized document processor
            
        Raises:
            ValueError: If file format is not supported
        """
        ext = Path(file_path).suffix.lower()
        logger.debug(f"Creating processor for file type: {ext}")

        if ext == '.pdf':
            # Import here to avoid circular import
            from doc_parse_convert.extraction.pdf import PDFProcessor
            processor = PDFProcessor(config)
        # elif ext in ['.epub']:  # Future implementation
        #     processor = EPUBProcessor(config)
        else:
            error_msg = f"Unsupported file format: {ext}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Load the document
        processor.load(file_path)
        logger.info(f"Successfully created and loaded processor for {file_path}")
        return processor
