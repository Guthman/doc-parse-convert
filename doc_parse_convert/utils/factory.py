"""
Factory classes for creating document processors.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Union, BinaryIO

from doc_parse_convert.config import ProcessingConfig, logger
from doc_parse_convert.extraction.base import DocumentProcessor

# Avoid circular import
if TYPE_CHECKING:
    from doc_parse_convert.extraction.pdf import PDFProcessor


class ProcessorFactory:
    """Factory for creating document processors."""

    @staticmethod
    def create_processor(
        file_input: Union[str, Path, BinaryIO],
        config: ProcessingConfig
    ) -> DocumentProcessor:
        """Create and initialize appropriate processor based on file type.

        Args:
            file_input: Path to document file OR file-like object
            config: Processing configuration

        Returns:
            Initialized document processor

        Raises:
            ValueError: If file format is not supported

        Security Note:
            If passing file paths, validate against your secure base
            directory first. See io_helpers.py for examples.
        """
        # Determine file type
        if isinstance(file_input, (str, Path)):
            ext = Path(file_input).suffix.lower()
        elif hasattr(file_input, 'name'):
            ext = Path(file_input.name).suffix.lower()
        else:
            raise ValueError("Cannot determine file type from file-like object without 'name' attribute")

        logger.debug(f"Creating processor for file type: {ext}")

        if ext == '.pdf':
            from doc_parse_convert.extraction.pdf import PDFProcessor
            processor = PDFProcessor(config)
        else:
            error_msg = f"Unsupported file format: {ext}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        processor.load(file_input)
        logger.info(f"Successfully created and loaded processor")
        return processor
