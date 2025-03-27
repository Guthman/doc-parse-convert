"""Guthman Document Extraction Utilities

A package for document parsing and conversion.
"""

__version__ = "0.3.0"

# Expose key modules and classes
from doc_parse_convert.content_extraction import (
    ProcessingConfig,
    ExtractionStrategy,
    Chapter,
    ChapterContent,
    PageContent,
    TextBox,
    Table,
    Figure,
    PDFProcessor,
    ProcessorFactory
)

from doc_parse_convert.content_conversion import (
    convert_epub_to_html,
    convert_epub_to_txt,
    convert_epub_to_pdf,
    convert_html_to_markdown
)
