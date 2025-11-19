"""
Document parsing and conversion utilities.
"""

# Import main components to expose at package level
from doc_parse_convert.config import ProcessingConfig, ExtractionStrategy, logger
from doc_parse_convert.models.document import Chapter, DocumentSection
from doc_parse_convert.models.content import (
    ChapterContent, PageContent, TextBox, Table, Figure
)
from doc_parse_convert.extraction import (
    DocumentProcessor, PDFProcessor, DocumentStructureExtractor
)
# Import utilities separately to avoid circular imports
from doc_parse_convert.utils.image import ImageConverter
from doc_parse_convert.utils.factory import ProcessorFactory
from doc_parse_convert.conversion import (
    convert_epub_to_html,
    convert_epub_to_txt,
    convert_epub_to_pdf,
    convert_html_to_markdown
)

__version__ = "0.6.0"
