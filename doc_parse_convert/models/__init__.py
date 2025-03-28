"""
Data models for document processing.
"""

from typing import TYPE_CHECKING

# Bring all models to the models namespace
from doc_parse_convert.models.document import Chapter, DocumentSection
from doc_parse_convert.models.content import (
    Figure, TextBox, Table, PageContent, ChapterContent
)

# Resolve circular references
if TYPE_CHECKING:
    from doc_parse_convert.models.content import ChapterContent
