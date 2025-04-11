"""
Document content models.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Figure:
    """Represents a figure in a document."""
    description: Optional[str] = None
    byline: Optional[str] = None


@dataclass
class TextBox:
    """Represents a text box or side note in a document."""
    content: str
    type: str  # e.g., 'text_box', 'side_note', 'callout'


@dataclass
class Table:
    """Represents a table in a document."""
    content: str
    caption: Optional[str] = None


@dataclass
class PageContent:
    """Represents the structured content of a page."""
    chapter_text: str
    text_boxes: List[TextBox] = None
    tables: List[Table] = None
    figures: List[Figure] = None

    def __post_init__(self):
        self.text_boxes = self.text_boxes or []
        self.tables = self.tables or []
        self.figures = self.figures or []


@dataclass
class ChapterContent:
    """Represents the structured content of a chapter."""
    title: str
    pages: List[PageContent]
    start_page: int
    end_page: int
