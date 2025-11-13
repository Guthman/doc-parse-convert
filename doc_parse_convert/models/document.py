"""
Document structure models.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, ForwardRef, TYPE_CHECKING

# Use string annotation for forward reference
if TYPE_CHECKING:
    from doc_parse_convert.models.content import ChapterContent
else:
    ChapterContent = ForwardRef('ChapterContent')


@dataclass
class Chapter:
    """Represents a chapter in a document."""
    title: str
    start_page: int
    end_page: Optional[int] = None
    level: int = 1
    content: Optional[ChapterContent] = None


@dataclass
class DocumentSection:
    """Represents a section in a document structure hierarchy with physical and logical page information."""
    title: str
    start_page: int  # 0-based physical page index
    end_page: Optional[int] = None  # 0-based physical page index
    level: int = 0  # Depth in the document hierarchy (0 for document root, 1 for chapters, etc.)
    children: List["DocumentSection"] = None  # Subsections
    logical_start_page: Optional[int] = None  # As displayed in the document (e.g., "Page 1")
    logical_end_page: Optional[int] = None  # As displayed in the document
    section_type: Optional[str] = None  # E.g., "chapter", "section", "appendix"
    identifier: Optional[str] = None  # E.g., "Chapter 1", "Appendix A"

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

    def add_child(self, child: "DocumentSection") -> None:
        """Add a child section to this section."""
        if self.children is None:
            self.children = []
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the section to a dictionary representation."""
        result = {
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "level": self.level,
        }

        if self.logical_start_page is not None:
            result["logical_start_page"] = self.logical_start_page
        if self.logical_end_page is not None:
            result["logical_end_page"] = self.logical_end_page
        if self.section_type:
            result["section_type"] = self.section_type
        if self.identifier:
            result["identifier"] = self.identifier
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]

        return result
