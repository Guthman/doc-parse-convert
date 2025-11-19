"""
Image type classification enum for image content extraction.
"""

from enum import Enum


class ImageType(str, Enum):
    """Classification of image content types."""

    TABLE = "table"
    """Data presented in rows and columns."""

    CHART_OR_GRAPH = "chart_or_graph"
    """Visual representations of data (line, bar, pie charts, etc.)."""

    DIAGRAM = "diagram"
    """Flowcharts, schemas, architecture diagrams, or technical drawings."""

    PHOTOGRAPH = "photograph"
    """Real-world photographs."""

    TEXT_BLOCK = "text_block"
    """Images that are primarily text content."""

    COMPOUND = "compound"
    """Images containing multiple distinct sub-parts (e.g., Figure 1a, 1b)."""

    OTHER = "other"
    """Any other type of image content."""
