"""
Pydantic schemas and Gemini response schemas for image content extraction.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

from doc_parse_convert.image_extraction.image_types import ImageType


class TableData(BaseModel):
    """Extracted data from a table image."""

    markdown_content: str = Field(
        description="The full content of the table in GitHub-flavored Markdown format."
    )


class ChartData(BaseModel):
    """Extracted data from a chart or graph image."""

    chart_type: str = Field(
        description="The type of chart, e.g., 'line graph', 'bar chart', 'pie chart', 'scatter plot'."
    )
    summary: str = Field(
        description="A detailed summary of the chart, including its title, axes labels, trends, and key insights."
    )
    data_points: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Extracted numeric data points, if discernible with high confidence. E.g., [{'x': 4.5, 'y': 3.5}, ...]."
    )


class CompoundData(BaseModel):
    """Extracted data from a compound image with multiple sub-parts."""

    elements: List["ExtractedImageData"] = Field(
        description="A list of extraction results for each sub-image found."
    )


class ExtractedImageData(BaseModel):
    """Main result of image content extraction."""

    image_type: ImageType = Field(
        description="The classification of the image content."
    )
    description: str = Field(
        description="A brief, one-sentence description of the overall image."
    )
    content: Optional[Union[TableData, ChartData, CompoundData]] = Field(
        None,
        description="Structured data extracted from the image. Present for TABLE, CHART_OR_GRAPH, and COMPOUND types."
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Extraction quality score 0.0-1.0. Low values indicate blurry images or uncertain data."
    )


# Update forward references for recursive model
CompoundData.model_rebuild()


def get_image_extraction_schema() -> dict:
    """
    Get the response schema for image content extraction.

    Returns:
        dict: Image extraction response schema for Gemini API
    """
    return {
        "type": "OBJECT",
        "properties": {
            "image_type": {
                "type": "STRING",
                "enum": [t.value for t in ImageType]
            },
            "description": {
                "type": "STRING",
                "description": "A brief, one-sentence description of the overall image."
            },
            "confidence": {
                "type": "NUMBER",
                "description": "Extraction quality 0.0-1.0. Use 1.0 for clear images, 0.5 for partially readable, 0.0 for blurry/illegible."
            },
            "content": {
                "type": "OBJECT",
                "description": "Structured data extracted based on image type.",
                "properties": {
                    "markdown_content": {
                        "type": "STRING",
                        "description": "For TABLE type: full table in Markdown format."
                    },
                    "chart_type": {
                        "type": "STRING",
                        "description": "For CHART_OR_GRAPH type: the type of chart."
                    },
                    "summary": {
                        "type": "STRING",
                        "description": "For CHART_OR_GRAPH type: detailed summary of the chart."
                    },
                    "data_points": {
                        "type": "ARRAY",
                        "description": "For CHART_OR_GRAPH type: extracted numeric data points.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "label": {"type": "STRING"},
                                "x": {"type": "NUMBER"},
                                "y": {"type": "NUMBER"},
                                "value": {"type": "NUMBER"}
                            }
                        }
                    },
                    "elements": {
                        "type": "ARRAY",
                        "description": "For COMPOUND type: extraction results for each sub-image.",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "image_type": {
                                    "type": "STRING",
                                    "enum": [t.value for t in ImageType]
                                },
                                "description": {"type": "STRING"},
                                "content": {"type": "OBJECT"}
                            }
                        }
                    }
                }
            }
        },
        "required": ["image_type", "description"]
    }
