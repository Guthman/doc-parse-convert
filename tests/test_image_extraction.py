"""
Tests for the image content extraction module.
"""

import base64
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from doc_parse_convert.image_extraction import (
    ImageType,
    ImageContentExtractor,
    ExtractedImageData,
    TableData,
    ChartData,
    CompoundData,
)
from doc_parse_convert.image_extraction.schemas import get_image_extraction_schema
from doc_parse_convert.image_extraction.extractor import SUPPORTED_IMAGE_TYPES
from doc_parse_convert.config import ProcessingConfig
from doc_parse_convert.ai.client import AIClient


# --- Fixtures ---

@pytest.fixture
def test_files_dir():
    """Return the path to the test files directory."""
    return Path(__file__).parent.parent / "test_files"


@pytest.fixture
def table_image_path(test_files_dir):
    """Return the path to the table test image."""
    image_path = test_files_dir / "table_test.png"
    if not image_path.exists():
        pytest.skip(f"Table test image not found at {image_path}")
    return image_path


@pytest.fixture
def mock_ai_client():
    """Create a mock AI client for unit tests."""
    mock_client = Mock(spec=AIClient)
    mock_client.config = Mock()
    return mock_client


@pytest.fixture
def mock_config():
    """Create a mock processing config."""
    return Mock(spec=ProcessingConfig)


@pytest.fixture
def extractor(mock_ai_client, mock_config):
    """Create an ImageContentExtractor with mocked dependencies."""
    return ImageContentExtractor(mock_ai_client, mock_config)


# --- Unit Tests ---

class TestImageType:
    """Tests for the ImageType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert ImageType.TABLE.value == "table"
        assert ImageType.CHART_OR_GRAPH.value == "chart_or_graph"
        assert ImageType.DIAGRAM.value == "diagram"
        assert ImageType.PHOTOGRAPH.value == "photograph"
        assert ImageType.TEXT_BLOCK.value == "text_block"
        assert ImageType.COMPOUND.value == "compound"
        assert ImageType.OTHER.value == "other"

    def test_enum_is_string(self):
        """Test that ImageType inherits from str."""
        assert isinstance(ImageType.TABLE, str)
        assert ImageType.TABLE == "table"


class TestSchemas:
    """Tests for Pydantic schemas."""

    def test_table_data(self):
        """Test TableData schema."""
        data = TableData(markdown_content="| A | B |\n|---|---|\n| 1 | 2 |")
        assert data.markdown_content == "| A | B |\n|---|---|\n| 1 | 2 |"

    def test_chart_data_with_data_points(self):
        """Test ChartData schema with data points."""
        data = ChartData(
            chart_type="line graph",
            summary="A line graph showing growth over time.",
            data_points=[{"x": 1, "y": 10}, {"x": 2, "y": 20}]
        )
        assert data.chart_type == "line graph"
        assert data.summary == "A line graph showing growth over time."
        assert len(data.data_points) == 2

    def test_chart_data_without_data_points(self):
        """Test ChartData schema without data points (optional field)."""
        data = ChartData(
            chart_type="bar chart",
            summary="A bar chart showing categories."
        )
        assert data.data_points is None

    def test_extracted_image_data(self):
        """Test ExtractedImageData schema."""
        data = ExtractedImageData(
            image_type=ImageType.TABLE,
            description="A table showing fertilizer data.",
            content=TableData(markdown_content="| Col1 | Col2 |")
        )
        assert data.image_type == ImageType.TABLE
        assert data.description == "A table showing fertilizer data."
        assert isinstance(data.content, TableData)

    def test_compound_data(self):
        """Test CompoundData with nested elements."""
        element1 = ExtractedImageData(
            image_type=ImageType.CHART_OR_GRAPH,
            description="Sub-chart 1",
            content=ChartData(chart_type="line", summary="First chart")
        )
        element2 = ExtractedImageData(
            image_type=ImageType.CHART_OR_GRAPH,
            description="Sub-chart 2",
            content=ChartData(chart_type="line", summary="Second chart")
        )
        compound = CompoundData(elements=[element1, element2])
        assert len(compound.elements) == 2

    def test_get_image_extraction_schema(self):
        """Test that the Gemini schema is valid."""
        schema = get_image_extraction_schema()
        assert schema["type"] == "OBJECT"
        assert "image_type" in schema["properties"]
        assert "description" in schema["properties"]
        assert "content" in schema["properties"]
        assert "image_type" in schema["required"]
        assert "description" in schema["required"]


class TestImageContentExtractor:
    """Tests for the ImageContentExtractor class."""

    def test_supported_image_types(self):
        """Test that common image formats are supported."""
        assert ".png" in SUPPORTED_IMAGE_TYPES
        assert ".jpg" in SUPPORTED_IMAGE_TYPES
        assert ".jpeg" in SUPPORTED_IMAGE_TYPES
        assert SUPPORTED_IMAGE_TYPES[".png"] == "image/png"
        assert SUPPORTED_IMAGE_TYPES[".jpg"] == "image/jpeg"

    def test_load_from_file(self, extractor, table_image_path):
        """Test loading image from file path."""
        image_bytes, mime_type = extractor._load_from_file(table_image_path)

        assert len(image_bytes) > 0
        assert mime_type == "image/png"
        # Check PNG magic bytes
        assert image_bytes[:8] == b'\x89PNG\r\n\x1a\n'

    def test_load_from_file_not_found(self, extractor):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            extractor._load_from_file(Path("/nonexistent/image.png"))

    def test_load_from_file_unsupported_extension(self, extractor, tmp_path):
        """Test error for unsupported file extension."""
        fake_file = tmp_path / "image.xyz"
        fake_file.write_bytes(b"fake data")

        with pytest.raises(ValueError, match="Unsupported image extension"):
            extractor._load_from_file(fake_file)

    def test_normalize_bytes_input(self, extractor):
        """Test normalizing bytes input."""
        test_bytes = b"fake image data"

        # Should require mime_type
        with pytest.raises(ValueError, match="mime_type is required"):
            extractor._normalize_image_input(test_bytes)

        # Should work with mime_type
        result_bytes, result_mime = extractor._normalize_image_input(
            test_bytes, mime_type="image/png"
        )
        assert result_bytes == test_bytes
        assert result_mime == "image/png"

    def test_normalize_base64_input(self, extractor):
        """Test normalizing base64 input."""
        original_bytes = b"fake image data"
        base64_str = base64.b64encode(original_bytes).decode()

        # Should require mime_type
        with pytest.raises(ValueError, match="mime_type is required"):
            extractor._normalize_image_input(base64_str)

        # Should work with mime_type
        result_bytes, result_mime = extractor._normalize_image_input(
            base64_str, mime_type="image/jpeg"
        )
        assert result_bytes == original_bytes
        assert result_mime == "image/jpeg"

    def test_normalize_path_input(self, extractor, table_image_path):
        """Test normalizing Path object input."""
        result_bytes, result_mime = extractor._normalize_image_input(table_image_path)

        assert len(result_bytes) > 0
        assert result_mime == "image/png"

    def test_normalize_string_path_input(self, extractor, table_image_path):
        """Test normalizing string file path input."""
        result_bytes, result_mime = extractor._normalize_image_input(str(table_image_path))

        assert len(result_bytes) > 0
        assert result_mime == "image/png"

    def test_parse_response_table(self, extractor):
        """Test parsing a table response."""
        response_data = {
            "image_type": "table",
            "description": "A table with data",
            "content": {
                "markdown_content": "| A | B |\n|---|---|\n| 1 | 2 |"
            }
        }

        result = extractor._parse_response(response_data)

        assert result.image_type == ImageType.TABLE
        assert result.description == "A table with data"
        assert isinstance(result.content, TableData)
        assert "| A | B |" in result.content.markdown_content

    def test_parse_response_chart(self, extractor):
        """Test parsing a chart response."""
        response_data = {
            "image_type": "chart_or_graph",
            "description": "A line graph",
            "content": {
                "chart_type": "line graph",
                "summary": "Shows trends over time",
                "data_points": [{"x": 1, "y": 10}]
            }
        }

        result = extractor._parse_response(response_data)

        assert result.image_type == ImageType.CHART_OR_GRAPH
        assert isinstance(result.content, ChartData)
        assert result.content.chart_type == "line graph"
        assert len(result.content.data_points) == 1

    def test_parse_response_no_content(self, extractor):
        """Test parsing a response without content (e.g., photograph)."""
        response_data = {
            "image_type": "photograph",
            "description": "A photo of a plant"
        }

        result = extractor._parse_response(response_data)

        assert result.image_type == ImageType.PHOTOGRAPH
        assert result.content is None


# --- Integration Tests (require API credentials) ---

@pytest.fixture
def image_extractor(processing_config):
    """Create an ImageContentExtractor with real AI client."""
    ai_client = AIClient(processing_config)
    return ImageContentExtractor(ai_client, processing_config)


class TestImageExtractionIntegration:
    """Integration tests that make real API calls.

    These tests require GOOGLE_APPLICATION_CREDENTIALS to be set.
    They will incur API costs.
    """

    def test_extract_table_from_file(self, image_extractor, table_image_path):
        """Test extracting a table from the test image file."""
        result = image_extractor.extract(table_image_path)

        # Verify classification
        assert result.image_type == ImageType.TABLE
        assert result.description  # Should have a description

        # Verify table content
        assert result.content is not None
        assert isinstance(result.content, TableData)
        assert result.content.markdown_content  # Should have markdown

        # Check for expected table content
        markdown = result.content.markdown_content.lower()
        assert "calcium" in markdown or "carbonate" in markdown
        assert "fertilizer" in markdown or "nitrogen" in markdown

    def test_extract_table_from_bytes(self, image_extractor, table_image_path):
        """Test extracting from raw bytes."""
        with open(table_image_path, "rb") as f:
            image_bytes = f.read()

        result = image_extractor.extract(image_bytes, mime_type="image/png")

        assert result.image_type == ImageType.TABLE
        assert result.content is not None

    def test_extract_table_from_base64(self, image_extractor, table_image_path):
        """Test extracting from base64-encoded data."""
        with open(table_image_path, "rb") as f:
            image_bytes = f.read()

        base64_data = base64.b64encode(image_bytes).decode()

        result = image_extractor.extract(base64_data, mime_type="image/png")

        assert result.image_type == ImageType.TABLE
        assert result.content is not None
