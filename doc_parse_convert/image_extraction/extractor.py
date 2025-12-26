"""
Image content extractor for analyzing and extracting structured data from images.
"""

import base64
import json
import os
from pathlib import Path
from typing import Union

from vertexai.generative_models import Part

from doc_parse_convert.ai.client import AIClient
from doc_parse_convert.config import ProcessingConfig, logger
from doc_parse_convert.exceptions import AIExtractionError
from doc_parse_convert.image_extraction.image_types import ImageType
from doc_parse_convert.image_extraction.schemas import (
    ExtractedImageData,
    TableData,
    ChartData,
    CompoundData,
    get_image_extraction_schema,
)
from doc_parse_convert.image_extraction.prompts import get_image_extraction_prompt

# Supported image extensions and their MIME types
SUPPORTED_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ImageContentExtractor:
    """
    Extracts structured content from images using AI.

    Supports extraction from:
    - Raw image bytes
    - Base64-encoded image data
    - Local file paths (PNG, JPG, JPEG, etc.)

    Example:
        >>> from doc_parse_convert.config import ProcessingConfig
        >>> from doc_parse_convert.ai.client import AIClient
        >>> from doc_parse_convert.image_extraction import ImageContentExtractor
        >>>
        >>> config = ProcessingConfig(
        ...     project_id="your-project",
        ...     use_application_default_credentials=True
        ... )
        >>> ai_client = AIClient(config)
        >>> extractor = ImageContentExtractor(ai_client, config)
        >>>
        >>> # Extract from file path
        >>> result = extractor.extract("path/to/image.png")
        >>> print(result.image_type)
        >>> print(result.description)
    """

    def __init__(self, ai_client: AIClient, config: ProcessingConfig):
        """
        Initialize the image content extractor.

        Args:
            ai_client: AIClient instance for Gemini API calls
            config: Processing configuration
        """
        self.ai_client = ai_client
        self.config = config

    def extract(
            self,
            image: Union[bytes, str, Path],
            mime_type: str = None,
            metadata: dict = None
    ) -> ExtractedImageData:
        """
        Analyze an image and extract structured content.

        Args:
            image: Image data as one of:
                - bytes: Raw image bytes
                - str: Either a file path or base64-encoded image data
                - Path: Path object to an image file
            mime_type: MIME type of the image. Required for bytes/base64 input,
                      auto-detected for file paths.
            metadata: Optional dictionary containing contextual information about the image.
                     Can include fields like 'chapter', 'page_number', 'document_title', etc.

        Returns:
            ExtractedImageData with classification and extracted content

        Raises:
            AIExtractionError: If extraction fails
            ValueError: If image input is invalid or mime_type cannot be determined
            FileNotFoundError: If file path doesn't exist
        """
        image_bytes, detected_mime_type = self._normalize_image_input(image, mime_type)

        logger.info(f"Extracting content from image ({detected_mime_type})")
        logger.debug(f"Image size: {len(image_bytes)} bytes")
        if metadata:
            logger.debug(f"Metadata: {metadata}")

        # Create image part for Gemini
        image_part = Part.from_data(data=image_bytes, mime_type=detected_mime_type)

        # Get prompt and schema
        prompt = get_image_extraction_prompt(metadata)
        response_schema = get_image_extraction_schema()

        # Create parts list
        parts = [image_part, Part.from_text(prompt)]

        try:
            response = self.ai_client._call_model_with_retry(
                parts,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=response_schema
            )

            # Parse response
            response_data = json.loads(response.text)
            logger.debug(f"Raw response: {response_data}")

            # Convert to Pydantic model
            result = self._parse_response(response_data)

            logger.info(f"Successfully extracted content: type={result.image_type.value}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise AIExtractionError(f"Invalid JSON response from AI: {e}") from e
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            raise AIExtractionError(f"Failed to extract image content: {e}") from e

    def _normalize_image_input(
            self,
            image: Union[bytes, str, Path],
            mime_type: str = None
    ) -> tuple[bytes, str]:
        """
        Normalize various image input formats to bytes and mime type.

        Args:
            image: Image input in various formats
            mime_type: Optional MIME type override

        Returns:
            Tuple of (image_bytes, mime_type)

        Raises:
            ValueError: If input is invalid or mime_type cannot be determined
            FileNotFoundError: If file path doesn't exist
        """
        # Handle Path objects
        if isinstance(image, Path):
            return self._load_from_file(image, mime_type)

        # Handle bytes
        if isinstance(image, bytes):
            if not mime_type:
                raise ValueError(
                    "mime_type is required when providing raw bytes. "
                    "Use 'image/png', 'image/jpeg', etc."
                )
            return image, mime_type

        # Handle string - could be file path or base64
        if isinstance(image, str):
            # Check if it's a file path
            if os.path.exists(image):
                return self._load_from_file(Path(image), mime_type)

            # Try to decode as base64
            try:
                image_bytes = base64.b64decode(image)
            except Exception:
                raise ValueError(
                    f"String input is not a valid file path or base64 data: {image[:50]}..."
                )

            if not mime_type:
                raise ValueError(
                    "mime_type is required when providing base64 data. "
                    "Use 'image/png', 'image/jpeg', etc."
                )
            return image_bytes, mime_type

        raise ValueError(f"Unsupported image input type: {type(image)}")

    def _load_from_file(self, path: Path, mime_type: str = None) -> tuple[bytes, str]:
        """
        Load image from file path.

        Args:
            path: Path to image file
            mime_type: Optional MIME type override

        Returns:
            Tuple of (image_bytes, mime_type)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension is not supported
        """
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        # Determine MIME type from extension if not provided
        if not mime_type:
            ext = path.suffix.lower()
            mime_type = SUPPORTED_IMAGE_TYPES.get(ext)
            if not mime_type:
                supported = ", ".join(SUPPORTED_IMAGE_TYPES.keys())
                raise ValueError(
                    f"Unsupported image extension '{ext}'. "
                    f"Supported: {supported}"
                )

        logger.debug(f"Loading image from: {path}")
        with open(path, "rb") as f:
            image_bytes = f.read()

        return image_bytes, mime_type

    def _parse_response(self, response_data: dict) -> ExtractedImageData:
        """
        Parse the raw API response into a Pydantic model.

        Args:
            response_data: Raw JSON response from Gemini

        Returns:
            ExtractedImageData model
        """
        image_type = ImageType(response_data["image_type"])
        description = response_data["description"]
        confidence = response_data.get("confidence")
        content_data = response_data.get("content")

        content = None
        if content_data:
            if image_type == ImageType.TABLE:
                content = TableData(
                    markdown_content=content_data.get("markdown_content", "")
                )
            elif image_type == ImageType.CHART_OR_GRAPH:
                content = ChartData(
                    chart_type=content_data.get("chart_type", "unknown"),
                    summary=content_data.get("summary", ""),
                    data_points=content_data.get("data_points")
                )
            elif image_type == ImageType.COMPOUND:
                elements = []
                for elem in content_data.get("elements", []):
                    elements.append(self._parse_response(elem))
                content = CompoundData(elements=elements)

        return ExtractedImageData(
            image_type=image_type,
            description=description,
            content=content,
            confidence=confidence
        )
