"""
Configuration for document processing.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create file handler which logs even debug messages
file_handler = logging.FileHandler('content_extraction.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Safety settings for Gemini model
GEMINI_SAFETY_CONFIG = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


class ExtractionStrategy(Enum):
    """Strategies for extracting content from documents."""
    NATIVE = "native"
    AI = "ai"
    OCR = "ocr"


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    project_id: Optional[str] = None
    vertex_ai_location: Optional[str] = None
    gemini_model_name: Optional[str] = None
    service_account_file: Optional[str] = None
    use_application_default_credentials: bool = False
    toc_extraction_strategy: ExtractionStrategy = ExtractionStrategy.NATIVE  # Strategy for table of contents extraction
    content_extraction_strategy: ExtractionStrategy = ExtractionStrategy.NATIVE  # Strategy for chapter content extraction
    max_pages_for_preview: int = 200  # Default is to only look at first 200 pages
    image_quality: int = 300  # DPI for image conversion
