"""
Configuration for document processing.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

# Configure logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Determine log directory
log_dir = os.environ.get('DOC_PARSE_CONVERT_LOG_DIR')
if not log_dir:
    # Use a subdirectory in the system temp directory
    log_dir = os.path.join(tempfile.gettempdir(), 'doc_parse_convert_logs')
    
# Create log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Create file handler which logs even debug messages
log_file_path = os.path.join(log_dir, 'content_extraction.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Log the file location for user awareness
logger.info(f"Logging to file: {log_file_path}")

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
    # Google Cloud & Vertex AI configuration
    project_id: Optional[str] = None
    vertex_ai_location: Optional[str] = None
    gemini_model_name: Optional[str] = None
    service_account_file: Optional[str] = None
    use_application_default_credentials: bool = False
    
    # GCS & Jina configuration for HTML to Markdown conversion
    jina_api_token: Optional[str] = None
    gcs_bucket_name: Optional[str] = None
    service_account_json: Optional[str] = None  # Raw JSON string of service account credentials
    
    # Extraction strategy configuration
    toc_extraction_strategy: ExtractionStrategy = ExtractionStrategy.NATIVE  # Strategy for table of contents extraction
    content_extraction_strategy: ExtractionStrategy = ExtractionStrategy.NATIVE  # Strategy for chapter content extraction
    
    # Processing configuration
    max_pages_for_preview: int = 200  # Default is to only look at first 200 pages
    image_quality: int = 300  # DPI for image conversion
