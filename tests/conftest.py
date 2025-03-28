"""
Shared fixtures for the test suite.
"""

import os
import pytest
from pathlib import Path

# Import directly from the modules instead of content_extraction
from doc_parse_convert.config import ProcessingConfig, ExtractionStrategy
from doc_parse_convert.utils.factory import ProcessorFactory
from doc_parse_convert.ai.client import AIClient


@pytest.fixture(scope="session")
def test_files_dir():
    """Return the path to the test files directory."""
    # Get the path to the test_files directory
    return Path(__file__).parent.parent / "test_files"


@pytest.fixture(scope="session")
def pdf_sample_path(test_files_dir):
    """Return the path to the PDF sample file."""
    pdf_path = test_files_dir / "LLMAll_en-US_FINAL.pdf"
    if not pdf_path.exists():
        pytest.skip(f"PDF sample file not found at {pdf_path}")
    return str(pdf_path)


@pytest.fixture(scope="session")
def epub_sample_path(test_files_dir):
    """Return the path to the EPUB sample file."""
    epub_path = test_files_dir / "Quick Start Guide - John Schember.epub"
    if not epub_path.exists():
        pytest.skip(f"EPUB sample file not found at {epub_path}")
    return str(epub_path)


@pytest.fixture(scope="session")
def vertex_credentials():
    """Get Vertex AI credentials from environment variables.
    
    Skips tests if credentials are not available.
    """
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        pytest.skip("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    
    if not os.path.exists(creds_path):
        pytest.skip(f"Credentials file not found at {creds_path}")
        
    return creds_path


@pytest.fixture(scope="session")
def jina_api_key():
    """Get Jina API key from environment variables.
    
    Skips tests if API key is not available.
    """
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        pytest.skip("JINA_API_KEY environment variable not set")
    return api_key


@pytest.fixture(scope="session")
def processing_config(vertex_credentials):
    """Create a processing configuration for the tests."""
    # Extract project ID and location from credentials file
    import json
    with open(vertex_credentials, 'r') as f:
        creds_data = json.load(f)
    
    project_id = creds_data.get('project_id')
    
    # Create and return the config
    config = ProcessingConfig(
        project_id=project_id,
        vertex_ai_location="us-central1",  # Default location, adjust if needed
        gemini_model_name="gemini-1.5-flash-002",
        service_account_file=vertex_credentials,
        toc_extraction_strategy=ExtractionStrategy.AI,
        content_extraction_strategy=ExtractionStrategy.AI,
    )
    return config


@pytest.fixture(scope="function")
def pdf_processor(processing_config, pdf_sample_path):
    """Create a PDF processor with the sample PDF file."""
    processor = ProcessorFactory.create_processor(pdf_sample_path, processing_config)
    # Use a try-finally block to ensure the processor is closed
    try:
        yield processor
    finally:
        processor.close()


@pytest.fixture(scope="session")
def ai_client(processing_config):
    """Create an AI client for testing."""
    return AIClient(processing_config)


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    return tmp_path
