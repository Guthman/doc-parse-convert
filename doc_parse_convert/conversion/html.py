"""
HTML document conversion utilities.
"""

from pathlib import Path
import requests
from requests import Response

from doc_parse_convert.conversion.storage import upload_to_gcs
from doc_parse_convert.config import ProcessingConfig, logger


def convert_html_to_markdown(file_path: str | Path, config: ProcessingConfig) -> Response:
    """
    Converts HTML to Markdown by first uploading to GCS and then using Jina API.

    Args:
        file_path (str | Path): Path to the HTML file
        config (ProcessingConfig): Processing configuration with Jina and GCS settings

    Returns:
        Response: The API response from Jina

    Raises:
        ValueError: If required configuration parameters are missing
    """
    # Validate config
    if not config.jina_api_token:
        logger.error("Jina API token not provided in config")
        raise ValueError("Jina API token is required for HTML to Markdown conversion")

    if not config.service_account_json:
        logger.error("GCS service account JSON not provided in config")
        raise ValueError("Service account JSON is required for HTML to Markdown conversion")

    if not config.gcs_bucket_name:
        logger.error("GCS bucket name not provided in config")
        raise ValueError("GCS bucket name is required for HTML to Markdown conversion")

    # Upload file to GCS and get public URL
    logger.info(f"Uploading HTML file to GCS bucket: {config.gcs_bucket_name}")
    gcs_url = upload_to_gcs(file_path, config)

    # Call Jina API with the GCS URL
    logger.info("Calling Jina API for conversion")
    url = f"https://r.jina.ai/{gcs_url}"
    headers = {
        'Authorization': f'Bearer {config.jina_api_token}',
        'X-With-Generated-Alt': 'true'
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        logger.error(f"Jina API request failed with status code: {response.status_code}")
        logger.error(f"Response: {response.text}")
    else:
        logger.info("Successfully converted HTML to Markdown")

    return response
