"""
Cloud storage utilities for document processing.
"""

import json
from pathlib import Path
import requests
from google.oauth2 import service_account
import google.auth.transport.requests

from doc_parse_convert.config import ProcessingConfig, logger


def get_gcs_token(service_account_json: str) -> str:
    """
    Get an access token from a service account JSON string.

    Args:
        service_account_json (str): Service account credentials JSON as a string

    Returns:
        str: Access token for GCS API requests

    Raises:
        ValueError: If the service account JSON is invalid
    """
    try:
        # Parse the service account JSON string
        credentials_info = json.loads(service_account_json)

        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://www.googleapis.com/auth/devstorage.read_write']
        )

        # Get token
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)

        return credentials.token
    except json.JSONDecodeError as e:
        logger.error(f"Invalid service account JSON: {str(e)}")
        raise ValueError(f"Invalid service account JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting GCS token: {str(e)}")
        raise


def upload_to_gcs(file_path: str | Path, config: ProcessingConfig) -> str:
    """
    Uploads a file to Google Cloud Storage using the JSON API and returns the public URL.

    Args:
        file_path (str | Path): Path to the file to upload
        config (ProcessingConfig): Processing configuration with GCS settings

    Returns:
        str: The public URL for accessing the uploaded file

    Raises:
        ValueError: If required configuration parameters are missing
    """
    if not config.service_account_json:
        logger.error("Service account JSON not provided in config")
        raise ValueError("Service account JSON is required for GCS upload")

    if not config.gcs_bucket_name:
        logger.error("GCS bucket name not provided in config")
        raise ValueError("GCS bucket name is required for GCS upload")

    bucket_name = config.gcs_bucket_name
    upload_url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o"

    logger.info(f"Getting GCS token for bucket: {bucket_name}")
    access_token = get_gcs_token(config.service_account_json)

    with open(file_path, 'rb') as f:
        file_content = f.read()

    params = {
        'name': Path(file_path).name,
        'uploadType': 'media'
    }

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'text/html',
        'Content-Length': str(len(file_content))
    }

    upload_response = requests.post(
        upload_url,
        params=params,
        headers=headers,
        data=file_content
    )

    if upload_response.status_code != 200:
        raise Exception(f"Failed to upload file: {upload_response.text}")

    # Return the public URL
    object_name = Path(file_path).name
    public_url = f"https://storage.googleapis.com/{bucket_name}/{object_name}"
    return public_url
