"""
Cloud storage utilities for document processing.
"""

import json
from pathlib import Path
import requests
from google.oauth2 import service_account
import google.auth.transport.requests


def get_gcs_token(service_account_json: str) -> str:
    """
    Get an access token from a service account JSON string.
    
    Args:
        service_account_json (str): Service account credentials JSON as a string
        
    Returns:
        str: Access token for GCS API requests
    """
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


def upload_to_gcs(file_path: str | Path, service_account_json: str, bucket_name: str) -> str:
    """
    Uploads a file to Google Cloud Storage using the JSON API and returns the public URL.
    
    Args:
        file_path (str | Path): Path to the file to upload
        service_account_json (str): Service account credentials JSON as a string
        bucket_name (str): GCS bucket to upload to
        
    Returns:
        str: The public URL for accessing the uploaded file
    """
    upload_url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o"
    
    access_token = get_gcs_token(service_account_json)
    
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
