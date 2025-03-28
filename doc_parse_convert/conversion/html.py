"""
HTML document conversion utilities.
"""

from pathlib import Path
import requests
from requests import Response

from doc_parse_convert.conversion.storage import upload_to_gcs, get_gcs_token


def convert_html_to_markdown(file_path: str | Path, jina_api_token: str, service_account_json: str, bucket_name: str) -> Response:
    """
    Converts HTML to Markdown by first uploading to GCS and then using Jina API.
    
    Args:
        file_path (str | Path): Path to the HTML file
        jina_api_token (str): API token for authentication
        service_account_json (str): Service account credentials JSON as a string
        bucket_name (str): GCS bucket to upload to
        
    Returns:
        Response: The API response from Jina
    """
    # Upload file to GCS and get public URL
    gcs_url = upload_to_gcs(file_path, service_account_json, bucket_name)
    
    # Call Jina API with the GCS URL
    url = f"https://r.jina.ai/{gcs_url}"
    headers = {
        'Authorization': f'Bearer {jina_api_token}',
        'X-With-Generated-Alt': 'true'
    }
    
    response = requests.get(url, headers=headers)
    return response
