from pathlib import Path
import subprocess
import io
import zipfile
import tempfile
import shutil
import base64
import json
import requests
from requests import Response
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
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


def convert_epub_to_html(file_path: str | Path) -> list[str]:
    """
    Convert EPUB to HTML while preserving images as base64-encoded strings within the HTML.
    
    Args:
        file_path (str | Path): Path to the EPUB file
        
    Returns:
        list[str]: List of HTML strings, with images encoded as base64 within the HTML
    """
    book = epub.read_epub(file_path)
    output_html = []
    
    # Create a mapping of image IDs to their base64 encoded content
    image_map = {}
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            # Get image content and encode as base64
            image_content = item.get_content()
            b64_content = base64.b64encode(image_content).decode('utf-8')
            # Get the media type (e.g., 'image/jpeg', 'image/png')
            media_type = item.media_type
            # Store with multiple key variations to match possible paths
            image_name = item.get_name()
            # Store the full path
            image_map[image_name] = f'data:{media_type};base64,{b64_content}'
            # Store just the filename
            image_map[Path(image_name).name] = f'data:{media_type};base64,{b64_content}'
            # Store without 'images/' prefix if it exists
            if 'images/' in image_name:
                image_map[image_name.replace('images/', '')] = f'data:{media_type};base64,{b64_content}'

    # Process HTML documents
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find all images and replace src with base64 data
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    # Remove any parent directory references
                    clean_src = src.replace('../', '').replace('./', '')
                    
                    # Try different path variations
                    if clean_src in image_map:
                        img['src'] = image_map[clean_src]
                    elif Path(clean_src).name in image_map:
                        img['src'] = image_map[Path(clean_src).name]
                    elif clean_src.replace('images/', '') in image_map:
                        img['src'] = image_map[clean_src.replace('images/', '')]
            
            output_html.append(str(soup))
            
    return output_html


def convert_epub_to_txt(input_file_path: str | Path,
                        output_file_path: str | Path = None) -> str | io.StringIO:
    """
    Converts an EPUB file to plain text.

    If an output file path is provided, the text is written to that file and
    the raw content string is returned. If no output path is provided,
    the function returns a StringIO file-like object containing the text.

    Args:
        input_file_path (str | Path): Path to the input EPUB file.
        output_file_path (str | Path, optional): Path to the output text file.
            If None, a file-like object (StringIO) is returned instead of writing to disk.

    Returns:
        str: The extracted content as a single string.
        io.StringIO: A file-like object containing the text content,
            returned only if `output_file_path` is None.

    Raises:
        FileNotFoundError: If the input EPUB file does not exist.
        ValueError: If the input file is not a valid EPUB file.

    Example:
        # To save the text to a file
        convert_epub_to_txt('input.epub', 'output.txt')

        # To get the content as a file-like object
        content_file = convert_epub_to_txt('input.epub')
        print(content_file.getvalue())
    """
    book = epub.read_epub(input_file_path)
    filename = Path(input_file_path).stem
    content = []

    # Extract text content from the EPUB file
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            body_content = item.get_body_content().decode()
            soup = BeautifulSoup(body_content, features='lxml')
            text = soup.get_text(separator='\n', strip=True)
            content.append(text)

    content_str = '\n'.join(content)

    # If no output path provided, return a new StringIO object
    if output_file_path is None:
        output_file = io.StringIO()
        output_file.write(content_str)
        output_file.seek(0)
        return output_file

    # Otherwise, write to the output file path
    with open(f'{output_file_path}/{filename}.txt', 'w', encoding='utf-8') as f:
        f.write(content_str)

    return content_str


def extract_epub_css(epub_file: str | Path, css_output_dir: str | Path) -> Path | None:
    """
    Extracts the first found CSS file from the EPUB archive.

    Args:
        epub_file (str | Path): Path to the EPUB file.
        css_output_dir (str | Path): Directory where extracted CSS will be saved.

    Returns:
        Path | None: The path to the extracted CSS file, or None if no CSS is found.
    """
    css_output_dir = Path(css_output_dir)
    css_output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(epub_file, 'r') as epub_zip:
        for file_name in epub_zip.namelist():
            if file_name.endswith('.css'):
                extracted_css = css_output_dir / Path(file_name).name
                with epub_zip.open(file_name) as css_file:
                    with open(extracted_css, 'wb') as output_css:
                        shutil.copyfileobj(css_file, output_css)
                return extracted_css
    return None


def convert_epub_to_pdf(input_file_path: str | Path,
                        output_file_path: str | Path = None,
                        pdf_engine: str = "wkhtmltopdf",
                        use_embedded_css: bool = True,
                        standalone: bool = True,
                        pandoc_executable_path: str = "pandoc") -> str:
    """
    Convert an EPUB file to PDF using Pandoc, optionally using embedded CSS from the EPUB.

    Args:
        input_file_path (str | Path): Path to the input EPUB file.
        output_file_path (str | Path, optional): Path to the output PDF file. Defaults to the same name as the input with a .pdf extension.
        pdf_engine (str, optional): The engine to use for PDF generation. Defaults to 'wkhtmltopdf'.
        use_embedded_css (bool, optional): Whether to use CSS embedded in the EPUB file for styling. Defaults to True.
        standalone (bool, optional): If True, produces a standalone document with a title page, etc. Defaults to True.
        pandoc_executable_path (str | Path, optional): The path to the Pandoc executable. Defaults to "pandoc" (expects it to be in PATH).

    Returns:
        str: The path to the generated PDF file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        subprocess.CalledProcessError: If the Pandoc command fails.
    """

    input_file_path = Path(input_file_path)
    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file '{input_file_path}' not found.")

    # Set default output path if not provided
    if output_file_path is None:
        output_file_path = input_file_path.with_suffix('.pdf')
    filename = Path(input_file_path).stem
    output_file_path = Path(f'{output_file_path}/{filename}.pdf')

    # Extract CSS from the EPUB if requested
    css_file = None
    if use_embedded_css:
        with tempfile.TemporaryDirectory() as temp_dir:
            css_file = extract_epub_css(input_file_path, temp_dir)

    # Base Pandoc command, using the provided pandoc executable path
    pandoc_cmd = [str(pandoc_executable_path), str(input_file_path), "-o", str(output_file_path)]

    # Add PDF engine (LaTeX-based or other)
    if pdf_engine == 'wkhtmltopdf':
        pdf_engine = "wkhtmltopdf"  # Expect it to be in PATH
    pandoc_cmd += ["--pdf-engine", pdf_engine]

    # Add CSS if extracted or embedded CSS is found
    if css_file:
        pandoc_cmd += ["--css", str(css_file)]

    # Add standalone flag if required
    if standalone:
        pandoc_cmd.append("-s")

    try:
        # Run the Pandoc command
        subprocess.run(pandoc_cmd, check=True)
        print(f"Successfully converted {input_file_path} to {output_file_path}")
        return str(output_file_path)

    except subprocess.CalledProcessError as e:
        print(' '.join(pandoc_cmd))
        raise RuntimeError(f"Pandoc command failed: {e}")
