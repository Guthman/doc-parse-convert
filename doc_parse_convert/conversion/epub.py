"""
EPUB document conversion utilities.
"""

import io
import zipfile
import tempfile
import shutil
import base64
import subprocess
import os
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union, BinaryIO

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from doc_parse_convert.config import logger, ProcessingConfig
from doc_parse_convert.exceptions import ConversionError


@contextmanager
def _ensure_file_path(file_input: Union[str, Path, BinaryIO], suffix: str = ""):
    """Context manager to ensure input is a file path.

    If input is a file-like object, it's saved to a temporary file.
    """
    if isinstance(file_input, (str, Path)):
        yield file_input
    elif hasattr(file_input, 'read'):
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(file_input, tmp)
                tmp_path = tmp.name
            yield tmp_path
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        raise TypeError(f"Input must be a file path or file-like object, not {type(file_input)}")


def convert_epub_to_html(file_input: Union[str, Path, BinaryIO]) -> List[str]:
    """
    Convert EPUB to HTML while preserving images as base64-encoded strings.

    Args:
        file_input: Path to the EPUB file or a file-like object.

                   SECURITY WARNING: If using file paths, YOU MUST
                   validate them against your secure base directory
                   BEFORE passing to this method.

    Returns:
        list[str]: List of HTML strings, with images encoded as base64
    """
    with _ensure_file_path(file_input, suffix=".epub") as file_path:
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


def convert_epub_to_txt(
    file_input: Union[str, Path, BinaryIO],
    output_file_path: str | Path = None
) -> Union[str, io.StringIO]:
    """
    Converts an EPUB file to plain text.

    Args:
        file_input: Path to the input EPUB file or a file-like object.
        output_file_path (str | Path, optional): Path to the output text file.
            If None, a file-like object (StringIO) is returned.

                   SECURITY WARNING: If using file paths, YOU MUST
                   validate them against your secure base directory
                   BEFORE passing to this method.

    Returns:
        str or io.StringIO: Extracted content as a string or a file-like object.
    """
    with _ensure_file_path(file_input, suffix=".epub") as input_file_path:
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


def extract_epub_css(
    file_input: Union[str, Path, BinaryIO],
    css_output_dir: str | Path
) -> Optional[Path]:
    """
    Extracts the first found CSS file from the EPUB archive.

    Args:
        file_input: Path to the EPUB file or a file-like object.
        css_output_dir (str | Path): Directory where extracted CSS will be saved.

                   SECURITY WARNING: If using file paths, YOU MUST
                   validate them against your secure base directory
                   BEFORE passing to this method.

    Returns:
        Path | None: The path to the extracted CSS file, or None if no CSS is found.
    """
    css_output_dir = Path(css_output_dir)
    css_output_dir.mkdir(parents=True, exist_ok=True)

    with _ensure_file_path(file_input, suffix=".epub") as epub_file:
        with zipfile.ZipFile(epub_file, 'r') as epub_zip:
            for file_name in epub_zip.namelist():
                if file_name.endswith('.css'):
                    extracted_css = css_output_dir / Path(file_name).name
                    with epub_zip.open(file_name) as css_file:
                        with open(extracted_css, 'wb') as output_css:
                            shutil.copyfileobj(css_file, output_css)
                    return extracted_css
    return None


def convert_epub_to_pdf(
    file_input: Union[str, Path, BinaryIO],
    output_file_path: str | Path = None,
    pdf_engine: str = "wkhtmltopdf",
    use_embedded_css: bool = True,
    standalone: bool = True,
    config: ProcessingConfig = None  # NEW: require config for security
) -> str:
    """
    Convert an EPUB file to PDF using Pandoc.

    Args:
        file_input: Path to the input EPUB file or a file-like object.
        output_file_path: Path to the output PDF file (optional)
        pdf_engine: The engine to use for PDF generation (default: 'wkhtmltopdf')
        use_embedded_css: Whether to use CSS embedded in the EPUB (default: True)
        standalone: If True, produces a standalone document (default: True)
        config: Processing configuration with validated pandoc executable.
                If None, uses default configuration.

    Returns:
        str: The path to the generated PDF file

    Raises:
        FileNotFoundError: If the input file does not exist
        ConversionError: If the Pandoc command fails

    Security Note:
        Pandoc executable is validated against a whitelist to prevent
        command injection attacks.
    """
    if config is None:
        config = ProcessingConfig()  # Use defaults with validation

    with _ensure_file_path(file_input, suffix=".epub") as input_file_path_str:
        input_file_path = Path(input_file_path_str)
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

        # Base Pandoc command using validated executable
        pandoc_cmd = [
            config.pandoc_executable,  # VALIDATED
            str(input_file_path),
            "-o",
            str(output_file_path)
        ]

        # Add PDF engine
        if pdf_engine == 'wkhtmltopdf':
            pdf_engine = "wkhtmltopdf"
        pandoc_cmd += ["--pdf-engine", pdf_engine]

        # Add CSS if extracted
        if css_file:
            pandoc_cmd += ["--css", str(css_file)]

        # Add standalone flag if required
        if standalone:
            pandoc_cmd.append("-s")

        try:
            subprocess.run(
                pandoc_cmd,
                check=True,
                timeout=config.subprocess_timeout
            )
            logger.info(f"Successfully converted {input_file_path} to {output_file_path}")
            return str(output_file_path)

        except subprocess.TimeoutExpired:
            raise ConversionError(
                f"Pandoc conversion timed out after {config.subprocess_timeout} seconds"
            )
        except subprocess.CalledProcessError as e:
            logger.error(' '.join(pandoc_cmd))
            raise ConversionError(f"Pandoc command failed: {e}")
