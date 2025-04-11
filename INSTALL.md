# Installation Guide for doc_parse_convert

This guide explains how to install and use the `doc_parse_convert` package in other projects.

## Installation Options

### Option 1: Local Development Installation

If you're actively developing the package or want to use it from the local source:

```bash
# Navigate to the package directory
cd path/to/doc-parse-convert

# Install in development mode
pip install -e .
```

This creates a symbolic link to your source code, so any changes you make to the package are immediately available without reinstalling.

### Option 2: Install from GitHub

To install directly from a GitHub repository (once you've pushed it):

```bash
pip install git+https://github.com/Guthman/doc-parse-convert.git
```

### Option 3: Install from PyPI (future option)

If you later publish the package to PyPI:

```bash
pip install doc_parse_convert
```

## Usage Examples

After installation, you can import the package in your Python scripts:

```python
from doc_parse_convert import ProcessingConfig, PDFProcessor
from doc_parse_convert import convert_epub_to_html

# See example.py for more detailed usage examples
```

## Dependencies

### Python Dependencies

The package automatically installs all required Python dependencies, including:

- pymupdf (PyMuPDF)
- google-cloud-aiplatform
- pillow
- tenacity
- ebooklib
- beautifulsoup4
- requests

### System Dependencies

In addition to Python packages, certain functionality requires external tools:

#### Required for EPUB/PDF/HTML Conversion

- **Pandoc**: Used for EPUB to PDF conversion
  - **Windows**: Install from [pandoc.org/installing.html](https://pandoc.org/installing.html) or using `choco install pandoc`
  - **macOS**: Install using Homebrew: `brew install pandoc`
  - **Linux**: Install using package manager: `apt-get install pandoc` or `yum install pandoc`

- **wkhtmltopdf**: Used for HTML to PDF conversion
  - **Windows**: Install from [wkhtmltopdf.org/downloads.html](https://wkhtmltopdf.org/downloads.html) or using `choco install wkhtmltopdf`
  - **macOS**: Install using Homebrew: `brew install wkhtmltopdf`
  - **Linux**: Install using package manager: `apt-get install wkhtmltopdf` or `yum install wkhtmltopdf`

See the README.md for a complete feature-dependency matrix.
