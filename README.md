# Guthman's Document Parsing Utilities

A collection of utilities for document content extraction and conversion, including:

- PDF document processing and content extraction
- EPUB to HTML/TXT/PDF conversion
- Support for AI-assisted document content extraction
- Hierarchical document structure extraction with page ranges

## 🔒 Security

This repository uses [Gitleaks](https://github.com/gitleaks/gitleaks) to prevent accidentally committing secrets. See [SECURITY.md](SECURITY.md) for our security policy and [GITLEAKS_SETUP.md](GITLEAKS_SETUP.md) for setup instructions.

**Before contributing**: Install Gitleaks to scan for secrets automatically. See [INSTALLATION_INSTRUCTIONS.md](INSTALLATION_INSTRUCTIONS.md) for details.

## Installation

```bash
# Install directly from the repository
pip install git+https://github.com/Guthman/doc-parse-convert.git

# For development installation (from local clone)
pip install -e .
```

## System Dependencies

In addition to Python dependencies, this library requires the following external tools for certain functionality:

### Required for EPUB/PDF/HTML Conversion

- **Pandoc**: Used for EPUB to PDF conversion
  - **Windows**: Install from [pandoc.org/installing.html](https://pandoc.org/installing.html) or using `choco install pandoc`
  - **macOS**: Install using Homebrew: `brew install pandoc`
  - **Linux**: Install using package manager: `apt-get install pandoc` or `yum install pandoc`

- **wkhtmltopdf**: Used for HTML to PDF conversion
  - **Windows**: Install from [wkhtmltopdf.org/downloads.html](https://wkhtmltopdf.org/downloads.html) or using `choco install wkhtmltopdf`
  - **macOS**: Install using Homebrew: `brew install wkhtmltopdf`
  - **Linux**: Install using package manager: `apt-get install wkhtmltopdf` or `yum install wkhtmltopdf`

### Feature Dependency Matrix

| Feature | Required System Dependencies |
|---------|------------------------------|
| PDF content extraction | None |
| EPUB to HTML conversion | None |
| EPUB to TXT conversion | None |
| EPUB to PDF conversion | Pandoc |
| HTML to PDF conversion | wkhtmltopdf |
| HTML to Markdown conversion | None (but requires GCS bucket and Jina API credentials) |

## Configuration

The utilities require various configuration values and credentials. These can be provided in several ways:

1. Environment Variables:
   Create a `.env` file in your project root with the following variables:
   ```
   JINA_API_KEY=your_jina_api_key
   GCP_SERVICE_ACCOUNT=your_service_account_json
   AI_DEBUG_DIR=path/to/debug/directory  # For saving debug information when AI extraction fails
   ```

2. Processing Configuration:
   When using the document processors, provide a `ProcessingConfig` object with your settings:
   ```python
   from doc_parse_convert import ProcessingConfig, ExtractionStrategy

   config = ProcessingConfig(
       project_id="your-project-id",
       vertex_ai_location="your-location",
       gemini_model_name="gemini-2.5-flash",
       use_application_default_credentials=True,
       toc_extraction_strategy=ExtractionStrategy.NATIVE,
       content_extraction_strategy=ExtractionStrategy.AI
   )
   ```

3. Required Tools:
   - Pandoc: For EPUB to PDF conversion
   - wkhtmltopdf: For HTML to PDF conversion
   Make sure these tools are installed and available in your system PATH.

## Debugging AI Extraction

If you encounter issues with AI extraction, you can enable debugging by setting the AI_DEBUG_DIR environment variable:

```bash
# On Windows
$env:AI_DEBUG_DIR = "C:\path\to\debug\directory"

# On Linux/Mac
export AI_DEBUG_DIR=/path/to/debug/directory
```

When set, the library will save:
- Timestamped debug directories for each error
- Problematic images that caused API errors
- Error details and request information
- Complete API error diagnostics

This helps troubleshoot issues with the Vertex AI API, particularly "InvalidArgument" errors related to image sizes or content.

## Usage

### PDF Content Extraction

```python
from doc_parse_convert import ProcessingConfig, ExtractionStrategy, PDFProcessor

# Configure the processor
config = ProcessingConfig(
    toc_extraction_strategy=ExtractionStrategy.NATIVE,
    content_extraction_strategy=ExtractionStrategy.NATIVE
)

# Process a PDF file
processor = PDFProcessor(config)
processor.load("document.pdf")

# Extract table of contents
chapters = processor.get_table_of_contents()
for chapter in chapters:
    print(f"{chapter.title} (pages {chapter.start_page+1}-{chapter.end_page+1})")

# Extract content from a specific chapter
chapter_content = processor.extract_chapter_text(chapters[0])

# Don't forget to close the processor when finished
processor.close()
```

### EPUB Conversion

```python
from doc_parse_convert import convert_epub_to_html, convert_epub_to_txt, convert_epub_to_pdf

# Convert EPUB to HTML
html_content = convert_epub_to_html("book.epub")

# Convert EPUB to TXT
text_content = convert_epub_to_txt("book.epub")

# Convert EPUB to PDF
pdf_path = convert_epub_to_pdf("book.epub", output_folder="output_folder")
```

### Converting PDF Pages to Images

```python
from doc_parse_convert.utils.image import ImageConverter

# Using context manager (recommended)
with ImageConverter('document.pdf', format='png') as converter:
    for page_number, page_data in converter:
        with open(f'document_page_{page_number+1}.png', 'wb') as f:
            f.write(page_data)

# Alternative approach
converter = ImageConverter('document.pdf', format='jpg')
try:
    for page_number, page_data in converter:
        with open(f'document_page_{page_number+1}.jpg', 'wb') as f:
            f.write(page_data)
finally:
    converter.close()
```

### Document Structure Extraction

```python
from doc_parse_convert import ProcessingConfig, PDFProcessor, DocumentStructureExtractor

# Configure the processor
config = ProcessingConfig(
    toc_extraction_strategy=ExtractionStrategy.NATIVE
)

# Process a PDF file
processor = PDFProcessor(config)
processor.load("document.pdf")

# Extract hierarchical document structure with page ranges
structure_extractor = DocumentStructureExtractor(processor)
document_structure = structure_extractor.extract_structure()

# Export structure in different formats
json_structure = structure_extractor.export_structure("json")
xml_structure = structure_extractor.export_structure("xml")

# Extract text by sections
section_texts = structure_extractor.extract_text_by_section("output_folder")
```

### Using the Processor Factory

```python
from doc_parse_convert import ProcessingConfig, ProcessorFactory

# Configure processing options
config = ProcessingConfig()

# Automatically create the appropriate processor based on file type
processor = ProcessorFactory.create_processor("document.pdf", config)

# Use the processor
chapters = processor.get_table_of_contents()

# Always close the processor when done
processor.close()
```

## Examples

See the `examples/` directory for detailed usage examples:
- `usage_example.ipynb`: Jupyter notebook with example code and configuration
- `image_converter_example.py`: Example of converting PDF pages to PNG and JPG images
