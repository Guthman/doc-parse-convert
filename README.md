# Guthman's Document Parsing Utilities

A collection of utilities for document content extraction and conversion, including:

- PDF document processing and content extraction
- EPUB to HTML/TXT/PDF conversion
- Support for AI-assisted document content extraction

## Installation

```bash
# Install directly from the repository
pip install git+https://github.com/Guthman/doc-parse-convert.git

# For development installation (from local clone)
pip install -e .
```

## Configuration

The utilities require various configuration values and credentials. These can be provided in several ways:

1. Environment Variables:
   Create a `.env` file in your project root with the following variables:
   ```
   JINA_API_KEY=your_jina_api_key
   GCP_SERVICE_ACCOUNT=your_service_account_json
   ```

2. Processing Configuration:
   When using the document processors, provide a `ProcessingConfig` object with your settings:
   ```python
   from doc_parse_convert.content_extraction import ProcessingConfig, ExtractionStrategy
   
   config = ProcessingConfig(
       project_id="your-project-id",
       vertex_ai_location="your-location",
       gemini_model_name="gemini-1.5-flash-002",
       use_application_default_credentials=True,
       toc_extraction_strategy=ExtractionStrategy.NATIVE,
       content_extraction_strategy=ExtractionStrategy.AI
   )
   ```

3. Required Tools:
   - Pandoc: For EPUB to PDF conversion
   - wkhtmltopdf: For HTML to PDF conversion
   Make sure these tools are installed and available in your system PATH.

## Usage

```python
# Extract content from PDF documents
from doc_parse_convert.content_extraction import PDFProcessor, ProcessingConfig

# Configure the processor
config = ProcessingConfig(
    project_id="your-project-id",
    vertex_ai_location="your-location"
)

# Process a PDF file
processor = PDFProcessor(config)
processor.load("document.pdf")
chapters = processor.get_table_of_contents()

# Convert EPUB to other formats
from doc_parse_convert.content_conversion import convert_epub_to_html, convert_epub_to_pdf

# Convert EPUB to HTML
html_content = convert_epub_to_html("book.epub")

# Convert EPUB to PDF
pdf_path = convert_epub_to_pdf("book.epub", "output_folder")
```

## Examples

See the `examples/` directory for detailed usage examples:
- `usage_example.ipynb`: Jupyter notebook with example code and configuration
