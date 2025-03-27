# Guthman's Document Parsing Utilities

A collection of utilities for document content extraction and conversion, including:

- PDF document processing and content extraction
- EPUB to HTML/TXT/PDF conversion
- Support for AI-assisted document content extraction
- Hierarchical document structure extraction with page ranges

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
   AI_DEBUG_DIR=path/to/debug/directory  # For saving debug information when AI extraction fails
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

# Convert document pages to images (PNG or JPG)
from doc_parse_convert.content_extraction import ImageConverter

# Iterate through pages using context manager
with ImageConverter('document.pdf', format='png') as converter:
    for page_number, page_data in converter:
        with open(f'document_page_{page_number+1}.png', 'wb') as f:
            f.write(page_data)

# Change format to JPG
converter = ImageConverter('document.pdf', format='jpg')
for page_number, page_data in converter:
    with open(f'document_page_{page_number+1}.jpg', 'wb') as f:
        f.write(page_data)

# Extract document structure with page ranges
from doc_parse_convert.content_extraction import PDFProcessor, ProcessingConfig, DocumentStructureExtractor

# Configure the processor
config = ProcessingConfig(
    project_id="your-project-id",
    vertex_ai_location="your-location"
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

## Examples

See the `examples/` directory for detailed usage examples:
- `usage_example.ipynb`: Jupyter notebook with example code and configuration
- `image_converter_example.py`: Example of converting PDF pages to PNG and JPG images
