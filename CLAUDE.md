# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`doc_parse_convert` is a Python library for document content extraction and conversion. It supports PDF document processing with AI-assisted extraction (Google Vertex AI/Gemini), EPUB conversion, and hierarchical document structure analysis.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_image_converter.py

# Run with verbose output
pytest -v

# Run tests showing all output
pytest -v --no-header --no-summary
```

### Test Environment Variables

Tests that use AI features require these environment variables:
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud service account JSON file
- `JINA_API_KEY`: Jina API key for HTML to Markdown conversion

Tests requiring these credentials will be skipped if not set.

**Important**: Tests use real API calls (incur costs, no mocking).

## Development Setup

```bash
# Install in development mode
pip install -e .

# Required dependencies are auto-installed from setup.py
```

### External System Dependencies

Some features require external tools:
- **Pandoc**: EPUB to PDF conversion
- **wkhtmltopdf**: HTML to PDF conversion

## Architecture

### Core Processing Flow

1. **Document Loading**: `ProcessorFactory.create_processor()` auto-detects file type and creates appropriate processor (currently only PDF via `PDFProcessor`)
2. **Table of Contents Extraction**: Supports two strategies:
   - `ExtractionStrategy.NATIVE`: Uses PyMuPDF's built-in TOC extraction
   - `ExtractionStrategy.AI`: Uses Gemini vision model to analyze page images
3. **Content Extraction**: Extracts text/structure from chapters using same strategy options
4. **Structure Extraction**: `DocumentStructureExtractor` builds hierarchical document structure with page ranges

### Key Architectural Patterns

**Strategy Pattern for Extraction**: The `ExtractionStrategy` enum (NATIVE/AI/OCR) allows switching between extraction methods via `ProcessingConfig`. No automatic fallbacks - explicit strategy selection required.

**Processor Factory**: `ProcessorFactory` creates appropriate `DocumentProcessor` subclass based on file extension. Currently supports PDF; designed for future EPUB support.

**AI Client Abstraction**: `AIClient` wraps Vertex AI/Gemini interactions with:
- Retry logic (10 attempts with exponential temperature increase)
- Comprehensive error logging with debug artifact saving (set `AI_DEBUG_DIR` env var)
- Structured output via JSON schemas (see `doc_parse_convert/ai/schemas.py`)

**Image Conversion Layer**: `ImageConverter` (in `doc_parse_convert/utils/image.py`) converts PDF pages to images for AI processing. Supports context manager pattern and iteration.

### Module Structure

```
doc_parse_convert/
├── config.py              # ProcessingConfig, ExtractionStrategy, logging setup
├── extraction/
│   ├── base.py           # DocumentProcessor & ContentExtractor abstract base classes
│   ├── pdf.py            # PDFProcessor implementation
│   └── structure.py      # DocumentStructureExtractor for hierarchical analysis
├── ai/
│   ├── client.py         # AIClient for Vertex AI/Gemini interaction
│   ├── prompts.py        # AI prompt templates
│   └── schemas.py        # JSON schemas for structured AI output
├── models/
│   ├── document.py       # Chapter, DocumentSection data classes
│   └── content.py        # ChapterContent, PageContent, TextBox, etc.
├── conversion/
│   ├── epub.py          # EPUB conversion functions
│   ├── html.py          # HTML conversion & Jina integration
│   └── storage.py       # GCS upload for HTML→Markdown pipeline
└── utils/
    ├── factory.py       # ProcessorFactory
    └── image.py         # ImageConverter for PDF→image conversion
```

### Important Implementation Details

**Caching**: `PDFProcessor` caches TOC in `_chapters_cache` - cleared on `load()` or `close()`

**Page Numbering**: Internal page numbers are 0-based; user-facing are 1-based. `DocumentSection` supports both physical pages (0-based indexing) and logical pages (as displayed in PDF).

**AI Error Debugging**: When `AI_DEBUG_DIR` environment variable is set, the AI client saves timestamped debug directories containing:
- Problematic images that caused errors
- Complete error diagnostics
- Request parameters and response data

**Credential Handling**: Supports both service account files (`service_account_file`) and Application Default Credentials (`use_application_default_credentials=True`)

**Retry Logic**: AI calls use `tenacity` with 10 attempts, 5-second fixed wait, and progressive temperature increase (0.0 → 1.0) to handle transient failures

## Configuration

The library uses `ProcessingConfig` for all settings:

```python
config = ProcessingConfig(
    # Vertex AI settings
    project_id="your-project-id",
    vertex_ai_location="us-central1",
    gemini_model_name="gemini-1.5-flash-002",
    use_application_default_credentials=True,

    # Extraction strategies (NATIVE or AI)
    toc_extraction_strategy=ExtractionStrategy.NATIVE,
    content_extraction_strategy=ExtractionStrategy.AI,

    # Processing parameters
    max_pages_for_preview=200,
    image_quality=300  # DPI for AI processing
)
```

Environment variables (can be set in `.env` file):
- `JINA_API_KEY`: For HTML→Markdown via Jina API
- `AI_DEBUG_DIR`: Directory for saving AI error debug artifacts
- `DOC_PARSE_CONVERT_LOG_DIR`: Override default log directory (default: system temp)

## Logging

Logging is configured in `config.py`:
- Console: INFO level
- File: DEBUG level → `<temp>/doc_parse_convert_logs/content_extraction.log` (or `$DOC_PARSE_CONVERT_LOG_DIR`)
- AI client has extensive debug logging for troubleshooting API issues

## Common Patterns

### Always close processors
```python
processor = PDFProcessor(config)
processor.load("file.pdf")
try:
    # ... use processor ...
finally:
    processor.close()
```

### Use ProcessorFactory for automatic type detection
```python
processor = ProcessorFactory.create_processor("document.pdf", config)
# Factory automatically calls load()
```

### AI vs Native extraction
- **NATIVE**: Fast, no API costs, works if PDF has embedded TOC
- **AI**: Slower, incurs Vertex AI costs, works on any PDF with visual structure
