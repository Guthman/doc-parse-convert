# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3] - 2025-03-28

### Fixed

- Fixed circular import issues involving PDFProcessor, ImageConverter, and ProcessorFactory modules
- Restructured imports to avoid dependency cycles using dynamic imports
- Updated test files to match the new module structure
- Improved module organization by importing directly from specific modules rather than through content_extraction

## [0.5.2] - 2025-03-28

### Fixed

- Fixed InvalidArgument 400 errors from Vertex AI API by completely redesigning the response schema approach:
  - Removed nested structures (no more 'children' arrays)
  - Used ultra-short property names (single-letter identifiers) to reduce payload size
  - Eliminated required fields constraints
  - Implemented post-processing to reconstruct hierarchical structure from flat data
- Further reduced maximum number of pages processed to 20 for TOC extraction
- Improved prompt instructions with specific field name guidance
- Added additional error diagnostics mentioning structure simplification
- Implemented hierarchy reconstruction algorithm to rebuild document tree from flat data

## [0.5.1] - 2025-03-27

### Fixed

- Fixed InvalidArgument 400 errors from Vertex AI API by simplifying response schemas
- Limited maximum number of pages processed to 30 to avoid oversized requests
- Simplified property names in response schema to reduce complexity
- Improved error diagnostics for API request issues
- Reduced number of required fields in response schema
- Updated prompt instructions for better AI response compatibility

## [0.5.0] - 2025-03-27

### Added

- Enhanced error diagnostics for AI API calls with detailed error information
- Comprehensive debug system for saving problematic images when AI calls fail
- Timestamp-based debug directory structure to organize debug artifacts
- Size-based image quality adjustment for large documents
- Image size validation and warnings for potentially oversized requests
- Better error messages for specific API error types

### Improved

- More robust error handling in AI structure extraction
- Improved logging with detailed API request information
- Enhanced validation of extraction results
- Better handling of missing end_page values in document structure
- Enhanced prompt for document structure extraction
- Auto-adjusted image quality based on document page count

### Fixed

- Fixed handling of InvalidArgument API errors with detailed diagnostics
- Better error messages with root cause analysis
- Page number conversions from 1-based (API response) to 0-based (internal)

## [0.4.0] - 2025-03-27

### Added

- Enhanced AI error reporting with detailed request and response logging
- Debug option to save problematic images when AI API calls fail
- Separate test functions for individual extraction strategies (native and AI)

### Changed

- Removed automatic fallbacks when extraction strategies fail
- Modified `extract_structure` to strictly follow the specified extraction strategy
- Updated `get_table_of_contents` to use only the configured strategy
- Improved API error diagnostics and handling
- Test improvements:
  - Made end-to-end tests work without requiring AI credentials
  - Created synthetic chapters for testing PDFs without TOC
  - Added more robust extraction result validation
  - Properly skipped AI-dependent tests when credentials are unavailable

### Fixed

- Fixed error messages appearing in test logs from failed AI extraction attempts
- Improved error handling for different API response formats
- Added better validation of API responses

## [0.3.3] - 2025-03-27

### Improved

- Enhanced the TOC extraction prompt to better handle document hierarchy
- Improved ability to detect subheadings based on visual cues like font size, formatting, and indentation
- Fixed issues with heading level detection to properly identify parent-child relationships

## [0.3.2] - 2025-03-27

### Fixed

- Fixed AI TOC extraction to better handle multiple chapters on the same page
- Modified chapter end page assignment by sorting chapters by page number and level
- Updated tests to accommodate documents with non-ascending chapter page numbers
- Improved robustness when working with complex document structures