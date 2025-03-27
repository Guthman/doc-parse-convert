# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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