# Test Suite

This directory contains the test suite for the guthman-information-extraction-utilities package.

## Overview

The test suite is organized as follows:

- `conftest.py`: Shared fixtures and utilities
- `test_image_converter.py`: Tests for the ImageConverter class
- `test_document_processor.py`: Tests for document loading and processing
- `test_structure_extraction.py`: Tests for document structure extraction
- `test_content_extraction.py`: Tests for content extraction
- `test_content_conversion.py`: Tests for EPUB conversion functionality
- `test_end_to_end.py`: End-to-end tests for the full processing pipeline

## Running Tests

To run the tests:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_image_converter.py

# Run tests with verbose output
pytest -v

# Run tests and see output even for passing tests
pytest -v --no-header --no-summary
```

## Environment Variables

The following environment variables are required for tests that use the Vertex AI and Jina APIs:

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to the Google Cloud service account credentials JSON file
- `JINA_API_KEY`: Jina API key for HTML to Markdown conversion

Tests that require these credentials will be skipped if the environment variables are not set.

## Test Data

The tests use sample files from the `test_files` directory:

- `LLMAll_en-US_FINAL.pdf`: Sample PDF file
- `Quick Start Guide - John Schember.epub`: Sample EPUB file

## Important Notes

1. The tests use real API calls to Google Vertex AI and Jina. This means:
   - Tests will incur actual costs when run
   - Tests will fail if credentials are invalid
   - Tests may take longer to complete due to API latency

2. No mocking is used in the test suite, as the focus is on ensuring all components work together correctly.

3. The test suite includes proper setup and teardown to ensure resources are properly cleaned up after tests.

## Test Design Principles

This test suite follows several key best practices from pytest and general testing principles:

1. **Fast Tests**: Most tests are designed to be quick, using minimal sample data
2. **Independent Tests**: Each test is independent with proper setup/teardown via fixtures
3. **Single-Purpose Tests**: Each test validates one specific behavior
4. **Readable Tests**: Clear naming conventions and organization
5. **Deterministic Tests**: Tests with AI components validate structure rather than exact content
6. **Dependency Isolation**: Proper fixture usage with appropriate scopes

## Adding New Tests

When adding new tests, consider the following:

1. Use the existing fixtures in `conftest.py` when possible
2. Follow the naming convention `test_*` for test functions
3. Keep tests focused on a single behavior
4. Include proper cleanup in any test that creates resources
5. Use the `pytest.mark.skipif` decorator for tests that require specific environment variables
