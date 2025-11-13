"""
Response schemas for AI model calls.
"""


def get_toc_response_schema() -> dict:
    """
    Get the response schema for table of contents extraction.

    Returns:
        dict: TOC response schema
    """
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "t": {"type": "STRING"},     # Ultra-short property name for "title"
                "p": {"type": "INTEGER"},    # Ultra-short property name for "page"
                "l": {"type": "INTEGER"}     # Ultra-short property name for "level"
            }
        }
    }


def get_content_extraction_schema() -> dict:
    """
    Get the response schema for content extraction.

    Returns:
        dict: Content extraction response schema
    """
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "chapter_text": {
                    "type": "STRING",
                    "description": "The main text content formatted in markdown, preserving the original document structure and formatting"
                },
                "text_boxes": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "content": {"type": "STRING"},
                            "type": {"type": "STRING", "enum": ["text_box", "side_note", "callout"]}
                        }
                    }
                },
                "tables": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "content": {"type": "STRING"},
                            "caption": {"type": "STRING"}
                        }
                    }
                },
                "figures": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "description": {"type": "STRING"},
                            "byline": {"type": "STRING"}
                        }
                    }
                }
            },
            "required": ["chapter_text"]
        }
    }


def get_structure_extraction_schema() -> dict:
    """
    Get the response schema for document structure extraction.

    Returns:
        dict: Document structure response schema
    """
    return {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING"},
            "sections": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                        "start": {"type": "INTEGER"},
                        "end": {"type": "INTEGER"},
                        "level": {"type": "INTEGER"}
                    }
                }
            }
        },
        "required": ["title"]
    }
