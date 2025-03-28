"""
AI integration for document processing.
"""

from doc_parse_convert.ai.client import AIClient
from doc_parse_convert.ai.prompts import (
    get_toc_prompt,
    get_content_extraction_prompt,
    get_structure_extraction_prompt
)
from doc_parse_convert.ai.schemas import (
    get_toc_response_schema,
    get_content_extraction_schema,
    get_structure_extraction_schema
)
