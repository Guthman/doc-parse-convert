"""
AI prompts used for various extraction tasks.
"""


def get_toc_prompt() -> str:
    """
    Get the prompt for table of contents extraction.

    Returns:
        str: The TOC extraction prompt
    """
    return """Extract the table of contents from this document. Find all chapter titles and their page numbers.
            Format the output as a JSON array of chapters, where each item has these fields:
            - "t": The chapter title text (string)
            - "p": The page number where the chapter starts (number)
            - "l": The hierarchy level, where 1 is top level, 2 is subchapter, etc. (number)

            Important notes:
            - Use exactly these field names: t, p, l
            - Keep structure simple and flat (a plain array of objects)
            - Capture all chapters and sections you can find
            """


def get_content_extraction_prompt() -> str:
    """
    Get the prompt for content extraction.

    Returns:
        str: The content extraction prompt
    """
    return """Extract text VERBATIM from these images and format all output in markdown. Preserve the original document structure and formatting. REPRODUCE ALL TEXT WORD FOR WORD, DON'T PARAPHRASE.

            1. Main Chapter Text:
                • Extract and format the main text as markdown
                • Preserve all original formatting and structure
                • Output as 'chapter_text' in the response

            2. Supplemental Elements:
                • Extract text boxes, side notes, and callouts
                • Format their content in markdown as well
                • Label with appropriate type

            3. Tables:
                • Convert tables to markdown format
                • Include captions if present

            4. Figures:
                • Include descriptions and bylines

            5. Headers and Footers:
                • Exclude headers, footers, and page numbers

            Format the output as a JSON array of page objects."""


def get_structure_extraction_prompt() -> str:
    """
    Get the prompt for document structure extraction.

    Returns:
        str: The document structure extraction prompt
    """
    return """Analyze this document to extract its full hierarchical structure.

            Identify:
            1. The document title
            2. All sections and subsections with their:
               - Title
               - Starting page number
               - Ending page number (if determinable)
               - Hierarchy level (1 for top sections, 2 for subsections, etc.)

            Pay special attention to:
            - Chapter titles and numbers
            - Section headings and subheadings
            - Appendices
            - Front matter (preface, foreword, etc.)
            - Back matter (glossary, index, etc.)

            Format the response as a JSON object with this structure:
            {
              "title": "Main document title",
              "sections": [
                {
                  "title": "Section title",
                  "start": Starting page number (integer),
                  "end": Ending page number (integer, optional),
                  "level": Hierarchy level (integer)
                },
                ...more sections
              ]
            }
            """
