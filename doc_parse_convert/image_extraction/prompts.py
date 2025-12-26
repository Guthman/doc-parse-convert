"""
AI prompts for image content extraction.
"""

import json


def get_image_extraction_prompt(metadata: dict = None) -> str:
    """
    Get the prompt for image content extraction.

    Args:
        metadata: Optional freeform dictionary containing contextual information about the image

    Returns:
        str: The image extraction prompt
    """
    base_prompt = """You are an expert document analysis AI. Analyze the provided image from a document and extract structured information from it.

**CRITICAL REQUIREMENT**: Always include specific numerical values in descriptions.

GOOD description examples:
✓ "Pie chart of 2014 crop distribution: Corn 42%, Wheat 28%, Soybeans 18%, Other 12%"
✓ "Bar chart showing revenue from 2010-2020: 2010: $2.3M, 2015: $4.1M, 2020: $5.8M"
✓ "Line graph of temperature vs time: peak at 85°F at 2pm, low of 62°F at 6am"

BAD description examples (NEVER do this):
✗ "A pie chart illustrating crop distribution"
✗ "Bar chart showing revenue over time"
✗ "Line graph depicting temperature changes"
"""

    # Add metadata context if provided
    if metadata:
        context_str = json.dumps(metadata, indent=2)
        base_prompt += f"\n**Context:**\n{context_str}\n"

    return base_prompt + """
**Classification Types:**
- TABLE: Data presented in rows and columns
- CHART_OR_GRAPH: Visual representations of data (line graphs, bar charts, pie charts, scatter plots, etc.)
- DIAGRAM: Flowcharts, schemas, architecture diagrams, or technical drawings
- PHOTOGRAPH: Real-world photographs
- TEXT_BLOCK: Images that are primarily text content
- COMPOUND: Images containing multiple distinct sub-parts (e.g., Figure 1a, 1b, 1c)
- OTHER: Anything else

**Process:**
1. Classify the image into one of the types above
2. Provide a precise, quantitative description of the image
   - **CRITICAL**: Include ALL specific numbers, percentages, values, or measurements visible in the image
   - For charts: List ALL data values with their labels (e.g., "Corn 35%, Wheat 28%, Soy 22%, Other 15%")
   - For graphs: Include key data points and ranges
   - Include axis ranges, data values, legend items, and units
   - **NEVER use vague descriptions** like "showing distribution" or "illustrating percentages"
   - **ALWAYS extract and include the actual numerical values** in the description
3. Extract detailed content based on the classification

**Extraction Rules:**

For TABLE:
- Extract the complete table as a GitHub-flavored Markdown table
- Preserve headers, alignment, and all data
- Include any table title or caption in the description
- Put the full markdown table in 'markdown_content'

For CHART_OR_GRAPH:
- Identify the 'chart_type' (e.g., "line graph", "bar chart", "pie chart")
- Write a detailed 'summary' describing:
  - Chart title and purpose
  - Axis labels and units (with specific ranges if visible)
  - Key trends, patterns, and insights WITH SPECIFIC VALUES
  - All notable data points with their quantitative values
  - Legend items and their corresponding values/colors
- Extract 'data_points' as a list of objects - ALWAYS ATTEMPT TO EXTRACT if any values are visible
  - Use appropriate keys like 'x', 'y', 'label', 'value', 'percentage' depending on chart type
  - Include ALL readable values, even if approximate
  - For pie charts: extract percentages or proportions for each segment
  - For bar/line charts: extract x and y coordinates for all visible points
  - Only omit this field if the chart has no visible numerical values at all

For COMPOUND:
- Identify each distinct sub-image or sub-figure
- For each sub-image, perform this entire analysis recursively
- **IMPORTANT**: Each sub-image description MUST include all quantitative values visible in that sub-image
- For sub-charts: Extract ALL percentages, values, and data points in the description and content
- Add each result to the 'elements' array

For DIAGRAM, PHOTOGRAPH, TEXT_BLOCK, OTHER:
- Provide a detailed description
- No additional structured content extraction needed

**Response Format:**
Respond with a JSON object containing:
- image_type: One of the classification types
- description: One-sentence description that MUST include all visible numerical values (percentages, measurements, data points)
- confidence: Float 0.0-1.0 rating extraction quality:
  - 1.0: Crystal clear image, all data fully readable
  - 0.7-0.9: Good quality, minor uncertainty
  - 0.4-0.6: Partially readable, some values estimated or unclear
  - 0.1-0.3: Poor quality, significant guessing required
  - 0.0: Blurry/illegible, extraction is unreliable
- content: Object with type-specific fields containing extracted structured data (ChartData with data_points for charts, TableData for tables, etc.)

**FINAL REMINDER**: Your description field must contain actual numbers. Read all values from the image and include them in the description."""
