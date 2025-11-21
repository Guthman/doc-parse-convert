"""
Example script demonstrating how to use the doc_parse_convert package.

This example covers:
1. PDF extraction with native and AI strategies
2. Image content extraction from documents
3. Document structure analysis
4. EPUB conversion to various formats
"""

from pathlib import Path
from doc_parse_convert import (
    ProcessingConfig,
    ExtractionStrategy,
    ProcessorFactory,
    convert_epub_to_html,
    convert_epub_to_txt
)
from doc_parse_convert.image_extraction import ImageContentExtractor, ImageType
from doc_parse_convert.ai.client import AIClient


def demo_pdf_extraction_native():
    """
    Demonstrate how to extract content from a PDF file using native extraction.
    This is the fastest method and doesn't require AI API access.
    """
    print("\n--- Native PDF Extraction ---")

    # Configure for native extraction (no AI required)
    config = ProcessingConfig(
        toc_extraction_strategy=ExtractionStrategy.NATIVE,
        content_extraction_strategy=ExtractionStrategy.NATIVE,
    )

    pdf_path = "path/to/your/document.pdf"  # Replace with your PDF path
    processor = ProcessorFactory.create_processor(pdf_path, config)

    try:
        # Extract table of contents
        chapters = processor.get_table_of_contents()
        print(f"Found {len(chapters)} chapters:")
        for i, chapter in enumerate(chapters):
            print(f"  {i + 1}. {chapter.title} (pages {chapter.start_page + 1}-{chapter.end_page or '?'})")

        # Extract content from a specific chapter
        if chapters:
            chapter_content = processor.extract_chapter_text(chapters[0])
            print(f"\nExtracted content from '{chapter_content.title}':")
            print(f"  Pages: {len(chapter_content.pages)}")
            print(f"  Start page: {chapter_content.start_page + 1}")
            print(f"  End page: {chapter_content.end_page + 1}")

            # Print a sample of text from the first page
            if chapter_content.pages:
                sample_text = chapter_content.pages[0].chapter_text[:200] + "..."
                print(f"  Sample text: {sample_text}")

    finally:
        processor.close()


def demo_pdf_extraction_ai():
    """
    Demonstrate how to extract content from a PDF using AI-assisted extraction.
    This requires Google Vertex AI credentials and incurs API costs.
    """
    print("\n--- AI-Assisted PDF Extraction ---")

    # Configure for AI extraction
    config = ProcessingConfig(
        # Required for AI extraction
        project_id="your-google-project-id",
        vertex_ai_location="us-central1",
        gemini_model_name="gemini-2.5-flash",
        # Choose extraction strategies
        toc_extraction_strategy=ExtractionStrategy.AI,
        content_extraction_strategy=ExtractionStrategy.AI,
    )

    pdf_path = "path/to/your/document.pdf"
    processor = ProcessorFactory.create_processor(pdf_path, config)

    try:
        # AI extraction can find structure even without embedded TOC
        chapters = processor.get_table_of_contents()
        print(f"AI found {len(chapters)} chapters:")
        for i, chapter in enumerate(chapters):
            print(f"  {i + 1}. {chapter.title} (pages {chapter.start_page + 1}-{chapter.end_page or '?'})")

        if chapters:
            # AI extraction provides richer content including text boxes, tables, and figures
            chapter_content = processor.extract_chapter_text(chapters[0])
            print(f"\nAI-extracted content from '{chapter_content.title}':")

            for page in chapter_content.pages[:3]:  # Show first 3 pages
                print(f"\n  Page {page.page_number + 1}:")
                print(f"    Text boxes: {len(page.text_boxes)}")
                print(f"    Tables: {len(page.tables)}")
                print(f"    Figures: {len(page.figures)}")

                # Show text box content
                if page.text_boxes:
                    sample = page.text_boxes[0].text[:100] + "..."
                    print(f"    Sample text: {sample}")

    finally:
        processor.close()


def demo_document_structure():
    """
    Demonstrate how to extract hierarchical document structure.
    This shows parent-child relationships between sections.
    """
    print("\n--- Document Structure Analysis ---")

    config = ProcessingConfig(
        toc_extraction_strategy=ExtractionStrategy.NATIVE,
    )

    pdf_path = "path/to/your/document.pdf"
    processor = ProcessorFactory.create_processor(pdf_path, config)

    try:
        from doc_parse_convert.extraction.structure import DocumentStructureExtractor

        # Extract hierarchical structure
        extractor = DocumentStructureExtractor(processor)
        structure = extractor.extract_structure()

        print(f"Document: {structure.title}")
        print(f"Total pages: {structure.total_pages}")
        print(f"Root sections: {len(structure.sections)}")

        # Print structure tree
        def print_section(section, indent=0):
            prefix = "  " * indent
            print(f"{prefix}- {section.title}")
            print(f"{prefix}  Pages: {section.start_page + 1}-{section.end_page + 1 if section.end_page else '?'}")
            print(f"{prefix}  Level: {section.level}")
            if section.children:
                print(f"{prefix}  Children: {len(section.children)}")
                for child in section.children[:3]:  # Show first 3 children
                    print_section(child, indent + 1)

        print("\nDocument structure:")
        for section in structure.sections[:3]:  # Show first 3 root sections
            print_section(section)

        # Export structure as XML
        xml_output = extractor.to_xml()
        print(f"\nXML structure exported ({len(xml_output)} bytes)")

    finally:
        processor.close()


def demo_image_extraction_from_file():
    """
    Demonstrate extracting structured content from images using AI.
    This feature can identify tables, charts, diagrams, and other visual content.
    Requires Google Vertex AI credentials.
    """
    print("\n--- Image Content Extraction from File ---")

    # Configure AI client
    config = ProcessingConfig(
        project_id="your-google-project-id",
        vertex_ai_location="us-central1",
        gemini_model_name="gemini-2.5-flash",
    )

    # Initialize AI client and extractor
    ai_client = AIClient(config)
    extractor = ImageContentExtractor(ai_client, config)

    # Extract from a file path
    image_path = Path("path/to/your/image.png")  # PNG, JPG, JPEG supported

    try:
        result = extractor.extract(image_path)

        print(f"Image Type: {result.image_type}")
        print(f"Description: {result.description}")

        # Handle different content types
        if result.image_type == ImageType.TABLE and result.content:
            print("\nTable Content (Markdown):")
            print(result.content.markdown_content[:500] + "...")

        elif result.image_type == ImageType.CHART_OR_GRAPH and result.content:
            print(f"\nChart Type: {result.content.chart_type}")
            print(f"Summary: {result.content.summary}")
            if result.content.data_points:
                print(f"Data Points: {len(result.content.data_points)}")
                print(f"Sample: {result.content.data_points[:3]}")

        elif result.image_type == ImageType.COMPOUND and result.content:
            print(f"\nCompound Image with {len(result.content.elements)} elements:")
            for i, element in enumerate(result.content.elements):
                print(f"  {i+1}. {element.image_type}: {element.description[:50]}")

    except Exception as e:
        print(f"Error: {e}")


def demo_image_extraction_from_bytes():
    """
    Demonstrate extracting content from image bytes or base64 data.
    Useful when working with images from APIs or databases.
    """
    print("\n--- Image Content Extraction from Bytes ---")

    config = ProcessingConfig(
        project_id="your-google-project-id",
        vertex_ai_location="us-central1",
        gemini_model_name="gemini-2.5-flash",
    )

    ai_client = AIClient(config)
    extractor = ImageContentExtractor(ai_client, config)

    # Example 1: From raw bytes
    image_path = Path("path/to/your/image.png")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = extractor.extract(image_bytes, mime_type="image/png")
    print(f"From bytes - Type: {result.image_type}, Description: {result.description[:100]}")

    # Example 2: From base64-encoded data
    import base64
    base64_data = base64.b64encode(image_bytes).decode()
    result = extractor.extract(base64_data, mime_type="image/png")
    print(f"From base64 - Type: {result.image_type}, Description: {result.description[:100]}")


def demo_image_extraction_from_pdf():
    """
    Demonstrate extracting images from a PDF and analyzing their content.
    Combines PDF processing with image extraction.
    """
    print("\n--- Extract and Analyze Images from PDF ---")

    config = ProcessingConfig(
        project_id="your-google-project-id",
        vertex_ai_location="us-central1",
        gemini_model_name="gemini-2.5-flash",
        toc_extraction_strategy=ExtractionStrategy.NATIVE,
    )

    pdf_path = "path/to/your/document.pdf"
    processor = ProcessorFactory.create_processor(pdf_path, config)

    # Initialize image extractor
    ai_client = AIClient(config)
    image_extractor = ImageContentExtractor(ai_client, config)

    try:
        import fitz  # PyMuPDF

        # Get first page
        page = processor.doc[0]

        # Extract images from the page
        image_list = page.get_images()
        print(f"Found {len(image_list)} images on page 1")

        for img_index, img in enumerate(image_list[:3]):  # Analyze first 3 images
            # Get image bytes from PDF
            xref = img[0]
            base_image = processor.doc.extract_image(xref)
            image_bytes = base_image["image"]
            mime_type = f"image/{base_image['ext']}"

            print(f"\nAnalyzing image {img_index + 1}...")

            # Extract content from the image
            result = image_extractor.extract(image_bytes, mime_type=mime_type)
            print(f"  Type: {result.image_type}")
            print(f"  Description: {result.description[:100]}")

            if result.image_type == ImageType.TABLE and result.content:
                print(f"  Table found with markdown content ({len(result.content.markdown_content)} chars)")

    finally:
        processor.close()


def demo_epub_conversion():
    """
    Demonstrate how to convert EPUB files to other formats.
    """
    print("\n--- EPUB Conversion ---")

    epub_path = "path/to/your/book.epub"  # Replace with your EPUB path

    # Convert EPUB to HTML with embedded images
    print("Converting EPUB to HTML...")
    html_content = convert_epub_to_html(epub_path)
    print(f"Generated {len(html_content)} HTML documents")

    # Convert EPUB to text
    print("\nConverting EPUB to text...")
    text_content = convert_epub_to_txt(epub_path)
    if hasattr(text_content, 'getvalue'):
        sample_text = text_content.getvalue()[:200] + "..."
        print(f"Sample text: {sample_text}")


if __name__ == "__main__":
    print("=" * 70)
    print("doc_parse_convert - Comprehensive Examples")
    print("=" * 70)

    print("\nAvailable demos:")
    print("  1. demo_pdf_extraction_native() - Fast PDF extraction without AI")
    print("  2. demo_pdf_extraction_ai() - AI-powered PDF extraction (requires credentials)")
    print("  3. demo_document_structure() - Hierarchical document analysis")
    print("  4. demo_image_extraction_from_file() - Extract content from image files")
    print("  5. demo_image_extraction_from_bytes() - Extract from bytes/base64")
    print("  6. demo_image_extraction_from_pdf() - Extract and analyze PDF images")
    print("  7. demo_epub_conversion() - Convert EPUB to HTML/text")

    print("\n" + "=" * 70)
    print("Quick Start Examples")
    print("=" * 70)

    # Example 1: Basic PDF extraction (no AI required)
    print("\n1. Basic PDF Extraction (Native - No AI Required)")
    print("-" * 50)
    print("from doc_parse_convert import ProcessingConfig, ExtractionStrategy, ProcessorFactory")
    print("")
    print("config = ProcessingConfig(")
    print("    toc_extraction_strategy=ExtractionStrategy.NATIVE,")
    print("    content_extraction_strategy=ExtractionStrategy.NATIVE,")
    print(")")
    print("processor = ProcessorFactory.create_processor('document.pdf', config)")
    print("chapters = processor.get_table_of_contents()")
    print("content = processor.extract_chapter_text(chapters[0])")
    print("processor.close()")

    # Example 2: Image extraction
    print("\n2. Image Content Extraction (AI Required)")
    print("-" * 50)
    print("from doc_parse_convert import ProcessingConfig")
    print("from doc_parse_convert.image_extraction import ImageContentExtractor")
    print("from doc_parse_convert.ai.client import AIClient")
    print("")
    print("config = ProcessingConfig(")
    print("    project_id='your-project',")
    print("    vertex_ai_location='us-central1',")
    print("    gemini_model_name='gemini-2.5-flash'")
    print(")")
    print("ai_client = AIClient(config)")
    print("extractor = ImageContentExtractor(ai_client, config)")
    print("result = extractor.extract('image.png')  # or bytes, or base64")
    print(f"# Returns: image_type, description, and structured content")

    # Example 3: Document structure
    print("\n3. Document Structure Analysis")
    print("-" * 50)
    print("from doc_parse_convert.extraction.structure import DocumentStructureExtractor")
    print("")
    print("processor = ProcessorFactory.create_processor('document.pdf', config)")
    print("extractor = DocumentStructureExtractor(processor)")
    print("structure = extractor.extract_structure()")
    print("xml_output = extractor.to_xml()  # Export as XML")

    print("\n" + "=" * 70)
    print("\nTo run demos:")
    print("1. Replace placeholder paths with your actual files")
    print("2. For AI features, set up Google Vertex AI credentials")
    print("3. Uncomment the demo function you want to run")
    print("4. Run: python example.py")
    print("\n" + "=" * 70)

    # Uncomment any of these to run the demos:
    # demo_pdf_extraction_native()
    # demo_pdf_extraction_ai()
    # demo_document_structure()
    # demo_image_extraction_from_file()
    # demo_image_extraction_from_bytes()
    # demo_image_extraction_from_pdf()
    # demo_epub_conversion()
