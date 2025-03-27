"""
Example script demonstrating how to use the doc_parse_convert package.
"""

from doc_parse_convert import (
    ProcessingConfig,
    ExtractionStrategy,
    ProcessorFactory,
    convert_epub_to_html,
    convert_epub_to_txt
)


def demo_pdf_extraction():
    """
    Demonstrate how to extract content from a PDF file.
    """
    # Configure the processor with various options
    config = ProcessingConfig(
        # Set these values if using AI-assisted extraction with Google Vertex AI
        # project_id="your-google-project-id", 
        # vertex_ai_location="us-central1",
        # gemini_model_name="gemini-1.5-flash-002",
        # Use native extraction by default (faster, no AI costs)
        toc_extraction_strategy=ExtractionStrategy.NATIVE,
        content_extraction_strategy=ExtractionStrategy.NATIVE,
    )

    # Option 1: Use ProcessorFactory to create the appropriate processor
    pdf_path = "path/to/your/document.pdf"  # Replace with your PDF path
    processor = ProcessorFactory.create_processor(pdf_path, config)

    # Option 2: Create PDFProcessor directly if you know it's a PDF
    # processor = PDFProcessor(config)
    # processor.load(pdf_path)

    try:
        # Extract table of contents
        chapters = processor.get_table_of_contents()
        print(f"Found {len(chapters)} chapters:")
        for i, chapter in enumerate(chapters):
            print(f"  {i + 1}. {chapter.title} (pages {chapter.start_page + 1}-{chapter.end_page or '?'})")

        # Extract content from a specific chapter (e.g., the first one)
        if chapters:
            chapter_content = processor.extract_chapter_text(chapters[0])
            print(f"\nExtracted content from '{chapter_content.title}':")
            print(f"  {len(chapter_content.pages)} pages of content")

            # Print a sample of text from the first page
            if chapter_content.pages:
                sample_text = chapter_content.pages[0].chapter_text[:200] + "..."
                print(f"  Sample text: {sample_text}")

    finally:
        # Always close the processor when done
        processor.close()


def demo_epub_conversion():
    """
    Demonstrate how to convert EPUB files to other formats.
    """
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
    print("==== PDF Extraction Demo ====")
    # Uncomment to run the PDF demo
    # demo_pdf_extraction()

    print("\n==== EPUB Conversion Demo ====")
    # Uncomment to run the EPUB demo
    # demo_epub_conversion()

    print(
        "\nTo run these demos, replace the file paths with your actual PDF/EPUB files and uncomment the function calls.")
