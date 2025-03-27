#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating the use of DocumentStructureExtractor to extract 
the complete document structure with page ranges.
"""

import os
import logging
from typing import Optional, Dict, Any

from doc_parse_convert.content_extraction import (
    ProcessingConfig, 
    PDFProcessor,
    DocumentStructureExtractor,
    ExtractionStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_document_structure(
    pdf_path: str, 
    output_dir: Optional[str] = None,
    output_format: str = "json",
    use_ai: bool = True,
    extract_text: bool = False
) -> Dict[str, Any]:
    """
    Extract the complete document structure with page ranges and optionally extract text by sections.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files (structure and text content)
        output_format: Format to export structure ("json", "dict", "xml")
        use_ai: Whether to use AI for structure extraction
        extract_text: Whether to extract text content by section
        
    Returns:
        Dictionary containing the document structure and any extracted text
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create processing configuration
    extraction_strategy = ExtractionStrategy.AI if use_ai else ExtractionStrategy.NATIVE
    
    config = ProcessingConfig(
        # Set your project_id and credentials if using Vertex AI
        # project_id="your-project-id",
        # vertex_ai_location="us-central1",
        # gemini_model_name="gemini-1.5-flash-002",
        # use_application_default_credentials=True,
        toc_extraction_strategy=extraction_strategy,
        content_extraction_strategy=extraction_strategy,
        max_pages_for_preview=100  # Process more pages for better structure detection
    )
    
    # Initialize the processor
    pdf_processor = PDFProcessor(config)
    pdf_processor.load(pdf_path)
    
    try:
        # Create the document structure extractor
        structure_extractor = DocumentStructureExtractor(pdf_processor)
        
        # Extract the document structure
        logger.info("Extracting document structure...")
        document_structure = structure_extractor.export_structure(output_format)
        
        # Save structure to file if output directory provided
        if output_dir:
            file_extension = "json" if output_format == "dict" else output_format
            structure_path = os.path.join(output_dir, f"document_structure.{file_extension}")
            
            logger.info(f"Saving document structure to {structure_path}")
            with open(structure_path, "w", encoding="utf-8") as f:
                f.write(document_structure if isinstance(document_structure, str) else str(document_structure))
        
        # Extract text by sections if requested
        section_texts = {}
        if extract_text:
            logger.info("Extracting text content by section...")
            text_output_dir = os.path.join(output_dir, "sections") if output_dir else None
            section_texts = structure_extractor.extract_text_by_section(text_output_dir)
            
        return {
            "structure": document_structure,
            "section_texts": section_texts if extract_text else None
        }
        
    finally:
        # Always close the processor
        pdf_processor.close()


def main():
    """Run the document structure extraction example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract document structure with page ranges")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", help="Output directory", default="./output")
    parser.add_argument("--format", "-f", help="Output format (json, xml)", choices=["json", "xml"], default="json")
    parser.add_argument("--no-ai", help="Disable AI-based extraction", action="store_true")
    parser.add_argument("--extract-text", "-t", help="Extract text content by section", action="store_true")
    
    args = parser.parse_args()
    
    try:
        result = extract_document_structure(
            args.pdf_path,
            output_dir=args.output,
            output_format=args.format,
            use_ai=not args.no_ai,
            extract_text=args.extract_text
        )
        
        if result["structure"]:
            logger.info("Document structure extraction completed successfully")
            if args.output:
                logger.info(f"Results saved to {args.output}")
        else:
            logger.error("Failed to extract document structure")
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main() 