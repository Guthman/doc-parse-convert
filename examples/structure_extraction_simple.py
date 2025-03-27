#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified example of document structure extraction.
This example demonstrates basic usage without requiring AI services.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path if running script directly
if __name__ == "__main__":
    project_root = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, project_root)

from doc_parse_convert.content_extraction import (
    ProcessingConfig, 
    PDFProcessor,
    DocumentStructureExtractor,
    ExtractionStrategy
)


def extract_structure_simple(pdf_path):
    """
    Extract document structure using native methods only.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        The document structure as a dictionary
    """
    # Create a configuration that doesn't require AI services
    config = ProcessingConfig(
        toc_extraction_strategy=ExtractionStrategy.NATIVE,
        content_extraction_strategy=ExtractionStrategy.NATIVE
    )
    
    # Initialize the processor
    processor = PDFProcessor(config)
    processor.load(pdf_path)
    
    try:
        # Create the structure extractor
        extractor = DocumentStructureExtractor(processor)
        
        # Extract and return the structure
        structure = extractor.extract_structure()
        return structure.to_dict()
        
    finally:
        # Always close the processor
        processor.close()


def main():
    """Run the simplified example."""
    if len(sys.argv) < 2:
        print("Usage: structure_extraction_simple.py <pdf_file>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    output_dir = "output"
    
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found")
        sys.exit(1)
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract structure
    print(f"Extracting structure from {pdf_path}...")
    structure = extract_structure_simple(pdf_path)
    
    # Save to file
    output_file = os.path.join(output_dir, "structure.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=2)
    
    print(f"Structure saved to {output_file}")
    
    # Display basic information
    print("\nDocument Structure Summary:")
    print(f"- Document has {len(structure.get('children', []))} top-level sections")
    
    # Show first few sections
    for i, section in enumerate(structure.get('children', [])[:3]):
        print(f"  - {section.get('title')} (Pages {section.get('start_page')+1}-{section.get('end_page')+1})")
        
    if len(structure.get('children', [])) > 3:
        print(f"  - ... and {len(structure.get('children', [])) - 3} more sections")


if __name__ == "__main__":
    main() 