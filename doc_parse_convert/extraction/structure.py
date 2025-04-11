"""
Document structure extraction functionality.
"""

import os
import re
import json
from typing import Any, Dict, Optional

import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

from doc_parse_convert.config import logger, ExtractionStrategy
from doc_parse_convert.models.document import DocumentSection
from doc_parse_convert.utils.image import ImageConverter


class DocumentStructureExtractor:
    """Class for extracting hierarchical document structure with page ranges."""
    
    def __init__(self, processor):
        """
        Initialize the document structure extractor.
        
        Args:
            processor: The document processor to use for extraction
        """
        self.processor = processor
        self.doc = processor.doc
        self.config = processor.config
        self.ai_client = processor.ai_client
        
    def extract_structure(self) -> DocumentSection:
        """
        Extract the complete document structure with hierarchical sections and page ranges.
        
        This method analyzes the entire document to produce a comprehensive structure using
        the specified extraction strategy. No automatic fallbacks are used.
        
        Returns:
            DocumentSection: Root section containing the complete document hierarchy
            
        Raises:
            ValueError: If extraction strategy is invalid
            Exception: If extraction fails
        """
        logger.info("Extracting complete document structure")
        
        # Create root document section
        root = DocumentSection(
            title="Document Root",
            start_page=0,
            end_page=self.doc.page_count - 1,
            level=0
        )
        
        # Use the extraction strategy specified in the config
        if self.config.toc_extraction_strategy == ExtractionStrategy.AI:
            logger.info("Using AI to extract document structure")
            return self._extract_structure_with_ai(root)
        elif self.config.toc_extraction_strategy == ExtractionStrategy.NATIVE:
            logger.info("Using native methods to extract document structure")
            return self._extract_structure_with_native_enhancement(root)
        else:
            error_msg = f"Unsupported extraction strategy: {self.config.toc_extraction_strategy}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _extract_structure_with_ai(self, root: DocumentSection) -> DocumentSection:
        """
        Extract document structure using AI analysis of the entire document.
        
        Args:
            root: Root document section
            
        Returns:
            DocumentSection: Root section with populated hierarchy
            
        Raises:
            ValueError: If AI model is not initialized or extraction fails
        """
        # Check if AI client is available
        if not self.ai_client or not self.ai_client.model:
            error_msg = "AI model not initialized for structure extraction"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert all pages to images for AI processing
        logger.info("Converting all document pages to images for AI processing")
        try:
            images = ImageConverter.convert_to_images(
                self.doc,
                num_pages=self.doc.page_count,  # Process entire document
                start_page=0,
                image_quality=self.config.image_quality
            )
            
            if not images:
                error_msg = "No images were generated from document for AI processing"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Successfully converted {len(images)} document pages to images")
        except Exception as e:
            error_msg = f"Failed to convert document pages to images: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        # Import required modules from AI package
        from doc_parse_convert.ai.prompts import get_structure_extraction_prompt
        from doc_parse_convert.ai.schemas import get_structure_extraction_schema
        from vertexai.generative_models import Part, GenerationConfig
        
        # Prepare parts for the AI request
        parts = []
        
        # Limit number of images to process (preventing request size issues)
        max_images = 1000
        logger.info(f"Using {max_images} out of {len(images)} pages for structure extraction")
        
        for i, img in enumerate(images[:max_images]):
            try:
                parts.append(Part.from_data(data=img["data"], mime_type=img["_mime_type"]))
            except Exception as e:
                logger.warning(f"Failed to process image {i + 1}: {str(e)}")
        
        if not parts:
            error_msg = "No valid image parts were created for AI processing"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Created {len(parts)} image parts for AI processing")
        
        # Add instruction text
        try:
            parts.append(Part.from_text(get_structure_extraction_prompt()))
        except Exception as e:
            logger.error(f"Failed to create instruction text part: {str(e)}")
            raise ValueError(f"Failed to create instruction text part: {str(e)}") from e
        
        generation_config = GenerationConfig(
            temperature=0.15  # Low temperature for more deterministic results
        )
        
        # Define response schema
        response_schema = get_structure_extraction_schema()

        # Call AI with retry
        try:
            logger.debug("Calling AI model to extract document structure")
            
            # Try with minimal schema to avoid InvalidArgument errors
            response = self.ai_client._call_model_with_retry(
                parts,
                generation_config,
                response_mime_type="application/json",
                response_schema=response_schema
            )
            
            # Parse the response
            structure_data = json.loads(response.text)
            
            # Process the structure data into our DocumentSection objects
            root.title = structure_data.get("title", "Document Root")
            
            # Helper function to recursively build structure
            # This function now has to handle both flattened and nested data
            def build_structure(section_data):
                # Create a map of sections by level and start page for reconstruction
                sections_by_id = {}
                all_sections = []
                
                # First pass: create all sections
                for item in section_data:
                    # Validate and adjust page numbers
                    start_page = max(0, item.get("start", 1) - 1)  # Convert to 0-based
                    end_page = item.get("end")
                    if end_page is not None:
                        end_page = max(start_page, end_page - 1)  # Convert to 0-based
                    
                    level = item.get("level", 1)
                    
                    # Create the section
                    section = DocumentSection(
                        title=item["title"],
                        start_page=start_page,
                        end_page=end_page,
                        level=level
                    )
                    
                    # Store in our maps
                    section_id = f"{level}_{start_page}"
                    sections_by_id[section_id] = section
                    all_sections.append(section)
                
                # Sort sections by level (ascending) and start page
                all_sections.sort(key=lambda s: (s.level, s.start_page))
                
                # Infer hierarchy - this is the magic to reconstruct the tree structure
                top_level_sections = []
                for section in all_sections:
                    if section.level == 1:
                        top_level_sections.append(section)
                        continue
                        
                    # Find a parent for this section
                    parent = None
                    for potential_parent in reversed(all_sections):
                        # Handle the case where either end_page is None
                        if potential_parent.level < section.level and potential_parent.start_page <= section.start_page:
                            # If either end_page is None, skip the end_page comparison
                            if potential_parent.end_page is None or section.end_page is None:
                                parent = potential_parent
                                break
                            # Only compare end_pages if both are not None
                            elif potential_parent.end_page >= section.end_page:
                                parent = potential_parent
                                break
                    
                    if parent:
                        parent.add_child(section)
                    else:
                        # If no parent found, add to top level
                        top_level_sections.append(section)
                
                return top_level_sections
            
            # Build the complete structure
            root.children = build_structure(structure_data.get("sections", []))
            
            # Set any missing end_page values
            # First, sort top-level sections by start page
            root.children.sort(key=lambda s: s.start_page)
            
            for i, section in enumerate(root.children):
                if section.end_page is None:
                    if i < len(root.children) - 1:
                        section.end_page = root.children[i + 1].start_page - 1
                    else:
                        section.end_page = self.doc.page_count - 1
            
            logger.info(f"Successfully extracted document structure with {len(root.children)} top-level sections")
            return root
            
        except Exception as e:
            logger.error(f"Error in AI structure extraction: {str(e)}")
            # Additional diagnostic information
            logger.error(f"Document has {self.doc.page_count} pages")
            logger.error(f"Using model: {self.config.gemini_model_name}")
            logger.debug(f"Response schema: {json.dumps(response_schema, indent=2)}")
            
            # Check for specific error types
            error_class = e.__class__.__name__
            if 'InvalidArgument' in error_class:
                logger.error("The API request contains an invalid argument. This could be due to:")
                logger.error("- Images too large or too many images in the request")
                logger.error("- Malformed request structure or invalid parameters")
                logger.error("- Model limitations or incompatible response schema")
                logger.error("Attempt to use a smaller subset of pages or simplify the schema further")
                
                # Save debug information if enabled
                debug_dir = os.environ.get("AI_DEBUG_DIR")
                if not debug_dir:
                    logger.info("Set AI_DEBUG_DIR environment variable to save debug information")
                
            raise
    
    def _extract_structure_with_native_enhancement(self, root: DocumentSection) -> DocumentSection:
        """
        Extract document structure using native TOC extraction and enhance it with additional analysis.
        
        Args:
            root: Root document section
            
        Returns:
            DocumentSection: Root section with populated hierarchy
        """
        logger.info("Extracting and enhancing document structure using native methods")
        
        # Get native table of contents
        toc = self.doc.get_toc()
        
        if not toc:
            logger.warning("No native TOC found, attempting to infer structure from document")
            return self._infer_structure_from_document(root)
        
        # Convert TOC to DocumentSection objects
        sections_by_level = {}  # Dictionary to keep track of the latest section at each level
        
        # First pass: create all sections
        for level, title, page in toc:
            # Convert to 0-based page index
            page_idx = page - 1
            
            section = DocumentSection(
                title=title,
                start_page=page_idx,
                level=level,
                logical_start_page=page  # Store the logical page number as well
            )
            
            # Find parent and add as child
            if level > 1 and level - 1 in sections_by_level:
                parent = sections_by_level[level - 1]
                parent.add_child(section)
            else:
                # Top-level section or couldn't find parent, add to root
                root.add_child(section)
            
            # Update the latest section at this level
            sections_by_level[level] = section
        
        # Second pass: set end pages
        # Sort all sections by start page for processing
        all_sections = []
        
        def collect_sections(section):
            all_sections.append(section)
            for child in section.children:
                collect_sections(child)
        
        for child in root.children:
            collect_sections(child)
        
        all_sections.sort(key=lambda s: (s.start_page, -s.level))
        
        # Set end pages based on next section at same or higher level
        for i, section in enumerate(all_sections):
            # Find the next section at same or higher level that starts after this one
            for j in range(i + 1, len(all_sections)):
                next_section = all_sections[j]
                if next_section.level <= section.level and next_section.start_page > section.start_page:
                    section.end_page = next_section.start_page - 1
                    break
            
            # If no next section found, end at document end
            if section.end_page is None:
                section.end_page = self.doc.page_count - 1
                
        # Analyze document to enhance with section types and identifiers
        self._enhance_structure_with_text_analysis(root)
            
        return root
    
    def _infer_structure_from_document(self, root: DocumentSection) -> DocumentSection:
        """
        Infer document structure by analyzing page content when no TOC is available.
        
        Args:
            root: Root document section
            
        Returns:
            DocumentSection: Root section with inferred hierarchy
        """
        logger.info("Inferring document structure from page content")
        
        # This is a simplified approach - in a real implementation, you would use
        # more sophisticated text analysis to detect headings, etc.
        
        # Simple approach: look for potential headings (large text, centered, etc.)
        potential_sections = []
        
        for page_idx in range(self.doc.page_count):
            page = self.doc[page_idx]
            
            # Extract text blocks with their attributes
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    if "spans" not in line:
                        continue
                        
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        font_size = span.get("size", 0)
                        
                        # Heuristic: potential headings are larger text
                        if len(text) > 0 and len(text) < 100 and font_size > 12:
                            # Check if it looks like a heading (e.g., "Chapter 1", "1. Introduction")
                            if re.match(r"^(chapter|section|part|appendix|\d+\.)\s+\w+", text.lower()):
                                # Determine level based on font size (larger = higher level)
                                level = 1 if font_size > 16 else 2
                                
                                potential_sections.append({
                                    "title": text,
                                    "start_page": page_idx,
                                    "level": level
                                })
        
        # Sort by page and create structure
        potential_sections.sort(key=lambda s: s["start_page"])
        
        # Create sections and set end pages
        for i, section_data in enumerate(potential_sections):
            section = DocumentSection(
                title=section_data["title"],
                start_page=section_data["start_page"],
                level=section_data["level"]
            )
            
            # Set end page
            if i < len(potential_sections) - 1:
                section.end_page = potential_sections[i + 1]["start_page"] - 1
            else:
                section.end_page = self.doc.page_count - 1
                
            # Add to root
            if section.level == 1:
                root.add_child(section)
            else:
                # Find parent for this section
                parent = None
                for potential_parent in reversed(root.children):
                    if potential_parent.start_page <= section.start_page:
                        parent = potential_parent
                        break
                
                if parent:
                    parent.add_child(section)
                else:
                    root.add_child(section)
        
        return root
    
    def _enhance_structure_with_text_analysis(self, root: DocumentSection) -> None:
        """
        Enhance the document structure with additional information from text analysis.
        
        Args:
            root: Root document section to enhance
        """
        logger.info("Enhancing document structure with text analysis")
        
        def process_section(section):
            # Skip processing if this is the root
            if section.level == 0:
                for child in section.children:
                    process_section(child)
                return
                
            # Analyze the first page of the section to extract more information
            page = self.doc[section.start_page]
            text = page.get_text(0, 500)  # Get first 500 characters
            
            # Try to identify section type and identifier
            section_type = None
            identifier = None
            
            # Common patterns for section types
            if re.search(r"\bchapter\s+\d+", text.lower()):
                section_type = "chapter"
                match = re.search(r"(chapter\s+\d+)", text.lower())
                if match:
                    identifier = match.group(1).title()
            elif re.search(r"\bappendix\s+[a-z]", text.lower(), re.IGNORECASE):
                section_type = "appendix"
                match = re.search(r"(appendix\s+[a-z])", text, re.IGNORECASE)
                if match:
                    identifier = match.group(1).title()
            elif re.search(r"^\s*\d+\.\d+\s+", text):
                section_type = "subsection"
                match = re.search(r"(\d+\.\d+)", text)
                if match:
                    identifier = f"Section {match.group(1)}"
            elif re.search(r"^\s*\d+\.\s+", text):
                section_type = "section"
                match = re.search(r"(\d+\.)", text)
                if match:
                    identifier = f"Section {match.group(1)}"
            
            # Update section with extracted information
            if section_type:
                section.section_type = section_type
            if identifier:
                section.identifier = identifier
                
            # Process children recursively
            for child in section.children:
                process_section(child)
        
        # Process all sections starting from root
        process_section(root)

    def export_structure(self, output_format: str = "json") -> Any:
        """
        Export the document structure in various formats.
        
        Args:
            output_format: Format to export ("json", "dict", "xml")
            
        Returns:
            The document structure in the requested format
        """
        structure = self.extract_structure()
        
        if output_format == "dict":
            return structure.to_dict()
        elif output_format == "json":
            return json.dumps(structure.to_dict(), indent=2)
        elif output_format == "xml":
            # Simple XML conversion
            
            def section_to_xml(section, parent_elem):
                section_elem = ET.SubElement(parent_elem, "section")
                section_elem.set("title", section.title)
                section_elem.set("start_page", str(section.start_page))
                section_elem.set("end_page", str(section.end_page) if section.end_page is not None else "")
                section_elem.set("level", str(section.level))
                
                if section.logical_start_page is not None:
                    section_elem.set("logical_start_page", str(section.logical_start_page))
                if section.logical_end_page is not None:
                    section_elem.set("logical_end_page", str(section.logical_end_page))
                if section.section_type:
                    section_elem.set("section_type", section.section_type)
                if section.identifier:
                    section_elem.set("identifier", section.identifier)
                
                for child in section.children:
                    section_to_xml(child, section_elem)
                
                return section_elem
            
            root_elem = ET.Element("document")
            section_to_xml(structure, root_elem)
            
            xml_str = ET.tostring(root_elem, encoding='utf-8')
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
            return pretty_xml
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def extract_text_by_section(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Extract text content for each section in the document structure.
        
        Args:
            output_dir: Optional directory to save extracted text files
            
        Returns:
            Dictionary mapping section identifiers to extracted text
        """
        structure = self.extract_structure()
        result = {}
        
        def process_section(section, path=""):
            # Skip root
            if section.level == 0:
                for child in section.children:
                    process_section(child, path)
                return
                
            # Create path for this section
            section_path = f"{path}/{section.title}" if path else section.title
            section_path = re.sub(r'[\\/*?:"<>|]', "_", section_path)  # Remove invalid chars
            
            # Extract text from the section's page range
            text = ""
            if section.start_page is not None and section.end_page is not None:
                for page_idx in range(section.start_page, section.end_page + 1):
                    if page_idx < self.doc.page_count:
                        page = self.doc[page_idx]
                        text += page.get_text()
            
            # Save to result dictionary
            identifier = section.identifier or section_path
            result[identifier] = text
            
            # Save to file if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f"{section_path}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
            
            # Process children
            for child in section.children:
                process_section(child, section_path)
        
        # Process all sections
        process_section(structure)
        return result
