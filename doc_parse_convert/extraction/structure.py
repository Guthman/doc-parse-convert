"""
Document structure extraction functionality.
"""

import os
import re
import json
from typing import Any, Dict, Optional

import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from doc_parse_convert.config import logger, ExtractionStrategy
from doc_parse_convert.models.document import DocumentSection
from doc_parse_convert.utils.image import ImageConverter
from doc_parse_convert.exceptions import AIExtractionError
from doc_parse_convert.utils.validators import validate_page_range


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
            logger.info(f"Successfully created image generator for all document pages")
        except Exception as e:
            error_msg = f"Failed to create image generator: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        # Use the AI client to extract structure from the image generator
        try:
            chapters = self.ai_client.extract_structure_from_images(images)
            root.children = chapters
            logger.info(f"Successfully extracted document structure with {len(root.children)} top-level sections")
            return root
        except AIExtractionError as e:
            logger.error(f"Error in AI structure extraction: {str(e)}")
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

        # Get native table of contents with extended info (includes Y-positions)
        toc_extended = self.doc.get_toc(simple=False)

        if not toc_extended:
            logger.warning("No native TOC found, attempting to infer structure from document")
            return self._infer_structure_from_document(root)

        # Convert TOC to DocumentSection objects
        sections_by_level = {}  # Dictionary to keep track of the latest section at each level
        all_sections_list = []  # Track all sections with their Y-positions

        # First pass: create all sections
        for entry in toc_extended:
            level = entry[0]
            title = entry[1]
            page = entry[2]
            dest_dict = entry[3] if len(entry) > 3 else {}

            # Convert to 0-based page index
            page_idx = page - 1

            # Extract Y-position from destination info
            y_position = None
            if 'to' in dest_dict and hasattr(dest_dict['to'], 'y'):
                y_position = dest_dict['to'].y
            elif 'dest' in dest_dict:
                dest_str = dest_dict['dest']
                if '/FitH' in dest_str:
                    match = re.search(r'/FitH\s+([\d.]+)', dest_str)
                    if match:
                        y_position = float(match.group(1))
                elif '/XYZ' in dest_str:
                    match = re.search(r'/XYZ\s+[\d.]+\s+([\d.]+)', dest_str)
                    if match:
                        y_position = float(match.group(1))

            section = DocumentSection(
                title=title,
                start_page=page_idx,
                level=level,
                logical_start_page=page  # Store the logical page number as well
            )

            # Store Y-position for later use in determining end pages
            all_sections_list.append((section, y_position))

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
        # Create a mapping from section id to Y-position
        section_y_positions = {id(section): y_pos for section, y_pos in all_sections_list}

        # Get page height for Y-position calculations (PDF coords: 0,0 at bottom-left)
        # We'll use the first page as reference
        page_height = self.doc[0].rect.height if self.doc.page_count > 0 else 792  # Default letter size

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
                if next_section.level <= section.level and next_section.start_page >= section.start_page:
                    if next_section.start_page > section.start_page:
                        # Next section starts on a different page
                        # Check if next section has Y-offset (not at top of page)
                        next_y = section_y_positions.get(id(next_section))

                        # In PDF coordinates, Y increases upward from bottom
                        # A section at top of page would have Y near page_height
                        # If Y is significantly less than page_height, there's content above it
                        if next_y is not None and next_y < (page_height - 50):
                            # Next section starts partway down the page
                            # Include that page in current section
                            section.end_page = next_section.start_page
                        else:
                            # Next section starts at top of page
                            section.end_page = next_section.start_page - 1
                        break
                    elif next_section.start_page == section.start_page:
                        # Same page - check Y-positions to determine order
                        continue

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
                            if re.match(r"^(chapter|section|part|appendix|\d+\.)\s+\S+", text.lower()):
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
            text = page.get_text()[:500]  # Get first 500 characters

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
                """Convert a DocumentSection to XML element with proper escaping."""
                section_elem = ET.SubElement(parent_elem, "section")

                # ElementTree automatically escapes special characters in attribute values
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
                # ADD VALIDATION
                validate_page_range(
                    section.start_page,
                    section.end_page + 1,  # +1 because range is inclusive
                    self.doc.page_count
                )

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
