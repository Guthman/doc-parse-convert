import os
import re
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, TYPE_CHECKING, ForwardRef
from pathlib import Path
import datetime
import json

import fitz  # PyMuPDF
from PIL import Image
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    HarmCategory,
    HarmBlockThreshold,
)
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create file handler which logs even debug messages
file_handler = logging.FileHandler('content_extraction.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

GEMINI_SAFETY_CONFIG = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


class ExtractionStrategy(Enum):
    NATIVE = "native"
    AI = "ai"
    OCR = "ocr"


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    project_id: Optional[str] = None
    vertex_ai_location: Optional[str] = None
    gemini_model_name: Optional[str] = None
    service_account_file: Optional[str] = None
    use_application_default_credentials: bool = False
    toc_extraction_strategy: ExtractionStrategy = ExtractionStrategy.NATIVE  # Strategy for table of contents extraction
    content_extraction_strategy: ExtractionStrategy = ExtractionStrategy.NATIVE  # Strategy for chapter content extraction
    max_pages_for_preview: int = 200  # Default is to only look at first 200 pages
    image_quality: int = 300  # DPI for image conversion


@dataclass
class Figure:
    """Represents a figure in a document."""
    description: Optional[str] = None
    byline: Optional[str] = None


@dataclass
class TextBox:
    """Represents a text box or side note in a document."""
    content: str
    type: str  # e.g., 'text_box', 'side_note', 'callout'


@dataclass
class Table:
    """Represents a table in a document."""
    content: str
    caption: Optional[str] = None


@dataclass
class PageContent:
    """Represents the structured content of a page."""
    chapter_text: str
    text_boxes: List[TextBox] = None
    tables: List[Table] = None
    figures: List[Figure] = None

    def __post_init__(self):
        self.text_boxes = self.text_boxes or []
        self.tables = self.tables or []
        self.figures = self.figures or []


@dataclass
class ChapterContent:
    """Represents the structured content of a chapter."""
    title: str
    pages: List[PageContent]
    start_page: int
    end_page: int


@dataclass
class Chapter:
    """Represents a chapter in a document."""
    title: str
    start_page: int
    end_page: Optional[int] = None
    level: int = 1
    content: Optional[ChapterContent] = None


@dataclass
class DocumentSection:
    """Represents a section in a document structure hierarchy with physical and logical page information."""
    title: str
    start_page: int  # 0-based physical page index
    end_page: Optional[int] = None  # 0-based physical page index
    level: int = 0  # Depth in the document hierarchy (0 for document root, 1 for chapters, etc.)
    children: List["DocumentSection"] = None  # Subsections
    logical_start_page: Optional[int] = None  # As displayed in the document (e.g., "Page 1")
    logical_end_page: Optional[int] = None  # As displayed in the document
    section_type: Optional[str] = None  # E.g., "chapter", "section", "appendix"
    identifier: Optional[str] = None  # E.g., "Chapter 1", "Appendix A"

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []
            
    def add_child(self, child: "DocumentSection") -> None:
        """Add a child section to this section."""
        if self.children is None:
            self.children = []
        self.children.append(child)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the section to a dictionary representation."""
        result = {
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "level": self.level,
        }
        
        if self.logical_start_page is not None:
            result["logical_start_page"] = self.logical_start_page
        if self.logical_end_page is not None:
            result["logical_end_page"] = self.logical_end_page
        if self.section_type:
            result["section_type"] = self.section_type
        if self.identifier:
            result["identifier"] = self.identifier
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
            
        return result


# Create forward reference to DocumentProcessor for type hinting
if TYPE_CHECKING:
    from typing import Type
    DocumentProcessorType = Type['DocumentProcessor']
else:
    DocumentProcessorType = ForwardRef('DocumentProcessor')


class DocumentStructureExtractor:
    """Class for extracting hierarchical document structure with page ranges."""
    
    def __init__(self, processor: DocumentProcessorType):
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
        
        # Define a more comprehensive prompt for structure extraction
        parts = []
        
        # Limit number of images to process (preventing request size issues)
        # max_images = min(len(images), 30)  # Process at most 30 pages to avoid oversized requests
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
            parts.append(Part.from_text(
                """Analyze this document and extract its hierarchical structure. 
                Identify all sections and subsections with their titles, page numbers, and levels.
                
                Format your response as a JSON object with this structure:
                {
                  "title": "Document Title",
                  "sections": [
                    {
                      "title": "Section 1 Title",
                      "start": 1,
                      "end": 10,
                      "level": 1,
                      "children": [
                        {
                          "title": "Subsection 1.1",
                          "start": 2,
                          "end": 5,
                          "level": 2,
                          "children": []
                        }
                      ]
                    }
                  ]
                }
                
                Note: All page numbers should be 1-based (first page is 1, not 0)."""
            ))
        except Exception as e:
            logger.error(f"Failed to create instruction text part: {str(e)}")
            raise ValueError(f"Failed to create instruction text part: {str(e)}") from e
        
        generation_config = GenerationConfig(
            temperature=0.1  # Low temperature for more deterministic results
        )
        
        # Define simplified response schema (with shorter property names)
        response_schema = {
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
        
        # Call AI with retry
        try:
            logger.debug("Calling AI model to extract document structure")
            
            # Define a very simple schema with minimal nesting and property validation
            simplified_schema = {
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
                }
            }
            
            # Try with minimal schema to avoid InvalidArgument errors
            response = self.ai_client._call_model_with_retry(
                parts,
                generation_config,
                response_mime_type="application/json",
                response_schema=simplified_schema
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
                        if (potential_parent.level < section.level and 
                            potential_parent.start_page <= section.start_page and
                            (potential_parent.end_page is None or 
                             potential_parent.end_page >= section.end_page)):
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
            import xml.dom.minidom as minidom
            import xml.etree.ElementTree as ET
            
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


class DocumentProcessor(ABC):
    """Base class for document processors."""

    def __init__(self, config: ProcessingConfig):
        """Initialize the document processor.
        
        Args:
            config: Configuration for document processing
        """
        self.config = config
        self.doc = None  # Document object to be initialized by subclasses
        self.ai_client = None  # AI client to be initialized
        self._initialize_ai_client()

    def _initialize_ai_client(self) -> None:
        """Initialize AI client if credentials are provided."""
        if not self.config.project_id or not self.config.vertex_ai_location:
            logger.debug("No project ID or location provided, skipping AI client initialization")
            return

        try:
            if self.config.use_application_default_credentials:
                # Use application default credentials
                vertexai.init(
                    project=self.config.project_id,
                    location=self.config.vertex_ai_location
                )
            elif self.config.service_account_file:
                # Use service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.service_account_file
                )
                vertexai.init(
                    project=self.config.project_id,
                    location=self.config.vertex_ai_location,
                    credentials=credentials
                )
            
            # Initialize the AI client
            self.ai_client = AIClient(self.config)
            logger.info("Successfully initialized AI client")
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {str(e)}")
            self.ai_client = None

    @abstractmethod
    def load(self, file_path: str) -> None:
        """Load the document from the given path."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the document and free resources."""
        pass

    @abstractmethod
    def get_table_of_contents(self) -> List[Chapter]:
        """Extract the table of contents."""
        pass

    @abstractmethod
    def split_by_chapters(self, output_dir: str) -> None:
        """Split the document into separate files by chapters."""
        pass

    @abstractmethod
    def extract_chapter_text(self, chapter: Chapter) -> ChapterContent:
        """Extract text from a specific chapter."""
        pass


class ContentExtractor(ABC):
    """Base class for content extractors."""

    def __init__(self, config: ProcessingConfig):
        self.config = config

    @abstractmethod
    def extract_text(self, content: Any) -> str:
        """Extract text from the given content."""
        pass

    @abstractmethod
    def extract_structure(self, content: Any) -> List[Chapter]:
        """Extract structural information from the content."""
        pass


class ImageConverter:
    """Utility class for converting document pages to images.
    
    This class can be used in two ways:
    1. As an iterator to get raw image data:
       ```
       converter = ImageConverter('document.pdf')
       for page_number, page in converter:
           with open(f'document_{page_number}.png', 'wb') as f:
               f.write(page)
       ```
       
    2. Using the static method to get Vertex AI compatible image objects:
       ```
       images = ImageConverter.convert_to_images(document)
       ```
    """
    
    def __init__(self, file_path: str | Path, format: str = "png", dpi: int = 300):
        """Initialize the image converter.
        
        Args:
            file_path: Path to the document file to convert
            format: Image format to use ('png' or 'jpg')
            dpi: DPI for image conversion
        """
        self.file_path = file_path
        self.format = format.lower()
        if self.format not in ['png', 'jpg']:
            raise ValueError("Format must be either 'png' or 'jpg'")
        self.dpi = dpi
        self.doc = None
        self.current_page = 0
        
        # Open the document
        self._open_document()
    
    def _open_document(self):
        """Open the document for conversion."""
        if self.doc is not None:
            self.close()
        
        try:
            self.doc = fitz.open(self.file_path)
            self.current_page = 0
        except Exception as e:
            raise ValueError(f"Failed to open document: {str(e)}")
    
    def close(self):
        """Close the document and free resources."""
        if self.doc:
            self.doc.close()
            self.doc = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __iter__(self):
        """Return iterator for the converter."""
        self.current_page = 0
        return self
    
    def __next__(self):
        """Get the next page as an image.
        
        Returns:
            Tuple of (page_number, image_data_bytes)
        """
        if self.doc is None:
            raise ValueError("Document not open")
        
        if self.current_page >= self.doc.page_count:
            raise StopIteration
        
        page_number = self.current_page
        page = self.doc.load_page(page_number)
        pix = page.get_pixmap(dpi=self.dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG' if self.format == 'png' else 'JPEG')
        img_data = img_byte_arr.getvalue()
        
        self.current_page += 1
        return page_number, img_data
    
    def set_format(self, format: str):
        """Set the image format.
        
        Args:
            format: Image format to use ('png' or 'jpg')
        """
        format = format.lower()
        if format not in ['png', 'jpg']:
            raise ValueError("Format must be either 'png' or 'jpg'")
        self.format = format
        return self
    
    @staticmethod
    def convert_to_images(document: Any, num_pages: int = 20, start_page: int = 0, image_quality: int = 300) -> List[Dict[str, Any]]:
        """Convert document pages to images.
        
        Args:
            document: Document to convert (e.g., PDF)
            num_pages: Number of pages to convert
            start_page: Starting page number (0-based index)
            image_quality: DPI for image conversion (default: 300)
        
        Returns:
            List of dictionaries containing image data in format expected by Vertex AI
            
        Raises:
            ValueError: If document type is not supported or conversion fails
            NotImplementedError: If conversion is not implemented for the document type
        """
        if isinstance(document, fitz.Document):
            images = []
            # Calculate the end page, ensuring we don't exceed document bounds
            end_page = min(start_page + num_pages, document.page_count)
            
            # Estimate reasonable quality based on page count to prevent oversized requests
            # Adjust quality down if we have many pages to process
            adjusted_quality = image_quality
            # if num_pages > 50:
            #     adjusted_quality = min(image_quality, 200)  # Medium quality for many pages
            # if num_pages > 100:
            #     adjusted_quality = min(image_quality, 150)  # Lower quality for very many pages
            
            if adjusted_quality != image_quality:
                logger.info(f"Adjusting image quality from {image_quality} to {adjusted_quality} DPI due to large page count ({num_pages})")
            
            for i in range(start_page, end_page):
                try:
                    page = document.load_page(i)
                    
                    # Calculate matrix for the specified DPI
                    # 72 is the base DPI for PDF
                    zoom = adjusted_quality / 72
                    matrix = fitz.Matrix(zoom, zoom)
                    
                    pix = page.get_pixmap(matrix=matrix)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    
                    # Convert to bytes
                    img_byte_arr = io.BytesIO()
                    
                    # Use compression to reduce file size (quality=85 provides good balance)
                    img.save(img_byte_arr, format='PNG', optimize=True)
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Check image size and log warning if it's large
                    img_size_mb = len(img_byte_arr) / (1024 * 1024)
                    if img_size_mb > 10:
                        logger.warning(f"Page {i+1} image is very large: {img_size_mb:.2f}MB")
                    elif img_size_mb > 5:
                        logger.debug(f"Page {i+1} image is large: {img_size_mb:.2f}MB")
                    
                    # Create image object in format expected by Vertex AI
                    image_obj = {
                        "data": img_byte_arr,
                        "_mime_type": "image/png"
                    }
                    images.append(image_obj)
                    
                except Exception as e:
                    logger.error(f"Error converting page {i+1}: {str(e)}")
                    # Continue with other pages if possible
            
            if not images:
                raise ValueError(f"Failed to convert any pages from document")
            
            return images
        
        raise NotImplementedError(f"Conversion not implemented for {type(document)}")


class AIClient:
    """Manages interactions with AI APIs."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model = None
        if config.gemini_model_name:
            logger.info(f"Initializing AI model with {config.gemini_model_name}")
            self.model = GenerativeModel(config.gemini_model_name)
        else:
            logger.warning("No Gemini model name provided in config")

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(5))
    def _call_model_with_retry(self, parts: List[Part], generation_config: GenerationConfig, response_mime_type: str = None, response_schema: dict = None, attempt: int = 0) -> Any:
        """Call the AI model with retry logic."""
        if not self.model:
            logger.error("AI model not initialized")
            raise ValueError("AI model not initialized")

        try:
            # Log request details
            logger.debug(f"API Request - Parts count: {len(parts)}")
            logger.debug(f"API Request - Generation config: {generation_config}")
            if response_schema:
                logger.debug(f"API Request - Response schema: {response_schema}")
                
            # For text parts, log the content (truncated if too long)
            for i, part in enumerate(parts):
                if hasattr(part, 'text'):
                    text = part.text
                    if len(text) > 500:
                        text = text[:500] + "... [truncated]"
                    logger.debug(f"API Request - Text part {i}: {text}")
                elif hasattr(part, 'mime_type') and part.mime_type.startswith('image/'):
                    if hasattr(part, 'data'):
                        img_size = len(part.data) if part.data else 0
                        logger.debug(f"API Request - Image part {i}: {part.mime_type}, size: {img_size} bytes")
                    else:
                        logger.debug(f"API Request - Image part {i}: {part.mime_type}")

            # Create new config with adjusted temperature
            base_temp = 0.0  # Default temperature if not specified
            if hasattr(generation_config, 'temperature'):
                base_temp = generation_config.temperature
            
            adjusted_temp = min(base_temp + (attempt * 0.1), 1.0)
            logger.debug(f"Attempt {attempt + 1}/10 with temperature {adjusted_temp:.2f}")

            # Create new config with all parameters
            config_params = {
                'temperature': adjusted_temp,
                'candidate_count': 1,  # Required for structured output
            }
            
            if response_mime_type:
                config_params['response_mime_type'] = response_mime_type
            
            if response_schema:
                config_params['response_schema'] = response_schema

            adjusted_config = GenerationConfig(**config_params)

            logger.debug("Calling AI model with adjusted configuration")
            response = self.model.generate_content(
                parts,
                generation_config=adjusted_config,
                safety_settings=GEMINI_SAFETY_CONFIG
            )

            if not hasattr(response, 'text') or not response.text:
                logger.error("Received invalid or empty response from model")
                raise ValueError("Invalid or empty response from model")

            # Log response (truncated if too long)
            response_text = response.text
            if len(response_text) > 500:
                logger.debug(f"API Response: {response_text[:500]}... [truncated]")
            else:
                logger.debug(f"API Response: {response_text}")

            logger.debug("Successfully received valid response from model")
            return response

        except Exception as e:
            logger.error(f"Error during model call: {e.__class__.__name__} {str(e)}")
            
            # Log detailed error information if available
            if hasattr(e, 'response'):
                if hasattr(e.response, 'text'):
                    logger.error(f"Error response text: {e.response.text}")
                elif hasattr(e.response, 'content'):
                    logger.error(f"Error response content: {e.response.content}")
                elif hasattr(e.response, 'json'):
                    try:
                        logger.error(f"Error response JSON: {e.response.json()}")
                    except:
                        logger.error(f"Error response object: {e.response}")
            
            # For Google API errors, extract more details
            if hasattr(e, 'details'):
                logger.error(f"Error details: {e.details}")
            if hasattr(e, 'code'):
                logger.error(f"Error code: {e.code}")
            
            # Debug API request details for InvalidArgument errors
            if 'InvalidArgument' in e.__class__.__name__:
                logger.error(f"API request might contain invalid arguments. Check image sizes and request structure.")
                logger.debug(f"Using model: {self.config.gemini_model_name}")
                logger.debug(f"Project ID: {self.config.project_id}")
                logger.debug(f"Location: {self.config.vertex_ai_location}")
                
                # Check if any parts exceed size limits
                total_size = 0
                for i, part in enumerate(parts):
                    if hasattr(part, 'data') and part.data:
                        part_size = len(part.data)
                        total_size += part_size
                        if part_size > 50 * 1024 * 1024:  # 50MB
                            logger.error(f"Image part {i} exceeds 50MB limit: {part_size / (1024 * 1024):.2f}MB")
                
                logger.debug(f"Total request size: {total_size / (1024 * 1024):.2f}MB")
            
            # Save problematic images to disk for debugging
            debug_dir = os.environ.get("AI_DEBUG_DIR")
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_subdir = os.path.join(debug_dir, f"error_{timestamp}")
                os.makedirs(debug_subdir, exist_ok=True)
                
                # Save debug info
                with open(os.path.join(debug_subdir, "error_info.txt"), "w") as f:
                    f.write(f"Error: {e.__class__.__name__} - {str(e)}\n")
                    f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                    f.write(f"Model: {self.config.gemini_model_name}\n")
                    f.write(f"Temperature: {adjusted_temp}\n")
                    f.write(f"Attempt: {attempt + 1}/10\n")
                    
                    if response_schema:
                        f.write(f"\nResponse Schema:\n{json.dumps(response_schema, indent=2)}\n")
                
                # Save images
                for i, part in enumerate(parts):
                    if hasattr(part, 'mime_type') and getattr(part, 'mime_type', '').startswith('image/'):
                        try:
                            ext = part.mime_type.split('/')[-1]
                            debug_path = os.path.join(debug_subdir, f"image_{i}.{ext}")
                            with open(debug_path, 'wb') as f:
                                f.write(part.data)
                        except Exception as img_error:
                            logger.error(f"Failed to save debug image {i}: {str(img_error)}")
                
                logger.info(f"Saved debug information to {debug_subdir}")
            
            # Re-raise to allow retry
            raise

    def extract_structure_from_images(self, images: List[Dict[str, Any]]) -> List[Chapter]:
        """Extract structural information from document images using AI."""
        logger.info("Starting structure extraction from images")
        
        if not self.model:
            logger.error("AI model not initialized")
            raise ValueError("AI model not initialized")

        # Ultra-simplified schema with just the bare essentials
        # This minimalist approach avoids Vertex AI schema complexity limitations
        response_schema = {
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

        # Limit number of images to process (preventing request size issues)
        # max_images = min(len(images), 20)  # Process at most 20 pages to reduce API payload size
        max_images = 1000
        logger.info(f"Using {max_images} out of {len(images)} pages for TOC extraction")
        
        # Create Part objects from image data
        parts = []
        for i, img in enumerate(images[:max_images]):
            try:
                logger.debug(f"Processing image {i + 1}/{max_images}")
                parts.append(Part.from_data(data=img["data"], mime_type=img["_mime_type"]))
            except Exception as e:
                logger.warning(f"Failed to process image {i + 1}: {str(e)}")
                continue

        if not parts:
            logger.error("No valid images to process")
            raise ValueError("No valid images to process")

        logger.debug("Adding instruction text to parts")
        parts.append(Part.from_text(
            """Extract the table of contents from this document. Find all chapter titles and their page numbers.
            Format the output as a JSON array of chapters, where each item has these fields:
            - "t": The chapter title text (string)
            - "p": The page number where the chapter starts (number)
            - "l": The hierarchy level, where 1 is top level, 2 is subchapter, etc. (number)
            
            Important notes:
            - Use exactly these field names: t, p, l
            - Keep structure simple and flat (a plain array of objects)
            - Capture all chapters and sections you can find
            """
        ))

        generation_config = GenerationConfig(
            temperature=0.0  # Explicitly set temperature
        )

        try:
            logger.debug("Calling AI model with retry")
            response = self._call_model_with_retry(
                parts, 
                generation_config,
                response_mime_type="application/json",
                response_schema=response_schema
            )
            
            logger.debug("Parsing JSON response")
            response_text = response.text
            toc_data = json.loads(response_text)
            
            if not isinstance(toc_data, list):
                logger.error(f"Invalid response format: expected list, got {type(toc_data)}")
                raise ValueError(f"Expected list response, got {type(toc_data)}")
                
            chapters = []
            for i, item in enumerate(toc_data):
                if not isinstance(item, dict):
                    logger.warning(f"Skipping invalid item {i} in response: {item}")
                    continue
                    
                try:
                    logger.debug(f"Processing chapter item {i + 1}")
                    # Map shortened property names to our internal names
                    title = str(item.get("t", "")).strip()
                    page = int(item.get("p", 1))
                    level = int(item.get("l", 1))
                    
                    chapters.append(Chapter(
                        title=title,
                        start_page=page - 1,  # Convert to 0-based
                        level=level
                    ))
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to process chapter item {i + 1}: {str(e)}")
                    continue

            # Set end pages
            logger.debug("Setting chapter end pages")
            
            # Sort chapters by page number first, then by level if they appear on the same page
            # This handles cases where multiple chapters appear on the same page
            chapters.sort(key=lambda x: (x.start_page, x.level))
            
            for i in range(len(chapters) - 1):
                chapters[i].end_page = chapters[i + 1].start_page

            if not chapters:
                logger.warning("No valid chapters extracted from AI response")
            else:
                logger.info(f"Successfully extracted {len(chapters)} chapters")
                
            return chapters
            
        except Exception as e:
            logger.error(f"Error processing AI response: {str(e)}")
            # Only try to log response text if response exists and has the text attribute
            response_text = "No response text"
            try:
                if 'response' in locals() and hasattr(response, 'text'):
                    response_text = response.text
            except Exception:
                pass
            logger.debug(f"Raw response: {response_text}")
            
            # Additional diagnostics for InvalidArgument errors
            if 'InvalidArgument' in e.__class__.__name__:
                logger.error("API request might contain invalid arguments. This could be due to:")
                logger.error("- Images too large or too many images in the request")
                logger.error("- Malformed request structure or invalid parameters")
                logger.error("- Model limitations or incompatible response schema")
                logger.error("Try with fewer pages or further simplify the schema")
            
            return []


class PDFProcessor(DocumentProcessor):
    """PDF-specific implementation of DocumentProcessor."""

    def __init__(self, config: ProcessingConfig):
        logger.info("Initializing PDFProcessor")
        super().__init__(config)
        self.doc = None
        self.ai_client = AIClient(config)
        self.file_path = None
        self._chapters_cache = None

    def load(self, file_path: str) -> None:
        """Load the PDF document."""
        logger.info(f"Loading PDF from {file_path}")
        try:
            self.doc = fitz.open(file_path)
            self.file_path = file_path
            self._chapters_cache = None  # Reset cache on new document load
            logger.info(f"Successfully loaded PDF with {self.doc.page_count} pages")
        except Exception as e:
            logger.error(f"Failed to load PDF: {str(e)}")
            raise

    def close(self) -> None:
        """Close the PDF document."""
        if self.doc:
            logger.info("Closing PDF document")
            self.doc.close()
            self.doc = None
            self._chapters_cache = None  # Clear cache on close
        else:
            logger.debug("No document to close")

    def get_table_of_contents(self) -> List[Chapter]:
        """
        Extract the table of contents using the configured strategy without fallbacks.
        
        Returns:
            List[Chapter]: Table of contents as a list of chapters
            
        Raises:
            ValueError: If document not loaded or strategy is not supported
            Exception: If extraction fails
        """
        if self._chapters_cache is not None:
            return self._chapters_cache

        if not self.doc:
            logger.error("Document not loaded")
            raise ValueError("Document not loaded")

        if self.config.toc_extraction_strategy == ExtractionStrategy.NATIVE:
            logger.info("Using native TOC extraction")
            toc = self.doc.get_toc()
            if not toc:
                logger.error("No native TOC found in document")
                raise ValueError("No native TOC found in document")
                
            chapters = []
            for level, title, page in toc:
                if level == 1:  # Only top-level chapters
                    chapters.append(Chapter(
                        title=title,
                        start_page=page - 1,  # Convert to 0-based indexing
                        level=level
                    ))

            # Set end pages
            for i in range(len(chapters) - 1):
                chapters[i].end_page = chapters[i + 1].start_page
            if chapters:
                chapters[-1].end_page = self.doc.page_count

            self._chapters_cache = chapters
            logger.info(f"Successfully extracted {len(chapters)} chapters using native method")
            return chapters
            
        elif self.config.toc_extraction_strategy == ExtractionStrategy.AI:
            logger.info("Using AI for TOC extraction")
            if not self.ai_client or not self.ai_client.model:
                logger.error("AI client not initialized")
                raise ValueError("AI client not initialized")
                
            images = ImageConverter.convert_to_images(
                self.doc,
                num_pages=self.config.max_pages_for_preview,
                start_page=0
            )
            chapters = self.ai_client.extract_structure_from_images(images)
            
            if not chapters:
                logger.error("AI extraction failed to extract any chapters")
                raise ValueError("AI extraction failed to extract any chapters")
                
            self._chapters_cache = chapters
            logger.info(f"Successfully extracted {len(chapters)} chapters using AI method")
            return chapters
            
        else:
            error_msg = f"Unsupported extraction strategy: {self.config.toc_extraction_strategy}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def split_by_chapters(self, output_dir: str) -> None:
        """Split the PDF into separate files by chapters."""
        if not self.doc:
            raise ValueError("Document not loaded")

        chapters = self.get_table_of_contents()
        if not chapters:
            raise ValueError("No chapters found in document")

        base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        for i, chapter in enumerate(chapters):
            start_page = chapter.start_page
            end_page = chapter.end_page or self.doc.page_count

            # Ensure valid page range
            start_page = min(start_page, self.doc.page_count - 1)
            end_page = min(end_page, self.doc.page_count)

            # Create chapter document
            chapter_doc = fitz.open()
            chapter_doc.insert_pdf(self.doc, from_page=start_page, to_page=end_page - 1)

            # Save chapter
            chapter_title = re.sub(r'[^\w\-_\. ]', '_', chapter.title)
            output_filename = f"{base_filename}_{i + 1:02d}-{chapter_title}.pdf"
            output_path = os.path.join(output_dir, output_filename)

            chapter_doc.save(output_path)
            chapter_doc.close()

    def extract_chapter_text(self, chapter: Chapter) -> ChapterContent:
        """Extract text from a specific chapter using the configured strategy."""
        logger.info(f"Extracting text from chapter: {chapter.title}")
        logger.debug(f"Chapter details - Start page: {chapter.start_page}, End page: {chapter.end_page}")

        if not self.doc:
            logger.error("Document not loaded")
            raise ValueError("Document not loaded")

        if self.config.content_extraction_strategy == ExtractionStrategy.NATIVE:
            logger.debug("Using native extraction strategy")
            pages = []
            start_page = chapter.start_page
            end_page = chapter.end_page or self.doc.page_count

            for page_num in range(start_page, end_page):
                logger.debug(f"Processing page {page_num + 1}")
                page = self.doc[page_num]
                pages.append(PageContent(chapter_text=page.get_text()))

            logger.info(f"Successfully extracted {len(pages)} pages using native strategy")
            return ChapterContent(
                title=chapter.title,
                pages=pages,
                start_page=start_page,
                end_page=end_page
            )

        elif self.config.content_extraction_strategy == ExtractionStrategy.AI:
            logger.debug("Using AI extraction strategy")
            # Convert chapter pages to images
            images = ImageConverter.convert_to_images(
                self.doc,
                num_pages=(chapter.end_page or self.doc.page_count) - chapter.start_page,
                start_page=chapter.start_page
            )
            logger.debug(f"Converted {len(images)} pages to images")

            # Use AI to extract text
            if not self.ai_client.model:
                logger.error("AI model not initialized")
                raise ValueError("AI model not initialized")

            response_schema = {
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

            # Create Part objects from image data
            parts = []
            for i, img in enumerate(images):
                try:
                    logger.debug(f"Processing image {i + 1}/{len(images)}")
                    parts.append(Part.from_data(data=img["data"], mime_type=img["_mime_type"]))
                except Exception as e:
                    logger.warning(f"Failed to process image {i + 1}: {str(e)}")
                    continue

            if not parts:
                logger.error("No valid images to process")
                raise ValueError("No valid images to process")

            logger.debug("Adding instruction text to parts")
            parts.append(Part.from_text(
                """Extract text VERBATIM from these images and format all output in markdown. Preserve the original document structure and formatting. REPRODUCE ALL TEXT WORD FOR WORD, DON'T PARAPHRASE.

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
            ))

            generation_config = GenerationConfig(
                temperature=0.0  # Explicitly set temperature
            )

            response = None
            try:
                logger.debug("Calling AI model with retry")
                response = self.ai_client._call_model_with_retry(
                    parts, 
                    generation_config,
                    response_mime_type="application/json",
                    response_schema=response_schema
                )

                logger.debug("Parsing JSON response")
                response_text = response.text
                pages_data = json.loads(response_text)

                pages = []
                for i, page_data in enumerate(pages_data):
                    logger.debug(f"Processing page data {i + 1}/{len(pages_data)}")
                    try:
                        text_boxes = [
                            TextBox(content=tb["content"], type=tb["type"])
                            for tb in page_data.get("text_boxes", [])
                        ]
                        
                        tables = [
                            Table(content=t["content"], caption=t.get("caption"))
                            for t in page_data.get("tables", [])
                        ]
                        
                        figures = [
                            Figure(description=f.get("description"), byline=f.get("byline"))
                            for f in page_data.get("figures", [])
                        ]
                        
                        pages.append(PageContent(
                            chapter_text=page_data["chapter_text"],
                            text_boxes=text_boxes,
                            tables=tables,
                            figures=figures
                        ))
                    except Exception as e:
                        logger.error(f"Error processing page {i + 1}: {str(e)}")
                        continue

                logger.info(f"Successfully processed {len(pages)} pages")
                return ChapterContent(
                    title=chapter.title,
                    pages=pages,
                    start_page=chapter.start_page,
                    end_page=chapter.end_page or self.doc.page_count
                )

            except Exception as e:
                logger.error(f"Error in AI text extraction: {str(e)}\nAPI response: {response}")
                raise

        else:
            logger.error(f"Unsupported extraction strategy: {self.config.content_extraction_strategy}")
            raise ValueError(f"Unsupported extraction strategy: {self.config.content_extraction_strategy}")

    def extract_chapters(self, chapter_indices: Optional[List[int]] = None) -> List[Chapter]:
        """Extract content from specified chapters.
        
        Args:
            chapter_indices: List of chapter indices to extract. If None, extracts all chapters.
            
        Returns:
            List of Chapter objects with their content populated.
        """
        if not self.doc:
            raise ValueError("Document not loaded")
            
        # Use cached chapters if available, otherwise get them
        chapters = self.get_table_of_contents()
        if not chapters:
            raise ValueError("No chapters found in document")
            
        # If no specific chapters requested, process all chapters
        if chapter_indices is None:
            chapter_indices = list(range(len(chapters)))
            
        # Validate indices
        if not all(0 <= i < len(chapters) for i in chapter_indices):
            raise ValueError(f"Invalid chapter index. Valid range is 0-{len(chapters)-1}")
        # Extract content for specified chapters
        for i in chapter_indices:
            chapter = chapters[i]
            chapter.content = self.extract_chapter_text(chapter)
            
        return [chapters[i] for i in chapter_indices]


class ProcessorFactory:
    """Factory for creating document processors."""

    @staticmethod
    def create_processor(file_path: str, config: ProcessingConfig) -> DocumentProcessor:
        """Create and initialize appropriate processor based on file extension.
        
        Args:
            file_path: Path to the document file
            config: Processing configuration
            
        Returns:
            Initialized document processor
            
        Raises:
            ValueError: If file format is not supported
        """
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            processor = PDFProcessor(config)
        # elif ext in ['.epub']:  # Future implementation
        #     processor = EPUBProcessor(config)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        # Load the document
        processor.load(file_path)
        return processor


# Resolve forward references
if not TYPE_CHECKING:
    DocumentProcessorType.__forward_arg__ = 'DocumentProcessor'
    DocumentProcessorType._evaluate(globals(), None, recursive_guard=frozenset())
