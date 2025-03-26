import os
import re
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path

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
    max_pages_for_preview: int = 20  # Default is to only look at first 20 pages
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


class DocumentProcessor(ABC):
    """Base class for document processors."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._initialize_ai_client()

    def _initialize_ai_client(self) -> None:
        """Initialize AI client if credentials are provided."""
        if not self.config.project_id or not self.config.vertex_ai_location:
            return

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
    """Utility class for converting document pages to images."""

    @staticmethod
    def convert_to_images(document: Any, num_pages: int = 20, start_page: int = 0) -> List[Dict[str, Any]]:
        """Convert document pages to images.
        
        Args:
            document: Document to convert (e.g., PDF)
            num_pages: Number of pages to convert
            start_page: Starting page number (0-based index)
        
        Returns:
            List of dictionaries containing image data in format expected by Vertex AI
        """
        if isinstance(document, fitz.Document):
            images = []
            # Calculate the end page, ensuring we don't exceed document bounds
            end_page = min(start_page + num_pages, document.page_count)
            
            for i in range(start_page, end_page):
                page = document.load_page(i)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Create image object in format expected by Vertex AI
                image_obj = {
                    "data": img_byte_arr,
                    "_mime_type": "image/png"
                }
                images.append(image_obj)
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

            logger.debug("Successfully received valid response from model")
            return response

        except Exception as e:
            logger.error(f"Error during model call: {str(e)}")
            raise

    def extract_structure_from_images(self, images: List[Dict[str, Any]]) -> List[Chapter]:
        """Extract structural information from document images using AI."""
        logger.info("Starting structure extraction from images")
        
        if not self.model:
            logger.error("AI model not initialized")
            raise ValueError("AI model not initialized")

        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "chapter": {"type": "STRING"},
                    "page": {"type": "INTEGER"},
                    "level": {"type": "INTEGER"}
                },
                "required": ["chapter", "page"]
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
            "Extract the table of contents from these PDF images, including chapter titles and page numbers. "
            "Skip pages which do not contain any parts of the table of contents. "
            "Provide the output in JSON format."
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
            import json
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
                    chapters.append(Chapter(
                        title=str(item.get("chapter", "")).strip(),
                        start_page=int(item.get("page", 1)) - 1,
                        level=int(item.get("level", 1))
                    ))
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to process chapter item {i + 1}: {str(e)}")
                    continue

            # Set end pages
            logger.debug("Setting chapter end pages")
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
        """Extract the table of contents from the PDF."""
        if self._chapters_cache is not None:
            return self._chapters_cache

        if not self.doc:
            logger.error("Document not loaded")
            raise ValueError("Document not loaded")

        # Always try native extraction first
        logger.debug("Attempting native TOC extraction")
        toc = self.doc.get_toc()
        if toc:
            logger.info("Successfully found native TOC")
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

        # If native extraction failed and AI is allowed, try AI extraction
        if self.config.toc_extraction_strategy == ExtractionStrategy.AI:
            logger.info("Native TOC extraction failed, falling back to AI extraction")
            images = ImageConverter.convert_to_images(
                self.doc,
                num_pages=self.config.max_pages_for_preview,
                start_page=0
            )
            chapters = self.ai_client.extract_structure_from_images(images)
            if chapters:
                self._chapters_cache = chapters
                return chapters
            else:
                logger.warning("AI TOC extraction failed")

        # If we get here, either:
        # 1. Native failed and AI is not allowed
        # 2. Both native and AI failed
        if not self.config.toc_extraction_strategy == ExtractionStrategy.AI:
            logger.warning("Native TOC extraction failed and AI extraction not enabled")
        else:
            logger.warning("Both native and AI TOC extraction methods failed")
        
        return []

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
                import json
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
