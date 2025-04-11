"""
PDF-specific document processing.
"""

import os
import re
import json
from typing import List, Optional

import fitz

from doc_parse_convert.config import logger, ProcessingConfig, ExtractionStrategy
from doc_parse_convert.extraction.base import DocumentProcessor
from doc_parse_convert.models.document import Chapter
from doc_parse_convert.models.content import ChapterContent, PageContent, TextBox, Table, Figure
from doc_parse_convert.utils.image import ImageConverter
from doc_parse_convert.ai.client import AIClient


class PDFProcessor(DocumentProcessor):
    """PDF-specific implementation of DocumentProcessor."""

    def __init__(self, config: ProcessingConfig):
        logger.info("Initializing PDFProcessor")
        super().__init__(config)
        self.doc = None
        self.ai_client = None
        self.file_path = None
        self._chapters_cache = None

    def load(self, file_path: str) -> None:
        """Load the PDF document."""
        logger.info(f"Loading PDF from {file_path}")
        try:
            self.doc = fitz.open(file_path)
            self.file_path = file_path
            self._chapters_cache = None  # Reset cache on new document load
            # Initialize AI client here instead of in constructor to ensure it's created after doc is loaded
            self.ai_client = AIClient(self.config)
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
                logger.warning("No native TOC found in document, returning empty list")
                self._chapters_cache = []
                return []
                
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
            if not self.ai_client or not self.ai_client.model:
                logger.error("AI model not initialized")
                raise ValueError("AI model not initialized")

            # Import required modules
            from doc_parse_convert.ai.prompts import get_content_extraction_prompt
            from doc_parse_convert.ai.schemas import get_content_extraction_schema
            from vertexai.generative_models import Part, GenerationConfig

            # Get schema and prompt
            response_schema = get_content_extraction_schema()

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
            parts.append(Part.from_text(get_content_extraction_prompt()))

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
                logger.error(f"Error in AI text extraction: {str(e)}")
                if response:
                    logger.error(f"API response: {response}")
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
