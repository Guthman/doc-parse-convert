"""
PDF-specific document processing.
"""

import os
import re
import json
from typing import List, Optional, Union, BinaryIO, Dict
from pathlib import Path

import fitz

from doc_parse_convert.config import logger, ProcessingConfig, ExtractionStrategy
from doc_parse_convert.extraction.base import DocumentProcessor
from doc_parse_convert.models.document import Chapter
from doc_parse_convert.models.content import ChapterContent, PageContent, TextBox, Table, Figure
from doc_parse_convert.utils.image import ImageConverter
from doc_parse_convert.ai.client import AIClient
from doc_parse_convert.utils.io_helpers import validate_file_input
from doc_parse_convert.utils.validators import validate_page_range
from pathvalidate import sanitize_filename


class PDFProcessor(DocumentProcessor):
    """PDF-specific implementation of DocumentProcessor."""

    def __init__(self, config: ProcessingConfig):
        logger.info("Initializing PDFProcessor")
        super().__init__(config)
        self.doc = None
        self.file_path = None
        self._chapters_cache = None

    def load(self, file_input: Union[str, Path, BinaryIO]) -> None:
        """Load the PDF document.

        Args:
            file_input: Either a file path or file-like object.

                       SECURITY WARNING: If using file paths, YOU MUST
                       validate them against your secure base directory
                       BEFORE passing to this method. See io_helpers.py
                       for validation examples.

        Raises:
            ValueError: If document cannot be loaded
            FileNotFoundError: If file path doesn't exist
        """
        logger.info(f"Loading PDF")
        try:
            # Preserve original path if string or Path was passed
            if isinstance(file_input, str):
                self.file_path = file_input
            elif isinstance(file_input, Path):
                self.file_path = str(file_input)
            else:
                self.file_path = getattr(file_input, 'name', '<stream>')

            file_obj = validate_file_input(file_input)
            # Read content and open as stream to support file-like objects
            content = file_obj.read()
            self.doc = fitz.open(stream=content, filetype="pdf")
            self._chapters_cache = None
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
        Extract the table of contents using the configured strategy.

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

            # ADD VALIDATION
            validate_page_range(start_page, end_page, self.doc.page_count)

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

        start_page = chapter.start_page
        end_page = chapter.end_page or self.doc.page_count

        # ADD VALIDATION
        validate_page_range(start_page, end_page, self.doc.page_count)

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
            logger.debug(f"Created image generator for chapter pages")

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

    def split_by_structure(
        self,
        output_dir: str,
        max_depth: Optional[int] = None,
        skip_empty_sections: bool = True,
        use_hierarchical_names: bool = False
    ) -> Dict[str, str]:
        """
        Split PDF by hierarchical document structure from TOC.

        Creates separate PDF files for each section in the document hierarchy,
        up to the specified depth. Uses naive approach where same-page sections
        create duplicate PDFs.

        Args:
            output_dir: Directory for output PDF files
            max_depth: Maximum hierarchy depth to split (None = all levels, 1 = chapters only, etc.)
            skip_empty_sections: Skip sections with no pages (default True)
            use_hierarchical_names: If True, use heading texts in filename (e.g., "Chapter_Section_Subsection.pdf")
                                   If False, use numbers (e.g., "01.02.03_Title.pdf")

        Returns:
            Dict mapping section titles to output file paths

        Raises:
            ValueError: If document not loaded or no TOC found
        """
        if not self.doc:
            logger.error("Document not loaded")
            raise ValueError("Document not loaded")

        logger.info(f"Splitting PDF by structure (max_depth={max_depth})")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Extract hierarchical structure
        from doc_parse_convert.extraction.structure import DocumentStructureExtractor
        extractor = DocumentStructureExtractor(self)
        root = extractor.extract_structure()

        if not root.children:
            logger.warning("No sections found in document structure")
            raise ValueError("No sections found in document")

        # Initialize tracking
        results = {}
        section_counters = {}

        # Process each top-level section
        for section in root.children:
            self._process_section_recursive(
                section=section,
                output_dir=output_dir,
                max_depth=max_depth,
                current_depth=1,
                prefix="",
                section_counters=section_counters,
                results=results,
                skip_empty=skip_empty_sections,
                use_hierarchical_names=use_hierarchical_names
            )

        logger.info(f"Successfully created {len(results)} PDF files")
        return results

    def _process_section_recursive(
        self,
        section,  # DocumentSection
        output_dir: str,
        max_depth: Optional[int],
        current_depth: int,
        prefix: str,
        section_counters: Dict[int, int],
        results: Dict[str, str],
        skip_empty: bool,
        use_hierarchical_names: bool = False
    ) -> None:
        """
        Recursively process section and create PDF files.

        Args:
            section: Current DocumentSection to process
            output_dir: Output directory path
            max_depth: Maximum depth to process (None = unlimited)
            current_depth: Current recursion depth (1-based)
            prefix: Hierarchical prefix - either number (e.g., "1.2") or title path (e.g., "Chapter_Section")
            section_counters: Counter dict by depth level
            results: Accumulator for output file paths
            skip_empty: Whether to skip empty sections
            use_hierarchical_names: If True, use heading texts in prefix; if False, use numbers
        """
        # Check depth limit
        if max_depth is not None and current_depth > max_depth:
            return

        # Validate page range
        if section.start_page is None:
            logger.warning(f"Section '{section.title}' has no start_page, skipping")
            return

        # Use end_page or document end
        end_page = section.end_page if section.end_page is not None else self.doc.page_count - 1

        # Skip empty sections if requested
        if skip_empty and section.start_page > end_page:
            logger.debug(f"Skipping empty section '{section.title}'")
            return

        # Increment counter for this level
        section_counters[current_depth] = section_counters.get(current_depth, 0) + 1
        section_num = section_counters[current_depth]

        # Sanitize title for use in filename
        safe_title = sanitize_filename(section.title, replacement_text="_")

        # Build hierarchical prefix
        if use_hierarchical_names:
            # Use title-based hierarchy
            if prefix:
                full_prefix = f"{prefix} - {safe_title}"
            else:
                full_prefix = safe_title
        else:
            # Use number-based hierarchy
            if prefix:
                full_prefix = f"{prefix}.{section_num:02d}"
            else:
                full_prefix = f"{section_num:02d}"

        # Create PDF for this section
        try:
            output_path = self._create_section_pdf(
                section, output_dir, full_prefix, end_page, use_hierarchical_names
            )
            results[section.title] = output_path
        except Exception as e:
            logger.error(f"Failed to create PDF for section '{section.title}': {str(e)}")
            # Continue processing other sections

        # Recurse into children
        if section.children:
            for child in section.children:
                self._process_section_recursive(
                    child, output_dir, max_depth, current_depth + 1,
                    full_prefix, section_counters, results, skip_empty,
                    use_hierarchical_names
                )

    def _create_section_pdf(
        self,
        section,  # DocumentSection
        output_dir: str,
        prefix: str,
        end_page: int,
        use_hierarchical_names: bool = False
    ) -> str:
        """
        Create PDF file for a section.

        Args:
            section: DocumentSection to create PDF for
            output_dir: Output directory path
            prefix: Hierarchical prefix for filename (number or title path)
            end_page: Calculated end page for section
            use_hierarchical_names: If True, prefix is the full filename; if False, append title

        Returns:
            Path to created PDF file
        """
        if use_hierarchical_names:
            # Prefix already contains the full hierarchical title
            filename = f"{prefix}.pdf"
        else:
            # Build filename: {number}_{title}.pdf
            safe_title = sanitize_filename(section.title, replacement_text="_")
            filename = f"{prefix}_{safe_title}.pdf"

        output_path = os.path.join(output_dir, filename)

        # Validate page range
        validate_page_range(
            section.start_page,
            end_page + 1,  # +1 for exclusive end in validation
            self.doc.page_count
        )

        # Create new PDF with pages from section
        section_doc = fitz.open()
        section_doc.insert_pdf(
            self.doc,
            from_page=section.start_page,
            to_page=end_page  # inclusive
        )

        # Save and close
        section_doc.save(output_path)
        section_doc.close()

        logger.info(f"Created {filename} (pages {section.start_page + 1}-{end_page + 1})")
        return output_path
