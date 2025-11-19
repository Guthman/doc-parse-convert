"""
Client for interacting with AI APIs (Vertex AI/Gemini).
"""

import json
from typing import List, Any

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
)
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from doc_parse_convert.config import ProcessingConfig, GEMINI_SAFETY_CONFIG, logger
from doc_parse_convert.models.document import Chapter
from doc_parse_convert.exceptions import AIExtractionError


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

    def _call_model_with_retry(
        self,
        parts: List[Part],
        temperature: float = 0.0,
        response_mime_type: str = None,
        response_schema: dict = None,
    ) -> Any:
        """
        Call the AI model with configurable retry logic.

        Uses exponential backoff for more robust error handling.
        Retry behavior can be configured via ProcessingConfig.
        """
        if not self.model:
            logger.error("AI model not initialized")
            raise ValueError("AI model not initialized")

        # Create retry decorator with config values
        retry_decorator = retry(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait
            ),
            reraise=True
        )

        @retry_decorator
        def _call():
            try:
                # Log request details
                logger.debug(f"API Request - Parts count: {len(parts)}")
                logger.debug(f"API Request - Temperature: {temperature}")
                if response_schema:
                    logger.debug(f"API Request - Response schema present")

                # Build generation config
                config_params = {
                    'temperature': temperature,
                    'candidate_count': 1,
                }

                if response_mime_type:
                    config_params['response_mime_type'] = response_mime_type

                if response_schema:
                    config_params['response_schema'] = response_schema

                adjusted_config = GenerationConfig(**config_params)

                logger.debug("Calling AI model")
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
                logger.error(f"Error during model call: {e.__class__.__name__} {str(e)}")

                # Log detailed error information
                if hasattr(e, 'details'):
                    logger.error(f"Error details: {e.details}")
                if hasattr(e, 'code'):
                    logger.error(f"Error code: {e.code}")

                # Debug API request for InvalidArgument errors
                if 'InvalidArgument' in e.__class__.__name__:
                    logger.error("API request contains invalid arguments")
                    logger.error("Check image sizes and request structure")

                    # Save debug info if enabled
                    self._save_debug_info(e, parts, response_schema)

                raise

        try:
            return _call()
        except RetryError as e:
            logger.error(f"All {self.config.retry_attempts} retry attempts failed")
            raise AIExtractionError(
                f"Failed after {self.config.retry_attempts} attempts"
            ) from e.last_attempt.exception()

    def _save_debug_info(self, e, parts, response_schema):
        # Placeholder for the _save_debug_info logic mentioned in the plan
        pass

    def extract_structure_from_images(self, images) -> List[Chapter]:
        """
        Extract structural information from document images using AI.

        Args:
            images: Either a list or generator of image dictionaries

        Returns:
            List of extracted chapters

        Note:
            For memory efficiency, pass a generator from
            convert_to_images_generator() instead of a list.
        """
        logger.info("Starting structure extraction from images")

        if not self.model:
            logger.error("AI model not initialized")
            raise ValueError("AI model not initialized")

        from doc_parse_convert.ai.schemas import get_toc_response_schema
        response_schema = get_toc_response_schema()

        # Convert to generator if it's a list
        if isinstance(images, list):
            images = iter(images)

        # Process in batches
        all_chapters = []
        batch = []

        for img in images:
            batch.append(img)

            # Process when batch is full
            if len(batch) >= self.config.image_batch_size:
                chapters = self._process_image_batch(batch, response_schema)
                all_chapters.extend(chapters)
                batch = []

        # Process remaining images
        if batch:
            chapters = self._process_image_batch(batch, response_schema)
            all_chapters.extend(chapters)

        logger.info(f"Successfully extracted {len(all_chapters)} chapters")
        return all_chapters

    def _process_image_batch(self, batch: List[dict], response_schema: dict) -> List[Chapter]:
        """Process a batch of images for structure extraction."""
        from doc_parse_convert.ai.prompts import get_toc_prompt
        from vertexai.generative_models import Part

        # Create Part objects from batch
        parts = []
        for i, img in enumerate(batch):
            try:
                logger.debug(f"Processing image {i + 1}/{len(batch)}")
                parts.append(Part.from_data(data=img["data"], mime_type=img["_mime_type"]))
            except Exception as e:
                logger.warning(f"Failed to process image {i + 1}: {str(e)}")
                continue

        if not parts:
            logger.error("No valid images to process in batch")
            return []

        # Add instruction text
        logger.debug("Adding instruction text to parts")
        parts.append(Part.from_text(get_toc_prompt()))

        try:
            logger.debug("Calling AI model with retry")
            response = self._call_model_with_retry(
                parts,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=response_schema
            )

            # Parse and return chapters
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

            chapters.sort(key=lambda x: (x.start_page, x.level))

            for i in range(len(chapters) - 1):
                chapters[i].end_page = chapters[i + 1].start_page

            return chapters

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise AIExtractionError(
                "Failed to extract structure from AI response. "
                "Check logs for detailed error information."
            ) from e
