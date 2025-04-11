"""
Client for interacting with AI APIs (Vertex AI/Gemini).
"""

import os
import json
import datetime
from typing import List, Any

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
)
from tenacity import retry, stop_after_attempt, wait_fixed

from doc_parse_convert.config import ProcessingConfig, GEMINI_SAFETY_CONFIG, logger
from doc_parse_convert.models.document import Chapter


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

    def extract_structure_from_images(self, images: List[dict]) -> List[Chapter]:
        """Extract structural information from document images using AI."""
        logger.info("Starting structure extraction from images")
        
        if not self.model:
            logger.error("AI model not initialized")
            raise ValueError("AI model not initialized")

        # Ultra-simplified schema with just the bare essentials
        # This minimalist approach avoids Vertex AI schema complexity limitations
        from doc_parse_convert.ai.schemas import get_toc_response_schema
        response_schema = get_toc_response_schema()

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

        # Add instruction text from prompts module
        from doc_parse_convert.ai.prompts import get_toc_prompt
        logger.debug("Adding instruction text to parts")
        parts.append(Part.from_text(get_toc_prompt()))

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
