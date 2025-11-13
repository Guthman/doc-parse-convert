"""
Base classes for document extraction.
"""

from abc import ABC, abstractmethod
from typing import List, Any, TYPE_CHECKING, ForwardRef

from doc_parse_convert.config import ProcessingConfig
from doc_parse_convert.models.document import Chapter
from doc_parse_convert.models.content import ChapterContent

# Forward reference for type hinting
if TYPE_CHECKING:
    from doc_parse_convert.ai.client import AIClient
    DocumentProcessorType = 'DocumentProcessor'
else:
    DocumentProcessorType = ForwardRef('DocumentProcessor')


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
        from doc_parse_convert.ai.client import AIClient
        import vertexai
        from google.oauth2 import service_account

        if not self.config.project_id or not self.config.vertex_ai_location:
            from doc_parse_convert.config import logger
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
            from doc_parse_convert.config import logger
            logger.info("Successfully initialized AI client")
        except Exception as e:
            from doc_parse_convert.config import logger
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
