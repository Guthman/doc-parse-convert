"""Custom exceptions for doc_parse_convert library."""


class DocParseConvertError(Exception):
    """Base exception for all doc_parse_convert errors."""
    pass


class ValidationError(DocParseConvertError):
    """Raised when input validation fails."""
    pass


class AIExtractionError(DocParseConvertError):
    """Raised when AI extraction fails."""
    pass


class ConversionError(DocParseConvertError):
    """Raised when document conversion fails."""
    pass


class FileAccessError(DocParseConvertError):
    """Raised when file access is denied or fails."""
    pass
