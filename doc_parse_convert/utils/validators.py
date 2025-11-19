"""Input validation helpers."""

from doc_parse_convert.exceptions import ValidationError


def validate_page_range(start_page: int, end_page: int, max_pages: int) -> None:
    """
    Validate that a page range is within document bounds.

    Args:
        start_page: Starting page number (0-based index)
        end_page: Ending page number (0-based index, exclusive)
        max_pages: Total number of pages in document

    Raises:
        ValidationError: If page range is invalid with descriptive message

    Examples:
        >>> validate_page_range(0, 10, 100)  # OK
        >>> validate_page_range(-1, 10, 100)  # ValidationError
        >>> validate_page_range(0, 101, 100)  # ValidationError
    """
    if start_page < 0:
        raise ValidationError(
            f"Start page cannot be negative: {start_page}. "
            f"Page numbers are 0-based indices."
        )

    if end_page < start_page:
        raise ValidationError(
            f"End page ({end_page}) must be greater than or equal to start page ({start_page})"
        )

    if start_page >= max_pages:
        raise ValidationError(
            f"Start page ({start_page}) exceeds document length. "
            f"Document has {max_pages} pages (0-{max_pages-1})."
        )

    if end_page > max_pages:
        raise ValidationError(
            f"End page ({end_page}) exceeds document length. "
            f"Document has {max_pages} pages (0-{max_pages-1})."
        )


def validate_page_number(page: int, max_pages: int, parameter_name: str = "page") -> None:
    """
    Validate that a single page number is within bounds.

    Args:
        page: Page number (0-based index)
        max_pages: Total number of pages
        parameter_name: Name of parameter for error messages

    Raises:
        ValidationError: If page number is invalid
    """
    if page < 0:
        raise ValidationError(
            f"{parameter_name} cannot be negative: {page}"
        )

    if page >= max_pages:
        raise ValidationError(
            f"{parameter_name} ({page}) exceeds document length. "
            f"Document has {max_pages} pages (0-{max_pages-1})."
        )
