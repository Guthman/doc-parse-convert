from doc_parse_convert.utils.validators import validate_page_range
from doc_parse_convert.exceptions import ValidationError
import pytest


def test_page_range_validation():
    """Test page range validation catches errors."""
    # Valid range should pass
    validate_page_range(0, 10, 100)

    # Invalid ranges should raise
    with pytest.raises(ValidationError):
        validate_page_range(-1, 10, 100)  # Negative start

    with pytest.raises(ValidationError):
        validate_page_range(10, 5, 100)  # End < start

    with pytest.raises(ValidationError):
        validate_page_range(0, 101, 100)  # Exceeds max

    with pytest.raises(ValidationError):
        validate_page_range(100, 101, 100) # Start exceeds max


def test_exception_messages_descriptive():
    """Test that exception messages help debugging."""
    try:
        validate_page_range(-1, 10, 100)
    except ValidationError as e:
        assert 'negative' in str(e).lower()
        assert '-1' in str(e)
