"""I/O helper functions for safe file handling."""

from typing import Union, BinaryIO
from pathlib import Path


def validate_file_input(file_input: Union[str, Path, BinaryIO]) -> BinaryIO:
    """
    Accept either a validated file path OR a file-like object.

    SECURITY WARNING: If passing a file path, callers MUST validate
    the path against their own secure base directory BEFORE calling
    this function. This library does NOT perform path validation.

    Example of proper validation:
        ```python
        from pathlib import Path

        SECURE_BASE = Path("/secure/data").resolve()
        user_path = Path(user_input).resolve()

        if not user_path.is_relative_to(SECURE_BASE):
            raise PermissionError("Access denied")

        processor.load(user_path)
        ```

    Args:
        file_input: Either a file path (str/Path) or file-like object (BinaryIO)

    Returns:
        File-like object ready for reading

    Raises:
        FileNotFoundError: If path doesn't exist
        PermissionError: If file cannot be opened
    """
    if isinstance(file_input, (str, Path)):
        return open(file_input, 'rb')
    elif hasattr(file_input, 'read'):
        return file_input
    else:
        raise TypeError(
            f"file_input must be str, Path, or file-like object, got {type(file_input)}"
        )
