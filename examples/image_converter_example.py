"""
Example demonstrating how to use the ImageConverter class.

This example shows how to:
1. Convert a PDF to PNG images and save them to disk
2. Convert a PDF to JPG images and save them to disk
3. Use the context manager pattern for automatic resource cleanup
"""

import os
from doc_parse_convert.content_extraction import ImageConverter

def save_document_as_images(input_path, output_dir, format="png"):
    """Convert a document to images and save them to disk.
    
    Args:
        input_path: Path to the input document
        output_dir: Directory to save images to
        format: Image format ('png' or 'jpg')
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Using the context manager pattern for automatic cleanup
    with ImageConverter(input_path, format=format) as converter:
        for page_number, page_image in converter:
            # Save each page as an image
            extension = format.lower()
            output_path = os.path.join(output_dir, f"page_{page_number+1:03d}.{extension}")
            with open(output_path, 'wb') as f:
                f.write(page_image)
            print(f"Saved {output_path}")

def main():
    # Example usage - replace with your actual PDF path
    pdf_path = "your_document.pdf"  # Replace with actual path
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} not found. Please provide a valid PDF path.")
        return
    
    # Save as PNG images
    output_dir_png = "output_png"
    print(f"Converting {pdf_path} to PNG images...")
    save_document_as_images(pdf_path, output_dir_png, format="png")
    
    # Save as JPG images
    output_dir_jpg = "output_jpg"
    print(f"Converting {pdf_path} to JPG images...")
    save_document_as_images(pdf_path, output_dir_jpg, format="jpg")
    
    # Manual iteration example
    print("\nManual iteration example:")
    converter = ImageConverter(pdf_path)
    try:
        # Get only the first 3 pages
        for i in range(3):
            try:
                page_number, page_image = next(converter)
                print(f"Page {page_number+1} has {len(page_image)} bytes")
            except StopIteration:
                print("No more pages")
                break
    finally:
        # Ensure we close the document
        converter.close()

if __name__ == "__main__":
    main()
