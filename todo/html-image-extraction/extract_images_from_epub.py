"""
Extract images from EPUB XHTML files with comprehensive metadata.

This script extracts base64-encoded images from XHTML files and creates a manifest
with rich metadata for later alignment with extracted text content.

Key Features:
- Container-first caption detection (more robust than sibling-only)
- Flexible chapter/section detection (Chapter, Appendix, Acknowledgments, etc.)
- Automatic ID generation for images without IDs
- Rich metadata including preceding text snippets for alignment
- Support for both figure and table images

Usage:
    from extract_images_from_epub import extract_images_from_xhtml

    # Pass list of HTML strings (e.g., from docling)
    manifest = extract_images_from_xhtml(html_strings, output_dir="images")

    # Save manifest
    import json
    with open("image_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
"""

import os
import base64
import re
import json
from bs4 import BeautifulSoup


def extract_images_from_xhtml(
    html_strings,
    output_dir="images",
    chapter_pattern=r'^(chapter|appendix|acknowledgments|preface|section)',
    caption_classes_pattern=r'(fig-caption|tab-caption)',
    container_classes_pattern=r'(fig-img|tab-img)',
    snippet_length=150,
    parser='lxml'
):
    """
    Extract images and metadata from a list of XHTML strings.

    Args:
        html_strings (list): List of HTML/XHTML content strings
        output_dir (str): Directory to save extracted images
        chapter_pattern (str): Regex pattern for matching chapter/section div IDs.
            Default matches: chapter, appendix, acknowledgments, preface, section
        caption_classes_pattern (str): Regex pattern for caption CSS classes.
            Default matches: fig-caption, tab-caption
        container_classes_pattern (str): Regex pattern for image container CSS classes.
            Default matches: fig-img, tab-img
        snippet_length (int): Max characters for preceding_text_snippet. Default: 150
        parser (str): BeautifulSoup parser to use. Default: 'lxml'.
            Alternatives: 'html.parser', 'html5lib'

    Returns:
        list: Image manifest with metadata for each image

    Example manifest entry:
        {
            "source_html_index": 0,
            "image_index_in_html": 2,
            "image_id": "fig_A1",
            "alt_text": "An illustrative diagram",
            "chapter_id": "ChapterA",
            "context_type": "figure",
            "caption": "Figure A1. The architecture of the system.",
            "preceding_text_snippet": "As we can see in the following diagram...",
            "filename": "ChapterA_fig_A1.jpeg"
        }
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_manifest = []

    # Compile regex patterns from parameters
    chapter_regex = re.compile(chapter_pattern, re.IGNORECASE)
    caption_classes = re.compile(caption_classes_pattern)
    container_classes = re.compile(container_classes_pattern)

    for i, html in enumerate(html_strings):
        soup = BeautifulSoup(html, parser)
        img_tags = soup.find_all('img')

        for img_idx, img in enumerate(img_tags):
            # Initialize metadata with None values
            metadata = {
                "source_html_index": i,
                "image_index_in_html": img_idx,
                "context_type": None,
                "caption": None,
                "preceding_text_snippet": None
            }

            # 1. Extract direct attributes and handle missing ID
            image_id = img.get('id')
            if not image_id:
                image_id = f"generated_id_{i}_{img_idx}"
            metadata['image_id'] = image_id
            metadata['alt_text'] = img.get('alt', '')

            # 2. Find chapter/section context
            chapter_div = img.find_parent('div', id=chapter_regex)
            metadata['chapter_id'] = chapter_div.get('id') if chapter_div else 'unknown_section'

            # 3. Find container, caption, and context type (container-first approach)
            container = img.find_parent(['p', 'div'], class_=container_classes)
            if not container:
                # Broader fallback - includes table cells
                container = img.find_parent(['p', 'div', 'td'])

            if container:
                # Search for caption relative to the container
                # Try next sibling first (caption after image)
                caption_tag = container.find_next_sibling('p', class_=caption_classes)

                # If not found, try previous sibling (caption before image)
                if not caption_tag:
                    caption_tag = container.find_previous_sibling('p', class_=caption_classes)

                if caption_tag:
                    metadata['caption'] = caption_tag.get_text(strip=True)
                    p_class = ' '.join(caption_tag.get('class', []))
                    if 'tab-caption' in p_class:
                        metadata['context_type'] = 'table'
                    elif 'fig-caption' in p_class:
                        metadata['context_type'] = 'figure'

                # Capture preceding text for alignment anchoring
                prev_element = container.find_previous_sibling()
                if prev_element:
                    metadata['preceding_text_snippet'] = prev_element.get_text(strip=True)[:snippet_length]

            # 4. Process base64 data
            src = img.get('src')
            if src and src.startswith('data:image'):
                try:
                    header, encoded_data = src.split(',', 1)
                    # Extract image format (jpeg, png, etc.)
                    img_format = header.split(';')[0].split('/')[1]

                    filename = f"{metadata['chapter_id']}_{metadata['image_id']}.{img_format}"
                    metadata['filename'] = filename

                    # Decode and save image
                    image_data = base64.b64decode(encoded_data)
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'wb') as f:
                        f.write(image_data)

                    image_manifest.append(metadata)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse src for image {metadata['image_id']}: {e}")
            else:
                print(f"Warning: Skipping image {metadata['image_id']} with non-base64 src: {src[:50] if src else 'None'}")

    return image_manifest


def save_manifest(manifest, output_path="image_manifest.json"):
    """
    Save the image manifest to a JSON file.

    Args:
        manifest (list): Image manifest from extract_images_from_xhtml
        output_path (str): Path to save JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest with {len(manifest)} images to {output_path}")


def main():
    """
    Example usage - processes XHTML files from a directory.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_images_from_epub.py <xhtml_directory> [output_dir]")
        print("\nExample:")
        print('  python extract_images_from_epub.py "manual_conversion/Ball Redbook - Crop Production_files" extracted_images')
        sys.exit(1)

    xhtml_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_images"

    if not os.path.exists(xhtml_dir):
        print(f"Error: Directory not found: {xhtml_dir}")
        sys.exit(1)

    # Read all XHTML files
    html_strings = []
    xhtml_files = sorted([f for f in os.listdir(xhtml_dir) if f.endswith(('.xhtml', '.html', '.htm'))])

    print(f"Found {len(xhtml_files)} XHTML/HTML files")

    for filename in xhtml_files:
        filepath = os.path.join(xhtml_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                html_strings.append(f.read())
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")

    print(f"Processing {len(html_strings)} HTML files...")

    # Extract images
    manifest = extract_images_from_xhtml(html_strings, output_dir=output_dir)

    # Save manifest
    manifest_path = os.path.join(output_dir, "image_manifest.json")
    save_manifest(manifest, manifest_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print("EXTRACTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total images extracted: {len(manifest)}")

    # Count by context type
    context_counts = {}
    for img in manifest:
        ctx = img.get('context_type') or 'unknown'
        context_counts[ctx] = context_counts.get(ctx, 0) + 1

    print(f"\nImages by type:")
    for ctx, count in sorted(context_counts.items()):
        print(f"  {ctx}: {count}")

    # Count by chapter
    chapter_counts = {}
    for img in manifest:
        chapter = img.get('chapter_id') or 'unknown'
        chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1

    print(f"\nImages by chapter (top 10):")
    for chapter, count in sorted(chapter_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {chapter}: {count}")


if __name__ == "__main__":
    main()
