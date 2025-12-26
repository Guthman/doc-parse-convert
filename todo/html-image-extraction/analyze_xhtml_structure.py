"""
Analyze XHTML files to extract unique combinations of tags and attributes.
Uses BeautifulSoup to handle large files.
"""
import os
from collections import defaultdict
from bs4 import BeautifulSoup
import json


def analyze_xhtml_files(directory):
    """
    Extract unique tag and attribute combinations from XHTML files.

    Args:
        directory: Path to directory containing XHTML files

    Returns:
        Dictionary mapping tag names to sets of attribute combinations
    """
    # Store unique combinations: {tag_name: {frozenset of (attr, value) tuples}}
    tag_attributes = defaultdict(set)

    # Get all XHTML/HTML files
    xhtml_files = [f for f in os.listdir(directory)
                   if f.endswith(('.xhtml', '.html', '.htm'))]

    print(f"Found {len(xhtml_files)} XHTML/HTML files to process")

    for idx, filename in enumerate(xhtml_files, 1):
        filepath = os.path.join(directory, filename)
        print(f"Processing {idx}/{len(xhtml_files)}: {filename}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read file in chunks to be more memory efficient
                soup = BeautifulSoup(f, 'lxml')

            # Find all tags
            for elem in soup.find_all(True):  # True finds all tags
                tag = elem.name

                # Get all attributes as a tuple of tuples
                if elem.attrs:
                    # Convert attribute dict to sorted tuple of tuples
                    # Handle list values (like class) by converting to tuples
                    attr_items = []
                    for key, value in elem.attrs.items():
                        if isinstance(value, list):
                            # Convert list to space-separated string (standard for class, etc.)
                            value = ' '.join(value)

                        # Skip base64 data in src attributes - just note the format
                        if key == 'src' and value.startswith('data:image'):
                            # Extract just the image format
                            value = value.split(';')[0]  # e.g., "data:image/jpeg"

                        attr_items.append((key, value))

                    attr_tuple = tuple(sorted(attr_items))
                    tag_attributes[tag].add(attr_tuple)
                else:
                    # Tag with no attributes
                    tag_attributes[tag].add(())

            # Clear soup to free memory
            soup.decompose()

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    return tag_attributes


def format_results(tag_attributes):
    """
    Format results in a readable way, grouping by attribute names.

    Returns structured data suitable for analysis.
    """
    results = {}

    for tag, attr_sets in sorted(tag_attributes.items()):
        tag_info = {
            'count': len(attr_sets),
            'combinations': []
        }

        # Group by attribute names to see different values
        attr_name_groups = defaultdict(set)

        for attr_tuple in attr_sets:
            if attr_tuple:  # If there are attributes
                attr_dict = dict(attr_tuple)
                tag_info['combinations'].append(attr_dict)

                # Group values by attribute name
                for attr_name, attr_value in attr_tuple:
                    attr_name_groups[attr_name].add(attr_value)
            else:
                tag_info['combinations'].append({})

        # Add attribute value summaries
        if attr_name_groups:
            tag_info['attribute_values'] = {
                attr_name: sorted(list(values))
                for attr_name, values in sorted(attr_name_groups.items())
            }

        results[tag] = tag_info

    return results


def write_results(results, output_file):
    """Write results to JSON."""

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote JSON results to: {json_file}")


def main():
    # Directory containing XHTML files
    directory = r"manual_conversion\Ball Redbook -  Crop Production_files"

    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return

    print("Starting XHTML structure analysis...")
    print(f"Directory: {directory}\n")

    # Analyze files
    tag_attributes = analyze_xhtml_files(directory)

    # Format results
    print("\nFormatting results...")
    results = format_results(tag_attributes)

    # Write output
    output_file = "xhtml_structure_analysis.json"
    write_results(results, output_file)

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total unique tags found: {len(results)}")
    print(f"\nMost common tags:")
    for tag in list(sorted(results.keys()))[:20]:
        print(f"  - <{tag}>: {results[tag]['count']} unique attribute combinations")


if __name__ == "__main__":
    main()
