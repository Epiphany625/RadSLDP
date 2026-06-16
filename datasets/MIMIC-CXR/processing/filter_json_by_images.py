#!/usr/bin/env python3
"""
Filter LLaVA-Rad training JSON to only include images that exist in your p10 folder.

This script:
1. Scans your p10 image folder to get all available image files
2. Reads the PhysioNet annotation JSON
3. Filters to only include entries where the image file exists in p10
4. Saves the filtered JSON

Usage:
    python filter_json_by_images.py \
        --image-folder /path/to/files/p10 \
        --input-json data/llava-rad-mimic-cxr-annotations-1.0.0/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json \
        --output-json chat_train_p10_filtered.json
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def scan_image_folder(image_folder):
    """
    Scan the image folder and build a set of all available image files.

    Args:
        image_folder: Path to the p10 folder containing images

    Returns:
        Set of image filenames (dicom_ids with .jpg extension)
        Dictionary mapping dicom_id to full relative path
    """
    print(f"Scanning image folder: {image_folder}")

    image_files = set()  # Set of filenames (e.g., "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg")
    image_paths = {}     # Map dicom_id to relative path

    # Walk through the p10 directory
    # Structure: p10/p10XXXXXX/sXXXXXXXX/*.jpg
    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.JPG'):
                # Get the dicom_id (filename without extension)
                dicom_id = Path(filename).stem
                image_files.add(filename)

                # Build relative path from p10
                # e.g., "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
                rel_path = os.path.relpath(os.path.join(root, filename),
                                          os.path.dirname(image_folder))
                image_paths[dicom_id] = rel_path

    print(f"Found {len(image_files):,} images in {image_folder}")
    return image_files, image_paths


def filter_json_by_images(input_json, image_files, image_paths, output_json):
    """
    Filter the annotation JSON to only include images that exist.

    Args:
        input_json: Path to the input annotation JSON
        image_files: Set of available image filenames
        image_paths: Dictionary mapping dicom_id to relative path
        output_json: Path to save the filtered JSON
    """
    print(f"\nReading annotation JSON: {input_json}")

    with open(input_json, 'r') as f:
        data = json.load(f)

    print(f"Total annotations in input: {len(data):,}")

    # Filter the data
    filtered_data = []
    stats = defaultdict(int)

    for entry in data:
        stats['total'] += 1

        # Get the image path from the annotation
        image_path = entry.get('image', '')
        if image_path.startswith("mimic/p11"):
            break
        # Handle different path formats
        # Could be: "mimic/p10/p10000032/s50414267/dicom_id.jpg"
        # Or: "p10/p10000032/s50414267/dicom_id.jpg"
        if image_path.startswith('mimic/'):
            image_path = image_path[len('mimic/'):]

        # Extract the filename (dicom_id.jpg)
        filename = os.path.basename(image_path)
        dicom_id = Path(filename).stem

        # Check if this image exists in our p10 folder
        if filename in image_files:
            # Update the image path to match our folder structure
            if dicom_id in image_paths:
                entry['image'] = image_paths[dicom_id]

            filtered_data.append(entry)
            stats['included'] += 1
        else:
            stats['excluded'] += 1

    # Save filtered JSON
    print(f"\nWriting filtered JSON: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    # Print statistics
    print("\n" + "="*70)
    print("FILTERING RESULTS")
    print("="*70)
    print(f"Total annotations in input:     {stats['total']:,}")
    print(f"Images found in p10:            {stats['included']:,}")
    print(f"Images not in p10 (excluded):   {stats['excluded']:,}")
    print(f"Percentage kept:                {stats['included']/stats['total']*100:.1f}%")
    print("="*70)
    print(f"\nFiltered data saved to: {output_json}")

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Filter LLaVA-Rad JSON to only include images in p10 folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter training data
  python filter_json_by_images.py \\
    --image-folder /path/to/files/p10 \\
    --input-json data/llava-rad-mimic-cxr-annotations-1.0.0/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json \\
    --output-json chat_train_p10_filtered.json

  # Filter dev/test data
  python filter_json_by_images.py \\
    --image-folder /path/to/files/p10 \\
    --input-json data/llava-rad-mimic-cxr-annotations-1.0.0/chat_dev_MIMIC_CXR_all_gpt4extract_rulebased_v1.json \\
    --output-json chat_dev_p10_filtered.json
        """
    )

    parser.add_argument(
        '--image-folder',
        required=True,
        help='Path to the p10 image folder (e.g., /path/to/files/p10)'
    )
    parser.add_argument(
        '--input-json',
        required=True,
        help='Path to the input annotation JSON file'
    )
    parser.add_argument(
        '--output-json',
        required=True,
        help='Path to save the filtered JSON file'
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.image_folder):
        print(f"ERROR: Image folder not found: {args.image_folder}")
        exit(1)

    if not os.path.exists(args.input_json):
        print(f"ERROR: Input JSON not found: {args.input_json}")
        exit(1)

    # Scan for images
    image_files, image_paths = scan_image_folder(args.image_folder)

    if not image_files:
        print("ERROR: No images found in the specified folder!")
        print("Please check that the image folder path is correct.")
        exit(1)

    # Filter the JSON
    filtered_data = filter_json_by_images(
        args.input_json,
        image_files,
        image_paths,
        args.output_json
    )

    print("\n✅ Done! You can now use the filtered JSON for training.")


if __name__ == "__main__":
    main()
