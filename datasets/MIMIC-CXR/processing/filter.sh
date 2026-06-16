#!/bin/bash
#
# Filter all annotation JSONs (train/dev/test) to only include images in p10
#
# Usage:
#   bash filter_all_jsons.sh /path/to/files/p10
#
# Example:
#   bash filter_all_jsons.sh /Users/xurunhui/Desktop/USC/medical\ research/models/files/p10

# Check if image folder argument is provided
# if [ -z "$1" ]; then
#     echo "ERROR: Please provide the path to your p10 image folder"
#     echo ""
#     echo "Usage: bash filter_all_jsons.sh /path/to/files/p10"
#     echo ""
#     echo "Example:"
#     echo "  bash filter_all_jsons.sh /Users/xurunhui/Desktop/USC/medical\ research/models/files/p10"
#     exit 1
# fi

IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/mimic-cxr-jpg/mimic-cxr-jpg/2.1.0/files/p10"
DATA_DIR="/project2/ruishanl_1185/SDP_for_VLM/datasets/physionet.org/files/llava-rad-mimic-cxr-annotation/1.0.0"

echo "================================================================="
echo "Filtering LLaVA-Rad JSONs for p10 images only"
echo "================================================================="
echo "Image folder: $IMAGE_FOLDER"
echo "Data directory: $DATA_DIR"
echo ""

# Check if directories exist
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "ERROR: Image folder not found: $IMAGE_FOLDER"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Make sure you're running this from the llava-rad directory"
    exit 1
fi

# Filter training data
echo "================================================================="
echo "1. Filtering TRAINING data..."
echo "================================================================="
python filter_json_by_images.py \
    --image-folder "$IMAGE_FOLDER" \
    --input-json "$DATA_DIR/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json" \
    --output-json "chat_train_p10_filtered.json"

echo ""
echo ""

# Filter dev data
echo "================================================================="
echo "2. Filtering DEV data..."
echo "================================================================="
python filter_json_by_images.py \
    --image-folder "$IMAGE_FOLDER" \
    --input-json "$DATA_DIR/chat_dev_MIMIC_CXR_all_gpt4extract_rulebased_v1.json" \
    --output-json "chat_dev_p10_filtered.json"

echo ""
echo ""

# Filter test data
echo "================================================================="
echo "3. Filtering TEST data..."
echo "================================================================="
python filter_json_by_images.py \
    --image-folder "$IMAGE_FOLDER" \
    --input-json "$DATA_DIR/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json" \
    --output-json "chat_test_p10_filtered.json"

echo ""
echo ""
echo "================================================================="
echo "SUMMARY"
echo "================================================================="
echo "Filtered files created:"
echo "  - chat_train_p10_filtered.json"
echo "  - chat_dev_p10_filtered.json"
echo "  - chat_test_p10_filtered.json"
echo ""
echo "You can now use these files for training!"
echo "================================================================="
