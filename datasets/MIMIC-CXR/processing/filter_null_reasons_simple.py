#!/usr/bin/env python3
"""
Simple script to filter out entries with null reasons
"""

import json
import re

input_file = 'chat_train_p10_filtered.json'
output_file = 'chat_train_p10_with_reasons.json'

print(f"Loading {input_file}...")

# Load JSON - use decoder that's more lenient
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()
    data = json.loads(content)

print(f"Loaded {len(data):,} entries")

# Filter out null reasons
print("Filtering entries with null reasons...")
filtered_data = [entry for entry in data if entry.get('reason') is not None and entry['reason'].lower().strip() != 'on']

print(f"\nResults:")
print(f"  Original entries: {len(data):,}")
print(f"  With valid reasons: {len(filtered_data):,}")
print(f"  Filtered out (null): {len(data) - len(filtered_data):,}")
print(f"  Retention rate: {len(filtered_data)/len(data)*100:.2f}%")

# Save
print(f"\nSaving to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print("Done!")
