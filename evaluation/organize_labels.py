#!/usr/bin/env python3
import os
import re
import shutil
import argparse

# mapping from the suffix in the filename to the desired output name
MODALITY_MAP = {
    'left_rgb': 'left_rgb.txt',
    'scaled_left_rgb': 'scaled_left_rgb.txt',
    'thermal_png': 'thermal.txt',
}

def move_and_rename_files(labels_dir, subset_dir):
    # regex to capture: anything_timestamp_modality.rf.hash.txt
    pattern = re.compile(
        r'.+_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+?)_'
        r'(left_rgb|scaled_left_rgb|thermal_png)\.rf\w+\.txt$'
    )

    for fname in os.listdir(labels_dir):
        match = pattern.match(fname)
        if not match:
            continue  # skip files that don't match the expected pattern

        timestamp_id, modality = match.groups()
        dest_name = MODALITY_MAP[modality]

        # build source and destination paths
        src_path = os.path.join(labels_dir, fname)
        dest_dir = os.path.join(subset_dir, timestamp_id)
        dest_path = os.path.join(dest_dir, dest_name)

        # ensure destination directory exists
        os.makedirs(dest_dir, exist_ok=True)

        # move & rename
        print(f"Moving {src_path} → {dest_path}")
        shutil.move(src_path, dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize label files into their timestamped subdirectories."
    )
    parser.add_argument(
        "--labels-dir",
        default="labels",
        help="Path to the directory containing the .txt label files.",
    )
    parser.add_argument(
        "--subset-dir",
        default="exemplary_subset_labeled",
        help="Path to the root of the exemplary_subset_labeled directory.",
    )
    args = parser.parse_args()

    move_and_rename_files(args.labels_dir, args.subset_dir)
