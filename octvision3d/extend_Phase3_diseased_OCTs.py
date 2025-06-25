"""
This script processes .seg.nrrd segmentation files to append new segments with predefined
colored and enforce consistent label structure. It verifies that each file contains only
the expected original labels (in correct order), removes extra labels (optionally with
--force), and appends new segments with predefined colors. It updates both the NRRD header
and data, ensuring compatibility with 3D Slicer and other medical image analysis tools.
"""

import nrrd
import os
import re
import argparse
import numpy as np
import tifffile as tif
from tqdm import tqdm
from glob import glob
from collections import OrderedDict
from pprint import pprint
from octvision3d.utils import (get_filenames,
                               segment_name_already_exists,
                               add_segmentation_to_header,
                               check_duplicate_labels,
                               delete_by_name)

            
def check_order_and_remove_extra_labels(data, header, original_labels):
    """
    Ensure that all original labels are present in the NRRD header and in the correct order.
    Optionally removes any extra labels not in the original list, if --force is enabled.

    Parameters:
    - data: np.ndarray, 3D segmentation volume with shape [num_segments, H, W]
    - header: collections.OrderedDict, metadata of the NRRD file
    - original_labels: list of str, the expected label names in the required order

    Returns:
    - new_data: np.ndarray, updated segmentation volume with only the original labels
    - new_header: OrderedDict, updated header with original labels and removed extras

    Raises:
    - ValueError: if extra labels are present and --force is not set,
                  or if original labels are out of order.
    """
    name_tracker = list(map(list, zip(original_labels, [-1]*len(original_labels))))
    idx = len(original_labels) - 1
    new_header = header.copy()
    new_data = data.copy()
    for key, val in reversed(header.items()):
        match = re.match(r"Segment(\d+)_", key)
        if match:
            if key.endswith("_Name"):
                segment_number = int(match.group(1))
                if val not in original_labels:
                    if FLAGS.force:
                        print(f"{val} is not in original labels. Force deleting...")
                        new_data, new_header = delete_by_name(new_data, new_header, val, original_labels)
                    else:
                        raise ValueError(f"{val} not one of the original labels. Use --force to delete")
                else:
                    name_tracker[idx][1] = segment_number
                    idx -= 1
    for i, name in enumerate(name_tracker):
        if i != name[1]:
            raise ValueError("Original labels not in proper order")
    return new_data, new_header

def get_corrected_header(data, header, original_labels):
    """
    Validate that all original labels are present in the NRRD header and in the correct order.
    Optionally removes extra labels (if --force is enabled via FLAGS).

    Parameters:
    - data: np.ndarray, 3D segmentation volume (shape: [num_segments, height, width])
    - header: collections.OrderedDict, the metadata from the NRRD file
    - original_labels: list of str, expected label names in the required order

    Returns:
    - new_data: np.ndarray, updated segmentation volume with only valid original labels
    - new_header: OrderedDict, updated header with validated (and possibly trimmed) segments

    Raises:
    - AssertionError: if any expected label is missing from the header
    - ValueError: if labels are out of order or extras exist and --force is not set
    """
    # These labels should be present in every segmentation
    for i in original_labels:
        try:
            assert segment_name_already_exists(header, i)
        except AssertionError as e:
            raise AssertionError(f"{i} label not found in header")
    # Check that the labels are in the correct order
    new_data, new_header = check_order_and_remove_extra_labels(data, header, original_labels)
    return new_data, new_header 

def main():
    original_labels = ["CNV", "DRU", "EX", "FLU", "GA", "HEM", "RPE"]
    cmap = OrderedDict([
        ("RET", "0.635 0.0 1.0"),
        ("CHO", "0.56 0.56 0.44"),
        ("VIT", "0.88 0.94 0.99"),
        ("HYA", "0.46 0.98 0.99"),
        ("SHS", "0.69 0.99 0.82"),
        ("ART", "0.99 0.99 0.33"),
        ("ERM", "0.22 0.49 0.97"),
        ("SES", "0.392 0.196 0.0"),
    ])

    # Load the NRRD file
    for f in tqdm(get_filenames(FLAGS.path, "seg.nrrd")):
        startedAddingSegments = False

        data, header = nrrd.read(f)

        # run tests to ensure seg.nrrd header is correct
        # deletes labels not in the original_labels
        corrected_data, corrected_header = get_corrected_header(data, header, original_labels)
        nrrd.write(f, corrected_data, corrected_header)

        # Add new labels
        for k, v in cmap.items():
            data, header = nrrd.read(f)
            success = add_segmentation_to_header(data, header, f, k, v, startedAddingSegments)
            if success and not startedAddingSegments:
                startedAddingSegments = True

        data, header = nrrd.read(f)
        header_vals = set([i for i in header.values() if type(i)==str])
        labels = ["CNV", "DRU", "EX", "FLU", "GA", "HEM", "RPE", "RET",\
                  "CHO", "VIT", "HYA", "SHS", "ART", "ERM", "SES"]
        err_s = ""
        for label in labels:
            if label not in header_vals:
                err_s += f", {label}" if len(err_s) != 0 else f"{label}"
        if len(err_s) > 0:
            raise ValueError(f"ERROR: Labels {err_s} not in final header")

        # check for duplicates at the end as well
        check_duplicate_labels(header)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to .seg.nrrd files to add new segments to"
    )
    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Delete labels not in original labels"
    )
    FLAGS, _ = parser.parse_known_args()
    main()
