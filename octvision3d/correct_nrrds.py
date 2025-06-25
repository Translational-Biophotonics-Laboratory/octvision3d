"""
This script appends a predefined set of new segment labels with associated RGB colors
to each .seg.nrrd file in a specified directory. It updates both the header and data
accordingly using the `add_segmentation_to_header` function.

After processing, it verifies that all expected labels (original + new) are present
in the header. If any are missing, it raises an error.

Usage:
    python script.py --path /path/to/nrrd/files [--force]
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
                               generate_new_segment_data,
                               find_last_segment_position,
                               segment_name_already_exists,
                               add_segmentation_to_header)

def main():
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
        type=bool,
        default=False,
        help="Ignore duplicate and just skip over it"
    )
    FLAGS, _ = parser.parse_known_args()
    main()

