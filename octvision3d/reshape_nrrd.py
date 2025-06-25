"""
This script crops a `.seg.nrrd` segmentation volume by specified pixel amounts along
the x and y axes (left, right, up, down), and updates the metadata accordingly.

Functionality:
- Loads a NRRD segmentation file
- Crops the volume spatially using the provided margins
- Adjusts the header fields (e.g., `sizes`, `space origin`, `Segment*_Extent`)
- Saves the cropped volume and updated header to a new file

Usage:
    python script.py --path /path/to/file.seg.nrrd --left 20 --right 20 --up 10 --down 10

Output:
    Creates a new file named <original-name>-new.seg.nrrd in the same directory.
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

def reshape_nrrd():
    """
    Crop a NRRD segmentation volume spatially along x and y axes and update its metadata accordingly.

    Assumes global `FLAGS` contains:
    - FLAGS.path: str, path to the input NRRD file
    - FLAGS.left: int, number of pixels to crop from the left (x-axis)
    - FLAGS.right: int, number of pixels to crop from the right (x-axis)
    - FLAGS.up: int, number of pixels to crop from the top (y-axis)
    - FLAGS.down: int, number of pixels to crop from the bottom (y-axis)

    Also assumes a global `output_path` is defined for saving the modified file.

    Modifies:
    - Crops the volume spatially on x/y axes
    - Updates header's "sizes", "space origin", and segment "_Extent" fields

    Returns:
    - None. Saves the cropped volume and updated header to `output_path`.
    """
    # Load the NRRD file
    data, header = nrrd.read(FLAGS.path)
    print("Original shape:", data.shape)

    right_shift = -1*FLAGS.right if FLAGS.right > 0 else data.shape[1]
    down_shift = -1 * FLAGS.down if FLAGS.down > 0 else data.shape[2]

    new_bitmap_data = data[:, FLAGS.left:right_shift, FLAGS.up:down_shift, :]
    print("Output shape:", new_bitmap_data.shape)

    # Add new segmentation information to the header
    pattern = re.compile(r"Segment(\d+)_Extent")

    copied_odict = OrderedDict()
    for key, value in header.items():
        if key == "sizes":
            copied_odict[key] = new_bitmap_data.shape
        elif key == "space origin":
            # Update the origin to reflect cropping along x and y dimensions
            copied_odict[key] = np.array([0,0,0])
        elif re.match(pattern, key):
            # Adjust segment extents for cropped dimensions in x and y
            x_dim, y_dim = map(int, value.split()[1:3])
            new_x_dim, new_y_dim = new_bitmap_data.shape[1:3]
            if x_dim > new_x_dim:
                value = value.replace(str(x_dim), str(new_x_dim), 1)
            if y_dim > new_y_dim:
                value = value.replace(str(y_dim), str(new_y_dim), 1)
            copied_odict[key] = value
        else:
            copied_odict[key] = value

    nrrd.write(output_path, new_bitmap_data, copied_odict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to .seg.nrrd file to add new segments to"
    )
    parser.add_argument(
        "--left",
        type=int,
        default=0,
        help="index to skip on the left"
    )
    parser.add_argument(
        "--right",
        type=int,
        default=0,
        help="index to skip on the right"
    )
    parser.add_argument(
        "--up",
        type=int,
        default=0,
        help="index to skip at the top (y-axis)"
    )
    parser.add_argument(
        "--down",
        type=int,
        default=0,
        help="index to skip at the bottom (y-axis)"
    )
    FLAGS, _ = parser.parse_known_args()

    output_path = FLAGS.path.split(".")[0] + "-new.seg.nrrd"

