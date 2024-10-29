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

    # Load the NRRD file
    data, header = nrrd.read(FLAGS.path)
    print("Original shape:", data.shape)
    # for i in range(data.shape[-1]-1):
    #     for j in range(i+1, data.shape[-1]):
    #         print("isEqual?", i, j, np.array_equal(data[:, :, :, i], data[:, :, :, j]))

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
