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
    FLAGS, _ = parser.parse_known_args()

    output_path = FLAGS.path.split(".")[0] + "-new.seg.nrrd"

    # Load the NRRD file
    data, header = nrrd.read(FLAGS.path)
    print("Original shape:", data.shape)
    # for i in range(data.shape[-1]-1):
    #     for j in range(i+1, data.shape[-1]):
    #         print("isEqual?", i, j, np.array_equal(data[:, :, :, i], data[:, :, :, j]))

    right_shift = -1*FLAGS.right if FLAGS.right > 0 else data.shape[1]

    new_bitmap_data = data[:, FLAGS.left: right_shift]
    new_bitmap_data = new_bitmap_data[:, :, :, 190:209]
    print("Output shape:", new_bitmap_data.shape)

    # Add new segmentation information to the header
    pattern = re.compile(r"Segment(\d+)_Extent")

    copied_odict = OrderedDict()
    for key, value in header.items():
        if key == "sizes":
            copied_odict[key] = new_bitmap_data.shape
        elif key == "space origin":
            copied_odict[key] = np.array([0,0,0])
        elif re.match(pattern, key):
            x_dim = int(value.split()[1])
            new_x_dim = new_bitmap_data.shape[1]
            if x_dim > new_x_dim:
                copied_odict[key] = value.replace(str(x_dim), str(new_x_dim))
            copied_odict[key] = value
        else:
            copied_odict[key] = value

    nrrd.write(output_path, new_bitmap_data, copied_odict)
