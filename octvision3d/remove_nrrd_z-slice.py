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
        "--slice",
        type=int,
        default=0,
        help="index to remove"
    )
    parser.add_argument(
        "--dryrun",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable for dry run"

    )
    FLAGS, _ = parser.parse_known_args()

    output_path = FLAGS.path.split(".")[0] + "-new.seg.nrrd"

    # Load the NRRD file
    data, header = nrrd.read(FLAGS.path)
    print("Original shape:", data.shape)

    new_bitmap_data = np.delete(data, FLAGS.slice, axis=-1)
    print("New shape:", new_bitmap_data.shape)

    copied_odict = OrderedDict()
    for key, value in header.items():
        if key == "sizes":
            copied_odict[key] = new_bitmap_data.shape
        elif key == "space origin":
            copied_odict[key] = np.array([0,0,0])
        else:
            copied_odict[key] = value

    if not FLAGS.dryrun:
        nrrd.write(output_path, new_bitmap_data, copied_odict)
    else:
        print("dryrun finished")
