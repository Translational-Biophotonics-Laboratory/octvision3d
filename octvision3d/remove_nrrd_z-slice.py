"""
This script removes a specified slice (along the last axis) from a .seg.nrrd file 
and updates the file's metadata accordingly. The modified NRRD is saved under a new filename 
unless dry run mode is enabled.

Usage:
    python script.py --path /path/to/file.seg.nrrd --slice 42 [--dryrun]

Arguments:
    --path    : Path to the input .seg.nrrd file
    --slice   : Index of the slice to remove (default: 0)
    --dryrun  : If set, does not save output, only prints changes

Output:
    Saves the new NRRD file as <original-name>-new.seg.nrrd unless --dryrun is specified.
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

def remove_slice():
    """
    Remove a specific slice from a NRRD volume along the last axis and update its metadata.

    Assumes global `FLAGS` contains:
    - FLAGS.path: str, path to the input NRRD file
    - FLAGS.slice: int, index of the slice to remove along the last axis
    - FLAGS.dryrun: bool, if True, does not save the output file
    - output_path: str, path to save the modified NRRD (must be globally defined)

    Returns:
    - None. Prints original and new shapes. Writes modified NRRD unless dryrun is enabled.
    """
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
    remove_slice()

