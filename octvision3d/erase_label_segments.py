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
from octvision3d.utils import get_filenames, get_idx_of_label

def remove_data_from_label(data, header, label):
    label_idx = get_idx_of_label(header, label)
    data[label_idx] = np.zeros(data[label_idx].shape)
    return data

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
    if os.path.isdir(FLAGS.path):
        for f in tqdm(get_filenames(FLAGS.path, "seg.nrrd")):
            data, header = nrrd.read(f)
            new_data = remove_data_from_label(data, header, FLAGS.label)

            nrrd.write(f, new_data, header)
    else:
        data, header = nrrd.read(FLAGS.path)
        new_data = remove_data_from_label(data, header, FLAGS.label)

        nrrd.write(FLAGS.path, new_data, header)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to .seg.nrrd files to add new segments to"
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Name of label to empty. Does not remove the label but deletes the labels within"
    )
    FLAGS, _ = parser.parse_known_args()
    main()
