"""
This script downsamples a volumetric TIFF stack to a target number of slices (default: 19)
and saves the result to a specified output path.

Functionality:
- Supports loading either a single multi-slice TIFF file or a directory of 2D image slices
- Selects evenly spaced slices based on the target depth
- Ensures exactly `target` unique slices, even with rounding edge cases
- Saves the downsampled volume as a new TIFF file

Usage:
    For a single multi-page TIFF:
        python script.py --path input.tif --output_dir ./out --output_name output.tif

    For a directory of 2D slices:
        python script.py --path ./folder --multifile --ext tif --output_dir ./out --output_name output.tif

Notes:
- Supports grayscale images; uses OpenCV for reading 2D slices and tifffile for TIFF stacks
- Will raise an error if output path and filename are not provided
"""

import os
import cv2
import numpy as np
import tifffile as tif
from scipy.ndimage import zoom
from argparse import ArgumentParser
from utils import get_filenames, create_directory
from pprint import pprint

def downsample(data, target=19):
    """
    Downsample a 3D volume to a fixed number of evenly spaced slices along the first axis.

    Parameters:
    - data: np.ndarray, 3D volume of shape [depth, height, width]
    - target: int, desired number of output slices (default: 19)

    Returns:
    - downsampled_data: np.ndarray, volume with shape [target, height, width]

    Notes:
    - Ensures unique slice indices (avoiding duplicates from rounding)
    - If necessary, over-generates indices and trims back to exactly `target` slices
    """
    # Calculate the exact step size
    step_size = data.shape[0] / target

    # Generate indices using the step size
    selected_indices = np.array([int(round(i * step_size)) for i in range(target)])

    # Ensure unique indices (in case of rounding duplicates)
    unique_indices = np.unique(selected_indices)

    # Handle cases where the number of unique indices is less than the target
    while len(unique_indices) < target:
        missing_count = target - len(unique_indices)
        additional_indices = np.array([int(round(i * step_size)) for i in range(target + missing_count)])
        unique_indices = np.unique(np.concatenate((unique_indices, additional_indices)))

    # Trim to exact number of target slices
    selected_indices = list(unique_indices[:target])
    downsampled_data = data[selected_indices, :, :]
    return downsampled_data

def main():
    if FLAGS.multifile:
        filenames = get_filenames(FLAGS.path, ext=f"{FLAGS.ext}*")
        if len(filenames) == 0:
            print(f"No files with ext {FLAGS.ext} found in {os.path.abspath(FLAGS.path)}")
            return
        if FLAGS.ext == "tif":
            vol = np.array([tif.imread(f) for f in filenames])
        else:
            vol = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filenames])
    else:
        vol = tif.imread(FLAGS.path)

    if vol.shape[0] > 19:
        vol = downsample(vol)

    if FLAGS.output_dir and FLAGS.output_name:
        create_directory(FLAGS.output_dir)
        tif.imwrite(f"{FLAGS.output_dir}/{FLAGS.output_name}", vol)
        print(f"Saved to {FLAGS.output_dir}/{FLAGS.output_name} with shape: {vol.shape}")
    else:
        raise ValueError("Need to specify --output_dir and --output_name in order to save new TIFF file")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to TIFF image if single-file or path to folder if multi-file"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=19,
        help="Number of target z-slices"
    )
    parser.add_argument(
        "--multifile",
        action="store_true",
        help="Set True when loading multifile tiff image"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Path to put output volumes"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="Name of output TIFF file including the extension"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="tif",
        help="Choose file extension of image files"
    )
    FLAGS, _ = parser.parse_known_args()

    main()
