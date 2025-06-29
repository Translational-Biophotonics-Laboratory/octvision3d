"""
This script verifies that all pixels in segmentation overlays have been labeled (i.e., not black)
for each slice in a set of .seg.nrrd segmentation files.

Functionality:
- Loads .seg.nrrd segmentation masks and corresponding TIFF images
- Applies color overlays to the segmentation using label metadata
- Checks for and reports any unlabeled (black) pixels in each slice of the overlay

Usage:
    python script.py --path /path/to/seg/files [--ext seg.nrrd]

Notes:
- TIFF images must exist alongside .seg.nrrd files with matching filenames
- Expects segmentations to contain exactly 14 labels
- Prints location and count of unlabeled pixels per slice, if any are found
"""

from argparse import ArgumentParser
import numpy as np
import os
import nrrd
from octvision3d.utils import overlay_segments, sorted_rgb_colors,\
                              get_filenames
import tifffile as tiff

def check_unlabeled_pixels(overlay, seg_path):
    """
    Reports the number of unlabeled (black) pixels in each image slice of an overlay.

    Parameters:
    - overlay (list of numpy.ndarray): List of RGB image slices.
    - seg_path (str): Path to the segmentation overlay for reporting.

    Returns:
    - None
    """
    unlabeled = []
    for i, image in enumerate(overlay):
        # Sum the RGB values of each pixel. Black pixels will sum to 0.
        pixel_sums = np.sum(image, axis=-1)

        # Identify black pixels (sum == 0) and count them
        n_unlabeled_pixels = np.count_nonzero(pixel_sums == 0)
        x, y = np.where(pixel_sums == 0)
        locations = list(zip(x.tolist(), y.tolist()))

        if n_unlabeled_pixels > 0:
            unlabeled.append(f"{os.path.basename(seg_path)}, Slice {i}: {n_unlabeled_pixels} pixels unlabeled")
            unlabeled.append(f"shape: {pixel_sums.shape}, locations {locations}")

    if unlabeled:
        for s in unlabeled:
            print(s)
    else:
        print(f"No unlabeled pixels found in {os.path.basename(seg_path)}")

def main():
    filenames = get_filenames(FLAGS.path, ext=FLAGS.ext)
    tif_filenames = get_filenames(FLAGS.path, ext="tif")

    if len(filenames) == 0:
        raise AssertionError(f"No files with found at {FLAGS.path} ending with {FLAGS.ext}")

    for filename, tif_filename in zip(filenames, tif_filenames):
        # load from .seg.nrrd file
        bitmap, header = nrrd.read(filename)
        tif_img = tiff.imread(tif_filename)

        if tif_img.shape != bitmap.shape[1:][::-1]:
            raise AssertionError(f"TIF and seg.nrrd bitmap do not have the same shape: {filename}, tif shape: {tif_img.shape}, label: {bitmap.shape[1:][::-1]}")

        if bitmap.shape[0] != 14:
            raise AssertionError(f"segmentation shape should have 14 labels. {filename} has {bitmap.shape[0]}")

        # get decimal rgb colors (0-1) from header file sorted (segment0, segment1,...)
        rgb_colors = sorted_rgb_colors(header)

        # overlay segmentations in different channels into one rgb image per 2d slice
        overlay = overlay_segments(bitmap, rgb_colors)

        check_unlabeled_pixels(overlay, filename)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to directory containing .seg.nrrd files"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="seg.nrrd",
        help="file extension"
    )
    FLAGS, _ = parser.parse_known_args()
    main()
