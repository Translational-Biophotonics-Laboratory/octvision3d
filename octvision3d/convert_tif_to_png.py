"""
This script converts a TIFF volume into PNG slices and optionally overlays the corresponding 
segmentation stored in a .seg.nrrd file.

Functionality:
- If --save_label is not set: saves each TIFF slice as an individual PNG in a PNG_output folder.
- If --save_label is set: loads the .seg.nrrd file with the same base name, generates color-coded
  overlays using label metadata, and saves overlay images to overlay_output.

Usage:
    python script.py --path /path/to/image.tif [--save_label]
"""

import os
import nrrd
import numpy as np
import tifffile as tif
from argparse import ArgumentParser
from utils import create_directory, overlay_segments, sorted_rgb_colors
import cv2

def save_png(vol, output_dir, filename, seg=False):
    """
    Save a 3D volume as individual PNG slices.

    Parameters:
    - vol: np.ndarray, 3D array representing the volume (shape: [depth, height, width] or [depth, height, width, channels])
    - output_dir: str, directory where the PNG files will be saved
    - filename: str, base name used to generate the output PNG filenames
    - seg: bool, optional flag (currently unused) indicating if the volume is a segmentation

    Returns:
    - None. Writes PNG images to disk as <filename>-<z>.png for each slice z.
    """
    for z in range(vol.shape[0]):
        cv2.imwrite(os.path.join(output_dir, f"{filename}-{z}.png"), vol[z])

def save_overlay(vol, overlay, output_dir, filename):
    """
    Save overlay images by concatenating grayscale volume slices with color segmentation overlays.

    Parameters:
    - vol: np.ndarray, 3D grayscale volume of shape [depth, height, width]
    - overlay: np.ndarray, 4D RGB segmentation overlays of shape [depth, height, width, 3]
    - output_dir: str, directory to save the overlay images
    - filename: str, base name for output files

    Returns:
    - None. Writes side-by-side overlay PNGs as <filename>-overlay-<z>.png for each slice z.
    """
    vol = np.stack((vol,)*3, axis=-1)
    print(vol.shape, overlay.shape)
    combined_vol = np.concatenate((vol, overlay[:,:,:,::-1]), axis=2) # change axis to 1 for top-bottom config
    for z in range(combined_vol.shape[0]):
        cv2.imwrite(os.path.join(output_dir, f"{filename}-overlay-{z}.png"), combined_vol[z]) 

def main():
    if os.path.isdir(FLAGS.path) or not FLAGS.path.endswith((".tif", ".tiff")):
        raise ValueError("--path should point to a .tif file")

    seg_path = f"{os.path.splitext(FLAGS.path)[0]}.seg.nrrd"

    if FLAGS.save_label and not os.path.exists(seg_path):
        raise ValueError(f"Could not find segmentation file {seg_path}")

    vol = tif.imread(FLAGS.path)
    png_dir = os.path.join(os.path.dirname(FLAGS.path), "PNG_output/")
    overlay_dir = os.path.join(os.path.dirname(FLAGS.path), "overlay_output/")
    filename_base = os.path.splitext(os.path.basename(FLAGS.path))[0]
    
    if not FLAGS.save_label:
        create_directory(png_dir)
        save_png(vol, png_dir, filename_base)
        print(f"Conversion completed: TIF output in {png_dir}")

    else:
        create_directory(overlay_dir)
        bitmap, header = nrrd.read(seg_path)
        rgb_colors = sorted_rgb_colors(header)
        print(bitmap.shape)
        overlay = overlay_segments(bitmap, rgb_colors)
        save_overlay(vol, overlay, overlay_dir, filename_base)
        print(f"Conversion completed: overlay output in {overlay_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to TIFF image"
    )
    parser.add_argument(
        '--save_label',
        default=False,
        action='store_true',
        help="If enabled, will also save segmentations from .seg.nrrd file of the same name"
    )

    FLAGS, _ = parser.parse_known_args()
    main()


