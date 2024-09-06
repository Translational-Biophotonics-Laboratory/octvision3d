import os
import nrrd
import numpy as np
import tifffile as tif
from argparse import ArgumentParser
from utils import create_directory, overlay_segments, sorted_rgb_colors
import cv2

def save_png(vol, output_dir, filename, seg=False):
    for z in range(vol.shape[0]):
        cv2.imwrite(os.path.join(output_dir, f"{filename}-{z}.png"), vol[z])

def save_overlay(vol, overlay, output_dir, filename):
    vol = np.stack((vol,)*3, axis=-1)
    print(vol.shape, overlay.shape)
    combined_vol = np.concatenate((vol, overlay[:,:,:,::-1]), axis=2) # change axis to 1 for top-bottom config
    for z in range(combined_vol.shape[0]):
        cv2.imwrite(os.path.join(output_dir, f"{filename}-overlay-{z}.png"), combined_vol[z]) 
        
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

