import argparse
import cv2
import numpy as np
from octvision3d.utils import get_filenames, create_directory
import tifffile as tiff
import cv2
import os
import shutil
from tqdm import tqdm

def reshape(image, target_height=496, target_width=1024, mode="bilinear"):
    """
    Reshape a 3D NumPy image volume by resizing height and width per slice.

    Parameters:
        image (np.ndarray): Input image array of shape (z, y, x).
        target_height (int): Desired height (y-dimension) of each slice.
        target_width (int): Desired width (x-dimension) of each slice.
        mode (str): Interpolation method: 'nearest', 'bilinear', or 'bicubic'.

    Returns:
        np.ndarray: Resized image array of shape (z, target_height, target_width).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if image.ndim != 3:
        raise ValueError("Input image must be a 3D array of shape (z, y, x).")

    interpolation_modes = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC
    }

    if mode not in interpolation_modes:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from {list(interpolation_modes.keys())}.")

    z_dim = image.shape[0]
    resized_slices = []

    for z in range(z_dim):
        slice_2d = image[z].astype(np.uint8)
        resized = cv2.resize(
            slice_2d,
            (target_width, target_height),
            interpolation=interpolation_modes[mode]
        )
        resized_slices.append(resized)

    return np.stack(resized_slices, axis=0)  # shape: (z, target_height, target_width)

def reshape_images():
    image_output = os.path.join(FLAGS.image, FLAGS.output_dir)
    label_output = os.path.join(FLAGS.label, FLAGS.output_dir)
    create_directory(image_output)
    create_directory(label_output)

    image_files = get_filenames(FLAGS.image, ext="tif")
    label_files = get_filenames(FLAGS.label, ext="tif")
    image_jsons = get_filenames(FLAGS.image, ext="json")
    label_jsons = get_filenames(FLAGS.label, ext="json")
    assert(len(image_files) > 0)
    assert(len(image_files) == len(label_files) == len(image_jsons) == len(label_jsons))

    for image_f, label_f, image_j, label_j in tqdm(zip(image_files, label_files, image_jsons, label_jsons), total=len(image_files)):
        image = tiff.imread(image_f)
        label = tiff.imread(label_f)
        image_reshaped = reshape(image)
        label_reshaped = reshape(label)
        tiff.imwrite(os.path.join(image_output, os.path.basename(image_f)), image_reshaped.astype(np.uint8))
        tiff.imwrite(os.path.join(label_output, os.path.basename(label_f)), label_reshaped.astype(np.int64))
        shutil.copy(image_j, os.path.join(image_output, os.path.basename(image_j)))
        shutil.copy(label_j, os.path.join(label_output, os.path.basename(label_j)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to TIFF image folder"
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Path to TIFF label folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reshaped",
        help="name of output folder"
    )

    FLAGS, _ = parser.parse_known_args()
    reshape_images()
