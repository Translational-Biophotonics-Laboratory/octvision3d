from glob import glob
import numpy as np
import cv2
import re
import os

def get_filenames(path, ext):
    return sorted(glob(f"{path}/*.{ext}"))

def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def _natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]

def natural_sort(l):
    return sorted(l, key=lambda x: _natural_sort_key(x[0]))

def overlay_segments(segments, colors):
    """
    Overlay binary masks onto a blank image with specified colors.

    :param masks: List of binary masks (numpy arrays).
    :param colors: List of colors corresponding to each mask.
    :return: Image with masks overlaid.
    """
    # Create a blank image

    final_images = np.zeros(segments.T.shape[:-1] + (3,), dtype=np.uint8)

    for segment, color in zip(segments, colors):
        for i, slice in enumerate(segment.T):
            bgr_image = cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)
            final_images[i] += bgr_image * color

    return final_images

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

        if n_unlabeled_pixels > 0:
            unlabeled.append(f"{os.path.basename(seg_path)}, Slice {i}: {n_unlabeled_pixels} pixels unlabeled")

    if unlabeled:
        for s in unlabeled:
            print(s)
    else:
        print(f"No unlabeled pixels found in {os.path.basename(seg_path)}")