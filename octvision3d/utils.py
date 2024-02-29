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

def sorted_rgb_colors(header):
    """
    Extracts and sorts RGB colors defined in a .seg.nrrd header OrderedDict.

    Parses the header dictionary for keys ending with "Color", extracts their RGB values, sorts them naturally
    based on their keys, and then converts these values to a numpy array of uint8 type.

    Parameters:
    - header (dict): Dictionary containing segment names as keys and color strings as values.

    Returns:
    - numpy.ndarray: Sorted array of RGB colors, with each color represented as a list of uint8 values.
    """
    segment_colors = {k.split("_")[0]: v for k, v in header.items() if k.endswith("Color")}
    sorted_color_map = natural_sort(segment_colors.items())
    _, sorted_colors = list(zip(*sorted_color_map))
    rgb_colors = np.array([[round(255.*float(c)) for c in i.split(" ")] for i in sorted_colors], dtype=np.uint8)
    return rgb_colors

def overlay_segments(bitmap, colors):
    """
    Overlay binary masks onto a blank image with specified colors.

    :param masks: List of binary masks (numpy arrays).
    :param colors: List of colors corresponding to each mask.
    :return: Image with masks overlaid.
    """
    # Create a blank image

    final_images = np.zeros(bitmap.T.shape[:-1] + (3,), dtype=np.uint8)

    for segment2d, color in zip(bitmap, colors):
        for i, slice in enumerate(segment2d.T):
            bgr_image = cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)
            final_images[i] += bgr_image * color

    return final_images
