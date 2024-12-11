from glob import glob
import numpy as np
import json
import cv2
import re
import os

def get_OCT_colors():
    colors = np.array([
        [0, 0, 0], # background
        [241, 214, 145],
        [177, 122, 101],
        [111, 184, 210],
        [216, 101, 79],
        [221, 130, 101],
        [144, 238, 144],
        [162, 0, 255],
        [143, 143, 112],
        [224, 240, 252],
        [117, 250, 252],
        [176, 252, 209],
        [252, 252, 84],
        [56, 125, 247],
        [100, 50, 0],
    ])
    return colors

def get_filenames(path, ext):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    ext = "".join(map(either, ext))
    return sorted(glob(f"{path}/*.{ext}"), key=_natural_sort_key)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

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

def create_dataset_dirs(path):
    for i in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
        if not os.path.exists(os.path.join(path, i)):
            os.makedirs(os.path.join(path, i), exist_ok=True)

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
                          description: str = None):
    
    # channel names need strings as keys
    # label values need ints as values
    for k in list(channel_names.keys()):
        assert isinstance(k, str)
    for k, v in labels.items():
        assert isinstance(v, int)
    
    dataset_json = {
        'channel_names': channel_names,
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
    }
    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if license is not None:
        dataset_json['licence'] = license
    if description is not None:
        dataset_json['description'] = description
    
    save_json(dataset_json, os.path.join(output_folder, 'dataset.json'), sort_keys=False)
