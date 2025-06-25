"""
This module provides utility functions for working with the OCTAVE segmentation dataset 
"""

from glob import glob
import numpy as np
import json
import cv2
import re
import os

def get_OCT_colors():
    """
    Return the standard RGB colormap for OCT tissue segmentation labels.

    Returns:
    - colors: np.ndarray of shape (15, 3), where each row is an RGB triplet (0–255)
      corresponding to a specific class label. The first entry (index 0) is background.
    """
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

def generate_new_segment_data(next_segment_number, name, color):
    """
    Generate a dictionary containing metadata for a new segment entry.

    Parameters:
    - next_segment_number: int, the number to be used in the segment key prefix (e.g., Segment1_)
    - name: str, the name to assign to the segment (e.g., "Optic Nerve")
    - color: str, the RGB color value as a space-separated string (e.g., "255 0 0")

    Returns:
    - new_segment_data: dict, keys and values formatted to represent a new segment's properties,
      including color, ID, name, and default extent.
    """
    base_key = f"Segment{next_segment_number}_"
    new_segment_data = {
        f"{base_key}Color": color,
        f"{base_key}ColorAutoGenerated": "0",
        f"{base_key}Extent": "0 -1 -167 -168 0 -1",
        f"{base_key}ID": f"Segment_{next_segment_number + 1}",
        f"{base_key}Name": name,
        f"{base_key}NameAutoGenerated": "0",
    }
    return new_segment_data

def get_num_segments(odict):
    """
    Count the number of unique segments defined in the NRRD header.

    Parameters:
    - odict: collections.OrderedDict, the NRRD header containing keys like 'Segment0_Name', 'Segment1_Color', etc.

    Returns:
    - num_segments: int, the number of unique segments based on 'Segment<number>_' key prefixes
    """
    return len(set([
        key.split("_")[0]                      # Extract the 'Segment<number>' prefix
        for key in odict.keys()
        if re.match(r"Segment(\d+)_", key)     # Only consider keys starting with 'Segment<number>_'
    ]))

def find_last_segment_position(odict):
    """
    Find the highest segment number and the corresponding key from an OrderedDict.

    Parameters:
    - odict: collections.OrderedDict, dictionary where keys are expected to follow
             the pattern 'Segment<number>_' (e.g., 'Segment3_Color')

    Returns:
    - last_segment_number: int, the highest segment number found (e.g., 3)
    - last_segment_key: str or None, the key associated with the last segment number
                        (e.g., 'Segment3_Color') — None if no segment keys are found
    """
    last_segment_number = -1
    last_segment_key = None
    for key in odict.keys():
        match = re.match(r"Segment(\d+)_", key)
        if match:
            segment_number = int(match.group(1))
            if segment_number > last_segment_number:
                last_segment_number = segment_number
                last_segment_key = key
    return last_segment_number, last_segment_key

def segment_name_already_exists(odict, new_segment_name):
    """
    Check if a segment with the specified name already exists in the OrderedDict.

    Parameters:
    - odict: collections.OrderedDict, dictionary containing segment metadata,
             where keys follow the format 'Segment<number>_Name'
    - new_segment_name: str, the segment name to check for existence

    Returns:
    - exists: bool, True if a segment with the given name already exists, False otherwise
    """
    for key, value in odict.items():
        if key.endswith("_Name") and value == new_segment_name:
            return True
    return False

def add_segmentation_to_header(data, header, nrrd_file_path, new_segment_name, new_segment_color, startedAddingSegments):
    """
    Add a new segmentation layer to an existing NRRD header and data volume.

    Parameters:
    - data: np.ndarray, the existing segmentation bitmap data (3D volume: [segments, height, width])
    - header: collections.OrderedDict, the metadata from the NRRD file
    - nrrd_file_path: str, path to the NRRD file to overwrite with updated segmentation
    - new_segment_name: str, the name of the new segment to add (must be unique unless resuming)
    - new_segment_color: str, the RGB color string for the new segment (e.g., "255 0 0")
    - startedAddingSegments: bool, if True, allows continuing insertion assuming segments have already been appended;
                              if False, skips duplicates silently

    Returns:
    - success: bool, True if the new segment was added successfully, False if skipped due to duplication

    Raises:
    - ValueError: if the segment name already exists but is being inserted mid-header (not at the end)
    """
	# Check if the new segment name already exists
    if segment_name_already_exists(header, new_segment_name):
        if not startedAddingSegments:
            print(f"WARNING: {new_segment_name} already exists in {nrrd_file_path}. Skipping...")
            return False
        else:
            raise ValueError(f"ERROR: {new_segment_name} exists out of position. Check manually in 3DSlicer")

	# Get the next segment number by finding the current highest
    last_segment_number, last_segment_key = find_last_segment_position(header)
    next_segment_number = last_segment_number + 1

    # Generate new header fields for the new segment
    new_segment_data = generate_new_segment_data(next_segment_number, new_segment_name, new_segment_color)

	# Append an empty segmentation mask to the data array (shape: [new_segment, H, W])
    new_bitmap_data = np.append(data, np.zeros([1] + list(data.shape[1:]), dtype=data.dtype), axis=0)

	# Regex pattern to match any 'Segment<number>'-style keys
    pattern = re.compile(r"Segment(\d+)")

    onLastSegment = False
    newSegmentAdded = False
    copied_odict = OrderedDict()

	# Reconstruct the header with the new segment injected after the last existing one
    for key, value in header.items():
        if key == "sizes":
            copied_odict[key] = new_bitmap_data.shape # Update size to reflect new segment count
        elif re.match(pattern, key):
            current_segment_num = int(re.match(pattern, key).group(1))
            if current_segment_num == last_segment_number and not newSegmentAdded:
                onLastSegment = True # We're at the end of existing segments
            copied_odict[key] = value
        else:
            if onLastSegment and not newSegmentAdded:
				# Inject new segment metadata here
                copied_odict.update(new_segment_data)
                newSegmentAdded = True
            copied_odict[key] = value

    # Save the NRRD file with the updated header
    nrrd.write(nrrd_file_path, new_bitmap_data, copied_odict)
    return True

def check_duplicate_labels(header):
    """
    Check for duplicate segment names in the NRRD header.

    Parameters:
    - header: collections.OrderedDict, NRRD metadata containing segment keys like 'Segment0_Name'

    Returns:
    - True if duplicates are found (raises ValueError), otherwise False

    Raises:
    - ValueError: if duplicate segment names are detected
    """
    # Collect all values from keys that define segment names
    labels = [
        val for key, val in header.items()
        if re.match(r"Segment(\d+)_", key) and key.endswith("_Name")
    ]

    # Raise an error if duplicates exist
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate labels found after processing: {labels}")

    return False  # Explicit return for clarity

def get_idx_of_label(header, name):
    """
    Retrieve the segment index for a given segment name from the NRRD header.

    Parameters:
    - header: collections.OrderedDict, NRRD metadata containing segment information
    - name: str, the segment name to look up

    Returns:
    - segment_number: int, the index of the segment (e.g., 0, 1, 2, ...)
      Returns None if the name is not found.
    """
    for key, val in header.items():
        match = re.match(r"Segment(\d+)_", key)
        if match:
            if key.endswith("_Name") and val == name:
                segment_number = int(match.group(1))
                return segment_number

def delete_by_name(data, header, name, original_labels):
    """
    Delete a segmentation label by name from both the NRRD data and header.

    Parameters:
    - data: np.ndarray, the 3D segmentation array where each slice along axis 0 is a label
    - header: collections.OrderedDict, the NRRD metadata containing segment definitions
    - name: str, the segment name to delete
    - original_labels: list of str, names of original segments that should not be deleted

    Returns:
    - new_data: np.ndarray, the data array with the specified label removed
    - new_header: OrderedDict, updated header with the label metadata removed

    Notes:
    - Labels in `original_labels` are protected from deletion.
    - Only segments with index >= 7 are allowed to be deleted.
    - If the label is not found, returns a warning string.
    """
    assert name not in original_labels
    segment_idx = get_idx_of_label(header, name)
    if segment_idx == None:
        return f"WARNING! label {name} not found in header"
    # number <7 should be reserved for original labels and not deleted
    assert segment_idx >= 7

    # Remove entries within the header OrderedDict
    new_header = header.copy()
    for key in header:
        if key.startswith(f"Segment{segment_idx}_"):
            new_header.pop(key)
    new_header["sizes"][0] -= 1

    # Remove corresponding segment channel
    print(f"Deleting idx {segment_idx} from {data.shape}")
    new_data = np.delete(data, segment_idx, axis=0)

    return new_data, new_header
