"""
This script converts MATLAB .mat files containing OCT volume data into multi-page TIFF (.tif) format.

Functionality:
- Loads .mat files from a specified directory
- Extracts the image volume using a specified dictionary key (default: "images")
- Transposes and rotates the volume for correct orientation
- Saves the result as a .tif file with the same base filename

Usage:
    python script.py --path /path/to/mat/files [--key images]

Notes:
- The image data must be stored as a NumPy array under the given key
- Output TIFFs are saved to the same directory as the input .mat files
"""

import os
import tifffile as tif
import numpy as np
import scipy.io
import argparse
from utils import get_filenames
from tqdm import tqdm

def convert_mat2tif(path, key="images"):
    """
    Convert MATLAB (.mat) files to TIFF (.tif) format.

    Args:
        path (str): Path to the directory containing .mat files.
        key (str, optional): Key to access the image data in the .mat file. Defaults to "images".

    Raises:
        AssertionError: Raised if the loaded data is not a NumPy array.

    Returns:
        None
    """
    # Get a list of .mat file paths
    mat_paths = get_filenames(path, ext="mat")
    
    # Loop through each .mat file
    for mat_path in tqdm(mat_paths):
        # Generate output filename by removing extension and appending .tif
        output_filename = os.path.splitext(os.path.basename(mat_path))[0] + ".tif"
        
        # Load .mat file data
        mat_data = scipy.io.loadmat(mat_path)

        # Ensure the loaded data is a NumPy array
        try:
            assert isinstance(mat_data[key], np.ndarray)
        except KeyError:
            raise KeyError(f"KeyError: {key}. mat_data has keys: {mat_data.keys()}")
        
        # Transpose numpy and rotate 90 degrees clockwise
        oct_volume = np.rot90(mat_data[key].T, k=3, axes=(1,2))
        
        # Write the transposed data to a TIFF file
        tif.imwrite(os.path.join(path, output_filename), oct_volume)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to .mat file directory"
    )
    parser.add_argument(
        "--key",
        type=str,
        default="images",
        help="dict key to load images from .mat file"
    )
    FLAGS, _ = parser.parse_known_args()
    
    # Convert .mat files to .tif format
    convert_mat2tif(FLAGS.path, key=FLAGS.key)
