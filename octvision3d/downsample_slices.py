import numpy as np
import tifffile as tif
from scipy.ndimage import zoom
from argparse import ArgumentParser
from utils import get_filenames
from pprint import pprint

def downsample(data, target=19):
    # Calculate the exact step size
    step_size = data.shape[2] / target

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
    downsampled_data = data[:, :, selected_indices]
    return downsampled_data

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
        help="Name of output TIFF file"
    )
    FLAGS, _ = parser.parse_known_args()

    if FLAGS.multifile:
        filenames = get_filenames(FLAGS.path, ext="tif")
        vol = np.array([tif.imread(f) for f in filenames]).T
    else:
        vol = tif.imread(FLAGS.path).T
    
    downsampled_data = downsample(vol)

    if FLAGS.output_dir and FLAGS.output_name:
        tif.imwrite(f"{FLAGS.output_dir}/{FLAGS.output_name}.tif", downsampled_data.T)
    else:
        raise ValueError("Need to specify --output_dir and --output_name in order to save new TIFF file")