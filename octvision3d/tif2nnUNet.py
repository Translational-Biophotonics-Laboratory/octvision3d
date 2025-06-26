# This script converts a directory of TIFF OCT volumes into an nnU-Net-compatible dataset.
# It creates the appropriate folder structure, saves image spacing metadata, and
# writes all volumes in TIFF format. It also generates the nnU-Net `dataset.json`
# metadata file with channel and label definitions.
#
# Used when only desiring to convert images to nnUNet format without corresponding
# .seg.nrrd 3DSlicer segmentation files. If you do, use segnrrd2nnUNet.py instead.
#
# Usage:
#   python tif2nnUNet.py --path /path/to/data
#   python tif2nnUNet.py --path /path/to/data

import os
import argparse
import tifffile as tif
from tqdm import tqdm
from octvision3d.utils import (get_filenames,
                               save_json,
                               create_directory,
                               generate_dataset_json)

def tif2nnUNet():
    """
    Converts segmentation NRRD files to nnU-Net compatible dataset.

    Args:
        path (str): The directory path containing the input files.
    """
    
    dataset_output_path = os.path.join(FLAGS.path, "converted_nnUNet")
    output_path = os.path.join(dataset_output_path, "imagesTs")
    
    create_directory(output_path)

    labels_dict = {
        "background": 0,
        "SRM": 1,  # Subretinal Material
        "HRM": 2,  # Hyperreflective Material
        "FLU": 3,  # Fluid
        "HTD": 4,  # Hypertransmission defect
        "RPE": 5,  # Retinal Pigment Epithelium
        "RET": 6,  # Retina
        "CHO": 7,  # Choroid
        "VIT": 8,  # Vitreous
        "HYA": 9,  # Hyaloid
        "SHS": 10, # Sub-Hyaloid Space
        "ART": 11, # Artifacts
        "ERM": 12, # Epiretinal Membrane
        "SES": 13  # Sub-ERM Space
    }
    # Retrieve paths for TIFF volumes and segmentation NRRD files, excluding those with "slo" in their names
    vol_paths = [i for i in get_filenames(FLAGS.path, ext="tif") if "slo" not in i]

    for vol_path in tqdm(vol_paths, total=len(vol_paths)):

        vol_name = os.path.splitext(os.path.basename(vol_path))[0]

        # Load TIFF volume and segmentation NRRD labels
        vol = tif.imread(vol_path)

        # Save spacing information as JSON
        spacing = [81.0, 1.0, 2.9]
        save_json({"spacing": spacing}, os.path.join(output_path, f"{vol_name}.json"))

        # Save volume and label images as TIFF files
        output_tif = os.path.join(output_path, f"{vol_name}_0000.tif")
        tif.imwrite(output_tif, vol, photometric='minisblack')

    # Generate the dataset JSON file required by nnU-Net
    generate_dataset_json(dataset_output_path,
                          channel_names={"0": "OCT"},
                          labels=labels_dict,
                          file_ending=".tif",
                          num_training_cases=len(vol_paths),
                          dataset_name=f"OCTAVE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--path",
            type=str,
            required=True,
            help="Path to tif volumes to convert",
    )
    FLAGS, _ = parser.parse_known_args()
    tif2nnUNet()

