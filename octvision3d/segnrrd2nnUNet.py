import os
import nrrd
import argparse
import numpy as np
import tifffile as tif
from tqdm import tqdm

from octvision3d.utils import (get_filenames,
                               create_dataset_dirs,
                               save_json,
                               generate_dataset_json)

def segnrrd2nnUNet(path):
    """
    Converts segmentation NRRD files to nnU-Net compatible dataset.

    Args:
        path (str): The directory path containing the input files.
    """
    
    output_path = os.path.join(path, "nnUNet_Dataset")
    
    # Create the necessary directories for the nnU-Net dataset
    create_dataset_dirs(output_path)
    
    imagesTr = os.path.join(output_path, "imagesTr")
    labelsTr = os.path.join(output_path, "labelsTr")

    # Dictionary mapping segmentation labels to their corresponding numeric values
    labels_dict = {
        "CNV": 1,  # Choroidal Neovascularization
        "DRU": 2,  # Drusen
        "EX": 3,   # Exudates
        "FLU": 4,  # Fluid
        "GA": 5,   # Geographic Atrophy
        "HEM": 6,  # Hemorrhage
        "RPE": 7,  # Retinal Pigment Epithelium
        "RET": 8,  # Retina
        "CHO": 9,  # Choroid
        "VIT": 10, # Vitreous
        "HYA": 11, # Hyaloid
        "SHS": 12, # Sub-Hyaloid Space
        "ART": 13, # Artifacts
        "ERM": 14, # Epiretinal Membrane
        "SES": 15  # Sub-ERM Space
    }

    # Retrieve paths for TIFF volumes and segmentation NRRD files, excluding those with "slo" in their names
    vol_paths = [i for i in get_filenames(path, ext="tif") if "slo" not in i]
    seg_paths = [i for i in get_filenames(path, ext="seg.nrrd") if "slo" not in i]
    
    for vol_path, seg_path in tqdm(zip(vol_paths, seg_paths), total=len(vol_paths)):
        # Ensure corresponding volume and segmentation files match by their basename
        assert vol_path.split(".")[0] == seg_path.split(".")[0]
        
        vol_name = os.path.splitext(os.path.basename(vol_path))[0]
        seg_name = os.path.basename(vol_path).split(".")[0]

        # Load TIFF volume and segmentation NRRD labels
        vol = tif.imread(vol_path)
        bitmap, header = nrrd.read(seg_path)

        rgb_vol = np.stack((vol,) * 3, axis=-1)

        # Convert one-hot encoded bitmap to label array, flipping axes from (X, Y, Z) to (Z, Y, X)
        labels = np.argmax(bitmap, axis=0).T
        
        # Save spacing information as JSON
        spacing = [81.0, 1.0, 2.9]
        save_json({"spacing": spacing}, os.path.join(imagesTr, f"{vol_name}.json"))
        save_json({"spacing": spacing}, os.path.join(labelsTr, f"{seg_name}.json"))

        # Save volume and label images as TIFF files
        output_tif = os.path.join(imagesTr, f"{vol_name}_0000.tif")
        output_labels = os.path.join(labelsTr, f"{seg_name}.tif")
        tif.imwrite(output_tif, rgb_vol, photometric='rgb')
        tif.imwrite(output_labels, labels, photometric='minisblack')

    # Generate the dataset JSON file required by nnU-Net
    generate_dataset_json(output_path, 
                          channel_names={"0": "OCT"},
                          labels=labels_dict,
                          file_ending=".tif",
                          num_training_cases=len(vol_paths),
                          dataset_name="3D OCT Dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--path",
            type=str,
            required=True,
            help="Path to .seg.nrrd segmentations",
    )
    FLAGS, _ = parser.parse_known_args()
    segnrrd2nnUNet(FLAGS.path)

