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

def replace_and_shift(arr):
    """
    Replace all values of 2 with 1, and shift down all numbers 3-15 by 1 in a numpy array.

    Parameters:
        arr (numpy.ndarray): Input array with values 1-15.

    Returns:
        numpy.ndarray: Modified array.
    """
    # Replace 2 with 1
    arr = np.where(arr == 2, 1, arr)

    # Shift values 3-15 down by 1
    arr = np.where(arr >= 3, arr - 1, arr)

    return arr

def segnrrd2nnUNet(path):
    """
    Converts segmentation NRRD files to nnU-Net compatible dataset.

    Args:
        path (str): The directory path containing the input files.
    """

    output_path = os.path.join(path, f"nnUNet_Dataset_v{FLAGS.version}")

    # Create the necessary directories for the nnU-Net dataset
    create_dataset_dirs(output_path)

    imagesTr = os.path.join(output_path, "imagesTr")
    labelsTr = os.path.join(output_path, "labelsTr")

    # Dictionary mapping segmentation labels to their corresponding numeric values
    if not FLAGS.combine_PED:
        labels_dict = {
            "background": 0,
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
    else:
        labels_dict = {
            "background": 0,
            "PED": 1,  # Pigement Epithelial Detachment
            "EX": 2,   # Exudates
            "FLU": 3,  # Fluid
            "GA": 4,   # Geographic Atrophy
            "HEM": 5,  # Hemorrhage
            "RPE": 6,  # Retinal Pigment Epithelium
            "RET": 7,  # Retina
            "CHO": 8,  # Choroid
            "VIT": 9,  # Vitreous
            "HYA": 10, # Hyaloid
            "SHS": 11, # Sub-Hyaloid Space
            "ART": 12, # Artifacts
            "ERM": 13, # Epiretinal Membrane
            "SES": 14  # Sub-ERM Space
        }


#    labels_dict = {
#        "background": 0,
#        "RPE": 1,  # Retinal Pigment Epithelium
#        "RET": 2,  # Retina
#        "CHO": 3,  # Choroid
#        "VIT": 4, # Vitreous
#        "HYA": 5, # Hyaloid
#        "SHS": 6, # Sub-Hyaloid Space
#        "ART": 7, # Artifacts
#        "ERM": 8, # Epiretinal Membrane
#        "SES": 9  # Sub-ERM Space
#    }

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

        # Remove first 6 labels as they are unused CNV DRU EX FLU GA HEM
        # bitmap = bitmap[6:]

        # Add an empty 0th layer to account for the expected background label at index 0
        bitmap = np.insert(bitmap, 0, np.zeros((1, *bitmap.shape[1:])), axis=0)

        # print number of pixels per label
        # print(bitmap.shape)
        # for i in range(bitmap.shape[0]):
            # print(i, bitmap[i].sum())

        # Convert one-hot encoded bitmap to label array, flipping axes from (X, Y, Z) to (Z, Y, X)
        labels = np.argmax(bitmap, axis=0).T

        # if combine_PED is true, combine CNV and DRU labels into PED label
        if FLAGS.combine_PED:
            labels = replace_and_shift(labels)

        # Save spacing information as JSON
        spacing = [81.0, 1.0, 2.9]
        save_json({"spacing": spacing}, os.path.join(imagesTr, f"{vol_name}.json"))
        save_json({"spacing": spacing}, os.path.join(labelsTr, f"{seg_name}.json"))

        # Save volume and label images as TIFF files
        output_tif = os.path.join(imagesTr, f"{vol_name}_0000.tif")
        output_labels = os.path.join(labelsTr, f"{seg_name}.tif")
        tif.imwrite(output_tif, vol, photometric='minisblack')
        tif.imwrite(output_labels, labels, photometric='minisblack')


    # Generate the dataset JSON file required by nnU-Net
    generate_dataset_json(output_path,
                          channel_names={"0": "OCT"},
                          labels=labels_dict,
                          file_ending=".tif",
                          num_training_cases=len(vol_paths),
                          dataset_name=f"nnUNet_Dataset_v{FLAGS.version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--path",
            type=str,
            required=True,
            help="Path to .seg.nrrd segmentations",
    )
    parser.add_argument(
            "--version",
            type=str,
            required=False,
            default=5,
            help="Dataset version number",
    )
    parser.add_argument(
            "--combine_PED",
            action="store_true",
            help="Combines DRU and CNV into a PED category",
    )
    FLAGS, _ = parser.parse_known_args()
    segnrrd2nnUNet(FLAGS.path)

