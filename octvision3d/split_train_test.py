import argparse
import os
import random
import shutil

# Function to get .tif and .json pairs in a directory
def get_file_pairs(directory, category):
    tif_files = [f for f in os.listdir(directory) if f.endswith(".tif") and f.split("-")[1] == category]
    file_pairs = []
    for tif_file in tif_files:
        file_split = os.path.splitext(tif_file)[0].split("_")
        if file_split[-1] == "0000":
            json_file = "_".join(file_split[:-1]) + ".json"
        else:
            json_file = "_".join(file_split) + ".json"
        if not json_file in os.listdir(directory):
            raise ValueError(f"{json_file} not in {directory}")
        file_pairs.append((tif_file, json_file))
    return file_pairs

def move_files(src_dir, dest_dir, file_pairs):
    for tif_file, json_file in file_pairs:
        # Move .tif file
        shutil.move(os.path.join(src_dir, tif_file), os.path.join(dest_dir, tif_file))
        # Move corresponding .json file
        shutil.move(os.path.join(src_dir, json_file), os.path.join(dest_dir, json_file))

def main():
    categories = ["NORMAL", "DME", "DRU", "CNV", "ERM"]
    for category in categories:
        # Get list of paired .tif and .json files for images and labels
        image_pairs = sorted(get_file_pairs(FLAGS.imagesTr, category))
        label_pairs = sorted(get_file_pairs(FLAGS.labelsTr, category))

        assert len(image_pairs) == len(label_pairs)

        # Ensure there are enough pairs in both imagesTr and labelsTr
        if len(image_pairs) < FLAGS.num_pairs or len(label_pairs) < FLAGS.num_pairs:
            print(f"Not enough pairs available. Found {len(image_pairs)} image pairs and {len(label_pairs)} label pairs.")
            return

        # Randomly select the specified number of pairs from imagesTr
        selected_pairs = random.sample(list(zip(image_pairs, label_pairs)), FLAGS.num_pairs)
        selected_image_pairs = [image_pair for image_pair, label_pair in selected_pairs]
        selected_label_pairs = [label_pair for image_pair, label_pair in selected_pairs]

        print(category)
        print(selected_image_pairs)
        print(selected_label_pairs)

        # Move selected image pairs to imagesTs
        move_files(FLAGS.imagesTr, FLAGS.imagesTs, selected_image_pairs)

        # Move corresponding label pairs to labelsTs
        move_files(FLAGS.labelsTr, FLAGS.labelsTs, selected_label_pairs)

        print(f"Moved {FLAGS.num_pairs} image-label pairs to test directories.")
        print("WARNING: Remember to modify dataset.json to reflect new training num value")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move random image-label pairs from training to test directories.")
    parser.add_argument("--imagesTr", required=True, help="Path to the imagesTr directory")
    parser.add_argument("--labelsTr", required=True, help="Path to the labelsTr directory")
    parser.add_argument("--imagesTs", required=True, help="Path to the imagesTs directory")
    parser.add_argument("--labelsTs", required=True, help="Path to the labelsTs directory")
    parser.add_argument("--num_pairs", type=int, default=4, help="Number of pairs to move (default: 4)")
    FLAGS = parser.parse_args()
    main()
