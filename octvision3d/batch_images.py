import os
import shutil
from argparse import ArgumentParser
from utils import get_filenames, create_directory
from tqdm import tqdm

def batch_list(input_list, batch_size=5):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

def main():
    tif_filenames = sorted(get_filenames(FLAGS.path, ext="tif*"))
    nrrd_filenames = sorted(get_filenames(FLAGS.path, ext="seg.nrrd"))
    assert len(tif_filenames) == len(nrrd_filenames) and len(tif_filenames) > 0

    batches = batch_list(list(zip(tif_filenames, nrrd_filenames)), batch_size=FLAGS.batch_size)
    for i, batch in tqdm(enumerate(batches, start=1), total=len(batches)):
        for tif, nrrd in batch:
            batch_dir = os.path.join(FLAGS.path, f"{os.path.basename(os.path.dirname(FLAGS.path))}-{i}")
            create_directory(batch_dir)

            shutil.move(tif, batch_dir)
            shutil.move(nrrd, batch_dir)
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to TIFF image if single-file or path to folder if multi-file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of image-label pairs for each batch"
    )
    FLAGS, _ = parser.parse_known_args()

    main()
