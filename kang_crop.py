# IMPORTS 
# Libraries
import os
import pandas as pd

# Modules
from utils.utils import *   # imports MAFAT_Dataset, cropPolygonImage, etc.

# VARIABLES
# Configuration
# filepaths for csv data
train_path = os.path.relpath('./dataset_v2/train.csv')
test_path  = os.path.relpath('./dataset_v2/test.csv')

# csv dataset
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# Global Variables
# root directories for images
TRAIN_ROOT = os.path.relpath('./dataset_v2/root/train/')
TEST_ROOT  = os.path.relpath('./dataset_v2/root/test/')

# FUNCTIONS
# Remove unwanted characters within labels
def edit_labels():
    # For train (used for class folders, etc.)
    train.replace(' ', '_', regex=True, inplace=True)
    train.replace('/', '_', regex=True, inplace=True)
    # For test (won't hurt even if there are fewer label columns)
    test.replace(' ', '_', regex=True, inplace=True)
    test.replace('/', '_', regex=True, inplace=True)

def generate_cropped_test_images(pad=5, fixedshape=64):
    """
    Create cropped images for all entries in the TEST dataset.
    Crops are saved into TEST_ROOT/cropped/.
    """
    # Make sure output folder exists
    cropped_dir = os.path.join(TEST_ROOT, 'cropped')
    os.makedirs(cropped_dir, exist_ok=True)

    # Build test dataset using the class from utils.py
    test_dataset = MAFAT_Dataset(
        csv_file=test_path,
        root_dir=TEST_ROOT
    )

    n_samples = len(test_dataset)
    print(f"Generating cropped images for TEST: {n_samples} samples...")

    for i in range(n_samples):
        sample = test_dataset[i]

        image    = sample['image']
        xyRaw    = sample['xyRaw']           # [x0,y0,x1,y1,x2,y2,x3,y3]
        base_fn  = sample['filename']
        tagid    = sample['tagid']

        # Build x and y vectors from xyRaw
        xvec = xyRaw[0::2]                   # [x0, x1, x2, x3]
        yvec = xyRaw[1::2]                   # [y0, y1, y2, y3]

        # Optional: skip if all coords are zero (no object)
        if (xvec == 0).all() and (yvec == 0).all():
            continue

        # Filename pattern consistent with train cropping
        out_name = f"{base_fn}_{tagid}.png"

        # Use cropPolygonImage from utils.py
        cropPolygonImage(
            image=image,
            xvec=xvec,
            yvec=yvec,
            pad=pad,
            root=TEST_ROOT,
            filename=out_name,
            pltIm=None,        # no plotting while batch processing
            fixedshape=fixedshape
        )

        if (i + 1) % 250 == 0:
            print(f"Cropped {i + 1} / {n_samples} test images")

    print("Done generating cropped TEST images.")


def main():
    # 1. Clean labels (same preprocessing as train script)
    edit_labels()

    # 2. Generate cropped images for TEST set only
    generate_cropped_test_images(pad=5, fixedshape=64)


if __name__ == '__main__':
    main()
