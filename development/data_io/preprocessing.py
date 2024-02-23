"""Preprocess training data to optimize dataloading"""
import sys
from pathlib import Path

from cv2 import cv2
from tqdm import tqdm

SCALES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
OUTPUT_FOLDER = Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed")


def preprocess(folder: Path):
    """Copy and scale the images of the folder to the data/preprocessed folder.

    This function will copy all images in the provided folder.
    While copying, it will also create scaled versions of the images.

    Args:
        folder: source folder containing training images.
    """
    print(f"Using '{folder}'")
    all_images = list(folder.iterdir())
    output_folder = OUTPUT_FOLDER / folder.name
    print(f"Copy to '{output_folder}'")
    cont = input("Continue ([y]/n)?")
    if cont and cont != "y":
        print("aborted")
        return
    output_folder.mkdir(exist_ok=True, parents=True)
    for image in tqdm(all_images):
        img = cv2.imread(str(image))

        for scale in SCALES:
            resized = cv2.resize(img, (scale, scale))
            scale_folder = output_folder / f"scale_{scale}"
            scale_folder.mkdir(exist_ok=True)
            cv2.imwrite(str(scale_folder / image.name), resized)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        preprocess(Path(sys.argv[1]))
    else:
        print("Please provide the source folder containing the images.")
