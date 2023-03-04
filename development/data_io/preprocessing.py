"""Preprocess training data to optimize dataloading"""
from pathlib import Path

from cv2 import cv2
from tqdm import tqdm

SCALES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
OUTPUT_FOLDER = Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed")


def preprocess(folder: Path):
    all_images = list(folder.iterdir())
    output_folder = OUTPUT_FOLDER / folder.name
    output_folder.mkdir(exist_ok=True, parents=True)
    for image in tqdm(all_images):
        img = cv2.imread(str(image))

        for scale in SCALES:
            resized = cv2.resize(img, (scale, scale))
            scale_folder = output_folder / f"scale_{scale}"
            scale_folder.mkdir(exist_ok=True)
            cv2.imwrite(str(scale_folder / image.name), resized)



if __name__ == "__main__":
    preprocess(Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\extracted\person0"))
