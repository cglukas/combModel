"""Face extractor that processes image and extracts faces from them"""
from pathlib import Path
from typing import Union

import PIL.Image
import cv2
import numpy as np
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN


class Extractor:
    def __init__(self):
        self.export_path = Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\extracted")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # keep_all keeps all the detected faces
        self.model = MTCNN(keep_all=True, device=device)
        self.default_corners = np.array(
            [
                [0.3, 0.3],
                [0.7, 0.3],
                [0.5, 0.7],
            ],
            dtype=np.float32,
        )
        self.image_size = 1024

    def extract(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        all_jpgs = list(filepath.glob("*.jpg"))

        export_path = self.export_path / filepath.name
        export_path.mkdir(exist_ok=True)
        for jpg in tqdm(all_jpgs):
            images = self.extract_single(jpg)
            for i, img in enumerate(images):
                export_file = export_path / f"{i}_{jpg.name}"
                cv2.imshow("hello", img/255)
                cv2.waitKey(10)
                cv2.imwrite(str(export_file), img)

    def extract_single(self, jpg: Path) -> list:
        img = PIL.Image.open(str(jpg)).convert("RGB")
        bounding_boxes, confidence, landmarks = self.model.detect(img, landmarks=True)
        images = []
        if landmarks is None:
            return images
        for this_marks in landmarks:
            corners = np.zeros((3, 2), dtype=np.float32)
            corners[0] = this_marks[0]  # left eye
            corners[1] = this_marks[1]  # right eye
            corners[2] = (this_marks[3] + this_marks[4]) / 2  # mouth
            size = self.default_corners * self.image_size
            affine_trans = cv2.getAffineTransform(corners, size)

            mat = np.array(img, dtype=np.float32)
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            mat = cv2.warpAffine(mat, affine_trans, (self.image_size, self.image_size))
            images.append(mat)
        return images


if __name__ == "__main__":
    extractor = Extractor()
    extractor.extract(Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\source_clips\person2"))
