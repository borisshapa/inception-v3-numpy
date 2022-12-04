import glob

import cv2
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class Cars(Dataset):
    def __init__(self):
        self.image_paths = [
            f"data/cars_dataset/cars_train/{str(i).zfill(5)}.jpg"
            for i in range(1, 8145)
        ]
        mat = scipy.io.loadmat("data/cars_dataset/cars_train_annos.mat")
        annotations = mat["annotations"]
        annotations = np.transpose(annotations)

        self.labels = [a[0][4][0][0] for a in annotations]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
        image = torch.tensor(image) / 255

        label = self.labels[item] - 1
        return torch.moveaxis(image, 2, 0), torch.tensor(label)


if __name__ == "__main__":
    dataset = Cars()
