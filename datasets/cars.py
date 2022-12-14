import glob
import os.path
import tarfile
from typing import Tuple

import cv2
import numpy as np
import scipy
from tqdm import tqdm


def load_dataset(image_size: Tuple[int, int] = (640, 480)):
    cars_dataset_dir = "data/cars_dataset"

    def load_cars_images(filename: str):
        images_dir = filename.split(".")[0]
        if not os.path.exists(images_dir):
            with tarfile.open(filename) as f:
                f.extractall(cars_dataset_dir)

        images = []
        for filename in tqdm([f"data/cars_dataset/cars_train/{str(i).zfill(5)}.jpg" for i in range(1, 8145)]):
            image = cv2.imread(filename)
            images.append(
                cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_CUBIC)
            )

        images = np.array(images) / 255
        images = np.moveaxis(images, -1, 1)
        return images

    def load_cars_labels(filename: str):
        mat = scipy.io.loadmat(filename)
        annotations = mat["annotations"]
        annotations = np.transpose(annotations)

        return np.array([a[0][4][0][0] for a in annotations])

    x = load_cars_images(os.path.join(cars_dataset_dir, "cars_train.tgz"))
    y = load_cars_labels(os.path.join(cars_dataset_dir, "cars_train_annos.mat"))
    y = y - 1

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    train_indices, test_indices = indices[:-800], indices[-800:]

    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, y_train, x_test, y_test
