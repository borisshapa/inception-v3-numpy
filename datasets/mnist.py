import gzip
import os.path
from urllib.request import urlretrieve

import cv2
import numpy as np

MNIST_URL = "http://yann.lecun.com/exdb/mnist/"


def load_dataset(resize: bool = True):
    def download(filename: str, source=MNIST_URL):
        print(f"Downloading {filename}")
        urlretrieve(source + filename, os.path.join("data", "mnist", filename))

    def load_mnist_images(filename: str):
        path = os.path.join("data", "mnist", filename)
        if not os.path.exists(path):
            download(filename)
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28, 1)
        if resize:
            resized_images = np.zeros((data.shape[0], 1, 75, 75))
            for im_id, image in enumerate(data):
                resized_image = cv2.resize(image, dsize=(75, 75), interpolation=cv2.INTER_CUBIC)
                resized_images[im_id] = np.expand_dims(resized_image, axis=0)

            return resized_images / 255
        else:
            return np.moveaxis(data, -1, 1) / 255

    def load_mnist_labels(filename):
        path = os.path.join("data", "mnist", filename)
        if not os.path.exists(path):
            download(filename)
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    x_train = load_mnist_images("train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels("train-labels-idx1-ubyte.gz")
    x_test = load_mnist_images("t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels("t10k-labels-idx1-ubyte.gz")

    return x_train, y_train, x_test, y_test
