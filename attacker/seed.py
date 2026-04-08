# attacker/seed.py
# Load small balanced seed samples from already downloaded MNIST data

from config import MNIST_PATH
import numpy as np
from torchvision import datasets, transforms


def get_seed_samples(n_per_class: int = 10):
    """
    Load n_per_class samples for each digit (0-9).
    Returns:
        images: np.ndarray of shape (N, 28, 28) float32
        labels: list of int (true MNIST labels, NOT victim labels)
    """
    dataset = datasets.MNIST(
        root=str(MNIST_PATH),
        train=False,
        download=False,      # already downloaded
        transform=transforms.ToTensor()
    )

    images_by_class = {i: [] for i in range(10)}

    for img, label in dataset:
        if len(images_by_class[label]) < n_per_class:
            images_by_class[label].append(img.numpy().squeeze())  # (28,28)
        if all(len(v) == n_per_class for v in images_by_class.values()):
            break

    images = []
    labels = []
    for i in range(10):
        images.extend(images_by_class[i])
        labels.extend([i] * n_per_class)

    return np.array(images, dtype=np.float32), labels