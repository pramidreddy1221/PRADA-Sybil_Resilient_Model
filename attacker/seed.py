from pathlib import Path
import numpy as np
from torchvision import datasets, transforms
from config import MNIST_PATH

def get_seed_samples(n_per_class: int = 10):
    dataset = datasets.MNIST(
        root=str(MNIST_PATH),
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )

    images_by_class = {i: [] for i in range(10)}

    for img, label in dataset:
        if len(images_by_class[label]) < n_per_class:
            images_by_class[label].append(img.numpy().squeeze())
        if all(len(v) == n_per_class for v in images_by_class.values()):
            break

    images = []
    labels = []
    for i in range(10):
        images.extend(images_by_class[i])
        labels.extend([i] * n_per_class)

    return np.array(images, dtype=np.float32), labels
