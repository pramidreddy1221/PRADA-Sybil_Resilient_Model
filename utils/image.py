import numpy as np
from PIL import Image
from pathlib import Path


def save_image(array: np.ndarray, path: Path) -> None:
    """
    Save a numpy array (28x28, float32, values in [0,1]) as a .png file.
    """
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(path)


def load_image(path: Path) -> np.ndarray:
    """
    Load a .png file and return as numpy array (28x28, float32, values in [0,1]).
    """
    img = Image.open(path).convert("L")  # L = grayscale
    return np.array(img, dtype=np.float32) / 255.0