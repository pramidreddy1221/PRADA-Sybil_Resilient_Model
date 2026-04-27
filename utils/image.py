import numpy as np
from PIL import Image
from pathlib import Path


def save_image(array: np.ndarray, path: Path) -> None:
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(path)


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0
