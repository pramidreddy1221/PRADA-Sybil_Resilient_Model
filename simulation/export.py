from pathlib import Path
from torchvision import datasets, transforms
from utils.image import save_image
from config import MNIST_PATH
from config import ROOT

OUTPUT_DIR = ROOT / "images" / "seed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PER_CLASS = 30

def export_seed_images():
    dataset = datasets.MNIST(
        root=str(MNIST_PATH),
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )

    count = {i: 0 for i in range(10)}

    for img, label in dataset:
        if count[label] < N_PER_CLASS:
            arr = img.numpy().squeeze()
            path = OUTPUT_DIR / f"class_{label}_sample_{count[label]}.png"
            save_image(arr, path)
            count[label] += 1
        if all(v == N_PER_CLASS for v in count.values()):
            break

    print(f"Saved {N_PER_CLASS * 10} seed images → {OUTPUT_DIR}")

if __name__ == "__main__":
    export_seed_images()