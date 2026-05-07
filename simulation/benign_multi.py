import sys
import random
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.image import load_image
from config import API_URL

IMAGE_DIR = ROOT / "images" / "seed"
ACCOUNTS = [f"benign_{i:03d}" for i in range(2, 7)]


def run(accounts=ACCOUNTS, subset_size=500):
    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    if not image_paths:
        print(f"ERROR: no images found in {IMAGE_DIR}")
        return

    print(f"image pool: {len(image_paths)} images")

    for idx, account_id in enumerate(accounts):
        random.seed(idx)
        subset = random.sample(image_paths, min(subset_size, len(image_paths)))
        print(f"{account_id}: sending {len(subset)} images")
        for i, path in enumerate(subset):
            arr = load_image(path)
            payload = {"account_id": account_id, "image": arr.tolist()}
            resp = requests.post(API_URL, json=payload)
            if resp.status_code != 200:
                print(f"  [{i + 1}/{len(subset)}] {path.name} ERROR {resp.status_code}")
            elif (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(subset)}] sent")
        print(f"{account_id}: done")


if __name__ == "__main__":
    run()
