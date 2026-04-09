# Simulates a benign user sending real MNIST images to the API

from pathlib import Path
from utils.image import load_image
from config import API_URL
import requests

IMAGE_DIR = Path("images/seed")

def simulate_benign(account_id: str = "benign_001"):
    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    
    if not image_paths:
        print("No images found. Run simulation/export.py first.")
        return

    print(f"Simulating benign user: {account_id}")
    print(f"Sending {len(image_paths)} images...")

    for path in image_paths:
        arr = load_image(path)
        payload = {
            "account_id": account_id,
            "image": arr.tolist()
        }
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"  {path.name} → pred: {data['pred']}")
        else:
            print(f"  {path.name} → failed: {response.status_code}")

if __name__ == "__main__":
    simulate_benign()