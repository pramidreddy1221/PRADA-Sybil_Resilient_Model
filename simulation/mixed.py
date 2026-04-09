# Simulates a single account sending a mix of normal and synthetic images
# This tests whether PRADA can detect an attacker who blends in with normal queries

from pathlib import Path
from utils.image import load_image, save_image
from config import API_URL, LAMBDA
import requests
import numpy as np
import torch
import torch.nn as nn
from attacker.substitute_model import SubstituteCNN
from config import SAVE_PATH, DEVICE

IMAGE_DIR  = Path("images/seed")
SYNTH_DIR  = Path("images/synthetic")
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

def generate_synthetic(model, arr: np.ndarray, label: int) -> np.ndarray:
    """Apply FGSM to a single image."""
    model.eval()
    X = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    X.requires_grad_(True)
    y = torch.tensor([label], dtype=torch.long).to(DEVICE)
    loss = nn.CrossEntropyLoss()(model(X), y)
    loss.backward()
    with torch.no_grad():
        synthetic = X + LAMBDA * X.grad.sign()
        synthetic = torch.clamp(synthetic, 0.0, 1.0)
    return synthetic.squeeze().cpu().numpy()

def simulate_mixed(account_id: str = "mixed_001", normal_ratio: float = 0.5):
    """
    Send a mix of normal and synthetic images from a single account.
    normal_ratio: proportion of normal images (0.5 = 50% normal, 50% synthetic)
    """
    # Load substitute model
    model = SubstituteCNN().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))

    image_paths = sorted(IMAGE_DIR.glob("*.png"))

    if not image_paths:
        print("No images found. Run simulation/export.py first.")
        return

    print(f"Simulating mixed user: {account_id}")
    print(f"Normal ratio: {normal_ratio*100}% normal, {(1-normal_ratio)*100}% synthetic")
    print(f"Total images: {len(image_paths)}")
    print("-" * 40)

    for i, path in enumerate(image_paths):
        arr = load_image(path)
        label = int(path.stem.split("_")[1])  # extract label from filename

        # Decide whether to send normal or synthetic
        if np.random.random() < normal_ratio:
            # Send normal image
            image_type = "normal"
            send_arr = arr
        else:
            # Generate and send synthetic image
            image_type = "synthetic"
            send_arr = generate_synthetic(model, arr, label)
            synth_path = SYNTH_DIR / f"synth_{path.name}"
            save_image(send_arr, synth_path)

        payload = {
            "account_id": account_id,
            "image": send_arr.tolist()
        }
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"  [{image_type}] {path.name} → pred: {data['pred']}")
        else:
            print(f"  {path.name} → failed: {response.status_code}")

if __name__ == "__main__":
    simulate_mixed()