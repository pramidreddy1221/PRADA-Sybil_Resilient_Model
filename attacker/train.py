import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import DEVICE

def train_substitute(model, images: np.ndarray, labels: list, epochs: int = 10):
    """
    Train substitute model on current labelled dataset.

    Args:
        model:   SubstituteCNN instance
        images:  np.ndarray (N, 28, 28) float32
        labels:  list of int (victim predicted labels)
        epochs:  number of training epochs (paper uses 10)

    Returns:
        model: trained substitute model
    """
    model.train()

    X = torch.tensor(images).unsqueeze(1)           # (N, 1, 28, 28)
    y = torch.tensor(labels, dtype=torch.long)      # (N,)

    dataset  = TensorDataset(X, y)
    loader   = DataLoader(dataset, batch_size=32, shuffle=True)

    # PAPERNOT fixed training strategy (paper page 6)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn   = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"    epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    return model


def evaluate_substitute(model, images: np.ndarray, victim_labels: list) -> float:
    """
    Measure agreement between substitute and victim predictions.
    Agreement = how often substitute predicts same class as victim.

    Args:
        model: trained SubstituteCNN
        images: np.ndarray (N, 28, 28)
        victim_labels: list of int (victim predicted labels)

    Returns:
        agreement: float between 0 and 1
    """
    model.eval()

    X = torch.tensor(images).unsqueeze(1).to(DEVICE)   # (N,1,28,28)

    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1).cpu().numpy()

    agreement = float(np.mean(preds == np.array(victim_labels)))
    return agreement