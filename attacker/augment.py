import torch
import torch.nn as nn
import numpy as np

from config import DEVICE, LAMBDA


def jacobian_augment(model, images: np.ndarray, labels: list) -> np.ndarray:
    """
    Generate synthetic samples using FGSM on substitute model.

    Paper (page 4, Algorithm 1 row 13):
    x_new = x + λ · sign(∇x L(F'(x, ci)))

    This is FGSM applied to the substitute model F' using the victim's
    predicted label ci — probes the decision boundary of the victim.

    Args:
        model:   trained SubstituteCNN
        images:  np.ndarray (N, 28, 28) float32
        labels:  list of int (victim predicted labels)

    Returns:
        synthetic: np.ndarray (N, 28, 28)
    """
    model.eval()

    X = torch.tensor(images).unsqueeze(1).to(DEVICE)   # (N,1,28,28)
    X.requires_grad_(True)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    logits  = model(X)
    loss    = loss_fn(logits, y)

    loss.backward()

    # x_new = x + λ · sign(∇x L(F'(x, ci)))
    with torch.no_grad():
        synthetic = X + LAMBDA * X.grad.sign()
        synthetic = torch.clamp(synthetic, 0.0, 1.0)

    return synthetic.squeeze(1).cpu().numpy()
