# Jacobian-based dataset augmentation from Papernot et al. — three variants implemented.
import torch
import torch.nn as nn
import numpy as np

from config import DEVICE, LAMBDA


# FGSM: single gradient step in the sign direction
def jacobian_augment(model, images: np.ndarray, labels: list) -> np.ndarray:
    model.eval()

    X = torch.tensor(images).unsqueeze(1).to(DEVICE)
    X.requires_grad_(True)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    logits = model(X)
    loss = loss_fn(logits, y)

    loss.backward()

    with torch.no_grad():
        synthetic = X + LAMBDA * X.grad.sign()
        synthetic = torch.clamp(synthetic, 0.0, 1.0)

    return synthetic.squeeze(1).cpu().numpy()


# I-FGSM: iterative gradient steps, total perturbation bounded by LAMBDA
def jacobian_augment_ifgsm(model, images: np.ndarray, labels: list, n_steps: int = 10) -> np.ndarray:
    model.eval()
    step_size = LAMBDA / n_steps

    X = torch.tensor(images).unsqueeze(1).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    X_adv = X.clone().detach()

    for _ in range(n_steps):
        X_adv = X_adv.clone().detach().requires_grad_(True)
        logits = model(X_adv)
        loss = loss_fn(logits, y)
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + step_size * X_adv.grad.sign()
            X_adv = torch.clamp(X_adv, 0.0, 1.0)

    return X_adv.squeeze(1).cpu().numpy()


# MI-FGSM: momentum iterative — accumulates gradient direction across steps to stabilise updates
def jacobian_augment_mifgsm(model, images: np.ndarray, labels: list, n_steps: int = 10, mu: float = 1.0) -> np.ndarray:
    model.eval()
    step_size = LAMBDA / n_steps

    X = torch.tensor(images).unsqueeze(1).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    X_adv = X.clone().detach()
    momentum = torch.zeros_like(X_adv)

    for _ in range(n_steps):
        X_adv = X_adv.clone().detach().requires_grad_(True)
        logits = model(X_adv)
        loss = loss_fn(logits, y)
        loss.backward()

        with torch.no_grad():
            grad = X_adv.grad
            l1_norm = grad.abs().sum(dim=(1, 2, 3), keepdim=True)
            grad_norm = grad / (l1_norm + 1e-8)
            momentum = mu * momentum + grad_norm
            X_adv = X_adv + step_size * momentum.sign()
            X_adv = torch.clamp(X_adv, 0.0, 1.0)

    return X_adv.squeeze(1).cpu().numpy()
