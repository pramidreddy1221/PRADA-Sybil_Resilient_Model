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


def jacobian_augment_ifgsm(model, images: np.ndarray, labels: list, n_steps: int = 10) -> np.ndarray:
    """
    Iterative FGSM augmentation (I-FGSM / Basic Iterative Method).

    Spreads the total perturbation budget LAMBDA across n_steps steps,
    clamping to [0,1] after each step. Produces stronger perturbations
    than single-step FGSM by following the gradient repeatedly.

    step_size = LAMBDA / n_steps
    x_{t+1} = clip(x_t + step_size · sign(∇x L(F'(x_t, ci))), 0, 1)
    """
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


def jacobian_augment_mifgsm(model, images: np.ndarray, labels: list, n_steps: int = 10, mu: float = 1.0) -> np.ndarray:
    """
    Momentum Iterative FGSM augmentation (MI-FGSM, Dong et al. 2018).

    Accumulates a momentum buffer g of L1-normalised gradient directions
    across steps, then updates with sign(g). Momentum stabilises the
    gradient direction and escapes flat regions better than I-FGSM.

    g_{t+1} = μ · g_t + ∇x L / ||∇x L||₁
    x_{t+1} = clip(x_t + step_size · sign(g_{t+1}), 0, 1)
    """
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
