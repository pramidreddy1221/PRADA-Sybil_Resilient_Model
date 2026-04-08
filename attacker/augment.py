# attacker/augment.py
# Jacobian-based Dataset Augmentation (JbDA)
# Implements FGSM on substitute model to generate synthetic samples
# Paper reference: Section II-F and Section III-C (page 4-5)

import torch
import torch.nn as nn
import numpy as np

from config import DEVICE, LAMBDA


def jacobian_augment(model, images: np.ndarray, labels: list) -> np.ndarray:
    """
    Generate synthetic samples using FGSM on substitute model.

    Paper (page 4, Algorithm 1 row 13):
    x_new = x + λ · sign(∇x L(F'(x, ci)))

    This is identical to FGSM applied to the substitute model F'
    using the victim's predicted label ci.

    Args:
        model:   trained SubstituteCNN
        images:  np.ndarray (N, 28, 28) float32
        labels:  list of int (victim predicted labels)

    Returns:
        synthetic: np.ndarray (N, 28, 28) — augmented samples
    """
    model.eval()

    # Convert to tensors
    X = torch.tensor(images).unsqueeze(1).to(DEVICE)   # (N,1,28,28)
    X.requires_grad_(True)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

    # Forward pass through substitute model
    loss_fn = nn.CrossEntropyLoss()
    logits  = model(X)
    loss    = loss_fn(logits, y)

    # Compute gradients
    loss.backward()

    # FGSM: step in sign of gradient (paper equation page 4)
    # x_new = x + λ · sign(∇x L(F'(x, ci)))
    with torch.no_grad():
        synthetic = X + LAMBDA * X.grad.sign()
        synthetic = torch.clamp(synthetic, 0.0, 1.0)   # keep pixels in [0,1]

    # Return as numpy (N, 28, 28)
    return synthetic.squeeze(1).cpu().numpy()

# ```

# ---

# **What this does:**
# - Implements exactly equation from paper ✅
# - `λ = 25.5/255` as paper specifies ✅
# - Clamps output to `[0,1]` to keep valid images ✅
# - Uses victim's predicted labels not ground truth ✅

# ---

# **Easy to explain to professor:**
# ```
# "I take each image, compute the gradient 
# of the loss with respect to the input pixels
# using the substitute model, then step in the
# sign of that gradient — this is FGSM.
# This generates synthetic samples that probe
# the decision boundary of the victim model."