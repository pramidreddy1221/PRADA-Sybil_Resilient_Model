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


def train_substitute_cvsearch(model, images: np.ndarray, labels: list):
    """
    Train substitute model with 5-fold CV over a 3×3 hyperparameter grid.

    Grid:
        lr     ∈ {1e-4, 1e-3, 1e-2}
        epochs ∈ {10, 40, 160}

    For each combo, trains on 4 folds and evaluates on the held-out fold,
    repeating for all 5 folds.  Picks the combo with the highest mean
    validation accuracy, then retrains the passed-in model on all data
    using those hyperparameters.

    Args:
        model:   SubstituteCNN instance (trained on all data after CV)
        images:  np.ndarray (N, 28, 28) float32
        labels:  list of int (victim predicted labels)

    Returns:
        (model, best_lr, best_epochs): model trained on all data with the
        winning hyperparameters, plus the selected lr and epochs values so
        the caller can reuse them for subsequent training rounds.
    """
    LR_GRID     = [1e-4, 1e-3, 1e-2]
    EPOCHS_GRID = [10, 40, 160]
    N_FOLDS     = 5

    X_all = torch.tensor(images).unsqueeze(1)       # (N, 1, 28, 28)
    y_all = torch.tensor(labels, dtype=torch.long)  # (N,)
    n     = len(X_all)

    rng          = np.random.default_rng(42)
    indices      = rng.permutation(n)
    fold_indices = np.array_split(indices, N_FOLDS)

    best_acc    = -1.0
    best_lr     = LR_GRID[-1]
    best_epochs = EPOCHS_GRID[0]

    print(f"  [CV] 5-fold grid search: {len(LR_GRID)}×{len(EPOCHS_GRID)} combos...")

    for lr in LR_GRID:
        for epochs in EPOCHS_GRID:
            fold_accs = []

            for fold in range(N_FOLDS):
                val_idx   = fold_indices[fold]
                train_idx = np.concatenate(
                    [fold_indices[k] for k in range(N_FOLDS) if k != fold]
                )

                X_tr,  y_tr  = X_all[train_idx], y_all[train_idx]
                X_val, y_val = X_all[val_idx],   y_all[val_idx]

                # Fresh model per fold — same class, random init
                cv_model  = type(model)().to(DEVICE)
                optimizer = optim.SGD(cv_model.parameters(), lr=lr, momentum=0.9)
                loss_fn   = nn.CrossEntropyLoss()
                loader    = DataLoader(
                    TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True
                )

                cv_model.train()
                for _ in range(epochs):
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        optimizer.zero_grad()
                        loss_fn(cv_model(xb), yb).backward()
                        optimizer.step()

                cv_model.eval()
                with torch.no_grad():
                    preds = torch.argmax(
                        cv_model(X_val.to(DEVICE)), dim=1
                    ).cpu()
                fold_accs.append(float((preds == y_val).float().mean()))

            mean_acc = float(np.mean(fold_accs))
            print(f"    lr={lr:.0e}  epochs={epochs:3d}  val_acc={mean_acc*100:.2f}%")

            if mean_acc > best_acc:
                best_acc    = mean_acc
                best_lr     = lr
                best_epochs = epochs

    print(
        f"  [CV] Best: lr={best_lr:.0e}  epochs={best_epochs}"
        f"  val_acc={best_acc*100:.2f}%"
    )
    print(f"  [CV] Final training on all {n} samples...")

    # Retrain the passed-in model on all data with the winning hyperparameters
    optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=0.9)
    loss_fn   = nn.CrossEntropyLoss()
    loader    = DataLoader(TensorDataset(X_all, y_all), batch_size=32, shuffle=True)

    model.train()
    for epoch in range(best_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"    epoch {epoch+1}/{best_epochs}  loss={avg:.4f}")

    return model, best_lr, best_epochs


def train_substitute_fixed(
    model, images: np.ndarray, labels: list, lr: float, epochs: int
):
    """
    Train substitute model with explicit lr and epochs — no hardcoded defaults.

    Identical to train_substitute() except the learning rate is a required
    parameter rather than being fixed at 0.01.  Used by run_attack_cvsearch()
    to apply the hyperparameters found by train_substitute_cvsearch().

    Args:
        model:   SubstituteCNN instance
        images:  np.ndarray (N, 28, 28) float32
        labels:  list of int (victim predicted labels)
        lr:      learning rate for SGD
        epochs:  number of training epochs

    Returns:
        model: trained substitute model
    """
    model.train()

    X = torch.tensor(images).unsqueeze(1)           # (N, 1, 28, 28)
    y = torch.tensor(labels, dtype=torch.long)      # (N,)

    dataset   = TensorDataset(X, y)
    loader    = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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