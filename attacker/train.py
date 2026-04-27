import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import DEVICE

def train_substitute(model, images: np.ndarray, labels: list, epochs: int = 10):
    model.train()

    X = torch.tensor(images).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

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
    model.eval()

    X = torch.tensor(images).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1).cpu().numpy()

    agreement = float(np.mean(preds == np.array(victim_labels)))
    return agreement


def train_substitute_cvsearch(model, images: np.ndarray, labels: list):
    LR_GRID = [1e-4, 1e-3, 1e-2]
    EPOCHS_GRID = [10, 40, 160]
    N_FOLDS = 5

    X_all = torch.tensor(images).unsqueeze(1)
    y_all = torch.tensor(labels, dtype=torch.long)
    n = len(X_all)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_indices = np.array_split(indices, N_FOLDS)

    best_acc = -1.0
    best_lr = LR_GRID[-1]
    best_epochs = EPOCHS_GRID[0]

    print(f"  [CV] 5-fold grid search: {len(LR_GRID)}×{len(EPOCHS_GRID)} combos...")

    for lr in LR_GRID:
        for epochs in EPOCHS_GRID:
            fold_accs = []

            for fold in range(N_FOLDS):
                val_idx = fold_indices[fold]
                train_idx = np.concatenate(
                    [fold_indices[k] for k in range(N_FOLDS) if k != fold]
                )

                X_tr,  y_tr = X_all[train_idx], y_all[train_idx]
                X_val, y_val = X_all[val_idx],   y_all[val_idx]

                cv_model = type(model)().to(DEVICE)
                optimizer = optim.SGD(cv_model.parameters(), lr=lr, momentum=0.9)
                loss_fn = nn.CrossEntropyLoss()
                loader = DataLoader(
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
                best_acc = mean_acc
                best_lr = lr
                best_epochs = epochs

    print(
        f"  [CV] Best: lr={best_lr:.0e}  epochs={best_epochs}"
        f"  val_acc={best_acc*100:.2f}%"
    )
    print(f"  [CV] Final training on all {n} samples...")

    optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_all, y_all), batch_size=32, shuffle=True)

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
    model.train()

    X = torch.tensor(images).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

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
