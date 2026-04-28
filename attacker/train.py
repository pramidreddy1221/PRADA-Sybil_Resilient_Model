import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import DEVICE, MNIST_PATH

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
    from torchvision import datasets, transforms

    mnist_test = datasets.MNIST(
        root=str(MNIST_PATH),
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )

    eval_images = []
    eval_labels = []
    for img, label in mnist_test:
        eval_images.append(img.numpy().squeeze())
        eval_labels.append(label)
        if len(eval_images) >= 1000:
            break
    eval_images = np.array(eval_images, dtype=np.float32)

    model.eval()
    X = torch.tensor(eval_images).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1).cpu().numpy()

    return float(np.mean(preds == np.array(eval_labels)))


def train_substitute_cvsearch(model, images: np.ndarray, labels: list):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    N_FOLDS = 5
    LR_LOG_MIN, LR_LOG_MAX = np.log10(1e-4), np.log10(1e-2)  # -4.0, -2.0
    EP_LOG_MIN, EP_LOG_MAX = np.log10(10),   np.log10(320)    #  1.0, ~2.505

    X_all = torch.tensor(images).unsqueeze(1)
    y_all = torch.tensor(labels, dtype=torch.long)
    n = len(X_all)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_indices = np.array_split(indices, N_FOLDS)

    def eval_cv(lr, epochs):
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

        return float(np.mean(fold_accs))

    def to_params(log_lr, log_ep):
        lr = 10 ** log_lr
        epochs = max(10, min(320, int(round(10 ** log_ep))))
        return lr, epochs

    observed_X = []
    observed_y = []

    def observe(log_lr, log_ep):
        lr, epochs = to_params(log_lr, log_ep)
        acc = eval_cv(lr, epochs)
        observed_X.append([log_lr, log_ep])
        observed_y.append(acc)
        return lr, epochs, acc

    # Step 1: 4 corner points
    print("  [BO] Step 1: evaluating 4 corner points...")
    corners = [
        (LR_LOG_MIN, EP_LOG_MIN),
        (LR_LOG_MIN, EP_LOG_MAX),
        (LR_LOG_MAX, EP_LOG_MIN),
        (LR_LOG_MAX, EP_LOG_MAX),
    ]
    for log_lr, log_ep in corners:
        lr, ep, acc = observe(log_lr, log_ep)
        print(f"    lr={lr:.0e}  epochs={ep:3d}  val_acc={acc*100:.2f}%")

    # Step 2: 11 random points uniformly inside log space
    print("  [BO] Step 2: evaluating 11 random points...")
    random_pts = rng.uniform(
        [LR_LOG_MIN, EP_LOG_MIN],
        [LR_LOG_MAX, EP_LOG_MAX],
        size=(11, 2)
    )
    for log_lr, log_ep in random_pts:
        lr, ep, acc = observe(log_lr, log_ep)
        print(f"    lr={lr:.0e}  epochs={ep:3d}  val_acc={acc*100:.2f}%")

    # Candidate grid: 40×25 = 1000 points in log space
    g_lr, g_ep = np.meshgrid(
        np.linspace(LR_LOG_MIN, LR_LOG_MAX, 40),
        np.linspace(EP_LOG_MIN, EP_LOG_MAX, 25)
    )
    candidates = np.column_stack([g_lr.ravel(), g_ep.ravel()])  # (1000, 2)

    # Step 3: iterations 16 to 30, GP-guided (UCB acquisition: mean + std)
    print("  [BO] Step 3: 15 GP-guided iterations...")
    kernel = RBF()
    for iteration in range(16, 31):
        X_obs = np.array(observed_X)
        y_obs = np.array(observed_y)

        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, alpha=1e-3,
            n_restarts_optimizer=2, random_state=42
        )
        gp.fit(X_obs, y_obs)

        mu, std = gp.predict(candidates, return_std=True)
        next_log_lr, next_log_ep = candidates[np.argmax(mu + std)]

        lr, ep, acc = observe(next_log_lr, next_log_ep)
        print(f"    iter {iteration:2d}: lr={lr:.0e}  epochs={ep:3d}  val_acc={acc*100:.2f}%")

    best_idx = int(np.argmax(observed_y))
    best_log_lr, best_log_ep = observed_X[best_idx]
    best_lr, best_epochs = to_params(best_log_lr, best_log_ep)
    best_acc = observed_y[best_idx]

    print(
        f"  [BO] Best: lr={best_lr:.0e}  epochs={best_epochs}"
        f"  val_acc={best_acc*100:.2f}%"
    )
    print(f"  [BO] Final training on all {n} samples...")

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
