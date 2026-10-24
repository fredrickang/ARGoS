import argparse
import copy
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def load_dataset(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    x = np.array([d["x_state"] for d in data], dtype=np.float32)
    y = np.array([d["y_miss"] for d in data], dtype=np.float32)
    return x, y


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_true = y_true[order]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank = np.arange(1, len(y_true) + 1, dtype=np.float64)
    pos_rank_sum = np.sum(rank[y_true == 1])
    auc = (pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)
    return float(auc)


def fit_temperature(model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    """Temperature scaling: fit a single scalar T minimizing val BCE so sigmoid(logit/T)
    is a calibrated probability (important because the learned risk r = miss_prob is
    compared against the acceptance threshold lambda)."""
    model.eval()
    with torch.no_grad():
        logits = model(x_val)
    T = nn.Parameter(torch.ones(1, device=logits.device))
    bce = nn.BCEWithLogitsLoss()
    opt = optim.LBFGS([T], lr=0.05, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = bce(logits / T.clamp(min=1e-2), y_val)
        loss.backward()
        return loss

    opt.step(closure)
    # Clamp to soften-only [1, 10]. On (near-)separable data BCE is minimized as T->0,
    # which SHARPENS probabilities toward 0/1 — the opposite of what we want: RCRL needs
    # smooth miss-probs so lambda can modulate the accept boundary and the fallback ladder
    # stays graceful. Temperature scaling should only *soften* an overconfident model.
    return float(min(10.0, max(1.0, T.detach().item())))


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(y_prob.astype(np.float64))
    candidates = np.concatenate(([0.0], candidates, [1.0]))

    best_t = 0.5
    best_acc = -1.0
    best_dist = float("inf")
    for t in candidates:
        y_pred = (y_prob >= t).astype(np.float32)
        acc = float((y_pred == y_true).mean())
        dist = abs(float(t) - 0.5)
        if acc > best_acc or (acc == best_acc and dist < best_dist):
            best_acc = acc
            best_t = float(t)
            best_dist = dist
    return best_t, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP for y_miss prediction.")
    parser.add_argument("--train", default="./dataset/train.pkl")
    parser.add_argument("--val", default="./dataset/val.pkl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--hidden",
        default="64,32",
        help="Comma-separated hidden sizes (v2 features are informative -> small net).",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--out", default="./patrol/models/checkpoints/model.pt")
    parser.add_argument("--feature-set", default="v2",
                        help="Feature-set tag stored in the checkpoint (informational).")
    parser.add_argument("--mono-idx", default="4,5",
                        help="Comma-separated feature indices that miss-prob must be "
                             "non-decreasing in (default: est_util, est_util_tail). Empty to disable.")
    parser.add_argument("--mono-weight", type=float, default=1.0,
                        help="Weight of the monotonicity penalty.")
    parser.add_argument("--mono-delta", type=float, default=0.5,
                        help="Perturbation (in standardized units) for the monotonicity check.")
    parser.add_argument(
        "--latency-samples",
        type=int,
        default=100,
        help="Number of single-sample inferences to average for latency.",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=20,
        help="Number of single-sample warmup inferences before timing.",
    )
    parser.add_argument(
        "--lr-tmax",
        type=int,
        default=None,
        help="CosineAnnealingLR T_max (defaults to epochs).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_x, train_y = load_dataset(Path(args.train))
    val_x, val_y = load_dataset(Path(args.val))

    in_dim = train_x.shape[1]
    hidden = [int(h.strip()) for h in args.hidden.split(",") if h.strip()]
    mono_idx = [int(i) for i in args.mono_idx.split(",") if i.strip() != ""]
    mono_idx = [i for i in mono_idx if 0 <= i < in_dim]

    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim, hidden, args.dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    t_max = args.lr_tmax or args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
    n_pos = float((train_y == 1).sum())
    n_neg = float((train_y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y)
    )
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    best_val = -1.0
    best_threshold = 0.5
    best_state = copy.deepcopy(model.state_dict())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Inference latency measurement (single-sample).
    model.eval()
    total_samples = 0
    total_time = 0.0
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            for i in range(xb.size(0)):
                x1 = xb[i : i + 1]
                if total_samples < args.latency_warmup:
                    _ = model(x1)
                    total_samples += 1
                    continue
                start = time.perf_counter()
                _ = model(x1)
                end = time.perf_counter()
                total_time += end - start
                total_samples += 1
                if total_samples >= args.latency_warmup + args.latency_samples:
                    break
            if total_samples >= args.latency_warmup + args.latency_samples:
                break
    timed_samples = max(0, total_samples - args.latency_warmup)
    if timed_samples > 0:
        per_sample_ms = (total_time / timed_samples) * 1000.0
        print(
            f"inference_latency: {per_sample_ms:.4f} ms/sample "
            f"(warmup={args.latency_warmup}, timed={timed_samples})"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if mono_idx and args.mono_weight > 0:
                # Penalize decreases in miss-prob when a monotone feature (e.g. est_util)
                # increases -> guarantees a monotone fallback ladder (no saturated cliff).
                xp = xb.clone()
                xp[:, mono_idx] = xp[:, mono_idx] + args.mono_delta
                fp = model(xp)
                loss = loss + args.mono_weight * torch.relu(logits - fp).mean()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)

        model.eval()
        probs = []
        labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(yb.cpu().numpy())

        val_probs = np.concatenate(probs)
        val_labels = np.concatenate(labels)
        threshold, val_acc = find_best_threshold(val_labels, val_probs)
        val_auc = compute_auc(val_labels, val_probs)
        avg_loss = total_loss / max(1, len(train_ds))
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f} "
            f"val_auc={val_auc:.4f} val_thr={threshold:.4f} lr={lr_now:.6f}"
        )

        # Select by AUC (ranking quality) — robust to class imbalance; fall back to acc.
        score = val_auc if val_auc == val_auc else val_acc  # nan-safe
        if score > best_val:
            best_val = score
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()

    # Restore best model, calibrate temperature on val, then save once.
    model.load_state_dict(best_state)
    x_val_t = torch.from_numpy(val_x).to(device)
    y_val_t = torch.from_numpy(val_y).to(device)
    temperature = fit_temperature(model, x_val_t, y_val_t)
    torch.save(
        {
            "model_state": best_state,
            "input_dim": in_dim,
            "hidden": hidden,
            "dropout": args.dropout,
            "mean": mean,
            "std": std,
            "threshold": best_threshold,
            "temperature": temperature,
            "feature_set": args.feature_set,
        },
        out_path,
    )
    print(
        f"best_val_auc={best_val:.4f} best_threshold={best_threshold:.4f} "
        f"temperature={temperature:.3f} feature_set={args.feature_set} saved={out_path}"
    )


if __name__ == "__main__":
    main()
