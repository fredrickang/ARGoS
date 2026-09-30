import argparse
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


class TaskAttention(nn.Module):
    def __init__(self, in_dim: int, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.task_proj = nn.Linear(in_dim, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.task_proj(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class AttnModel(nn.Module):
    def __init__(
        self,
        global_dim: int,
        task_dim: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        blocks = [TaskAttention(task_dim, d_model, heads, dropout)]
        for _ in range(layers - 1):
            blocks.append(TaskAttention(d_model, d_model, heads, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, global_x: torch.Tensor, task_x: torch.Tensor) -> torch.Tensor:
        g = self.global_proj(global_x)
        x = task_x
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).mean(dim=1)
        z = torch.cat([g, x], dim=1)
        return self.head(z).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train attention model for y_miss prediction.")
    parser.add_argument("--train", default="./dataset/train.pkl")
    parser.add_argument("--val", default="./dataset/val.pkl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--out", default="./patrol/models/checkpoints/attn_model.pt")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr-tmax", type=int, default=None)
    parser.add_argument("--latency-samples", type=int, default=100)
    parser.add_argument("--latency-warmup", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)

    train_x, train_y = load_dataset(Path(args.train))
    val_x, val_y = load_dataset(Path(args.val))

    if train_x.shape[1] < 3:
        raise SystemExit("x_state must include window_size + task features.")

    num_tasks = (train_x.shape[1] - 1) // 2
    if 1 + 2 * num_tasks != train_x.shape[1]:
        raise SystemExit("x_state length must be 1 + 2 * num_tasks.")

    global_dim = 1
    task_dim = 2

    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    train_global = train_x[:, :1]
    train_tasks = train_x[:, 1:].reshape(-1, num_tasks, task_dim)
    val_global = val_x[:, :1]
    val_tasks = val_x[:, 1:].reshape(-1, num_tasks, task_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttnModel(
        global_dim=global_dim,
        task_dim=task_dim,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    t_max = args.lr_tmax or args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
    n_pos = float((train_y == 1).sum())
    n_neg = float((train_y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.from_numpy(train_global),
        torch.from_numpy(train_tasks),
        torch.from_numpy(train_y),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_global),
        torch.from_numpy(val_tasks),
        torch.from_numpy(val_y),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = 0.0

    # Inference latency measurement (single-sample).
    model.eval()
    total_samples = 0
    total_time = 0.0
    with torch.no_grad():
        for g, t, _ in val_loader:
            g = g.to(device)
            t = t.to(device)
            for i in range(g.size(0)):
                g1 = g[i : i + 1]
                t1 = t[i : i + 1]
                if total_samples < args.latency_warmup:
                    _ = model(g1, t1)
                    total_samples += 1
                    continue
                start = time.perf_counter()
                _ = model(g1, t1)
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
        for g, t, yb in train_loader:
            g = g.to(device)
            t = t.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(g, t)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * g.size(0)

        model.eval()
        correct = 0
        total = 0
        probs = []
        labels = []
        with torch.no_grad():
            for g, t, yb in val_loader:
                g = g.to(device)
                t = t.to(device)
                yb = yb.to(device)
                logits = model(g, t)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += int((preds == yb).sum().item())
                total += yb.numel()
                probs.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(yb.cpu().numpy())

        val_acc = correct / max(1, total)
        val_auc = compute_auc(
            np.concatenate(labels), np.concatenate(probs)
        )
        avg_loss = total_loss / max(1, len(train_ds))
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f} "
            f"val_auc={val_auc:.4f} lr={lr_now:.6f}"
        )

        score = val_auc if not np.isnan(val_auc) else val_acc
        if score > best_val:
            best_val = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_tasks": num_tasks,
                    "d_model": args.d_model,
                    "heads": args.heads,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "mean": mean,
                    "std": std,
                },
                out_path,
            )
        scheduler.step()

    print(f"best_val_score={best_val:.4f} saved={out_path}")


if __name__ == "__main__":
    main()
