import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from patrol.models.mlp import MLP


def load_samples(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    x = np.array([d["x_state"] for d in data], dtype=np.float32)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with trained MLP.")
    parser.add_argument("--model", default="./patrol/models/checkpoints/model.pt")
    parser.add_argument("--input", required=True, help="Pickle with x_state samples.")
    parser.add_argument("--out", default="./outputs/preds.npy")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    in_dim = ckpt["input_dim"]
    hidden = ckpt["hidden"]
    dropout = ckpt.get("dropout", 0.0)
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    threshold = float(ckpt.get("threshold", 0.1224))

    model = MLP(in_dim, hidden, dropout)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = load_samples(Path(args.input))
    if mean is not None and std is not None:
        x = (x - mean) / std
    x_t = torch.from_numpy(x)

    preds = []
    with torch.no_grad():
        for i in range(0, len(x_t), args.batch_size):
            xb = x_t[i : i + args.batch_size]
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
    preds = np.concatenate(preds)
    np.save(args.out, preds)
    labels = (preds >= threshold).astype(np.int64)
    label_out = str(Path(args.out).with_suffix("")) + "_labels.npy"
    np.save(label_out, labels)
    print(
        f"saved {len(preds)} probs to {args.out}, labels to {label_out}, "
        f"threshold={threshold:.4f}"
    )


if __name__ == "__main__":
    main()
