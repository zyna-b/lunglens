"""Re-score already-trained models with the current metrics and screening rule (no retraining).

    python rescore.py --data /path/to/tbx11k --out outputs
"""

import argparse
import json
import os

import torch

from lunglens.data import load_all, read_splits
from lunglens.models import MODELS, build_model
from train import final_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--size", type=int, default=384)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_splits(a.splits)
    used = [r for r in rows if r["split"] in ("val", "test")]
    X = torch.from_numpy(load_all([os.path.join(a.data, "images", r["fname"]) for r in used], a.size)).to(dev)
    y = torch.tensor([r["label"] for r in used], device=dev)
    idx = {s: torch.tensor([k for k, r in enumerate(used) if r["split"] == s], device=dev) for s in ("val", "test")}
    for m in MODELS:
        out = os.path.join(a.out, m)
        if not os.path.exists(os.path.join(out, "best.pt")):
            continue
        old = json.load(open(os.path.join(out, "metrics.json"))) if os.path.exists(os.path.join(out, "metrics.json")) else {}
        ck = torch.load(os.path.join(out, "best.pt"), map_location=dev, weights_only=False)
        model = build_model(m).to(dev).to(memory_format=torch.channels_last)
        model.load_state_dict(ck["state_dict"])
        meta = {k: old[k] for k in ("model", "best_epoch", "epochs", "lr", "batch", "image_size", "train_minutes") if k in old} | {"model": m}
        final_eval(model, X, y, idx, used, out, a.size, meta)


if __name__ == "__main__":
    main()
