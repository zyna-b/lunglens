"""Train one LungLens model on TBX11K and test it.

    python train.py --data /path/to/tbx11k-simplified --model densenet121_cxr --out outputs

Writes outputs/<model>/: best.pt, history.csv, metrics.json (val + test), test_predictions.csv,
test_cams_tb.npy. All images are cached on the GPU as uint8 and augmented there, so a run on a
single Colab GPU takes minutes.
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from lunglens import augment
from lunglens.data import load_all, make_splits, read_index, read_splits, write_splits
from lunglens.metrics import bootstrap_ci, pointing_game, screening_threshold, sensitivity_threshold, summarise
from lunglens.models import CLASSES, MODELS, build_model

DEFAULTS = {  # epochs, learning rate
    "cnn_scratch": (40, 2e-3),
    "densenet121_imagenet": (20, 3e-4),
    "densenet121_cxr": (20, 3e-4),
}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with data.csv and images/")
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="use only N images per split (smoke test)")
    ap.add_argument("--skip-done", action="store_true", help="skip if <out>/<model>/metrics.json exists")
    return ap.parse_args()


@torch.no_grad()
def predict(model, X, idx, size, batch=64, cams=False):
    model.eval()
    probs, maps = [], []
    for s in range(0, len(idx), batch):
        x = augment.eval_batch(X[idx[s:s + batch]], size).contiguous(memory_format=torch.channels_last)
        with autocast():
            logits, cam = model.forward_with_cam(x)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        if cams:
            maps.append(cam[:, CLASSES.index("tb")].float().cpu())
    return torch.cat(probs).numpy(), (torch.cat(maps).numpy() if cams else None)


def autocast():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return torch.autocast("cpu", enabled=False)


def final_eval(model, X, y, idx, used_rows, out, size, meta):
    """Best epoch -> screening threshold from validation -> test metrics, predictions, heatmaps."""
    pv, _ = predict(model, X, idx["val"], size)
    yv = y[idx["val"]].cpu().numpy()
    thr = screening_threshold(pv[:, 2], yv == 2)
    pt, cams = predict(model, X, idx["test"], size, cams=True)
    yt = y[idx["test"]].cpu().numpy()
    test_rows = [used_rows[i] for i in idx["test"].cpu().tolist()]
    tb = [k for k, r in enumerate(test_rows) if r["label"] == 2]
    result = meta | {
        "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "screening_rule": "TB probability >= threshold, threshold = 95% specificity on the validation set's non-TB X-rays",
        "val": summarise(pv, yv, thr),
        "test": summarise(pt, yt, thr) | bootstrap_ci(pt, yt),
        "test_at_val_90_sensitivity": summarise(pt, yt, sensitivity_threshold(pv[:, 2], yv == 2))["tb_vs_rest"],
        "pointing_game": pointing_game(cams[tb], [test_rows[k]["boxes"] for k in tb]),
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.save(os.path.join(out, "test_cams_tb.npy"), cams.astype(np.float16))
    np.save(os.path.join(out, "val_probs.npy"), pv.astype(np.float32))
    with open(os.path.join(out, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "label", "pred", *[f"p_{c}" for c in CLASSES], "screen_positive"])
        for r, p in zip(test_rows, pt):
            w.writerow([r["fname"], CLASSES[r["label"]], CLASSES[int(p.argmax())], *[f"{v:.5f}" for v in p], int(p[2] >= thr)])
    t = result["test"]
    print(f"\nTEST {meta['model']}: acc {t['accuracy']:.4f}  balanced {t['balanced_accuracy']:.4f}  TB AUC {t['tb_vs_rest']['auc']:.4f}  "
          f"sens {t['tb_vs_rest']['sensitivity']:.3f} spec {t['tb_vs_rest']['specificity']:.3f} @ {thr:.3f}  "
          f"pointing {result['pointing_game']['hit_rate']:.3f} (random {result['pointing_game']['random_hit_rate']:.3f})", flush=True)
    return result


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    epochs, lr = DEFAULTS[a.model]
    epochs, lr = a.epochs or epochs, a.lr or lr
    out = os.path.join(a.out, a.model)
    if a.skip_done and os.path.exists(os.path.join(out, "metrics.json")):
        print(f"{a.model}: already trained, skipping")
        return
    os.makedirs(out, exist_ok=True)

    if os.path.exists(a.splits):
        rows = read_splits(a.splits)
    else:
        rows = make_splits(read_index(a.data))
        write_splits(rows, a.splits)
    split = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("train", "val", "test")}
    if a.limit:
        split = {s: v[:: max(1, len(v) // a.limit)][: a.limit] for s, v in split.items()}
    used = sorted(set(sum(split.values(), [])))
    pos = {i: k for k, i in enumerate(used)}
    print({s: len(v) for s, v in split.items()}, "| device:", dev, torch.cuda.get_device_name() if dev == "cuda" else "")

    t0 = time.time()
    X = torch.from_numpy(load_all([os.path.join(a.data, "images", rows[i]["fname"]) for i in used], a.size)).to(dev)
    y = torch.tensor([rows[i]["label"] for i in used], device=dev)
    idx = {s: torch.tensor([pos[i] for i in v], device=dev) for s, v in split.items()}
    print(f"cached {len(used)} images at {a.size}px in {time.time() - t0:.0f}s")

    model = build_model(a.model, pretrained=True).to(dev).to(memory_format=torch.channels_last)
    counts = torch.bincount(y[idx["train"]], minlength=len(CLASSES)).float()
    weights = counts.sum() / (len(CLASSES) * counts)  # rare TB class counts ~5x more
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(idx["train"]) / a.batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda" and not torch.cuda.is_bf16_supported())

    best, history = -1.0, []
    for ep in range(1, epochs + 1):
        model.train()
        t1, total, seen = time.time(), 0.0, 0
        perm = idx["train"][torch.randperm(len(idx["train"]), device=dev)]
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            if len(b) < 2:
                continue
            x = augment.train_batch(X[b], a.size).contiguous(memory_format=torch.channels_last)
            with autocast():
                loss = F.cross_entropy(model(x).float(), y[b], weight=weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, seen = total + loss.item() * len(b), seen + len(b)
        pv, _ = predict(model, X, idx["val"], a.size)
        yv = y[idx["val"]].cpu().numpy()
        m = summarise(pv, yv, screening_threshold(pv[:, 2], yv == 2))
        history.append({"epoch": ep, "train_loss": total / max(seen, 1), "val_accuracy": m["accuracy"], "val_balanced_accuracy": m["balanced_accuracy"],
                        "val_macro_auc": m["macro_auc"], "val_tb_auc": m["tb_vs_rest"]["auc"], "seconds": time.time() - t1})
        flag = ""
        if m["balanced_accuracy"] > best:
            best, flag = m["balanced_accuracy"], "  *best"
            torch.save({"model": a.model, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "epoch": ep,
                        "classes": CLASSES, "input": {"size": a.size, "channels": 1, "range": [0, 1]}}, os.path.join(out, "best.pt"))
        print(f"ep {ep:2d}/{epochs}  loss {history[-1]['train_loss']:.4f}  val acc {m['accuracy']:.4f}  bal {m['balanced_accuracy']:.4f}  "
              f"tbAUC {m['tb_vs_rest']['auc']:.4f}  {history[-1]['seconds']:.0f}s{flag}", flush=True)

    with open(os.path.join(out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    ck = torch.load(os.path.join(out, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    final_eval(model, X, y, idx, [rows[i] for i in used], out, a.size, {
        "model": a.model, "best_epoch": ck["epoch"], "epochs": epochs, "lr": lr, "batch": a.batch, "image_size": a.size,
        "train_minutes": round(sum(h["seconds"] for h in history) / 60, 1)})

if __name__ == "__main__":
    main()
