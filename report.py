"""After training: compare the three models, pick the deployed one, export it, draw the figures.

    python report.py --runs outputs --data /path/to/tbx11k

The deployed model is the one with the best *validation* balanced accuracy (the test set is never
used to choose). Writes:
  weights/<model>.pt              PyTorch checkpoints (state_dict + metadata), all three models
  weights/lunglens.onnx           deployed model for ONNX Runtime / the website (probs + heatmaps)
  weights/lunglens_torchscript.pt deployed model as TorchScript (torch.jit.load, no repo code needed)
  model_card.json                 classes, input, threshold, metrics, file hashes
  results/summary.json, results/results.md, results/figures/*.png
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from lunglens.data import load_gray, read_splits  # noqa: E402
from lunglens.models import CLASSES, MODELS, ExportWrapper, load_checkpoint  # noqa: E402

NICE = {"cnn_scratch": "CNN from scratch", "densenet121_imagenet": "DenseNet-121 (ImageNet start)", "densenet121_cxr": "DenseNet-121 (chest X-ray start)"}
COLORS = {"cnn_scratch": "#9a9a9a", "densenet121_imagenet": "#4C78A8", "densenet121_cxr": "#E45756"}
SIZE = 384


def clean(o):
    """NaN/inf -> None so the JSON files load in browsers (JSON has no NaN)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(model, path):
    x = torch.rand(1, 1, SIZE, SIZE)
    w = ExportWrapper(model).eval()  # a fresh wrapper is in train mode, and export would restore that onto the model
    torch.onnx.export(w, x, path, input_names=["xray"], output_names=["probs", "cam"],
                      dynamic_axes={"xray": {0: "batch"}, "probs": {0: "batch"}, "cam": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = w(x)
    got = sess.run(None, {"xray": x.numpy()})
    err = max(float(np.abs(ref[0].numpy() - got[0]).max()), float(np.abs(ref[1].numpy() - got[1]).max()))
    t = []
    for _ in range(12):
        t0 = time.perf_counter()
        sess.run(None, {"xray": x.numpy()})
        t.append(time.perf_counter() - t0)
    return err, float(np.median(t[2:]) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs")
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="data/splits.csv")
    a = ap.parse_args()
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    runs = {m: json.load(open(os.path.join(a.runs, m, "metrics.json"))) for m in MODELS if os.path.exists(os.path.join(a.runs, m, "metrics.json"))}
    best = max(runs, key=lambda m: runs[m]["val"]["balanced_accuracy"])
    # the other DenseNet goes on the website too, so visitors can compare the two starting points' heatmaps
    alt = next((m for m in ("densenet121_cxr", "densenet121_imagenet") if m in runs and m != best), None)
    print("models:", list(runs), "| deployed:", best)

    tmp = "weights/_tmp.onnx"
    for m in runs:
        shutil.copy(os.path.join(a.runs, m, "best.pt"), f"weights/{m}.pt")
        model, _ = load_checkpoint(f"weights/{m}.pt")
        err, ms = export_onnx(model, tmp)
        runs[m]["onnx_cpu_ms"], runs[m]["onnx_max_abs_err"] = round(ms, 1), err
        runs[m]["onnx_mb"] = round(os.path.getsize(tmp) / 1e6, 1)
        if m == best:
            os.replace(tmp, "weights/lunglens.onnx")
            with torch.no_grad():
                torch.jit.trace(ExportWrapper(model).eval(), torch.rand(1, 1, SIZE, SIZE)).save("weights/lunglens_torchscript.pt")
        elif m == alt:
            os.replace(tmp, "weights/lunglens_alt.onnx")
        print(f"{m}: onnx {runs[m]['onnx_mb']} MB, {ms:.0f} ms/image on this CPU, max diff vs torch {err:.2e}")
    if os.path.exists(tmp):
        os.remove(tmp)

    hist = {m: list(csv.DictReader(open(os.path.join(a.runs, m, "history.csv")))) for m in runs}
    for m in runs:  # per-image test predictions of every model, for the report's error analysis
        shutil.copy(os.path.join(a.runs, m, "test_predictions.csv"), f"results/test_predictions_{m}.csv")
    preds = {m: list(csv.DictReader(open(os.path.join(a.runs, m, "test_predictions.csv")))) for m in runs}

    # 1. training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in runs:
        e = [int(h["epoch"]) for h in hist[m]]
        ax[0].plot(e, [float(h["train_loss"]) for h in hist[m]], color=COLORS[m], label=NICE[m])
        ax[1].plot(e, [float(h["val_balanced_accuracy"]) for h in hist[m]], color=COLORS[m], label=NICE[m])
    ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].set(title="Validation balanced accuracy", xlabel="epoch")
    ax[1].legend(frameon=False)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/figures/training_curves.png", dpi=150)

    # 2. ROC, TB vs everything else, test set
    fig, ax = plt.subplots(figsize=(5, 5))
    for m in runs:
        y = np.array([r["label"] == "tb" for r in preds[m]])
        p = np.array([float(r["p_tb"]) for r in preds[m]])
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS[m], label=f"{NICE[m]}  AUC {runs[m]['test']['tb_vs_rest']['auc']:.4f}")
        s = runs[m]["test"]["tb_vs_rest"]
        ax.plot(1 - s["specificity"], s["sensitivity"], "o", color=COLORS[m])
    ax.plot([0, 1], [0, 1], ":", color="#bbb")
    ax.set(xlabel="1 − specificity", ylabel="sensitivity", title="TB vs not-TB, 1,800 test X-rays")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/figures/roc_tb.png", dpi=150)

    # 3. confusion matrix of the deployed model
    cm = np.array(runs[best]["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    names = ["healthy", "sick, not TB", "TB"]
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / cm[i].sum():.0%}", ha="center", va="center", color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names, xlabel="predicted", ylabel="true", title=NICE[best])
    fig.tight_layout()
    fig.savefig("results/figures/confusion_matrix.png", dpi=150)

    # 4. heatmaps vs doctors' boxes: the same TB test X-rays, one row per DenseNet
    rows = {r["fname"]: r for r in read_splits(a.splits)}
    shown = [m for m in (best, alt) if m]
    cams = {m: np.load(os.path.join(a.runs, m, "test_cams_tb.npy")).astype(np.float32) for m in shown}
    order = [r["fname"] for r in preds[best]]
    both = [k for k, f in enumerate(order) if rows[f]["label"] == 2 and all(preds[m][k]["pred"] == "tb" for m in shown)][:6]
    fig, axes = plt.subplots(len(shown), 6, figsize=(15, 2.75 * len(shown) + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, m in enumerate(shown):
        for j, k in enumerate(both):
            ax, f = axes[i, j], order[k]
            img = load_gray(os.path.join(a.data, "images", f), 512)
            cam = torch.nn.functional.interpolate(torch.tensor(cams[m][k])[None, None], size=(512, 512), mode="bilinear")[0, 0].numpy()
            cam = np.clip((cam - cam.min()) / (np.ptp(cam) + 1e-8), 0, 1)
            ax.imshow(img, cmap="gray")
            ax.imshow(cam, cmap="jet", alpha=0.35)
            for x1, y1, x2, y2 in rows[f]["boxes"]:
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", lw=1.6))
            if j == 0:
                ax.text(-30, 256, NICE[m].replace(" (", "\n("), rotation=90, va="center", ha="right", fontsize=9)
            if i == 0:
                ax.set_title(f, fontsize=8)
    fig.suptitle("Where each model looked (heatmap) vs where radiologists marked TB (green boxes): the same test X-rays", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/figures/heatmaps_vs_boxes.png", dpi=120)

    # 5. pointing game
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ms = list(runs)
    ax.barh([NICE[m] for m in ms], [runs[m]["pointing_game"]["hit_rate"] for m in ms], color=[COLORS[m] for m in ms])
    rnd = np.mean([runs[m]["pointing_game"]["random_hit_rate"] for m in ms])
    ax.axvline(rnd, ls=":", color="#555")
    ax.text(rnd, -0.6, f" random point: {rnd:.0%}", fontsize=8)
    for i, m in enumerate(ms):
        ax.text(runs[m]["pointing_game"]["hit_rate"] + 0.01, i, f"{runs[m]['pointing_game']['hit_rate']:.0%}", va="center", fontsize=9)
    ax.set(xlim=(0, 1), title="Hottest heatmap point inside a doctor's TB box")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/figures/pointing_game.png", dpi=150)

    # summary + model card
    summary = {"deployed_model": best, "alternative_model": alt, "models": runs}
    json.dump(clean(summary), open("results/summary.json", "w"), indent=2)
    lines = ["| Model | Params | Test accuracy | Balanced acc. | TB AUC (95% CI) | Sensitivity | Specificity | Heatmap hits TB box | CPU ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in runs:
        t, s = runs[m]["test"], runs[m]["test"]["tb_vs_rest"]
        ci = t["tb_auc_95ci"]
        lines.append(f"| {NICE[m]}{' **(deployed)**' if m == best else ''} | {runs[m]['params_million']}M | {t['accuracy']:.1%} | {t['balanced_accuracy']:.1%} | "
                     f"{s['auc']:.4f} ({ci[0]:.4f}–{ci[1]:.4f}) | {s['sensitivity']:.1%} | {s['specificity']:.1%} | {runs[m]['pointing_game']['hit_rate']:.0%} | {runs[m]['onnx_cpu_ms']:.0f} |")
    open("results/results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    b = runs[best]
    card = {
        "name": "LungLens", "task": "chest X-ray screening for tuberculosis (3 classes)",
        "classes": CLASSES,
        "input": {"name": "xray", "shape": [1, 1, SIZE, SIZE], "dtype": "float32", "range": [0, 1],
                  "preprocessing": "grayscale, resize to 384x384 bilinear, divide by 255; normalisation is inside the model"},
        "outputs": {"probs": "softmax over classes, shape [1, 3]", "cam": "class activation maps, shape [1, 3, 12, 12]"},
        "screening": {"rule": "refer for a lab test if probs[tb] >= threshold", "threshold": b["test"]["tb_vs_rest"]["threshold"],
                      "threshold_chosen_on": "validation set: 95% specificity on the non-TB X-rays"},
        "deployed_model": best, "architecture": NICE[best],
        "alternative_model": {"model": alt, "architecture": NICE[alt], "file": "weights/lunglens_alt.onnx",
                              "threshold": runs[alt]["test"]["tb_vs_rest"]["threshold"], "pointing_game_hit_rate": runs[alt]["pointing_game"]["hit_rate"],
                              "test_balanced_accuracy": runs[alt]["test"]["balanced_accuracy"],
                              "why": "same architecture, other starting weights; shown on the website to compare heatmaps"} if alt else None,
        "test_metrics": {k: b["test"][k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "accuracy_95ci", "tb_auc_95ci")}
        | {"tb_auc": b["test"]["tb_vs_rest"]["auc"], "sensitivity": b["test"]["tb_vs_rest"]["sensitivity"], "specificity": b["test"]["tb_vs_rest"]["specificity"],
           "pointing_game_hit_rate": b["pointing_game"]["hit_rate"]},
        "dataset": {"name": "TBX11K (simplified)", "url": "https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified", "license": "CC BY 4.0",
                    "paper": "Liu et al., Rethinking Computer-aided Tuberculosis Diagnosis, CVPR 2020", "splits": "data/splits.csv"},
        "files": {f: {"bytes": os.path.getsize(f"weights/{f}"), "sha256": sha256(f"weights/{f}")} for f in sorted(os.listdir("weights"))},
        "intended_use": "research and teaching demo of TB screening; not a medical device",
    }
    json.dump(clean(card), open("model_card.json", "w"), indent=2)
    print("wrote model_card.json, results/, weights/")


if __name__ == "__main__":
    main()
