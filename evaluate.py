"""Test the trained model. Needs only numpy, pillow and onnxruntime.

    python evaluate.py                          # the 12 labelled X-rays in samples/ (no download)
    python evaluate.py --data /path/to/tbx11k   # the full 1,800-image test set from data/splits.csv

Prints accuracy, balanced accuracy, per-class recall, TB AUC, and TB screening sensitivity and
specificity at the threshold in model_card.json; writes the same to evaluation.json.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from predict import HERE, LungLens


def auc(y, s):
    """Area under the ROC curve (Mann-Whitney), no sklearn needed."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="TBX11K folder (with images/); default: bundled samples")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "lunglens.onnx"))
    ap.add_argument("--out", default="evaluation.json")
    a = ap.parse_args()
    model = LungLens(a.weights)

    if a.data:
        rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data", "splits.csv"))) if r["split"] == "test"]
        items = [(os.path.join(a.data, "images", r["fname"]), r["class"]) for r in rows]
    else:
        rows = list(csv.DictReader(open(os.path.join(HERE, "samples", "labels.csv"))))
        items = [(os.path.join(HERE, "samples", r["file"]), r["class"]) for r in rows]

    y, P = [], []
    for k, (path, cls) in enumerate(items, 1):
        res, _ = model(path)
        y.append(model.classes.index(cls))
        P.append([res["probabilities"][c] for c in model.classes])
        if k % 200 == 0:
            print(f"{k}/{len(items)}", file=sys.stderr)
    y, P = np.array(y), np.array(P)
    pred = P.argmax(1)
    tb = model.classes.index("tb")
    is_tb, flag = y == tb, P[:, tb] >= model.threshold
    recall = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(model.classes) if (y == i).any()}
    out = {
        "images": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "recall_per_class": recall,
        "tb_auc": auc(is_tb, P[:, tb]),
        "screening_threshold": model.threshold,
        "tb_sensitivity": float(flag[is_tb].mean()) if is_tb.any() else None,
        "tb_specificity": float((~flag[~is_tb]).mean()) if (~is_tb).any() else None,
        "confusion_matrix": {"rows=true, cols=predicted": model.classes,
                             "counts": [[int(((y == i) & (pred == j)).sum()) for j in range(len(model.classes))] for i in range(len(model.classes))]},
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
