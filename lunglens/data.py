"""TBX11K (simplified Kaggle release, CC BY 4.0): https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified

data.csv has one row per TB box (or one row with bbox "none"). We fold it into one row per image.

Splits (fixed, written to data/splits.csv):
  test   TBX11K's own validation set: 1,800 images (800 healthy, 800 sick-but-not-TB, 200 TB). Its
         labels are public; the official test set's are not, so this is the standard held-out set.
  train  90% of TBX11K's training set, stratified by class, seed 42
  val    the other 10%, for picking the best epoch and the screening threshold
"""

import ast
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .models import CLASSES

LABEL = {"healthy": 0, "sick_but_no_tb": 1, "tb": 2}
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/vbookshelf/tbx11k-simplified"


def read_index(root):
    rows = {}
    with open(os.path.join(root, "data.csv"), newline="") as f:
        for r in csv.DictReader(f):
            e = rows.setdefault(r["fname"], {
                "fname": r["fname"], "label": LABEL[r["image_type"]], "source": r["source"],
                "tb_type": r["tb_type"], "boxes": [], "size": int(r["image_width"]),
            })
            if r["bbox"] != "none":
                b = ast.literal_eval(r["bbox"])
                e["boxes"].append([round(v, 1) for v in (b["xmin"], b["ymin"], b["xmin"] + b["width"], b["ymin"] + b["height"])])
    return [rows[k] for k in sorted(rows)]


def make_splits(rows, val_frac=0.10, seed=42):
    rng = np.random.default_rng(seed)
    for r in rows:
        r["split"] = "test" if r["source"] == "val" else "train"
    for c in range(len(CLASSES)):
        idx = [i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == c]
        rng.shuffle(idx)
        for i in idx[: round(len(idx) * val_frac)]:
            rows[i]["split"] = "val"
    return rows


def write_splits(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "split", "label", "class", "tb_type", "tb_boxes_xyxy_512px"])
        for r in rows:
            w.writerow([r["fname"], r["split"], r["label"], CLASSES[r["label"]], r["tb_type"], json.dumps(r["boxes"]) if r["boxes"] else ""])


def read_splits(path):
    with open(path, newline="") as f:
        return [{"fname": r["fname"], "split": r["split"], "label": int(r["label"]), "tb_type": r["tb_type"],
                 "boxes": json.loads(r["tb_boxes_xyxy_512px"]) if r["tb_boxes_xyxy_512px"] else []} for r in csv.DictReader(f)]


def load_gray(path, size):
    """The one preprocessing step shared by training, evaluation and predict.py: grayscale, resized to size x size."""
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_all(paths, size, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(lambda p: load_gray(p, size), paths)))
