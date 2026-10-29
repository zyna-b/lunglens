"""Screen chest X-rays for TB with the trained model. Needs only numpy, pillow and onnxruntime.

    python predict.py samples/tb_1.png
    python predict.py samples/*.png --heatmaps heatmaps/

Prints one JSON object per X-ray: the three class probabilities, the screening decision
("refer for a lab test" if the TB probability is at or above the threshold in model_card.json),
and, with --heatmaps, saves the X-ray with the model's TB heatmap on top.
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def load_gray(path, size):
    """Same preprocessing as training (lunglens/data.py): grayscale, resized to size x size."""
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def jet(v):
    """v in [0, 1] -> RGB uint8, a small stand-in for matplotlib's jet colormap."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def heatmap_overlay(path, cam, out_path, alpha=0.4):
    img = Image.open(path).convert("RGB")
    c = Image.fromarray(np.float32(cam)).resize(img.size, Image.BILINEAR)
    c = np.asarray(c)
    c = np.clip((c - c.min()) / (c.max() - c.min() + 1e-8), 0, 1)
    out = (np.asarray(img) * (1 - alpha) + jet(c) * alpha).astype(np.uint8)
    Image.fromarray(out).save(out_path)


class LungLens:
    def __init__(self, weights=os.path.join(HERE, "weights", "lunglens.onnx"), card=os.path.join(HERE, "model_card.json")):
        self.card = json.load(open(card))
        self.classes = self.card["classes"]
        self.size = self.card["input"]["shape"][-1]
        self.threshold = self.card["screening"]["threshold"]
        self.sess = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])

    def __call__(self, path):
        x = load_gray(path, self.size).astype(np.float32)[None, None] / 255.0
        probs, cam = self.sess.run(None, {"xray": x})
        p = probs[0]
        tb = float(p[self.classes.index("tb")])
        return {
            "file": path,
            "prediction": self.classes[int(p.argmax())],
            "probabilities": {c: round(float(v), 4) for c, v in zip(self.classes, p)},
            "screening": {"tb_probability": round(tb, 4), "threshold": round(self.threshold, 4),
                          "result": "refer for a TB lab test (sputum / GeneXpert)" if tb >= self.threshold else "no TB signs found"},
        }, cam[0, self.classes.index("tb")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "lunglens.onnx"))
    ap.add_argument("--heatmaps", help="folder to save heatmap overlays in")
    a = ap.parse_args()
    model = LungLens(a.weights)
    if a.heatmaps:
        os.makedirs(a.heatmaps, exist_ok=True)
    for path in a.images:
        result, cam = model(path)
        if a.heatmaps:
            out = os.path.join(a.heatmaps, os.path.splitext(os.path.basename(path))[0] + "_heatmap.png")
            heatmap_overlay(path, cam, out)
            result["heatmap"] = out
        print(json.dumps(result))


if __name__ == "__main__":
    main()
