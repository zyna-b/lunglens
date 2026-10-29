"""Pictures for the website, drawn from the real results at build time (no screenshots of plots):
the hero heatmap, unit charts of every test X-ray, the TB ROC, the confusion matrix, the pointing game
and the class mix of each split. Called by build_site.py; returns {{PLACEHOLDER}} -> HTML/SVG."""

import csv
import json
import os

import numpy as np
from PIL import Image

NICE = {"cnn_scratch": "CNN from scratch", "densenet121_imagenet": "DenseNet-121 · ImageNet start", "densenet121_cxr": "DenseNet-121 · chest X-ray start"}
COLOR = {"densenet121_imagenet": "var(--ember)", "densenet121_cxr": "var(--ice)", "cnn_scratch": "var(--film)"}
CLASS_COLOR = {"healthy": "var(--clear)", "sick_but_not_tb": "var(--lilac)", "tb": "var(--ember)"}
CLASS_NAME = {"healthy": "Healthy", "sick_but_not_tb": "Sick, not TB", "tb": "TB"}


def hero(out, sample="tb_1.png"):
    """TB heatmap of the deployed model on one test X-ray, as a glowing RGBA layer, plus where it peaks."""
    import onnxruntime as ort
    from matplotlib import colormaps

    x = np.asarray(Image.open(f"samples/{sample}").convert("L").resize((384, 384), Image.BILINEAR), dtype=np.float32) / 255.0
    probs, cam = ort.InferenceSession("weights/lunglens.onnx").run(None, {"xray": x[None, None]})
    c = cam[0, 2]
    iy, ix = np.unravel_index(c.argmax(), c.shape)
    t = (c - c.min()) / (c.max() - c.min() + 1e-8)
    t = np.asarray(Image.fromarray(t.astype(np.float32), mode="F").resize((512, 512), Image.BICUBIC)).clip(0, 1)
    rgba = colormaps["inferno"](0.2 + 0.8 * t)
    a = np.clip((t - 0.3) / 0.65, 0, 1)
    rgba[..., 3] = a * a * (3 - 2 * a) * 0.95  # smoothstep: only the evidence glows
    os.makedirs(f"{out}/hero", exist_ok=True)
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(f"{out}/hero/heat.png", optimize=True)
    row = next(r for r in csv.DictReader(open("samples/labels.csv")) if r["file"] == sample)
    x1, y1, x2, y2 = json.loads(row["tb_boxes_xyxy_512px"])[0]
    pct = lambda v: f"{v / 512 * 100:.2f}%"
    return {
        "HERO_IMG": f"samples/{sample}",
        "HERO_PEAK": f"left:{(ix + .5) / 12 * 100:.2f}%;top:{(iy + .5) / 12 * 100:.2f}%",
        "HERO_BOX": f"left:{pct(x1)};top:{pct(y1)};width:{pct(x2 - x1)};height:{pct(y2 - y1)}",
        "HERO_FILE": row["tbx11k_file"],
    }


def dots(n_on, n_off, cols, r, gap, label):
    """A unit chart: one dot per X-ray, lit ones first, so the count can be checked by eye."""
    n = n_on + n_off
    rows = -(-n // cols)
    w, h = cols * gap, rows * gap
    out = [f'<svg class="dots" viewBox="0 0 {w} {h}" role="img" aria-label="{label}"><title>{label}</title>']
    for i in range(n):
        cx, cy = (i % cols) * gap + gap / 2, (i // cols) * gap + gap / 2
        out.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r}" class="{"on" if i < n_on else "off"}"/>')
    return "".join(out) + "</svg>"


def roc(runs, best):
    """TB-vs-rest ROC of all three models, zoomed on the top-left corner where they differ."""
    W, H, L, R, T, B = 560, 430, 56, 20, 16, 50
    x0, x1, y0, y1 = 0.0, 0.2, 0.8, 1.0
    X = lambda f: L + (f - x0) / (x1 - x0) * (W - L - R)
    Y = lambda s: T + (1 - (s - y0) / (y1 - y0)) * (H - T - B)
    g = [f'<svg class="roc" viewBox="0 0 {W} {H}" role="img" aria-label="ROC curves for TB versus not-TB on 1,800 test X-rays, zoomed on false-positive rates up to 20%">',
         f'<defs><clipPath id="plot"><rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}"/></clipPath></defs>']
    for k in range(5):
        f = x0 + k * 0.05
        g.append(f'<line class="grid" x1="{X(f):.1f}" x2="{X(f):.1f}" y1="{T}" y2="{H - B}"/><text class="tick" x="{X(f):.1f}" y="{H - B + 18}" text-anchor="middle">{f * 100:.0f}%</text>')
        s = y0 + k * 0.05
        g.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(s):.1f}" y2="{Y(s):.1f}"/><text class="tick" x="{L - 8}" y="{Y(s) + 4:.1f}" text-anchor="end">{s * 100:.0f}%</text>')
    g.append(f'<text class="axis" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">false-positive rate (1 − specificity)</text>')
    g.append(f'<text class="axis" transform="translate(14 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">sensitivity</text>')
    order = [m for m in runs if m != best] + [best]  # deployed model drawn last, on top
    for m in order:
        rows = list(csv.DictReader(open(f"results/test_predictions_{m}.csv")))
        y = np.array([r["label"] == "tb" for r in rows])
        p = np.array([float(r["p_tb"]) for r in rows])
        o = np.argsort(-p, kind="stable")
        tp, fp = np.cumsum(y[o]), np.cumsum(~y[o])
        tpr, fpr = np.r_[0, tp / y.sum()], np.r_[0, fp / (~y).sum()]
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(fpr, tpr) if a <= x1 + 0.01)
        g.append(f'<polyline clip-path="url(#plot)" class="curve{" best" if m == best else ""}" style="stroke:{COLOR[m]}" points="{pts}"/>')
        s = runs[m]["test"]["tb_vs_rest"]
        g.append(f'<circle class="op" cx="{X(1 - s["specificity"]):.1f}" cy="{Y(s["sensitivity"]):.1f}" r="5.5" style="stroke:{COLOR[m]}"/>')
    g.append("</svg>")
    legend = "".join(f'<li><i style="background:{COLOR[m]}"></i>{NICE[m]}<b>AUC {runs[m]["test"]["tb_vs_rest"]["auc"]:.4f}</b></li>' for m in runs)
    return "".join(g) + f'<ul class="legend">{legend}</ul>'


def confusion(cm, classes):
    cm = np.array(cm)
    cells = ['<div class="cm" role="table" aria-label="Confusion matrix of the deployed model on 1,800 test X-rays">',
             '<div class="cm-corner" role="columnheader">true ↓ · predicted →</div>']
    cells += [f'<div class="cm-h" role="columnheader">{CLASS_NAME[c]}</div>' for c in classes]
    for i, c in enumerate(classes):
        cells.append(f'<div class="cm-h row" role="rowheader">{CLASS_NAME[c]}</div>')
        for j in range(len(classes)):
            f = cm[i, j] / cm[i].sum()
            tone = "var(--clear)" if i == j else "var(--ember)"
            cells.append(f'<div class="cm-c{" diag" if i == j else ""}" role="cell" style="--f:{f:.3f};--tone:{tone}"><b>{cm[i, j]:,}</b><span>{f * 100:.1f}%</span></div>')
    return "".join(cells) + "</div>"


def pointing(runs, best):
    rand = runs[best]["pointing_game"]["random_hit_rate"]
    items = sorted(runs, key=lambda m: -runs[m]["pointing_game"]["hit_rate"])
    rows = [f'<div class="pg-row"><span>{NICE[m]}</span><div class="pg-track"><div class="pg-fill" style="width:{runs[m]["pointing_game"]["hit_rate"] * 100:.1f}%;background:{COLOR[m]}"></div></div><b>{runs[m]["pointing_game"]["hit_rate"] * 100:.0f}%</b></div>' for m in items]
    rows.append(f'<div class="pg-row rand"><span>A random point</span><div class="pg-track"><div class="pg-fill" style="width:{rand * 100:.1f}%"></div></div><b>{rand * 100:.0f}%</b></div>')
    return '<div class="pg">' + "".join(rows) + "</div>"


def splits():
    from collections import Counter
    c = Counter((r["split"], r["class"]) for r in csv.DictReader(open("data/splits.csv")))
    out = []
    for sp, name in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
        tot = sum(c[(sp, k)] for k in CLASS_NAME)
        seg = "".join(f'<i style="width:{c[(sp, k)] / tot * 100:.2f}%;background:{CLASS_COLOR[k]}" title="{CLASS_NAME[k]}: {c[(sp, k)]:,}"></i>' for k in CLASS_NAME)
        out.append(f'<div class="mix"><span>{name}</span><div class="mix-bar">{seg}</div><b>{tot:,}</b></div>')
    key = "".join(f'<li><i style="background:{CLASS_COLOR[k]}"></i>{CLASS_NAME[k]}</li>' for k in CLASS_NAME)
    return "".join(out) + f'<ul class="legend small">{key}</ul>'


def build(out, runs, best):
    s = runs[best]["test"]["tb_vs_rest"]
    v = hero(out)
    v["DOTS_TB"] = dots(s["tp"], s["fn"], 20, 4.2, 12, f"{s['tp']} of {s['tp'] + s['fn']} TB X-rays flagged for a lab test")
    v["DOTS_NEG"] = dots(s["fp"], s["tn"], 50, 2.1, 6, f"{s['fp']} of {s['fp'] + s['tn']} X-rays without TB referred for a lab test")
    v["TP"], v["N_TB"], v["FP"], v["N_NEG"] = str(s["tp"]), str(s["tp"] + s["fn"]), str(s["fp"]), f"{s['fp'] + s['tn']:,}"
    v["ROC_SVG"] = roc(runs, best)
    v["CONFUSION"] = confusion(runs[best]["test"]["confusion_matrix"], ["healthy", "sick_but_not_tb", "tb"])
    v["POINTING"] = pointing(runs, best)
    v["SPLIT_BARS"] = splits()
    return v
