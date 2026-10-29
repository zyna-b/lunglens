"""Metrics: the usual three-class ones, plus the two that matter for TB screening.

Screening question: "should this person get a lab test for TB?" The rule is: refer if the TB
probability is at least a threshold. We fix the threshold on the validation set at 95% specificity
(5% of people without TB get an unnecessary test). We set it from the 600 non-TB validation X-rays,
not from the 60 TB ones: a cut-off estimated from 600 images is far more stable, and a
sensitivity-based cut-off from 60 images turned out to be badly over-fitted (see report).
Then we report sensitivity and specificity on the test set at that threshold. WHO's minimum target
for a TB triage test is 90% sensitivity and 70% specificity.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

from .models import CLASSES

TB = CLASSES.index("tb")


def screening_threshold(p_tb, is_tb, target_specificity=0.95):
    """Lowest threshold that refers at most (1 - target) of the non-TB images."""
    neg = np.sort(np.asarray(p_tb)[~np.asarray(is_tb)])
    k = int(np.ceil(target_specificity * len(neg)))
    return float(np.nextafter(neg[k - 1], 1)) if k > 0 else 0.5


def sensitivity_threshold(p_tb, is_tb, target_sensitivity=0.90):
    """Highest threshold whose sensitivity on these images is >= target (the first rule we tried)."""
    pos = np.sort(np.asarray(p_tb)[np.asarray(is_tb)])[::-1]
    k = int(np.ceil(target_sensitivity * len(pos)))
    return float(pos[k - 1]) if k > 0 else 0.5


def summarise(probs, labels, threshold):
    probs, labels = np.asarray(probs), np.asarray(labels)
    pred = probs.argmax(1)
    p, r, f, n = precision_recall_fscore_support(labels, pred, labels=range(len(CLASSES)), zero_division=0)
    is_tb, flag = labels == TB, probs[:, TB] >= threshold
    tp, fn = int((flag & is_tb).sum()), int((~flag & is_tb).sum())
    tn, fp = int((~flag & ~is_tb).sum()), int((flag & ~is_tb).sum())
    return {
        "n": int(len(labels)),
        "accuracy": float((pred == labels).mean()),
        "balanced_accuracy": float(r.mean()),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "macro_auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "n": int(n[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(labels, pred, labels=range(len(CLASSES))).tolist(),
        "tb_vs_rest": {
            "auc": float(roc_auc_score(is_tb, probs[:, TB])),
            "threshold": float(threshold),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        },
    }


def bootstrap_ci(probs, labels, n=1000, seed=0):
    """95% confidence intervals for accuracy and TB AUC by resampling the test set."""
    rng = np.random.default_rng(seed)
    probs, labels = np.asarray(probs), np.asarray(labels)
    acc, auc = [], []
    for _ in range(n):
        i = rng.integers(0, len(labels), len(labels))
        acc.append((probs[i].argmax(1) == labels[i]).mean())
        auc.append(roc_auc_score(labels[i] == TB, probs[i, TB]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy_95ci": q(acc), "tb_auc_95ci": q(auc)}


def pointing_game(cams, boxes, image_size=512):
    """Does the hottest point of the TB heatmap fall inside a doctor-drawn TB box?

    cams: [N, h, w] TB-class maps for TB images; boxes: per image a list of [x1, y1, x2, y2] in the
    512 px image. Also reports how much of the positive heat lies inside the boxes, and what
    fraction of the image the boxes cover (the hit rate a random point would get).
    """
    import torch
    import torch.nn.functional as F

    up = F.interpolate(torch.as_tensor(np.asarray(cams))[:, None].float(), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0].numpy()
    hits, energy, area = [], [], []
    for m, bs in zip(up, boxes):
        if not bs:
            continue
        mask = np.zeros_like(m, dtype=bool)
        for x1, y1, x2, y2 in bs:
            mask[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = True
        y, x = np.unravel_index(m.argmax(), m.shape)
        hits.append(bool(mask[y, x]))
        pos = np.clip(m, 0, None)
        energy.append(float(pos[mask].sum() / max(pos.sum(), 1e-8)))
        area.append(float(mask.mean()))
    return {"n": len(hits), "hit_rate": float(np.mean(hits)), "energy_in_boxes": float(np.mean(energy)), "random_hit_rate": float(np.mean(area))}
