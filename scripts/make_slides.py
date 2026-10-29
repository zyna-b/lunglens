"""Build the presentation: docs/slides/LungLens_slides.{pptx,pdf,mp4} + voiceover script.
    python scripts/make_slides.py [--voice en-US-AndrewMultilingualNeural]"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "tools"))
from slides import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--voice", default="en-US-AndrewMultilingualNeural")
a = ap.parse_args()
links = json.load(open("links.json"))
S = json.load(open("results/summary.json"))
runs, best = S["models"], S["deployed_model"]
B = runs[best]
T, TB = B["test"], B["test"]["tb_vs_rest"]
NICE = {"cnn_scratch": "CNN from scratch", "densenet121_imagenet": "DenseNet-121 · ImageNet start", "densenet121_cxr": "DenseNet-121 · X-ray start"}
SAY = {"cnn_scratch": "a CNN trained from scratch", "densenet121_imagenet": "DenseNet-121 starting from ImageNet", "densenet121_cxr": "DenseNet-121 starting from chest X-rays"}
pct = lambda v, d=1: f"{v * 100:.{d}f}%"
site = links.get("site") or "lunglens.vercel.app"
gh = links["github"].replace("https://", "")
author = links["author"]
first = B.get("test_at_val_90_sensitivity")

CSS = """
body{background:#dfe6ea}
.slide{background:#f6f8fa;color:#0f1b2d;padding:80px 120px 110px;display:flex;flex-direction:column;justify-content:center}
.slide:after{content:"LungLens";position:absolute;left:110px;bottom:44px;font-size:22px;color:#7d8896}
.bar{position:absolute;left:0;top:0;bottom:0;width:22px;background:#0b7a83}
h1{font-size:110px;line-height:1;margin:0 0 26px;letter-spacing:-.03em;font-weight:900}
h2{font-size:70px;line-height:1.08;margin:0 0 50px;letter-spacing:-.02em;font-weight:850}
.lead{font-size:44px;color:#3b4656;line-height:1.4;max-width:1500px}
ul{margin:0;padding-left:48px} li{font-size:43px;line-height:1.42;margin:14px 0;color:#1d2838}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:60px;align-items:center}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:36px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:28px}
.stat{background:#fff;border:3px solid #e1e7ee;border-radius:28px;padding:34px 36px}
.stat .n{font-size:116px;font-weight:900;letter-spacing:-.03em;line-height:1}
.stat .l{font-size:33px;color:#556;margin-top:14px;line-height:1.3}
.teal{background:#0b7a83;border-color:#0b7a83;color:#fff} .teal .l{color:#d7f0f2}
.warn{background:#fff6e0;border-color:#f3d48a}
img.fig{width:100%;max-height:820px;object-fit:contain;border-radius:18px;border:3px solid #e1e7ee;background:#fff}
.muted{color:#6b7686;font-size:28px}
.tag{display:inline-block;background:#0b7a83;color:#fff;font-weight:800;font-size:26px;padding:8px 18px;border-radius:12px;margin-bottom:28px;letter-spacing:.04em}
table{border-collapse:collapse;font-size:35px;width:100%} td,th{padding:18px 18px;border-bottom:3px solid #e1e7ee;text-align:left} th{font-size:22px;color:#7d8896;text-transform:uppercase;letter-spacing:.06em}
tr.dep td{background:#dff1f2}
"""

rows = "".join(f"<tr class='{'dep' if m == best else ''}'><td>{NICE[m]}</td><td>{pct(r['test']['accuracy'])}</td><td>{pct(r['test']['balanced_accuracy'])}</td>"
               f"<td>{r['test']['tb_vs_rest']['auc']:.4f}</td><td>{pct(r['test']['tb_vs_rest']['sensitivity'])}</td><td>{pct(r['test']['tb_vs_rest']['specificity'])}</td>"
               f"<td>{pct(r['pointing_game']['hit_rate'], 0)}</td></tr>" for m, r in runs.items())
cmp_say = ""
if "densenet121_cxr" in runs and "densenet121_imagenet" in runs:
    c, i = runs["densenet121_cxr"], runs["densenet121_imagenet"]
    better = "densenet121_cxr" if c["test"]["balanced_accuracy"] >= i["test"]["balanced_accuracy"] else "densenet121_imagenet"
    other = "densenet121_imagenet" if better == "densenet121_cxr" else "densenet121_cxr"
    ci_c, ci_i = c["test"]["tb_auc_95ci"], i["test"]["tb_auc_95ci"]
    overlap = not (ci_c[0] > ci_i[1] or ci_i[0] > ci_c[1])
    close = abs(c["test"]["balanced_accuracy"] - i["test"]["balanced_accuracy"]) < 0.02
    cmp_say = (f" {'The two DenseNets are close: ' if close else ''}{SAY[better]} did better on balanced accuracy, "
               f"{pct(runs[better]['test']['balanced_accuracy'])} against {pct(runs[other]['test']['balanced_accuracy'])}"
               + (", and their TB AUC confidence intervals overlap." if overlap else ", and their TB AUC confidence intervals do not overlap."))
scratch_say = ""
if "cnn_scratch" in runs:
    sb = runs["cnn_scratch"]["test"]["balanced_accuracy"]
    scratch_say = f" Training from scratch gives {pct(sb)} balanced accuracy" + (", so pretraining clearly helps." if T["balanced_accuracy"] - sb > 0.02 else ", close to the pretrained models on this dataset.")

pg_models = [m for m in ("densenet121_cxr", "densenet121_imagenet") if m in runs]
pg_best = max(pg_models, key=lambda m: runs[m]["pointing_game"]["hit_rate"])
pg_say = ("Accuracy isn't enough for a medical tool, so I checked where each model looks. Radiologists boxed every TB lesion, and I measured how often "
          "the hottest point of the heatmap lands inside a box. " + ", ".join(f"{SAY[m]}: {pct(runs[m]['pointing_game']['hit_rate'], 0)}" for m in pg_models)
          + f", against {pct(B['pointing_game']['random_hit_rate'], 0)} for a random point.")
if len(pg_models) == 2 and pg_best != best:
    pg_say += " So the model that has already seen chest X-rays looks in the right place far more often, even though it is slightly less accurate. The website lets you switch between the two and compare."

slides = [
    {"html": f"""<div class=bar></div><div class=grid2><div><div class=tag>DEEP LEARNING · SEMESTER PROJECT</div><h1>LungLens</h1>
<p class=lead>TB screening from a chest X-ray, in your browser, with a heatmap you can check.</p>
<p class=muted style="margin-top:60px">{author} · BSCS-7A · Superior University, Lahore<br>Instructor: Dr. Munir Ahmad</p></div>
<div><img class=fig src="figures/heatmaps_vs_boxes.png"></div></div>""",
     "say": f"Assalam o alaikum. I'm {author}. This is LungLens: a deep learning model that screens chest X-rays for tuberculosis, right in the browser, and shows where it looked."},
    {"html": """<div class=bar></div><h2>Pakistan: one of the world's largest TB burdens</h2><div class=grid3>
<div class="stat teal"><div class=n>~670k</div><div class=l>TB cases in 2024 (WHO 2025)</div></div>
<div class=stat><div class=n>5th</div><div class=l>highest in the world, 6.3% of all cases</div></div>
<div class=stat><div class=n>2021</div><div class=l>WHO: software may replace the human X-ray reader for TB screening, age 15+</div></div></div>
<p class=lead style="margin-top:46px">X-rays are cheap and everywhere. People who can read them for TB are not.</p>""",
     "say": "Pakistan had around six hundred and seventy thousand TB cases in 2024, the fifth highest in the world. X-rays are cheap and available almost everywhere, but people trained to read them for TB are scarce. Since 2021, the WHO has accepted software as the X-ray reader for TB screening."},
    {"html": """<div class=bar></div><h2>Most TB student projects have a hidden shortcut</h2><div class=grid2><div><ul>
<li>Popular Kaggle TB dataset: TB and normal X-rays from <b>different hospitals</b></li><li>A model can learn <b>"which hospital"</b> instead of "which disease"</li>
<li>So I used <b>TBX11K</b> (CVPR 2020): shared sources, and a class for <b>sick but not TB</b></li></ul></div>
<div class="stat warn"><div class=l style="font-size:34px;color:#7a5200">The real question is not "TB or healthy?"<br><br><b>It's "TB, or some other lung disease?"</b></div></div></div>""",
     "say": "Most student TB projects use a Kaggle dataset where the TB and normal X-rays come from different hospitals, so a model can learn which hospital an image came from instead of which disease it shows. I used TBX11K instead, which also includes patients who are sick but don't have TB. That's the real question: TB, or some other lung disease?"},
    {"html": """<div class=bar></div><h2>Data: TBX11K, 8,399 labelled X-rays</h2>
<table><tr><th>Split</th><th>Healthy</th><th>Sick, not TB</th><th>TB</th><th>Total</th></tr>
<tr><td>Train</td><td>2,700</td><td>2,700</td><td>539</td><td>5,939</td></tr><tr><td>Validation</td><td>300</td><td>300</td><td>60</td><td>660</td></tr>
<tr class=dep><td><b>Test (untouched)</b></td><td>800</td><td>800</td><td>200</td><td><b>1,800</b></td></tr></table>
<ul style="margin-top:40px"><li>TB areas boxed by radiologists · CC BY 4.0</li><li>384 × 384 grayscale · TB weighted ~5× in the loss · <b>no left-right flips</b>: the heart is on the left</li></ul>""",
     "say": "TBX11K has eight thousand four hundred labelled X-rays, and radiologists drew boxes around every TB lesion. I kept eighteen hundred X-rays completely untouched for the final test. TB is rare in the training set, so the loss weights it about five times more. And I never flip X-rays left to right: that would put the heart on the wrong side."},
    {"html": f"""<div class=bar></div><h2>Three models, one question at a time</h2><div class=grid3>
<div class=stat><div class=l><b style="font-size:38px;color:#0f1b2d">CNN from scratch</b><br><br>5 conv blocks, 2.4M params. The baseline.</div></div>
<div class=stat><div class=l><b style="font-size:38px;color:#0f1b2d">DenseNet-121</b><br><br>starts from <b>ImageNet</b> photos. The CheXNet architecture.</div></div>
<div class="stat teal"><div class=l><b style="font-size:38px;color:#fff">DenseNet-121</b><br><br>starts from <b>chest X-rays</b> of 7 public datasets.</div></div></div>
<p class=lead style="margin-top:46px">Same data, same augmentation, same schedule. The two DenseNets differ only in their starting weights.</p>""",
     "say": "I trained three models with identical data and settings. A small CNN from scratch as the baseline, and two DenseNet-121 networks that differ only in their starting weights: one starts from ImageNet photos, the other from chest X-rays. That isolates one question: does it help to start from a model that has already seen lungs?"},
    {"html": f"""<div class=bar></div><h2>Results on 1,800 test X-rays</h2>
<table><tr><th>Model</th><th>Accuracy</th><th>Balanced</th><th>TB AUC</th><th>Sensitivity</th><th>Specificity</th><th>Pointing</th></tr>{rows}</table>
<p class=muted style="margin-top:26px">Highlighted: deployed model, chosen on validation. Sensitivity / specificity: TB vs not-TB at the screening threshold.</p>""",
     "say": f"On eighteen hundred X-rays it had never seen, the deployed model, {SAY[best]}, is {pct(T['accuracy'])} accurate across three classes, with a TB AUC of {TB['auc']:.4f}." + cmp_say + scratch_say},
    {"html": f"""<div class=bar></div><h2>Does it look in the right place?</h2><div class=grid2><img class=fig src="figures/heatmaps_vs_boxes.png">
<div>{''.join(f'<div class="stat{" teal" if m == pg_best else ""}" style="margin-bottom:22px"><div class=n>{pct(runs[m]["pointing_game"]["hit_rate"], 0)}</div><div class=l>{NICE[m]}: hottest heatmap point inside a doctor&#39;s TB box</div></div>' for m in pg_models)}
<p class=lead style="margin-top:10px">A random point: {pct(B['pointing_game']['random_hit_rate'], 0)}. Switch between the two on the website.</p></div></div>""",
     "say": pg_say},
    {"html": f"""<div class=bar></div><h2>A lesson about thresholds</h2><div class=grid2><div><ul>
<li>First try: threshold for 90% sensitivity on <b>60</b> validation TB X-rays</li>
<li>On the test set: only <b>{pct(first['sensitivity']) if first else '–'}</b>. Too few positives, too confident</li>
<li>Fix: set it from the <b>600</b> non-TB X-rays (95% specificity)</li>
<li>Now: <b>{pct(TB['sensitivity'])}</b> sensitivity, <b>{pct(TB['specificity'])}</b> specificity</li></ul></div>
<img class=fig src="figures/roc_tb.png"></div>""",
     "say": (f"I also learned a lesson about thresholds. My first rule picked the threshold that caught ninety percent of the sixty TB cases in validation. On the test set it caught only {pct(first['sensitivity'])}. Sixty cases are too few to estimate that reliably. " if first else "")
            + f"So I set the threshold from the six hundred non-TB X-rays instead, at ninety-five percent specificity. Now it catches {pct(TB['sensitivity'])} of TB cases, with {pct(TB['specificity'])} specificity, well above the WHO minimum of ninety and seventy."},
    {"html": f"""<div class=bar></div><div class=grid2><div><h2>Live demo</h2><ul><li>Open <b>{site}</b></li><li>Upload an X-ray, or tap a sample</li><li>About a second, on your own device</li><li>Nothing is uploaded</li></ul></div>
<img class=fig src="figures/site_desktop.png"></div>""",
     "say": "Here's the website. I upload an X-ray or tap one of the test samples, and about a second later it gives the screening result, the three probabilities, and the heatmap, with the doctors' boxes on top for comparison. The model runs on my laptop; the X-ray never leaves it."},
    {"html": """<div class=bar></div><h2>Limits, and what's next</h2><div class=grid2><div><b style="font-size:34px">Limits</b><ul>
<li>Chinese hospital data: untested on Pakistani X-rays</li><li>Adults only (WHO: age 15+)</li><li>A screening aid, not a diagnosis</li></ul></div>
<div><b style="font-size:34px">Next</b><ul><li>Test on X-rays from Pakistani hospitals</li><li>Re-set the threshold for that population</li><li>Lung segmentation, so attention stays in the lungs</li></ul></div></div>""",
     "say": "Its limits: TBX11K comes from Chinese hospitals, so it still needs testing on X-rays from Pakistani hospitals. It's for adults, and it's a screening aid, not a diagnosis. Next steps: Pakistani data, a threshold set for that population, and lung segmentation."},
    {"html": f"""<div class=bar></div><h2>Everything is open</h2><ul style="margin-top:20px">
<li>Try it: <b>{site}</b></li><li>Code, weights, splits, notebook: <b>{gh}</b></li><li>Report, slides, model card: linked on the site</li></ul>
<p class=lead style="margin-top:70px">Thank you, Dr. Munir Ahmad.</p>""",
     "say": "The website, code, weights, data splits, notebook and report are all public and linked. Thank you, Dr. Munir Ahmad, and thank you for watching."},
]

os.makedirs("docs/slides", exist_ok=True)
build(slides, CSS, "docs/slides", voice=a.voice, assets={"results/figures": "figures"}, title="LungLens_slides", author=author)
