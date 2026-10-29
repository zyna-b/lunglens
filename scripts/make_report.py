"""Write the project report (HTML + PDF) from the results. Run after report.py.

    python scripts/make_report.py      -> docs/LungLens_report.pdf, docs/report/index.html
"""

import csv
import html
import json
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "tools"))

NICE = {"cnn_scratch": "CNN from scratch", "densenet121_imagenet": "DenseNet-121, ImageNet start", "densenet121_cxr": "DenseNet-121, chest-X-ray start"}
pct = lambda v, d=1: f"{v * 100:.{d}f}%"
links = json.load(open("links.json"))
S = json.load(open("results/summary.json"))
runs, best = S["models"], S["deployed_model"]
B = runs[best]
T, TB = B["test"], B["test"]["tb_vs_rest"]
splits = {r["fname"]: r for r in csv.DictReader(open("data/splits.csv"))}
preds = list(csv.DictReader(open(f"results/test_predictions_{best}.csv")))
thr = TB["threshold"]

# error analysis on the deployed model
tb_rows = [r for r in preds if r["label"] == "tb"]
by_type = {}
for t in ("active_tb", "latent_tb"):
    rows = [r for r in tb_rows if splits[r["fname"]]["tb_type"] == t]
    by_type[t] = (sum(float(r["p_tb"]) >= thr for r in rows), len(rows))
missed = [r for r in tb_rows if float(r["p_tb"]) < thr]
missed_as = {c: sum(r["pred"] == c for r in missed) for c in ("healthy", "sick_but_not_tb", "tb")}
false_ref = {c: sum(float(r["p_tb"]) >= thr for r in preds if r["label"] == c) for c in ("healthy", "sick_but_not_tb")}
n_cls = {c: sum(r["label"] == c for r in preds) for c in ("healthy", "sick_but_not_tb", "tb")}
cm = T["confusion_matrix"]
first_rule = B.get("test_at_val_90_sensitivity")

has = lambda m: m in runs
cmp_sentence = ""
if has("densenet121_cxr") and has("densenet121_imagenet"):
    c, i = runs["densenet121_cxr"], runs["densenet121_imagenet"]
    d_auc = c["test"]["tb_vs_rest"]["auc"] - i["test"]["tb_vs_rest"]["auc"]
    d_bal = c["test"]["balanced_accuracy"] - i["test"]["balanced_accuracy"]
    d_pg = c["pointing_game"]["hit_rate"] - i["pointing_game"]["hit_rate"]
    overlap = not (c["test"]["tb_auc_95ci"][0] > i["test"]["tb_auc_95ci"][1] or i["test"]["tb_auc_95ci"][0] > c["test"]["tb_auc_95ci"][1])
    cmp_sentence = (f"Starting from chest X-rays instead of ImageNet changed test balanced accuracy by {d_bal * 100:+.1f} points, TB AUC by {d_auc:+.4f} "
                    f"and the pointing-game hit rate by {d_pg * 100:+.1f} points. "
                    + ("The 95% confidence intervals of the two TB AUCs overlap, so on this dataset the two starting points are statistically "
                       "indistinguishable for detection; " if overlap else "The TB AUC confidence intervals do not overlap; ")
                    + ("the domain-pretrained network does, however, place its evidence inside the doctors' boxes more often." if d_pg > 0.02 else
                       "where the two networks look differs little." if abs(d_pg) <= 0.02 else
                       "the ImageNet network places its evidence inside the doctors' boxes more often."))
scratch_sentence = ""
if has("cnn_scratch"):
    s = runs["cnn_scratch"]
    scratch_sentence = (f"The CNN trained from scratch reaches {pct(s['test']['balanced_accuracy'])} balanced accuracy and a TB AUC of {s['test']['tb_vs_rest']['auc']:.4f}, "
                        f"against {pct(T['balanced_accuracy'])} and {TB['auc']:.4f} for the deployed model: pretraining is worth "
                        f"{(T['balanced_accuracy'] - s['test']['balanced_accuracy']) * 100:.1f} points of balanced accuracy here.")


TITLE = ("LungLens: chest X-ray tuberculosis screening with a domain-pretrained CNN whose heatmaps are checked against radiologists' boxes" if best == "densenet121_cxr"
         else "LungLens: chest X-ray tuberculosis screening, and what starting from chest X-rays changes about where a CNN looks")
alt = S.get("alternative_model")
choice = ""
if alt and alt in runs:
    pg_b, pg_a = B["pointing_game"]["hit_rate"], runs[alt]["pointing_game"]["hit_rate"]
    if pg_a > pg_b + 0.02:
        choice = (f"<p><b>Which model to deploy?</b> Our pre-declared rule (best validation balanced accuracy) selects the {NICE[best]} model, while the {NICE[alt]} model "
                  f"places its evidence inside the radiologists' boxes more often ({pct(pg_a)} vs {pct(pg_b)}). For a clinical tool, evidence that lines up with "
                  f"radiologists' marks can be worth a small loss in accuracy. We keep the rule rather than re-selecting after seeing test results, and the website ships "
                  f"both models with a switch, so users can compare the two heatmaps on the same X-ray.</p>")
    else:
        choice = (f"<p><b>Which model to deploy?</b> Our pre-declared rule selects the {NICE[best]} model, which also localises at least as well as the {NICE[alt]} model "
                  f"({pct(pg_b)} vs {pct(pg_a)} pointing-game hits). The website ships both, with a switch to compare their heatmaps.</p>")


def table(rows, head):
    return "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


results_rows = []
for m, r in runs.items():
    t, s = r["test"], r["test"]["tb_vs_rest"]
    results_rows.append([NICE[m] + (" <b>(deployed)</b>" if m == best else ""), f"{r['params_million']}M", pct(t["accuracy"]),
                         f"{pct(t['accuracy_95ci'][0])}–{pct(t['accuracy_95ci'][1])}", pct(t["balanced_accuracy"]), f"{t['macro_f1']:.3f}",
                         f"{s['auc']:.4f}", f"{t['tb_auc_95ci'][0]:.4f}–{t['tb_auc_95ci'][1]:.4f}", pct(s["sensitivity"]), pct(s["specificity"]),
                         pct(r["pointing_game"]["hit_rate"], 1), f"{r['onnx_cpu_ms']:.0f}"])
per_class = [[c.replace("_", " "), f"{T['per_class'][c]['precision']:.3f}", f"{T['per_class'][c]['recall']:.3f}", f"{T['per_class'][c]['f1']:.3f}", T["per_class"][c]["n"]]
             for c in ("healthy", "sick_but_not_tb", "tb")]
hp = [["Input", "grayscale, 384 × 384, values in [0, 1]"], ["Optimiser", "AdamW, weight decay 0.05"],
      ["Learning rate", "3e-4 (DenseNets), 2e-3 (scratch); 5% linear warm-up then cosine decay"],
      ["Epochs", "20 (DenseNets), 40 (scratch); best epoch by validation balanced accuracy"], ["Batch size", "32, bfloat16 mixed precision"],
      ["Loss", "cross-entropy, class weights ∝ 1 / frequency (TB ≈ 3.7, others ≈ 0.73)"],
      ["Augmentation (GPU)", "rotation ±7°, zoom 0.92–1.12, stretch ±7%, shift ±6%, gamma/contrast/brightness ±15–25%, noise, blur (p = 0.2); no flips"],
      ["Hardware", "1 × NVIDIA L4 (Google Colab); about 51 s per DenseNet epoch"], ["Seed", "42 (splits and training)"]]

missed_txt = ", ".join(f"{v} as {k.replace('_', ' ')}" for k, v in missed_as.items() if v)
ref = [
    "World Health Organization. <i>Global Tuberculosis Report 2025</i>. Geneva: WHO; 2025.",
    "World Health Organization. <i>WHO consolidated guidelines on tuberculosis. Module 2: screening – systematic screening for tuberculosis disease</i>. Geneva: WHO; 2021.",
    "World Health Organization. <i>High-priority target product profiles for new tuberculosis diagnostics: report of a consensus meeting</i>. Geneva: WHO; 2014.",
    "Liu Y, Wu Y-H, Ban Y, Wang H, Cheng M-M. Rethinking computer-aided tuberculosis diagnosis. <i>CVPR</i>, 2020.",
    "Rahman T, Khandakar A, Kadir MA, et al. Reliable tuberculosis detection using chest X-ray with deep learning, segmentation and visualization. <i>IEEE Access</i> 8, 2020.",
    "Rajpurkar P, Irvin J, Zhu K, et al. CheXNet: radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv:1711.05225, 2017.",
    "Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. <i>CVPR</i>, 2017.",
    "Cohen JP, Viviano JD, Bertin P, et al. TorchXRayVision: a library of chest X-ray datasets and models. <i>MIDL</i>, PMLR 172, 2022.",
    "Deng J, Dong W, Socher R, Li L-J, Li K, Fei-Fei L. ImageNet: a large-scale hierarchical image database. <i>CVPR</i>, 2009.",
    "Zhou B, Khosla A, Lapedriza A, Oliva A, Torralba A. Learning deep features for discriminative localization. <i>CVPR</i>, 2016.",
    "Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. <i>ICCV</i>, 2017.",
    "Zhang J, Bargal SA, Lin Z, Brandt J, Shen X, Sclaroff S. Top-down neural attention by excitation backprop. <i>IJCV</i> 126, 2018.",
    "Zech JR, Badgeley MA, Liu M, et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. <i>PLOS Medicine</i> 15(11), 2018.",
    "DeGrave AJ, Janizek JD, Lee S-I. AI for radiographic COVID-19 detection selects shortcuts over signal. <i>Nature Machine Intelligence</i> 3, 2021.",
    "Loshchilov I, Hutter F. Decoupled weight decay regularization. <i>ICLR</i>, 2019.",
    "ONNX Runtime contributors. ONNX Runtime Web. https://onnxruntime.ai, 2024.",
]

H = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>LungLens report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
<style>
body{{font-family:"Source Serif 4",Georgia,serif;font-size:10.6pt;line-height:1.5;color:#111;max-width:820px;margin:0 auto;padding:0 14px}}
h1,h2,h3,.meta,table,figcaption,.abstract b{{font-family:Inter,system-ui,sans-serif}}
h1{{font-size:21pt;line-height:1.15;margin:.2em 0 .25em;letter-spacing:-.01em}} h2{{font-size:13pt;margin:1.5em 0 .4em;border-bottom:1px solid #ddd;padding-bottom:3px}}
h3{{font-size:11pt;margin:1.1em 0 .3em}} .meta{{font-size:9pt;color:#444;margin-bottom:12px}} .meta a{{color:#0b7a83}}
.abstract{{background:#f3f7f8;border-left:4px solid #0b7a83;padding:10px 14px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:8.4pt;margin:8px 0 4px}} th,td{{border-bottom:1px solid #ddd;padding:4px 5px;text-align:left;vertical-align:top}} th{{background:#f5f5f5}}
figure{{margin:12px 0;break-inside:avoid}} figure img{{width:100%;border:1px solid #e5e5e5}} figcaption{{font-size:8.6pt;color:#444;margin-top:3px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} code{{font-size:8.6pt;background:#f2f2f2;padding:0 3px}}
ol.refs{{font-size:8.8pt;padding-left:18px}} .key{{background:#fff8e6;border-left:4px solid #d4a017;padding:8px 12px;margin:10px 0}}
@media print{{h2{{break-after:avoid}} table{{break-inside:avoid}}}}
</style></head><body>
<h1>{TITLE}</h1>
<div class="meta">{html.escape(links['author'])} · BS Computer Science (7A), Superior University, Lahore · Deep Learning, instructor Dr. Munir Ahmad · {links.get('month', 'November 2026')}<br>
Live demo: <a href="{links.get('site') or '#'}">{links.get('site') or '(Vercel link)'}</a> · Code &amp; weights: <a href="{links['github']}">{links['github']}</a>{(' · Video: <a href="' + links['youtube'] + '">' + links['youtube'] + '</a>') if links.get('youtube') else ''}</div>

<div class="abstract"><b>Abstract.</b> Pakistan carries the world's fifth-largest tuberculosis (TB) burden, and people trained to read chest X-rays for TB are scarce outside large cities. We build LungLens, a three-class screening model (healthy / sick but not TB / TB) trained on TBX11K, a dataset of 8,399 labelled X-rays whose TB lesions were boxed by radiologists. We compare a CNN trained from scratch with two DenseNet-121 networks that differ only in their starting weights: ImageNet photographs or chest X-rays from seven public datasets. On 1,800 held-out test X-rays the deployed model ({NICE[best]}) reaches {pct(T['accuracy'])} accuracy (95% CI {pct(T['accuracy_95ci'][0])}–{pct(T['accuracy_95ci'][1])}), {pct(T['balanced_accuracy'])} balanced accuracy and a TB-versus-rest AUC of {TB['auc']:.4f}. At a screening threshold fixed on validation data it detects {pct(TB['sensitivity'])} of TB cases at {pct(TB['specificity'])} specificity. Because every model ends in global average pooling, its class activation map comes from the same forward pass; the hottest point of the TB map falls inside a radiologist's box in {pct(B['pointing_game']['hit_rate'])} of TB test X-rays, against {pct(B['pointing_game']['random_hit_rate'])} for a random point. The model runs entirely in the browser through ONNX Runtime Web, so no X-ray leaves the user's device.</div>

<h2>1. Problem</h2>
<p>The WHO estimates about 670,000 new TB cases in Pakistan in 2024, 6.3% of the world total and the fifth-highest national burden [1]. Many cases are never diagnosed. Chest radiography is cheap and widely available, and since 2021 the WHO has recommended computer-aided detection (CAD) software as an alternative to a human reader when screening people aged 15 and over [2]. A screening tool must above all not miss TB: the WHO's minimum target for a triage test is 90% sensitivity with 70% specificity [3].</p>
<p>Most student TB projects use the Kaggle "TB Chest X-ray" database [5], in which TB and normal images come from different hospitals and countries. A model can then learn which hospital an image came from instead of which disease it shows, a shortcut documented for pneumonia and COVID-19 models [13, 14]. We avoid this by using TBX11K [4], whose classes share their sources and which includes a class of patients who are <i>sick but do not have TB</i>: the realistic difficulty is separating TB from other lung disease, not from a healthy chest.</p>
<p><b>Contributions.</b> (1) A controlled comparison of three starting points with identical data, augmentation and schedule. (2) A screening operating point chosen on validation data, and an account of why our first choice of rule failed. (3) A quantitative check of the heatmaps against radiologists' boxes (the pointing game [12]). (4) A deployed, private, in-browser tool with code, weights and data splits published.</p>

<h2>2. Related work</h2>
<p>CheXNet [6] showed that a DenseNet-121 [7] fine-tuned on ChestX-ray14 can match radiologists on pneumonia. Liu et al. [4] released TBX11K with bounding boxes and reported classification and detection baselines. Rahman et al. [5] report about 98% accuracy on their TB database, partly reflecting the source difference above. TorchXRayVision [8] provides DenseNet-121 weights trained on several large chest X-ray collections (NIH, CheXpert, MIMIC-CXR, PadChest and others), which we use as the domain-specific starting point. Class activation mapping (CAM) [10] and Grad-CAM [11] visualise evidence; the pointing game [12] turns such maps into a number.</p>

<h2>3. Data</h2>
<p>TBX11K (simplified Kaggle release, CC BY 4.0) contains 512 × 512 frontal X-rays labelled healthy, sick-but-not-TB, or TB (active or latent), with radiologist boxes around TB regions. The official test labels are not public, so we use TBX11K's labelled validation set as our untouched test set and split its training set 90/10 (stratified, seed 42) into training and validation. The exact file list is in <code>data/splits.csv</code>.</p>
{table([["Train", "2,700", "2,700", "539", "5,939"], ["Validation", "300", "300", "60", "660"], ["Test", "800", "800", "200 (with 1–4 boxes each)", "1,800"]], ["Split", "Healthy", "Sick, not TB", "TB", "Total"])}
<p>Images are converted to grayscale and resized to 384 × 384: TB lesions are small and lose detail at the 224 px usual for ImageNet models. TB is only 9% of the training set, so the loss weights each class by inverse frequency. Augmentation runs on the GPU per image: small rotations, shifts and zooms, exposure, contrast and gamma changes, noise and blur. We use no horizontal flips: a mirrored X-ray puts the heart on the wrong side, which never occurs in practice.</p>

<h2>4. Method</h2>
<p>All three models take a grayscale image in [0, 1], normalise it internally, and end in ReLU → global average pooling → one linear layer. For such a network the class activation map M<sub>c</sub> = Σ<sub>k</sub> w<sub>ck</sub> F<sub>k</sub> (F the final feature maps, w the linear weights) averages exactly to the class logit [10], so the deployed ONNX model returns probabilities and 12 × 12 maps in one pass, and the browser can draw the heatmap without gradients.</p>
<ul><li><b>CNN from scratch</b>: five blocks of two 3 × 3 convolutions with batch normalisation, 32→256 channels, 2.4M parameters.</li>
<li><b>DenseNet-121, ImageNet start</b> [7, 9]: the CheXNet architecture, 7.0M parameters.</li>
<li><b>DenseNet-121, chest-X-ray start</b>: the same architecture with a single-channel stem, initialised from TorchXRayVision's "all" weights [8].</li></ul>
{table(hp, ["Setting", "Value"])}
<p><b>Screening rule.</b> A person is referred for a lab test (sputum smear or GeneXpert) when the TB probability is at least a threshold. We fix the threshold at 95% specificity on the 600 non-TB validation X-rays, i.e. about 5% unnecessary referrals, and then read off sensitivity on the test set. The deployed threshold is {thr:.4f}.</p>
<div class="key"><b>A lesson on choosing thresholds.</b> Our first rule picked the threshold that gave 90% sensitivity on the 60 validation TB X-rays. {('On the test set that threshold gave only ' + pct(first_rule['sensitivity']) + ' sensitivity (' + pct(first_rule['specificity']) + ' specificity): with so few positives the estimate is noisy, and the model is more confident on validation images than on test images. ') if first_rule else ''}Anchoring the threshold on 600 negative images instead gives a far more stable estimate.</div>

<h2>5. Results</h2>
{table(results_rows, ["Model", "Params", "Accuracy", "95% CI", "Bal. acc.", "Macro F1", "TB AUC", "95% CI", "Sens.", "Spec.", "Pointing game", "CPU ms"])}
<p class="meta">Test set: 1,800 X-rays. Confidence intervals: 1,000 bootstrap resamples. Sensitivity and specificity for TB vs not-TB at each model's own validation threshold. Pointing game: the hottest point of the TB map lies inside a radiologist's box (random point: {pct(B['pointing_game']['random_hit_rate'])}). CPU ms: ONNX Runtime, one image, 2-core ARM server.</p>
<p>{cmp_sentence} {scratch_sentence}</p>
{choice}
<div class="two"><figure><img src="figures/roc_tb.png"><figcaption>Figure 1. TB-vs-rest ROC curves on the test set; dots mark each model's operating point.</figcaption></figure>
<figure><img src="figures/confusion_matrix.png"><figcaption>Figure 2. Confusion matrix of the deployed model (rows: true class).</figcaption></figure></div>
{table(per_class, ["Class", "Precision", "Recall", "F1", "Test images"])}
<figure><img src="figures/heatmaps_vs_boxes.png"><figcaption>Figure 3. Class activation maps of the deployed model on TB test X-rays it classified as TB, with the radiologists' boxes in green.</figcaption></figure>
<div class="two"><figure><img src="figures/pointing_game.png"><figcaption>Figure 4. Pointing game by model.</figcaption></figure>
<figure><img src="figures/training_curves.png"><figcaption>Figure 5. Training loss and validation balanced accuracy.</figcaption></figure></div>

<h2>6. Error analysis</h2>
<p>At the screening threshold the deployed model misses {len(missed)} of {len(tb_rows)} TB X-rays{(' (predicted ' + missed_txt + ')') if missed else ''}. It detects {by_type['active_tb'][0]} of {by_type['active_tb'][1]} <b>active</b> TB cases ({pct(by_type['active_tb'][0] / max(by_type['active_tb'][1], 1))}) and {by_type['latent_tb'][0]} of {by_type['latent_tb'][1]} <b>latent</b> TB cases ({pct(by_type['latent_tb'][0] / max(by_type['latent_tb'][1], 1))}); latent TB leaves subtler, often calcified traces, so the gap is clinically plausible. It refers {false_ref['healthy']} of {n_cls['healthy']} healthy and {false_ref['sick_but_not_tb']} of {n_cls['sick_but_not_tb']} sick-but-not-TB patients. In the three-class decision the most common confusion is between healthy and sick-but-not-TB ({cm[0][1]} healthy called sick, {cm[1][0]} sick called healthy), which matters less for TB screening than confusions involving TB ({cm[2][0] + cm[2][1]} TB called non-TB by the top class, {cm[0][2] + cm[1][2]} non-TB called TB).</p>

<h2>7. Deployment</h2>
<p>The deployed model is exported to ONNX (opset 17, {B['onnx_mb']} MB) with softmax and CAM inside, and to TorchScript. The website loads it once and runs it in the browser with ONNX Runtime Web (WebAssembly), about a second per X-ray on a laptop; the X-ray never leaves the device, which matters for medical images. The GitHub repository contains the code, the exact splits, all weights with SHA-256 hashes in <code>model_card.json</code>, the executed Colab notebook, and a GitHub Actions workflow that runs <code>predict.py</code> and <code>evaluate.py</code> on 12 bundled labelled X-rays at every push.</p>
<p><code>git clone {links['github']} &amp;&amp; pip install -r requirements.txt &amp;&amp; python evaluate.py</code></p>

<h2>8. Limitations and ethics</h2>
<ul><li>TBX11K was collected in Chinese hospitals; performance on Pakistani X-ray machines and populations is unknown and must be measured before any use.</li>
<li>The model is for adults; WHO's CAD recommendation covers ages 15 and over [2].</li>
<li>Our test set is TBX11K's validation set, so numbers are not directly comparable with the official (hidden-label) test leaderboard.</li>
<li>A heatmap shows where evidence is, not a lesion outline, and a high pointing-game score does not make the model a diagnostician.</li>
<li>LungLens is a teaching and research demo, not a medical device. A positive screen needs a lab test; a negative one does not rule TB out.</li></ul>

<h2>9. Conclusion</h2>
<p>A DenseNet-121 fine-tuned on TBX11K screens chest X-rays for TB with a TB AUC of {TB['auc']:.4f}, separates TB from other lung disease, and points at radiologist-marked lesions far more often than chance, while running privately in a browser. The next steps are a test on X-rays from Pakistani hospitals, calibration of the threshold on that population, and lung segmentation to keep the model's attention inside the lungs.</p>

<h2>References</h2><ol class="refs">{''.join(f'<li>{r}</li>' for r in ref)}</ol>
<p class="meta">This report was drafted with the help of an AI assistant (Claude) from the project's code and results, as the course requires, and checked against them.</p>
</body></html>"""

os.makedirs("docs/report/figures", exist_ok=True)
for f in os.listdir("results/figures"):
    shutil.copy(f"results/figures/{f}", f"docs/report/figures/{f}")
open("docs/report/index.html", "w", encoding="utf-8").write(H)
from htmlpdf import render  # noqa: E402

render("docs/report/index.html", "docs/LungLens_report.pdf")
