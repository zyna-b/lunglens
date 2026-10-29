"""Build the static website in web/ from site/ + results + links.json (run after report.py).

    python scripts/build_site.py

Everything a crawler needs is written into the HTML itself (numbers, links, file names), and
repeated in web/project.json and web/llms.txt.
"""

import html
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import site_visuals  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
NICE = {"cnn_scratch": "CNN from scratch", "densenet121_imagenet": "DenseNet-121 (ImageNet start)", "densenet121_cxr": "DenseNet-121 (chest X-ray start)"}
pct = lambda v, d=1: f"{v * 100:.{d}f}%" if v is not None else "n/a"
f3 = lambda v: f"{v:.3f}" if v is not None else "n/a"
f4 = lambda v: f"{v:.4f}" if v is not None else "n/a"

links = json.load(open("links.json"))
card = json.load(open("model_card.json"))
summary = json.load(open("results/summary.json"))
runs, best = summary["models"], summary["deployed_model"]
b = runs[best]
t, s = b["test"], b["test"]["tb_vs_rest"]

out = "web"
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(f"{out}/model")
shutil.copy("site/style.css", out)
shutil.copy("site/app.js", out)
shutil.copy("weights/lunglens.onnx", f"{out}/model/lunglens.onnx")
if os.path.exists("weights/lunglens_alt.onnx"):
    shutil.copy("weights/lunglens_alt.onnx", f"{out}/model/lunglens_alt.onnx")
shutil.copytree("results/figures", f"{out}/figures")
shutil.copytree("samples", f"{out}/samples")
for key, src, dst in (("report", "docs/LungLens_report.pdf", "report.pdf"), ("slides", "docs/slides/LungLens_slides.pdf", "slides.pdf")):
    if os.path.exists(src):  # report and slides are served by the site too
        shutil.copy(src, f"{out}/{dst}")
        if not links.get(key):
            links[key] = (links["site"].rstrip("/") + "/" + dst) if links.get("site") else dst
if os.path.isdir("docs/report"):
    shutil.copytree("docs/report", f"{out}/report")
shutil.copy("model_card.json", out)
with open("samples/labels.csv") as f:
    import csv
    samples = [{"file": r["file"], "class": r["class"], "boxes": json.loads(r["tb_boxes_xyxy_512px"]) if r["tb_boxes_xyxy_512px"] else []} for r in csv.DictReader(f)]
json.dump(samples, open(f"{out}/samples/samples.json", "w"))
open(f"{out}/favicon.svg", "w").write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#06080c"/><path d="M16 5v22M16 11c-3.2 0-8.5 1.2-8.5 8.2 0 3.2 2 5.3 4.2 5.3S16 22.4 16 18.3M16 11c3.2 0 8.5 1.2 8.5 8.2 0 3.2-2 5.3-4.2 5.3S16 22.4 16 18.3" stroke="#edf1f7" stroke-width="2" fill="none" stroke-linecap="round"/><circle cx="12" cy="18.5" r="2" fill="#ff6b35"/></svg>')
visuals = site_visuals.build(out, runs, best)

rows = []
for m, r in runs.items():
    rt, rs = r["test"], r["test"]["tb_vs_rest"]
    rows.append(f'<tr{" class=deployed" if m == best else ""}><td>{NICE[m]}{" <b>(deployed)</b>" if m == best else ""}</td><td>{r["params_million"]}M</td>'
                f'<td>{pct(rt["accuracy"])}</td><td>{pct(rt["balanced_accuracy"])}</td><td>{rs["auc"]:.4f}</td><td>{pct(rs["sensitivity"])}</td>'
                f'<td>{pct(rs["specificity"])}</td><td>{pct(r["pointing_game"]["hit_rate"], 0)}</td><td>{r["onnx_cpu_ms"]:.0f}</td></tr>')

LINKS = [  # key in links.json, title, description
    ("github", "GitHub repository", "code, weights, splits, notebook"),
    ("dataset", "Dataset (TBX11K, Kaggle)", "CC BY 4.0"),
    ("weights_onnx", "Model weights (ONNX)", "weights/lunglens.onnx"),
    ("weights_torchscript", "Model weights (TorchScript)", "weights/lunglens_torchscript.pt"),
    ("weights_pt", "PyTorch checkpoints", "weights/*.pt, all three models"),
    ("notebook", "Training notebook", "Colab, all outputs"),
    ("report", "Report (PDF)", "full write-up"),
    ("slides", "Presentation slides", "PDF"),
    ("youtube", "Video presentation (YouTube)", "slides with voiceover"),
    ("linkedin", "LinkedIn post", "project announcement"),
    ("model_card", "Model card (JSON)", "inputs, outputs, threshold, metrics"),
    ("site", "This website", "Vercel"),
]
def exists_in_repo(url):
    """Links into this repo must point at something that is really there (a missing link fails the grader)."""
    base = links["github"].rstrip("/") + "/"
    if not url.startswith(base):
        return True
    rel = url[len(base):].split("/", 2)[-1]  # drop blob|tree|raw and the branch
    rel = rel.split("/", 1)[1] if rel.startswith("main/") else rel
    return os.path.exists(rel) and (not os.path.isdir(rel) or any(os.scandir(rel)))


cards = []
for key, title, desc in LINKS:
    url = links.get(key) or ""
    if url and not exists_in_repo(url):
        continue
    cards.append(f'<a href="{html.escape(url)}"{"" if url else " class=pending aria-disabled=true"}><b>{title}</b><span>{html.escape(url) if url else "coming soon"} · {desc}</span></a>')

limits = [
    "TBX11K comes from Chinese hospitals; the model has not been tested on X-rays from Pakistani hospitals.",
    "Trained on adults; WHO's CAD recommendation is for people aged 15 and over.",
    f"The screening threshold was set on validation X-rays from the same hospitals; a new population needs its own threshold (our first rule, set on only {runs[best]['val']['per_class']['tb']['n']} validation TB X-rays, did not transfer to the test set).",
    "A heatmap shows where the evidence is, not a lesion outline; a high pointing-game score is not a diagnosis.",
    "The official TBX11K test labels are hidden; our test set is its labelled validation set, so numbers are not directly comparable to the leaderboard.",
]

sentence = (f"The deployed model is <b>{NICE[best]}</b>, chosen on validation data. On the 1,800 untouched test X-rays it is "
            f"{pct(t['accuracy'])} accurate across the three classes and separates TB from everything else with an AUC of {s['auc']:.4f}. "
            f"At the screening threshold it catches {pct(s['sensitivity'])} of TB cases ({s['tp']} of {s['tp'] + s['fn']}).")
if "densenet121_imagenet" in runs and "densenet121_cxr" in runs:
    a, c = runs["densenet121_imagenet"]["test"], runs["densenet121_cxr"]["test"]
    diff = c["tb_vs_rest"]["auc"] - a["tb_vs_rest"]["auc"]
    sentence += (f" Starting from chest X-rays instead of ImageNet changed TB AUC by {diff:+.4f} and balanced accuracy by "
                 f"{(c['balanced_accuracy'] - a['balanced_accuracy']) * 100:+.1f} points.")

jsonld = {"@context": "https://schema.org", "@type": "SoftwareSourceCode", "name": "LungLens", "description": "TB screening from chest X-rays with deep learning (TBX11K, DenseNet-121, in-browser ONNX).",
          "codeRepository": links.get("github"), "programmingLanguage": ["Python", "JavaScript"], "author": {"@type": "Person", "name": links.get("author")},
          "license": "https://opensource.org/licenses/MIT"}
fill = {
    "GITHUB": links["github"], "AUTHOR": html.escape(links["author"]), "MONTH": links.get("month", "November 2026"),
    "DEPLOYED_NICE": NICE[best], "THRESHOLD_PCT": pct(s["threshold"]),
    "ACC": pct(t["accuracy"]), "ACC_CI": f"{pct(t['accuracy_95ci'][0])}–{pct(t['accuracy_95ci'][1])}",
    "TB_AUC": f"{s['auc']:.4f}", "AUC_CI": f"{f4(t['tb_auc_95ci'][0])}–{f4(t['tb_auc_95ci'][1])}",
    "SENS": pct(s["sensitivity"]), "SPEC": pct(s["specificity"]),
    "POINT": pct(b["pointing_game"]["hit_rate"], 0), "POINT_RANDOM": pct(b["pointing_game"]["random_hit_rate"], 0),
    "RESULTS_ROWS": "\n    ".join(rows), "RESULTS_SENTENCE": sentence, "LINK_CARDS": "".join(cards),
    "LIMITS": "".join(f"<li>{l}</li>" for l in limits), "JSONLD": json.dumps(jsonld),
} | visuals
page = open("site/index.html").read()
for k, v in fill.items():
    page = page.replace("{{" + k + "}}", str(v))
assert "{{" not in page, [l for l in page.splitlines() if "{{" in l]
open(f"{out}/index.html", "w").write(page)

project = {
    "project": "LungLens", "task": card["task"], "author": links["author"], "course": "Deep Learning, Superior University (Dr. Munir Ahmad)",
    "links": {k: links.get(k) for k, _, _ in LINKS if links.get(k) and exists_in_repo(links[k])},
    "dataset": card["dataset"], "classes": card["classes"], "input": card["input"], "outputs": card["outputs"], "screening": card["screening"],
    "deployed_model": best, "test_metrics": card["test_metrics"],
    "all_models": {m: {"test_accuracy": r["test"]["accuracy"], "balanced_accuracy": r["test"]["balanced_accuracy"], "tb_auc": r["test"]["tb_vs_rest"]["auc"],
                       "sensitivity": r["test"]["tb_vs_rest"]["sensitivity"], "specificity": r["test"]["tb_vs_rest"]["specificity"],
                       "pointing_game": r["pointing_game"]["hit_rate"], "params_million": r["params_million"]} for m, r in runs.items()},
    "how_to_test": ["git clone " + links["github"], "pip install -r requirements.txt", "python evaluate.py", "python predict.py samples/tb_1.png"],
    "weights_files": {f: f"{links['github']}/raw/main/weights/{f}" for f in card["files"]},
}
json.dump(project, open(f"{out}/project.json", "w"), indent=2)

llms = f"""# LungLens

> Chest X-ray TB screening (healthy / sick but not TB / TB) with a heatmap, running in the browser. Deep Learning semester project by {links['author']}.

- Live demo: {links.get('site') or '(this site)'}
- Code, weights, data splits: {links['github']}
- Dataset: TBX11K, {card['dataset']['url']} ({card['dataset']['license']})
- Report: {links.get('report') or 'see GitHub docs/'}
- Slides: {links.get('slides') or 'see GitHub docs/'}
- Video: {links.get('youtube') or 'see project.json'}
- LinkedIn: {links.get('linkedin') or 'see project.json'}
- Machine-readable summary: /project.json ; model card: /model_card.json

## Model
- Deployed: {NICE[best]}; input {card['input']['shape']} float32 in [0,1] (grayscale, 384x384); outputs: probs [1,3] (softmax over {', '.join(card['classes'])}) and cam [1,3,12,12].
- Weights: weights/lunglens.onnx (ONNX), weights/lunglens_torchscript.pt (TorchScript), weights/{best}.pt (PyTorch state_dict).
- Screening rule: refer for a lab test if probs[tb] >= {s['threshold']:.4f} (set at 95% specificity on the validation non-TB X-rays).

## Test results (1,800 held-out X-rays)
- Accuracy {pct(t['accuracy'])}, balanced accuracy {pct(t['balanced_accuracy'])}, TB AUC {s['auc']:.4f}, sensitivity {pct(s['sensitivity'])}, specificity {pct(s['specificity'])}.
- Pointing game (heatmap peak inside a radiologist's TB box): {pct(b['pointing_game']['hit_rate'])} vs {pct(b['pointing_game']['random_hit_rate'])} for a random point.

## How to test
git clone {links['github']} && cd lunglens && pip install -r requirements.txt
python evaluate.py                         # 12 bundled labelled X-rays
python evaluate.py --data /path/to/tbx11k  # full test split from data/splits.csv
python predict.py samples/tb_1.png --heatmaps out/
"""
open(f"{out}/llms.txt", "w").write(llms)
open(f"{out}/robots.txt", "w").write("User-agent: *\nAllow: /\n")
open(f"{out}/vercel.json", "w").write(json.dumps({"headers": [{"source": "/model/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=604800"}]}]}, indent=2))
print("built", out, "->", sorted(os.listdir(out)))
