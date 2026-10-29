"""Write README.md from the results (run after report.py)."""
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
links = json.load(open("links.json"))
S = json.load(open("results/summary.json"))
card = json.load(open("model_card.json"))
runs, best = S["models"], S["deployed_model"]
B = runs[best]
T, TB = B["test"], B["test"]["tb_vs_rest"]
pct = lambda v, d=1: f"{v * 100:.{d}f}%"
gh = links["github"]
slug = gh.replace("https://github.com/", "")
example = subprocess.run([sys.executable, "predict.py", "samples/tb_1.png"], capture_output=True, text=True, check=True).stdout.strip()
example = json.dumps({k: v for k, v in json.loads(example).items() if k != "file"}, indent=1)
L = lambda k, text: f"[{text}]({links[k]})" if links.get(k) else f"{text} (coming soon)"

md = f"""# LungLens: TB screening from chest X-rays

[![test](https://github.com/{slug}/actions/workflows/test.yml/badge.svg)](https://github.com/{slug}/actions/workflows/test.yml)
![license](https://img.shields.io/badge/code-MIT-blue) ![data](https://img.shields.io/badge/data-TBX11K%20CC%20BY%204.0-teal)

**Upload a chest X-ray → healthy / sick but not TB / TB, plus a heatmap of where the model looked. Runs in the browser; the X-ray never leaves your device.**

{L('site', 'Live demo')} · {L('report', 'Report (PDF)')} · {L('slides', 'Slides')} · {L('youtube', 'Video')} · {L('linkedin', 'LinkedIn post')} · [Dataset (Kaggle)]({links['dataset']}) · [Model card](model_card.json)

Deep Learning semester project, BSCS-7A, Superior University, Lahore. Instructor: Dr. Munir Ahmad. Author: {links['author']}.

![Heatmaps vs radiologists' boxes](results/figures/heatmaps_vs_boxes.png)

## Results (1,800 held-out test X-rays)

{open('results/results.md').read()}
- Deployed: **{card['architecture']}**, chosen on validation balanced accuracy.
- Screening rule: refer for a lab test if `probs[tb] >= {TB['threshold']:.4f}` (95% specificity on validation non-TB X-rays). At that threshold: **{pct(TB['sensitivity'])} sensitivity, {pct(TB['specificity'])} specificity** (WHO triage minimum: 90% / 70%).
- Pointing game: the TB heatmap's hottest point lies inside a radiologist's box in **{pct(B['pointing_game']['hit_rate'])}** of TB test X-rays (random point: {pct(B['pointing_game']['random_hit_rate'])}).
- 95% bootstrap CIs: accuracy {pct(T['accuracy_95ci'][0])}–{pct(T['accuracy_95ci'][1])}, TB AUC {T['tb_auc_95ci'][0]:.4f}–{T['tb_auc_95ci'][1]:.4f}.

## Test it in 3 commands

```bash
git clone {gh} && cd lunglens
pip install -r requirements.txt          # or just: pip install numpy pillow onnxruntime
python evaluate.py                       # 12 bundled, labelled test X-rays in samples/
```

```bash
python predict.py samples/tb_1.png --heatmaps out/    # JSON + heatmap overlay
python evaluate.py --data /path/to/tbx11k-simplified  # the full 1,800-image test split
```

`predict.py` prints, per X-ray (real output for `samples/tb_1.png`):

```json
{example}
```

## Weights

| File | What | Load with |
|---|---|---|
| `weights/lunglens.onnx` | deployed model, outputs `probs` [1,3] and `cam` [1,3,12,12] | `onnxruntime.InferenceSession(...)` |
| `weights/lunglens_torchscript.pt` | same model, TorchScript | `torch.jit.load(...)` |
| `weights/<model>.pt` | PyTorch checkpoints of all three models | `lunglens.models.load_checkpoint(...)` |

Input: one grayscale X-ray, resized to 384×384, float32 in [0, 1], shape `[1, 1, 384, 384]`; normalisation is inside the model. Classes: `{card['classes']}`. SHA-256 of every file: [model_card.json](model_card.json).

## Data

[TBX11K](https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified) (Liu et al., CVPR 2020, CC BY 4.0): 8,399 labelled 512×512 X-rays, TB regions boxed by radiologists. Test = TBX11K's labelled validation set (800 / 800 / 200); its training set is split 90/10 into train / validation (stratified, seed 42). Exact files: [`data/splits.csv`](data/splits.csv).

## Method

Three models with identical data, augmentation (GPU; rotation, shift, zoom, exposure, noise, blur; **no flips**), class-weighted cross-entropy, AdamW + cosine schedule:

1. `cnn_scratch`: 5-block CNN from random weights (2.4M params)
2. `densenet121_imagenet`: DenseNet-121 from ImageNet weights
3. `densenet121_cxr`: the same DenseNet-121 from TorchXRayVision chest-X-ray weights

Every model ends in global average pooling + one linear layer, so the class activation map comes out of the same forward pass ([`lunglens/models.py`](lunglens/models.py)); the website draws it without gradients.

## Reproduce

Training ran on one NVIDIA L4 in Google Colab: [`notebooks/LungLens_train.ipynb`](notebooks/LungLens_train.ipynb) (all outputs kept).

```bash
python train.py --data /path/to/tbx11k-simplified --model densenet121_cxr --out outputs   # ~17 min on an L4
python rescore.py --data /path/to/tbx11k-simplified --out outputs
python report.py --runs outputs --data /path/to/tbx11k-simplified                          # weights/, figures, model card
```

## Repository

```
predict.py  evaluate.py          inference (numpy + pillow + onnxruntime only)
train.py  rescore.py  report.py  training, re-scoring, export + figures
lunglens/                        models, data, augmentation, metrics
data/splits.csv                  exact train / val / test files
weights/                         ONNX, TorchScript, PyTorch checkpoints
samples/                         12 labelled test X-rays (CC BY 4.0, TBX11K)
results/                         metrics, per-image predictions, figures
docs/                            report (PDF + HTML), slides (PPTX, PDF), voiceover script
site/  scripts/build_site.py     the website (deployed on Vercel from web/)
notebooks/                       executed Colab training notebook
```

## Limits

Trained and tested on TBX11K (Chinese hospitals); not validated on Pakistani X-rays. Adults only. A research and teaching demo, **not a medical device**: a positive screen needs a lab test, a negative one does not rule TB out.

## Credits

TBX11K: Liu, Wu, Ban, Wang, Cheng, *Rethinking Computer-aided Tuberculosis Diagnosis*, CVPR 2020 (CC BY 4.0). Chest X-ray weights: TorchXRayVision (Cohen et al., MIDL 2022). Code: MIT licence.
"""
open("README.md", "w").write(md)
print("wrote README.md")
