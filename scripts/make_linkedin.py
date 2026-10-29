"""Write docs/linkedin_post.md from the results."""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
links = json.load(open("links.json")); S = json.load(open("results/summary.json"))
B = S["models"][S["deployed_model"]]; T, TB = B["test"], B["test"]["tb_vs_rest"]
pct = lambda v, d=1: f"{v * 100:.{d}f}%"
pg = {m: r["pointing_game"]["hit_rate"] for m, r in S["models"].items()}
post = f"""Pakistan had about 670,000 TB cases in 2024, the 5th highest in the world. Chest X-rays are cheap and everywhere; people who can read them for TB are not.

For my Deep Learning semester project I built LungLens: upload a chest X-ray and, in about a second, it says healthy, sick but not TB, or TB, and shows where in the lungs it looked. It runs entirely in the browser, so the X-ray is never uploaded anywhere.

What I did differently:
• Used TBX11K, not the usual Kaggle TB set, where TB and normal X-rays come from different hospitals (a shortcut a model can learn)
• Three classes, because the hard part is telling TB from other lung disease
• Compared a CNN from scratch with DenseNet-121 starting from ImageNet vs from chest X-rays
• Checked the heatmaps against boxes drawn by radiologists: the hottest point lands inside a box in {pct(B['pointing_game']['hit_rate'], 0)} of TB X-rays (a random point: {pct(B['pointing_game']['random_hit_rate'], 0)})

Results on 1,800 test X-rays it never saw: {pct(T['accuracy'])} accuracy, TB AUC {TB['auc']:.4f}, {pct(TB['sensitivity'])} sensitivity at {pct(TB['specificity'])} specificity.

Try it: {links.get('site') or '[Vercel link]'}
Code, weights and data splits: {links['github']}
Walkthrough video: {links.get('youtube') or '[YouTube link]'}

Thanks to Dr. Munir Ahmad, and to Liu et al. for the TBX11K dataset.
It's a research demo, not a medical device.

#DeepLearning #ComputerVision #HealthcareAI #Tuberculosis #Pakistan #SuperiorUniversity
"""
os.makedirs("docs", exist_ok=True)
open("docs/linkedin_post.md", "w").write(post)
print(post)
