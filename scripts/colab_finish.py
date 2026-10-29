"""Last Colab step: re-score the trained models with the final screening rule, save a copy of the
executed notebook next to the results, and zip everything (run with %run inside the notebook)."""
import json
import subprocess
import sys

from google.colab import _message

OUT = "/content/drive/MyDrive/Deep Learning Projects/LungLens/outputs"
r = subprocess.run([sys.executable, "rescore.py", "--data", "/content/tbx11k", "--out", OUT], capture_output=True, text=True)
print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if "warn" not in l.lower()))
nb = _message.blocking_request("get_ipynb", timeout_sec=120)["ipynb"]
json.dump(nb, open(f"{OUT}/LungLens_train.ipynb", "w"))
r = subprocess.run(f'cd "{OUT}/.." && rm -f lunglens_outputs.zip && zip -qr lunglens_outputs.zip outputs && ls -la', shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr)
print("DONE")
