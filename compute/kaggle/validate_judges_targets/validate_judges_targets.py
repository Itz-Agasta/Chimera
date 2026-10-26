"""
Positive control for the ASR judges: generate the TARGET concepts (dog, cow, pizza, bird) with clean SD3-Medium and
SD-2 and check that both judges recognise them. Together with the source-concept check (no false alarms), this shows
the judges neither invent nor miss the target concept.

Output: /kaggle/working/judge_validation_targets.json (+ sample images).
"""
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "diffusers==0.42.0", "transformers", "accelerate", "sentencepiece", "protobuf"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)

import glob
import json
import os

_code = glob.glob("/kaggle/input/**/judge_check.py", recursive=True)
if not _code:
    raise SystemExit("chimera-code dataset not attached (no judge_check.py under /kaggle/input)")
sys.path.insert(0, os.path.dirname(_code[0]))

from huggingface_hub import login

from concepts import TARGETS
from judge import JUDGE_MODELS, load_judge

login(open(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)[0]).read().strip())
print("targets under test:", TARGETS, flush=True)

# Preflight: judges must work before GPU time goes into generation.
import torch
from PIL import Image

for _j in JUDGE_MODELS:
    _judge = load_judge(_j, device="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    print("preflight", _j, _judge.classify([Image.new("RGB", (512, 512), (200, 30, 30))])[0]["label"], flush=True)
    del _judge
    torch.cuda.empty_cache()

from judge_check import run

res = run(["sd3_medium", "sd2_base"], TARGETS, img_dir="/kaggle/working/samples")
with open("/kaggle/working/judge_validation_targets.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res["models"], indent=2))
