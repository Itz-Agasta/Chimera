"""
P0.5a full run on Kaggle T4 x2: gradient alignment of the existing cat->dog pools, 100 slots x 8 draws.
SD3 (nightshade_sd3, chimera_pred, chimera_anchor) on cuda:0, SD-2 (nightshade_sd2, nightshade_sd3) on cuda:1.
Headline: grad_closed excess over the random null (notes/failures.md C8). Pilot: compute/kaggle/p05_pilot/.
Outputs in /kaggle/working: grad_align_sd3.json, grad_align_sd2.json, sketches, logs, summary.json.
"""
import glob
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.42.0", "transformers", "accelerate",
                "safetensors", "sentencepiece", "protobuf"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)

W = "/kaggle/working"


def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


CODE = os.path.dirname(find("/kaggle/input/**/grad_align.py"))
DATA = os.path.dirname(os.path.dirname(find("/kaggle/input/**/chimera_coco/manifest.json")))
token = open(find("/kaggle/input/**/hf_token.txt")).read().strip()
ENV = {**os.environ, "HF_TOKEN": token, "PYTHONPATH": CODE, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
print("code", CODE, "data", DATA, flush=True)
summary = {"code": CODE, "data": DATA, "jobs": {}}


def save():
    json.dump(summary, open(f"{W}/summary.json", "w"), indent=2)


def launch(name, args, gpu):
    log = open(f"{W}/{name}.log", "w")
    env = {**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return name, subprocess.Popen([sys.executable, *args], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT), \
        time.time()


def wait(jobs):
    for name, proc, t0 in jobs:
        rc = proc.wait()
        summary["jobs"][name] = {"rc": rc, "seconds": round(time.time() - t0)}
        tail = open(f"{W}/{name}.log").read().splitlines()[-25:]
        summary["jobs"][name]["tail"] = tail
        print(f"===== {name} rc={rc} {time.time() - t0:.0f}s\n" + "\n".join(tail), flush=True)
        save()



wait([
    launch("ga_sd3", [f"{CODE}/grad_align.py", "--model", "sd3_medium",
                      "--pools", "nightshade_sd3,chimera_pred,chimera_anchor", "--data-root", DATA,
                      "--poison-root", f"{DATA}/poisons", "--draws", "8",
                      "--work", W, "--out", f"{W}/grad_align_sd3.json"], 0),
    launch("ga_sd2", [f"{CODE}/grad_align.py", "--model", "sd2_base", "--pools", "nightshade_sd2,nightshade_sd3",
                      "--data-root", DATA, "--poison-root", f"{DATA}/poisons", "--draws", "8",
                      "--work", W, "--out", f"{W}/grad_align_sd2.json"], 1),
])
