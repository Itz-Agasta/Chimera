"""
Mechanism theory tests on Kaggle T4 x2 (experiments/common/vae_theory.py), all four pairs, both Nightshade pools:
A = first-order displacement theory (gain vs distance, and the eps-sweep prediction), B = judge the decoded poison
latents (what the fine-tune is taught). SD-2 on cuda:0, SD3 on cuda:1; each runs a 4-image warm-up first.
Inputs: itzagasta/chimera-code, itzagasta/chimera-mech, itzagasta/chimera-secrets.
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
DATA = os.path.dirname(os.path.dirname(find("/kaggle/input/**/chimera-mech/**/chimera_coco/manifest.json")))
token = open(find("/kaggle/input/**/hf_token.txt")).read().strip()
ENV = {**os.environ, "HF_TOKEN": token, "PYTHONPATH": CODE, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
print("code", CODE, "data", DATA, flush=True)
summary = {"code": CODE, "data": DATA, "jobs": {}}


import threading

LOCK = threading.Lock()


def save():
    # Two build lanes finish jobs concurrently; serialise the dump so neither sees the dict mid-update.
    with LOCK:
        json.dump(summary, open(f"{W}/summary.json", "w"), indent=2)


def launch(name, args, gpu):
    log = open(f"{W}/{name}.log", "w")
    env = {**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return name, subprocess.Popen([sys.executable, *args], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT), \
        time.time()


def wait(jobs):
    for name, proc, t0 in jobs:
        rc = proc.wait()
        tail = open(f"{W}/{name}.log").read().splitlines()[-25:]
        with LOCK:
            summary["jobs"][name] = {"rc": rc, "seconds": round(time.time() - t0), "tail": tail}
        print(f"===== {name} rc={rc} {time.time() - t0:.0f}s\n" + "\n".join(tail), flush=True)
        save()



def lane(model, gpu):
    wait([launch(f"warm_{model}", [f"{CODE}/vae_theory.py", "--model", model, "--pairs", "cat->dog", "--limit", "4",
                                   "--data-root", DATA, "--out", f"{W}/vae_theory_warmup_{model}.json"], gpu)])
    if summary["jobs"][f"warm_{model}"]["rc"] == 0:
        wait([launch(f"full_{model}", [f"{CODE}/vae_theory.py", "--model", model, "--data-root", DATA,
                                       "--out", f"{W}/vae_theory_{model}.json"], gpu)])


import threading

# v2: SD-2 completed in v1 (outputs saved locally); rerun SD3 only, after re-uploading the one poison PNG that
# the v1 data upload had truncated.
lanes = [threading.Thread(target=lane, args=("sd3_medium", 0))]
for t in lanes:
    t.start()
for t in lanes:
    t.join()
