"""
P0.5c full run on Kaggle T4 x2: nightshade_sd3 cat->dog poisons at eps 0.08, 0.16, 0.32 on the Gate B cell's 100
ids, fp32 (notes/failures.md A9), then poison_diag on each. eps 0.04 is not rebuilt: the pilot reproduced the
existing L40S pool to closed 0.2313 vs 0.2309. cuda:0 builds 0.08 then 0.32, cuda:1 builds 0.16.
Outputs in /kaggle/working: poisons_eps*/, poison_diag_eps.json, logs, summary.json.
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




def build(eps, gpu):
    root = f"{W}/poisons_eps{eps}"
    wait([launch(f"gen_eps{eps}", [f"{CODE}/poison_gen.py", "--attack", "nightshade_sd3", "--pair", "cat->dog",
                                   "--data-root", DATA, "--anchors", f"{DATA}/anchors", "--eps", eps,
                                   "--select-seed", "0", "--select-count", "100", "--fp32",
                                   "--poison-root", root], gpu)])
    # One diag json per eps: two threads merging into one file could race (notes/failures.md A7).
    wait([launch(f"diag_eps{eps}", [f"{CODE}/poison_diag.py", "--attack", "nightshade_sd3", "--pair", "cat->dog",
                                    "--manifest", f"{DATA}/chimera_coco/manifest.json",
                                    "--images-root", f"{DATA}/chimera_coco/images", "--anchors", f"{DATA}/anchors",
                                    "--poison-root", root, "--tag", f"eps{eps}",
                                    "--out", f"{W}/poison_diag_eps{eps}.json"], gpu)])


lanes = [threading.Thread(target=lambda: [build("0.08", 0), build("0.32", 0)]),
         threading.Thread(target=lambda: build("0.16", 1))]
for t in lanes:
    t.start()
for t in lanes:
    t.join()
