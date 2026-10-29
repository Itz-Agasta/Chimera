"""
Phase 0.5 pilot on Kaggle T4 x2: does each job run, how long does it take, and are the numbers sane, before the
full P0.5a / P0.5c runs are launched.

  A. grad_align.py, 4 slots x 2 draws, SD3 on cuda:0 and SD-2 on cuda:1 in parallel.
     Checks: determinism_cos ~ 1, finite latents, s/backward, peak memory.
  B. poison_gen.py, nightshade_sd3 cat->dog at eps 0.04 on the Gate B cell's first 4 ids, once in fp16 (what a
     T4 falls back to) and once in fp32, then poison_diag.py on both and on the existing L40S (bf16) pool for the
     same 4 ids. Checks: no NaN, and whether the T4 can reproduce the L40S pool's `closed` -- if it cannot, the
     eps sweep has to be built at one precision throughout.

Inputs: itzagasta/chimera-code (experiments/common), itzagasta/chimera-p05 (the subset cut by
compute/modal/export_kaggle_subset.py), itzagasta/chimera-secrets (hf_token.txt).
Outputs in /kaggle/working: pilot_summary.json plus every job's own json and log.
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
    json.dump(summary, open(f"{W}/pilot_summary.json", "w"), indent=2)


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


# ---- A. gradient alignment pilot
wait([
    launch("ga_sd3", [f"{CODE}/grad_align.py", "--model", "sd3_medium",
                      "--pools", "nightshade_sd3,chimera_pred,chimera_anchor", "--data-root", DATA,
                      "--poison-root", f"{DATA}/poisons", "--limit", "4", "--draws", "2",
                      "--work", f"{W}/ga_work", "--out", f"{W}/grad_align_pilot_sd3.json"], 0),
    launch("ga_sd2", [f"{CODE}/grad_align.py", "--model", "sd2_base", "--pools", "nightshade_sd2,nightshade_sd3",
                      "--data-root", DATA, "--poison-root", f"{DATA}/poisons", "--limit", "4", "--draws", "2",
                      "--work", f"{W}/ga_work", "--out", f"{W}/grad_align_pilot_sd2.json"], 1),
])

# ---- B. eps 0.04 reproduction at fp16 and fp32
gen = ["--attack", "nightshade_sd3", "--pair", "cat->dog", "--data-root", DATA, "--anchors", f"{DATA}/anchors",
       "--eps", "0.04", "--select-seed", "0", "--select-count", "100", "--limit", "4"]
wait([
    launch("gen_fp16", [f"{CODE}/poison_gen.py", *gen, "--poison-root", f"{W}/poisons_fp16"], 0),
    launch("gen_fp32", [f"{CODE}/poison_gen.py", *gen, "--fp32", "--poison-root", f"{W}/poisons_fp32"], 1),
])
diag = ["--attack", "nightshade_sd3", "--pair", "cat->dog", "--manifest", f"{DATA}/chimera_coco/manifest.json",
        "--images-root", f"{DATA}/chimera_coco/images", "--anchors", f"{DATA}/anchors", "--limit", "4",
        "--out", f"{W}/poison_diag_pilot.json"]
for tag, root in (("l40s_bf16", f"{DATA}/poisons"), ("t4_fp16", f"{W}/poisons_fp16"),
                  ("t4_fp32", f"{W}/poisons_fp32")):
    wait([launch(f"diag_{tag}", [f"{CODE}/poison_diag.py", *diag, "--poison-root", root, "--tag", tag], 0)])

# Per-image comparison against the L40S pool's own optimiser record, for the same ids.
ref = {r["id"]: r for r in json.load(open(f"{DATA}/poisons/nightshade_sd3/cat-dog/poison_manifest.json"))["images"]}
for tag in ("fp16", "fp32"):
    path = f"{W}/poisons_{tag}/nightshade_sd3/cat-dog/poison_manifest.json"
    if os.path.exists(path):
        got = json.load(open(path))
        summary[f"gen_{tag}"] = {
            "compute_dtype": got.get("compute_dtype"), "seconds": got.get("seconds"), "peak_gb": got.get("peak_gb"),
            "per_image": [{"id": r["id"], "loss_q": r["loss_quantised"], "ref_loss_q": ref[r["id"]]["loss_quantised"],
                           "loss0": r["loss_initial"], "ref_loss0": ref[r["id"]]["loss_initial"],
                           "linf": r["linf_255"], "psnr": r["psnr_db"]} for r in got["images"]]}
if os.path.exists(f"{W}/poison_diag_pilot.json"):
    d = json.load(open(f"{W}/poison_diag_pilot.json"))
    summary["diag"] = {k: {"closed": v["scaled"]["closed"]["median"], "cosine": v["scaled"]["cosine"]["median"],
                           "ids": [r["id"] for r in v["images"]]} for k, v in d.items()}
save()
print(json.dumps({k: v for k, v in summary.items() if k != "jobs"}, indent=2))
