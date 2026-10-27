"""
Verify the published bundle and measure how fast a rented box can pull it.

Answers two questions before any money is spent on a GPU: is everything actually in the repo, and what download
throughput should bootstrap.sh expect? Runs on a CPU container whose network is comparable to a datacenter Vast
host, so the measured rate is a fair preview.

  modal run compute/modal/check_bundle.py --repo agasta/chimera-data
"""
import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")
app = modal.App("chimera-check-bundle")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==2.3.0")
    .env({"HF_HOME": "/tmp/hf"})
)


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], timeout=1800, cpu=4, memory=8192)
def check(repo: str, high_performance: bool = False) -> str:
    import os
    import shutil
    import time

    if high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ["HF_TOKEN"])
    files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    info = api.repo_info(repo_id=repo, repo_type="dataset", files_metadata=True)
    by_part, total = {}, 0
    for f in info.siblings:
        part = f.rfilename.split("/")[0]
        size = f.size or 0
        entry = by_part.setdefault(part, {"files": 0, "mb": 0.0})
        entry["files"] += 1
        entry["mb"] += size / 2**20
        total += size
    for entry in by_part.values():
        entry["mb"] = round(entry["mb"], 1)

    out = {"repo": repo, "total_gb": round(total / 2**30, 2), "by_part": by_part,
           "images": sum(1 for f in files if f.startswith("chimera_coco/images/")),
           "has_manifest": "chimera_coco/manifest.json" in files,
           "embed_indexes": [f for f in files if f.endswith("index.json")],
           "latents": [f for f in files if f.startswith("latents/")]}

    # Time one large shard: the per-file rate the whole 15.5 GB pull will run at.
    shard = next((f for f in files if "embeds/sd3_medium/shard_" in f), None)
    if shard:
        t0 = time.time()
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=shard,
                               token=os.environ["HF_TOKEN"], local_dir="/tmp/probe")
        seconds = time.time() - t0
        mb = pathlib.Path(path).stat().st_size / 2**20
        out["download_probe"] = {
            "file": shard, "mb": round(mb, 1), "seconds": round(seconds, 1),
            "mb_per_s": round(mb / seconds, 1),
            "high_performance": bool(high_performance),
            "projected_minutes_for_bundle": round((total / 2**20) / (mb / seconds) / 60, 1),
        }
        shutil.rmtree("/tmp/probe", ignore_errors=True)
    return json.dumps(out)


@app.local_entrypoint()
def main(repo: str = "agasta/chimera-data", high_performance: bool = False):
    out = json.loads(check.remote(repo, high_performance))
    print(json.dumps(out, indent=2))
