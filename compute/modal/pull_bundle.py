"""
Fill the chimera-data volume from the published bundle, for a Modal workspace that has never built it.

The bundle (agasta/chimera-data) holds exactly what the volume needs at the same paths -- chimera_coco/,
embeds/, latents/, anchors/, poisons/ -- so this is one download into /data, the same call bootstrap.sh makes
on a rented box. Rerunning it only fetches what is missing.

  modal run compute/modal/pull_bundle.py --repo agasta/chimera-data
"""
import modal

app = modal.App("chimera-pull-bundle")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==2.3.0")
    .env({"HF_HOME": "/tmp/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
)
data = modal.Volume.from_name("chimera-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": data}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=3600, cpu=8, memory=16384)
def pull(repo: str) -> dict:
    import os
    import pathlib
    import time

    from huggingface_hub import snapshot_download

    t0 = time.time()
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir="/data", token=os.environ["HF_TOKEN"])
    data.commit()
    parts = {}
    for p in pathlib.Path("/data").iterdir():
        if p.name.startswith("."):
            continue
        files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
        parts[p.name] = {"files": len(files), "gb": round(sum(f.stat().st_size for f in files) / 1e9, 2)}
    return {"seconds": round(time.time() - t0), "parts": parts}


@app.local_entrypoint()
def main(repo: str = "agasta/chimera-data"):
    import json

    print(json.dumps(pull.remote(repo), indent=2))
