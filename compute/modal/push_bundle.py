"""
Push the data bundle from the chimera-data volume to a private HuggingFace dataset repo, cloud to cloud.

Any rented box then pulls the bundle from HF's CDN instead of from a laptop, and a replacement box after a crash
pulls the same bytes. What travels:

  chimera_coco/images/<id>.jpg        5200 COCO crops at 512 px   (~600 MB)
  chimera_coco/manifest.json          the dataset splits
  embeds/<model>/                     cached caption embeddings   (~14.5 GB)
  latents/<model>_<res>.safetensors   VAE latents for the clean corpus (~0.4 GB)

Not pushed: model weights (pulled from their own HF repos) and checkpoints (transient, per run).

  modal run --detach compute/modal/push_bundle.py --repo chimera-data
  modal run compute/modal/push_bundle.py --repo chimera-data --dry-run
  modal run --detach compute/modal/push_bundle.py --repo chimera-data --only poisons,anchors
"""
import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")
app = modal.App("chimera-push-bundle")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==2.3.0")
    .env({"HF_HOME": "/cache"})   # transfers go through hf-xet, which is the default
)
cache = modal.Volume.from_name("chimera-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("chimera-data", create_if_missing=True)

# (path on the volume, path in the repo). Kept explicit so nothing unexpected is published.
#
# Ordered newest-first, because each part is its own commit and a push can be cut short (a workspace budget
# runs out, a container is preempted). Whatever is most expensive to regenerate goes first: the poisons cost
# ~12 GPU-hours, the embeddings a re-encode, and the COCO crops are a cheap re-export. Parts already on the
# repo cost almost nothing to re-send, since the hub dedupes by content hash.
PARTS = [
    # The optimised poisons and their anchors. The repo must stay private: the paper's ethics statement
    # commits to not releasing perturbations, and these are exactly that.
    ("poisons", "poisons"),
    ("anchors", "anchors"),
    ("embeds", "embeds"),
    ("latents", "latents"),
    ("chimera_coco/manifest.json", "chimera_coco/manifest.json"),
    ("chimera_coco/export_stats.json", "chimera_coco/export_stats.json"),
    ("chimera_coco/images", "chimera_coco/images"),
]


@app.function(image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=4 * 3600, cpu=4, memory=8192)
def push(repo: str, dry_run: bool = False, only: str = "") -> str:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    if "/" not in repo:
        # Resolve the namespace from the token rather than guessing the account name.
        repo = f"{api.whoami()['name']}/{repo}"
        print(f"repo resolved to {repo}", flush=True)

    parts = PARTS
    if only:
        wanted = [name.strip() for name in only.split(",") if name.strip()]
        unknown = [name for name in wanted if not any(src == name or src.startswith(name + "/")
                                                     for src, _ in PARTS)]
        if unknown:
            raise SystemExit(f"--only names nothing in the bundle: {unknown} "
                             f"(have {[src for src, _ in PARTS]})")
        parts = [(src, dst) for src, dst in PARTS
                 if any(src == name or src.startswith(name + "/") for name in wanted)]

    plan, total = [], 0
    for src, dst in parts:
        path = pathlib.Path("/data") / src
        if not path.exists():
            plan.append({"src": src, "status": "missing"})
            continue
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file()]
            size = sum(p.stat().st_size for p in files)
            plan.append({"src": src, "dst": dst, "files": len(files), "mb": round(size / 2**20, 1)})
        else:
            size = path.stat().st_size
            plan.append({"src": src, "dst": dst, "files": 1, "mb": round(size / 2**20, 1)})
        total += size
    summary = {"repo": repo, "parts": plan, "total_gb": round(total / 2**30, 2)}
    print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return json.dumps({**summary, "pushed": False})

    # Private: the corpus is derived from COCO images and the embeddings are large; nothing here is meant to be public.
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    for part in plan:
        if part.get("status") == "missing":
            print(f"skip {part['src']}: not on the volume", flush=True)
            continue
        path = pathlib.Path("/data") / part["src"]
        print(f"uploading {part['src']} ({part['mb']} MB)...", flush=True)
        if path.is_dir():
            api.upload_folder(folder_path=str(path), path_in_repo=part["dst"], repo_id=repo,
                              repo_type="dataset", commit_message=f"bundle: {part['dst']}")
        else:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=part["dst"], repo_id=repo,
                            repo_type="dataset", commit_message=f"bundle: {part['dst']}")
    return json.dumps({**summary, "pushed": True})


@app.local_entrypoint()
def main(repo: str = "", dry_run: bool = False, only: str = ""):
    if not repo:
        raise SystemExit("pass --repo <name> (the namespace is taken from the HF token) "
                         "or --repo <user>/<name>")
    print(json.dumps(json.loads(push.remote(repo, dry_run, only)), indent=2))
