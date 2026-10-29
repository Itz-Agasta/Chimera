"""
Cut the slice of the chimera-data volume that the Phase 0.5 Kaggle jobs read, so they run on a free T4 instead
of Modal credit.

P0.5a (gradient alignment) and P0.5c (eps-sweep poisons + latent diagnostic) only touch the cat->dog seed-0
cell: the 100 poisoned COCO ids, their 100 dirty-label dog partners, the four existing poison pools for those
ids, the SD3 dog anchor, and the text embeddings of the 100 captions. That is ~0.5 GB against a 13.7 GB text
cache, so it is cut here, CPU only, and written under /data/kaggle_subset/ with the volume's own layout --
chimera_coco/, poisons/, anchors/, embeds/<model>/ -- so every existing CLI works on it with --data-root.

  MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run compute/modal/export_kaggle_subset.py
  modal volume get chimera-data /kaggle_subset <local dir>
"""
import pathlib
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")

app = modal.App("chimera-kaggle-subset")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "safetensors", "numpy")
    .add_local_dir(ROOT / "experiments" / "common", "/root/common")
)
data = modal.Volume.from_name("chimera-data")

PAIR = "cat->dog"
POOLS = ("nightshade_sd2", "nightshade_sd3", "chimera_pred", "chimera_anchor")
MODELS = ("sd3_medium", "sd2_base")


@app.function(image=image, volumes={"/data": data}, timeout=1800, cpu=4, memory=16384)
def export() -> dict:
    import json
    import shutil

    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, "/root/common")
    from run_spec import RunSpec, build_inputs, image_path, pair_slug, poison_selection
    from text_cache import TextCache

    src, out = pathlib.Path("/data"), pathlib.Path("/data/kaggle_subset")
    if out.exists():
        shutil.rmtree(out)
    manifest = json.loads((src / "chimera_coco" / "manifest.json").read_text())
    images = src / "chimera_coco" / "images"

    # The Gate B cat->dog cell's poisons and its dirty-label partners, by the same functions training uses.
    spec = RunSpec(model="sd3_medium", regime="full_ft", attack="nightshade_sd3", pair=PAIR,
                   poison_count=100, seed=0)
    chosen = poison_selection(manifest, spec)
    dirty_spec = RunSpec(model="sd3_medium", regime="full_ft", attack="dirty_label", pair=PAIR,
                         poison_count=100, seed=0, clean_filler=0, source_clean=0)
    dirty, _ = build_inputs(manifest, dirty_spec, images)
    ids = [r["id"] for r in chosen]

    (out / "chimera_coco" / "images").mkdir(parents=True)
    shutil.copy(src / "chimera_coco" / "manifest.json", out / "chimera_coco" / "manifest.json")
    for r in chosen + dirty:
        shutil.copy(image_path(images, r), out / "chimera_coco" / "images" / f"{r['id']}.jpg")

    pools = {}
    for attack in POOLS:
        s, d = src / "poisons" / attack / pair_slug(PAIR), out / "poisons" / attack / pair_slug(PAIR)
        d.mkdir(parents=True)
        missing = [i for i in ids if not (s / f"{i}.png").exists()]
        if missing:
            raise FileNotFoundError(f"{attack}: {len(missing)} of the 100 ids missing, e.g. {missing[:3]}")
        for i in ids:
            shutil.copy(s / f"{i}.png", d / f"{i}.png")
        shutil.copy(s / "poison_manifest.json", d / "poison_manifest.json")
        pools[attack] = len(ids)

    for model in MODELS:
        (out / "anchors" / model).mkdir(parents=True)
        shutil.copy(src / "anchors" / model / "dog.png", out / "anchors" / model / "dog.png")

    # One shard in TextCache's own layout, holding only the captions these jobs condition on.
    captions = sorted({r["caption"] for r in chosen} | {r["caption"] for r in dirty})
    embeds = {}
    for model in MODELS:
        cache = TextCache(src / "embeds" / model)
        rows = [cache.get(c) for c in captions]
        blob = {k: torch.stack([r[k] for r in rows]).contiguous() for k in rows[0]}
        d = out / "embeds" / model
        d.mkdir(parents=True)
        save_file(blob, str(d / "shard_000.safetensors"))
        (d / "index.json").write_text(json.dumps(
            {"captions": captions, "shards": [{"file": "shard_000.safetensors", "start": 0,
                                               "stop": len(captions)}]}))
        embeds[model] = {k: list(v.shape) for k, v in blob.items()}

    data.commit()
    files = [f for f in out.rglob("*") if f.is_file()]
    return {"ids": len(ids), "dirty": len(dirty), "captions": len(captions), "pools": pools,
            "embeds": embeds, "files": len(files), "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1)}


@app.function(image=image, volumes={"/data": data}, timeout=1800, cpu=4, memory=8192)
def export_mech() -> dict:
    """
    The four-pair slice for the mechanism tests (vae_theory.py): both Nightshade pools of every pair on the
    ids they share, the clean originals of those ids, and every target's anchor for both models. No text
    embeddings -- nothing here runs a denoiser. Written to /data/kaggle_mech/ in the volume's layout.
    """
    import json
    import shutil

    sys.path.insert(0, "/root/common")
    from run_spec import RunSpec, image_path, pair_slug, poison_selection

    src, out = pathlib.Path("/data"), pathlib.Path("/data/kaggle_mech")
    if out.exists():
        shutil.rmtree(out)
    manifest = json.loads((src / "chimera_coco" / "manifest.json").read_text())
    (out / "chimera_coco" / "images").mkdir(parents=True)
    shutil.copy(src / "chimera_coco" / "manifest.json", out / "chimera_coco" / "manifest.json")
    pairs = {}
    for pair, info in manifest["pairs"].items():
        dirs = {a: src / "poisons" / a / pair_slug(pair) for a in ("nightshade_sd2", "nightshade_sd3")}
        shared = set.intersection(*({int(f.stem) for f in d.glob("*.png")} for d in dirs.values()))
        # cat->dog pools hold 233 images; keep the Gate B cell's 100 so every pair contributes 100.
        spec = RunSpec(model="sd3_medium", regime="full_ft", attack="nightshade_sd3", pair=pair,
                       poison_count=100, seed=0)
        ids = [r["id"] for r in poison_selection(manifest, spec) if r["id"] in shared]
        by_id = {r["id"]: r for r in info["poison_pool"]}
        for attack, d in dirs.items():
            dst = out / "poisons" / attack / pair_slug(pair)
            dst.mkdir(parents=True)
            for i in ids:
                shutil.copy(d / f"{i}.png", dst / f"{i}.png")
            shutil.copy(d / "poison_manifest.json", dst / "poison_manifest.json")
        for i in ids:
            shutil.copy(image_path(src / "chimera_coco" / "images", by_id[i]),
                        out / "chimera_coco" / "images" / f"{i}.jpg")
        for model in ("sd2_base", "sd3_medium"):
            (out / "anchors" / model).mkdir(parents=True, exist_ok=True)
            name = f"{info['target'].replace(' ', '_')}.png"
            shutil.copy(src / "anchors" / model / name, out / "anchors" / model / name)
        pairs[pair] = len(ids)
    data.commit()
    files = [f for f in out.rglob("*") if f.is_file()]
    return {"pairs": pairs, "files": len(files), "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1)}


@app.local_entrypoint()
def main(mech: bool = False):
    import json

    print(json.dumps((export_mech if mech else export).remote(), indent=2))
