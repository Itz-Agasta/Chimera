"""
Generate the optimised poison images on Modal, one container per (attack, pair).

Poison generation is embarrassingly parallel -- every image is its own optimisation problem -- so it belongs on
Modal's free credit and its fan-out, not on a rented Vast box where it would occupy one GPU serially while the
fine-tunes wait. The images land on the chimera-data volume next to the corpus they were made from and travel
to the rented box inside the same bundle.

Three steps, in order:

  1. anchors   one generated image per target concept, per model. The only step that loads text encoders.
       modal run --detach compute/modal/generate_poisons.py --step anchors --model sd3_medium

  2. probe     a handful of images with the loss recorded every N iterations: does the objective converge
               before 500 iterations, what does one image cost, and how far does batching help.
       modal run compute/modal/generate_poisons.py --step probe --attack chimera_pred --limit 4 --batch 4

  3. generate  the full pool for one (attack, pair), or every combination the matrix needs.
       modal run --detach compute/modal/generate_poisons.py --step generate --attack chimera_pred --pair 'cat->dog'
       modal run --detach compute/modal/generate_poisons.py --step generate --all

The volume keeps `poisons/<attack>/<pair>/` and `anchors/<model>/`; the run is resumable, so a container that
dies loses at most the batch in flight.
"""
import json
import pathlib
import subprocess
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")
if modal.is_local():
    sys.path.insert(0, str(ROOT / "experiments" / "common"))


def _load_shared(name: str, path: str):
    """Load one of experiments/common/*.py by path; see compute/modal/train.py for why by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


app = modal.App("chimera-poisons")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "diffusers==0.42.0", "transformers", "accelerate", "safetensors", "pillow",
                 "numpy", "sentencepiece", "protobuf", "huggingface_hub==2.3.0")
    .env({"HF_HOME": "/cache",
          # the VAE graph is rebuilt every iteration; let the allocator hand the blocks back
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(ROOT / "experiments" / "common", "/root/common")
)
cache = modal.Volume.from_name("chimera-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("chimera-data", create_if_missing=True)
TIMEOUT = 6 * 3600


def _run(fn, *args, **kwargs):
    """Turn a remote torch exception into text: the launching machine has no torch to unpickle it with."""
    import traceback

    try:
        return fn(*args, **kwargs)
    except Exception:
        raise RuntimeError("remote failure:\n" + traceback.format_exc()) from None


def _anchors(model: str) -> str:
    sys.path.insert(0, "/root/common")
    pg = _load_shared("chimera_poison_gen", "/root/common/poison_gen.py")
    from concepts import TARGETS

    out = _run(pg.make_anchors, model, TARGETS, "/data/anchors")
    data.commit()
    return json.dumps(out)


def _generate(attack: str, pair: str, commit: str, iters: int, batch: int, limit: int, sigma: float,
              eps: float, lpips_weight: float, log_every: int, only_ids: list | None = None,
              poison_root: str = "/data/poisons", fresh: bool = False) -> str:
    sys.path.insert(0, "/root/common")
    pg = _load_shared("chimera_poison_gen", "/root/common/poison_gen.py")

    if fresh:
        # A probe is scratch: each one runs a different configuration, and generate() rightly refuses to
        # resume a directory filled under another. Clearing it here keeps that guard strict where it matters,
        # which is the real poison directories -- those must never be silently mixed.
        import shutil

        shutil.rmtree(pg.poison_dir_for(poison_root, attack, pair), ignore_errors=True)

    manifest = json.loads(pathlib.Path("/data/chimera_coco/manifest.json").read_text())
    out = _run(pg.generate, attack, pair, manifest, "/data/chimera_coco/images", "/data/embeds",
               pg.poison_dir_for(poison_root, attack, pair), anchors_root="/data/anchors",
               iters=iters, eps=eps, sigma=sigma, lpips_weight=lpips_weight, batch=batch,
               limit=limit or None, log_every=log_every, git_commit=commit,
               only_ids=set(only_ids) if only_ids else None)
    data.commit()
    # The per-image records can be thousands of rows; they stay in the volume's poison_manifest.json.
    summary = {k: v for k, v in out.items() if k != "images"}
    summary["n_images"] = len(out["images"])
    summary["stealth"] = _stealth(out["images"])
    return json.dumps(summary)


def _stealth(images: list[dict]) -> dict:
    """Min/median/max of what the budget actually bought, so a probe answers the stealth question too."""
    def q(key):
        vals = sorted(r[key] for r in images if r.get(key) is not None)
        if not vals:
            return None
        return {"min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1]}

    return {k: q(k) for k in ("linf_255", "psnr_db", "loss_initial", "loss_quantised", "seconds")}


@app.function(gpu="L40S", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=TIMEOUT)
def anchors_l40s(model: str) -> str:
    return _anchors(model)


@app.function(gpu="L4", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=TIMEOUT)
def generate_l4(attack: str, pair: str, commit: str, iters: int = 500, batch: int = 4, limit: int = 0,
                sigma: float = 0.5, eps: float = 0.04, lpips_weight: float = 0.0,
                log_every: int = 0, only_ids: list = None,
                poison_root: str = "/data/poisons", fresh: bool = False) -> str:
    return _generate(attack, pair, commit, iters, batch, limit, sigma, eps, lpips_weight, log_every,
                     only_ids, poison_root, fresh)


@app.function(gpu="L40S", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=TIMEOUT)
def generate_l40s(attack: str, pair: str, commit: str, iters: int = 500, batch: int = 4, limit: int = 0,
                  sigma: float = 0.5, eps: float = 0.04, lpips_weight: float = 0.0,
                  log_every: int = 0, only_ids: list = None,
                poison_root: str = "/data/poisons", fresh: bool = False) -> str:
    return _generate(attack, pair, commit, iters, batch, limit, sigma, eps, lpips_weight, log_every,
                     only_ids, poison_root, fresh)


def _jobs(matrix_path: pathlib.Path, manifest_path: pathlib.Path) -> list[tuple[str, str, list]]:
    """
    The (attack, pair) combinations the matrix consumes, and exactly which ids each one needs.

    Read from the matrix rather than hardcoded, so adding a tier cannot leave a cell without its poisons, and
    narrowed to the union of the ids its cells actually draw -- most (attack, pair) pairs are wanted at one
    poison count and one seed, which is 100 of the 250-image pool.
    """
    from run_spec import PERTURBATION_ATTACKS, expand_matrix, required_poison_ids

    config = json.loads(matrix_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    seen = sorted({(s.attack, s.pair) for s in expand_matrix(config) if s.attack in PERTURBATION_ATTACKS})
    return [(a, p, sorted(required_poison_ids(manifest, config, a, p))) for a, p in seen]


@app.local_entrypoint()
def main(step: str = "generate", attack: str = "", pair: str = "cat->dog", model: str = "sd3_medium",
         matrix: str = "experiments/configs/matrix.json", all: bool = False, iters: int = 500,
         batch: int = 4, limit: int = 0, sigma: float = 0.5, eps: float = 0.04,
         lpips_weight: float = 0.0, log_every: int = 0, gpu: str = "L40S"):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True, cwd=ROOT).stdout.strip()

    if step == "anchors":
        print(json.dumps(json.loads(anchors_l40s.remote(model)), indent=2))
        return

    fn = {"L4": generate_l4, "L40S": generate_l40s}[gpu]

    if step == "probe":
        if not attack:
            raise SystemExit("--attack is required for a probe")
        out = json.loads(fn.remote(attack, pair, commit, iters, batch, limit or 4, sigma, eps,
                                   lpips_weight, log_every or 25, None, "/data/poison_probe", True))
        path = ROOT / "results" / "compute" / f"poison_probe_{attack}_{gpu.lower()}_b{batch}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        per_image = out.get("seconds", 0) / max(out.get("generated", 1), 1)
        print(json.dumps({k: v for k, v in out.items() if k != "curves"}, indent=2))
        print(f"\n{per_image:.1f} s/image at batch {batch} on {gpu}  ->  "
              f"{per_image * 250 / 3600:.1f} h per 250-image pool")
        print(f"probe -> {path}")
        return

    if not all and not attack:
        raise SystemExit("pass --attack (and --pair), or --all for every combination the matrix needs")
    jobs = (_jobs(ROOT / matrix, ROOT / "data" / "chimera_coco" / "manifest.json") if all
            else [(attack, pair, None)])
    total = sum(len(ids) for _, _, ids in jobs) if all else 0
    print(f"{len(jobs)} job(s)" + (f", {total} images in all:" if total else ":"))
    for a, p, ids in jobs:
        print(f"  {a}/{p}" + (f"  {len(ids)} images" if ids else ""))

    args = [(a, p, commit, iters, batch, limit, sigma, eps, lpips_weight, log_every, ids,
             "/data/poisons", False) for a, p, ids in jobs]
    results = {}
    for (a, p, _), raw in zip(jobs, fn.starmap(args)):
        out = json.loads(raw)
        results[f"{a}/{p}"] = out
        print(f"{a}/{p}: {out['n_images']} images, {out.get('generated', 0)} new, "
              f"{out.get('seconds', 0) / 60:.0f} min, peak {out.get('peak_gb')} GB")
        if out.get("stealth", {}).get("psnr_db"):
            s = out["stealth"]
            print(f"    Linf {s['linf_255']['median']}/255 median, PSNR {s['psnr_db']['median']} dB median")
    path = ROOT / "results" / "compute" / "poison_generation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"\nsummary -> {path}")
