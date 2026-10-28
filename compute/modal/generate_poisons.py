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
              poison_root: str = "/data/poisons", fresh: bool = False, step_scale: float = 1.0) -> str:
    sys.path.insert(0, "/root/common")
    pg = _load_shared("chimera_poison_gen", "/root/common/poison_gen.py")

    if fresh:
        # A probe is scratch: each one runs a different configuration, and generate() rightly refuses to
        # resume a directory filled under another. Clearing it here keeps that guard strict where it matters,
        # which is the real poison directories -- those must never be silently mixed.
        import shutil

        shutil.rmtree(pg.poison_dir_for(poison_root, attack, pair), ignore_errors=True)

    manifest = json.loads(pathlib.Path("/data/chimera_coco/manifest.json").read_text())

    last = [0.0]

    def checkpoint(done: int, total: int):
        """Make the finished batches durable, at most once a minute so committing is not the bottleneck."""
        import time as _time

        if _time.time() - last[0] > 60 or done >= total:
            data.commit()
            last[0] = _time.time()
            print(f"    committed {done}/{total}", flush=True)

    out = _run(pg.generate, attack, pair, manifest, "/data/chimera_coco/images", "/data/embeds",
               pg.poison_dir_for(poison_root, attack, pair), anchors_root="/data/anchors",
               iters=iters, eps=eps, sigma=sigma, lpips_weight=lpips_weight, batch=batch,
               limit=limit or None, log_every=log_every, git_commit=commit,
               only_ids=set(only_ids) if only_ids else None, step_scale=step_scale,
               on_batch=checkpoint)
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
                poison_root: str = "/data/poisons", fresh: bool = False, step_scale: float = 1.0) -> str:
    return _generate(attack, pair, commit, iters, batch, limit, sigma, eps, lpips_weight, log_every,
                     only_ids, poison_root, fresh, step_scale)


@app.function(gpu="L40S", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=TIMEOUT)
def generate_l40s(attack: str, pair: str, commit: str, iters: int = 500, batch: int = 4, limit: int = 0,
                  sigma: float = 0.5, eps: float = 0.04, lpips_weight: float = 0.0,
                  log_every: int = 0, only_ids: list = None,
                  poison_root: str = "/data/poisons", fresh: bool = False,
                  step_scale: float = 1.0) -> str:
    return _generate(attack, pair, commit, iters, batch, limit, sigma, eps, lpips_weight, log_every,
                     only_ids, poison_root, fresh, step_scale)


@app.function(gpu="L4", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=TIMEOUT)
def diag_l4(pools: list, limit: int, batch: int) -> str:
    """
    Per-image latent geometry of pools that already exist. VAE-only, so an L4 is plenty (peak ~6 GB during
    generation was with the optimiser graph; this is forward passes with no_grad).
    """
    sys.path.insert(0, "/root/common")
    pd = _load_shared("chimera_poison_diag", "/root/common/poison_diag.py")

    manifest = json.loads(pathlib.Path("/data/chimera_coco/manifest.json").read_text())
    out = {}
    for attack, pair in pools:
        print(f"  {attack}/{pair}", flush=True)
        out[f"{attack}/{pair}"] = _run(
            pd.measure_pool, attack, pair, manifest, "/data/chimera_coco/images", "/data/poisons",
            "/data/anchors", limit=limit or None, batch=batch)
    return json.dumps(out)


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
def main(step: str = "generate", attack: str = "", pair: str = "cat->dog", pairs: str = "",
         match_attack: str = "", model: str = "sd3_medium",
         matrix: str = "experiments/configs/matrix.json", all: bool = False, iters: int = 0,
         batch: int = 4, limit: int = 0, sigma: float = 0.5, eps: float = 0.04,
         lpips_weight: float = 0.0, log_every: int = 0, gpu: str = "L40S", step_scale: float = 0.0):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True, cwd=ROOT).stdout.strip()

    if step == "anchors":
        print(json.dumps(json.loads(anchors_l40s.remote(model)), indent=2))
        return

    if step == "diag":
        # Measures finished pools; generates nothing. Default is every nightshade pool, which is the pair of
        # arms the H1 mechanism claim compares.
        man = json.loads((ROOT / "data" / "chimera_coco" / "manifest.json").read_text())
        from poison_gen import ATTACK_SPEC

        if attack:
            attacks = [a.strip() for a in attack.split(",") if a.strip()]
        else:
            attacks = [a for a in sorted(ATTACK_SPEC) if a.startswith("nightshade")]
        wanted = [p.strip() for p in pairs.split(",") if p.strip()] if pairs else list(man["pairs"])
        pools = [(a, p) for a in attacks for p in wanted]
        print(f"measuring {len(pools)} pool(s): " + ", ".join(f"{a}/{p}" for a, p in pools))
        out = json.loads(diag_l4.remote(pools, limit, batch))

        path = ROOT / "results" / "compute" / "poison_diag.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = json.loads(path.read_text()) if path.exists() else {}
        merged.update(out)
        path.write_text(json.dumps(merged, indent=2))
        print(f"\n{'pool':34s} {'ch':>3s} {'closed':>18s} {'cosine':>8s} {'off-axis':>9s}")
        for key, o in sorted(merged.items()):
            c, cos = o["scaled"]["closed"], o["scaled"]["cosine"]
            print(f"{key:34s} {o['vae']['latent_channels']:>3d} "
                  f"{c['median']:>8.4f} [{c['p25']:+.3f},{c['p75']:+.3f}] "
                  f"{cos['median']:>8.4f} {o['scaled']['off_axis']['median']:>9.4f}")
        print(f"\ndiagnostics -> {path} ({len(out)} written, {len(merged)} total)")
        return

    fn = {"L4": generate_l4, "L40S": generate_l40s}[gpu]

    if step == "probe":
        if not attack:
            raise SystemExit("--attack is required for a probe")
        from poison_gen import SCHEDULE

        iters = iters or SCHEDULE[attack]["iters"]
        step_scale = step_scale if step_scale else SCHEDULE[attack]["step_scale"]
        # One scratch directory per configuration: concurrent probes each clear their own on start, so a
        # shared path would have them deleting each other's images mid-run.
        out = json.loads(fn.remote(attack, pair, commit, iters, batch, limit or 4, sigma, eps,
                                   lpips_weight, log_every or 25, None,
                                   f"/data/poison_probe/s{step_scale:g}_i{iters}", True, step_scale))
        # Every knob the probe varies goes in the name, or one measurement overwrites another.
        path = (ROOT / "results" / "compute"
                / f"poison_probe_{attack}_{gpu.lower()}_b{batch}_s{step_scale:g}_i{iters}.json")
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
    from poison_gen import SCHEDULE

    # --pairs runs several pairs of one attack as parallel containers under a single summary writer, so three
    # pairs cannot race each other merging poison_generation.json.
    #
    # --match-attack pins the ids to the ones another attack's pool already holds. Without it, a pool built by
    # --limit N draws a different N of the pair's poison_pool than the sibling attack got (measured: 37/100
    # overlap for car->cow), which would confound "optimised against which model" with "perturbed which
    # images" in exactly the comparison H1 rests on.
    if all:
        jobs = _jobs(ROOT / matrix, ROOT / "data" / "chimera_coco" / "manifest.json")
    else:
        wanted = [p.strip() for p in pairs.split(",") if p.strip()] if pairs else [pair]
        ids_for = {}
        if match_attack:
            from run_spec import required_poison_ids

            man = json.loads((ROOT / "data" / "chimera_coco" / "manifest.json").read_text())
            config = json.loads((ROOT / matrix).read_text())
            for p in wanted:
                got = sorted(required_poison_ids(man, config, match_attack, p))
                if not got:
                    raise SystemExit(f"--match-attack {match_attack!r} draws no ids for {p!r}: the matrix has "
                                     f"no cell for that (attack, pair), so there is nothing to match")
                ids_for[p] = got
            print(f"matching ids to {match_attack}: " + ", ".join(f"{p}={len(ids_for[p])}" for p in wanted))
        jobs = [(attack, p, ids_for.get(p)) for p in wanted]
    total = sum(len(ids) for _, _, ids in jobs if ids)
    print(f"{len(jobs)} job(s)" + (f", {total} images in all:" if total else ":"))

    # Each attack runs on its own measured schedule; an explicit --iters / --step-scale overrides it for
    # every job. Without this, one invocation of --all would give half the jobs the other family's schedule:
    # Nightshade's step saturates the velocity objective, and 0.02 leaves the latent objective barely moved.
    def plan_for(a):
        sched = SCHEDULE[a]
        return (iters or sched["iters"], step_scale if step_scale else sched["step_scale"])

    for a, p, ids in jobs:
        it, ss = plan_for(a)
        print(f"  {a}/{p}: {len(ids) if ids else '?'} images, {it} iters, step_scale {ss:g}")

    args = [(a, p, commit, plan_for(a)[0], batch, limit, sigma, eps, lpips_weight, log_every, ids,
             "/data/poisons", False, plan_for(a)[1]) for a, p, ids in jobs]
    results = {}
    for (a, p, _), raw in zip(jobs, fn.starmap(args)):
        out = json.loads(raw)
        results[f"{a}/{p}"] = out
        print(f"{a}/{p}: {out['n_images']} images, {out.get('generated', 0)} new, "
              f"{out.get('seconds', 0) / 60:.0f} min, peak {out.get('peak_gb')} GB")
        if out.get("stealth", {}).get("psnr_db"):
            s = out["stealth"]
            print(f"    Linf {s['linf_255']['median']}/255 median, PSNR {s['psnr_db']['median']} dB median")
    # Merge, never overwrite: this file is the measured record for every (attack, pair) ever generated, and the
    # H1 mechanism table is read straight out of it. A run that generates three pairs must not erase the other
    # seven entries.
    path = ROOT / "results" / "compute" / "poison_generation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(path.read_text()) if path.exists() else {}
    replaced = sorted(set(merged) & set(results))
    merged.update(results)
    path.write_text(json.dumps(merged, indent=2))
    print(f"\nsummary -> {path} ({len(results)} written, {len(merged)} entries total"
          + (f", replaced {', '.join(replaced)}" if replaced else "") + ")")
