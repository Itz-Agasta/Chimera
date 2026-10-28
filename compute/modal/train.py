"""
Run training cells from experiments/common/train.py on a Modal GPU, and write their run directories locally.

The container gets the repo's shared modules and the chimera-data volume (images, manifest, cached text
embeddings, latent cache, checkpoints); the local side only moves JSON, so no torch is needed here.

One cell:
  modal run compute/modal/train.py --model sd3_medium --regime full_ft --attack clean --pair 'cat->dog'

A whole tier (resumable -- runs whose metrics.json already exists are skipped):
  modal run compute/modal/train.py --matrix experiments/configs/matrix.json --tier clean_baseline

Smoke test (3 steps, warms the latent cache, writes results/compute/ not a run directory):
  modal run compute/modal/train.py --model sd3_medium --regime full_ft --attack clean --pair 'cat->dog' --smoke
"""
import json
import pathlib
import subprocess
import sys

import modal

# Module level runs again inside the container, where the repo layout does not exist.
ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")
if modal.is_local():
    sys.path.insert(0, str(ROOT / "experiments" / "common"))


def _load_shared(name: str, path: str):
    """
    Load one of experiments/common/*.py by path inside the container.

    Modal copies this launcher to /root/<its filename>.py and imports it under that name, so a plain
    `import train` (or `import evaluate`) resolves to the launcher already in sys.modules rather than to the
    shared module mounted at /root/common. Loading by path sidesteps the collision; sibling imports inside the
    shared module still resolve through /root/common on sys.path.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

app = modal.App("chimera-train")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "diffusers==0.42.0", "transformers", "accelerate", "bitsandbytes", "peft",
                 "safetensors", "pillow", "numpy", "sentencepiece", "protobuf",
                 "huggingface_hub==2.3.0")
    .env({"HF_HOME": "/cache",
          # the VAE pass fragments the heap before the transformer is built; let the allocator give it back
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(ROOT / "experiments" / "common", "/root/common")
)
cache = modal.Volume.from_name("chimera-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("chimera-data", create_if_missing=True)
GPU_TIMEOUT = 8 * 3600   # a 40-epoch probe run on SD3 is ~3.5 h on an L40S


def _run(fn, *args, **kwargs):
    """
    Call `fn`, turning any exception into a plain RuntimeError carrying the traceback as text.

    A torch exception (CUDA OOM above all) cannot be unpickled on the launching machine, which has no torch, so
    Modal reports a deserialization error instead of the real failure. Text always travels.
    """
    import traceback

    try:
        return fn(*args, **kwargs)
    except Exception:
        raise RuntimeError("remote failure:\n" + traceback.format_exc()) from None


def _train(spec_dict: dict, commit: str, poison_dir: str | None, checkpoint: bool) -> str:
    sys.path.insert(0, "/root/common")
    from run_spec import RunSpec
    train = _load_shared("chimera_train", "/root/common/train.py").train

    spec = RunSpec.from_dict(spec_dict)
    manifest = json.loads(pathlib.Path("/data/chimera_coco/manifest.json").read_text())
    probe_root = pathlib.Path("/data/probes")

    def on_probe(probes):
        # Persist after every probe: a run that dies or hits the budget cap still leaves its curve so far.
        out = probe_root / spec.run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "probe.json").write_text(json.dumps({"run_id": spec.run_id, "probe": probes}, indent=2))
        data.commit()

    _, config, inputs, metrics = _run(train,
        spec, manifest, "/data/chimera_coco/images", f"/data/embeds/{spec.model}",
        poison_dir=poison_dir, latent_cache=f"/data/latents/{spec.model}_{spec.resolution}.safetensors",
        commit=commit, checkpoint_dir="/data/checkpoints" if checkpoint else None,
        probe_dir=str(probe_root), on_probe=on_probe)
    data.commit()
    cache.commit()
    # Plain JSON: the launching machine has no torch to unpickle anything else.
    return json.dumps({"config": config, "inputs": inputs, "metrics": metrics})


@app.function(gpu="L4", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=GPU_TIMEOUT)
def train_l4(spec_dict: dict, commit: str, poison_dir: str = None, checkpoint: bool = False) -> str:
    return _train(spec_dict, commit, poison_dir, checkpoint)


@app.function(gpu="L40S", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=GPU_TIMEOUT)
def train_l40s(spec_dict: dict, commit: str, poison_dir: str = None, checkpoint: bool = False) -> str:
    return _train(spec_dict, commit, poison_dir, checkpoint)


@app.local_entrypoint()
def main(model: str = "sd3_medium", regime: str = "full_ft", attack: str = "clean", pair: str = "cat->dog",
         poison_count: int = 0, seed: int = 0, steps: int = 0, matrix: str = "", tier: str = "",
         poison_dir: str = "", checkpoint: bool = False, smoke: bool = False, gpu: str = "L4",
         learning_rate: float = 0.0, epochs: float = 0.0, clean_filler: int = -1, source_clean: int = -1,
         probe_epochs: str = "", probe_all_pairs: bool = False,
         dry_run: bool = False):
    from run_spec import RunSpec, estimate, expand_matrix

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True, cwd=ROOT).stdout.strip()
    recipe = {"steps": steps} if steps else {}
    if learning_rate:
        recipe["learning_rate"] = learning_rate
    if epochs:
        recipe["epochs"] = epochs
    if clean_filler >= 0:
        recipe["clean_filler"] = clean_filler
    if source_clean >= 0:
        recipe["source_clean"] = source_clean
    if probe_epochs:
        recipe["probe_epochs"] = tuple(float(e) for e in probe_epochs.split(","))
    if probe_all_pairs:
        recipe["probe_all_pairs"] = True
    if smoke:
        recipe["steps"] = 3

    if matrix:
        config = json.loads((ROOT / matrix).read_text())
        specs = expand_matrix(config)
        if tier:
            names = [t["name"] for t in config["tiers"]]
            if tier not in names:
                raise SystemExit(f"no tier {tier!r} in {matrix} (have {names})")
            wanted = {s.run_id for s in expand_matrix({**config, "tiers": [t for t in config["tiers"]
                                                                           if t["name"] == tier]})}
            specs = [s for s in specs if s.run_id in wanted]
        for key, value in recipe.items():
            for s in specs:
                setattr(s, key, value)
        print(json.dumps(estimate(specs, config, 3200), indent=2))
    else:
        specs = [RunSpec(model=model, regime=regime, attack=attack, pair=pair, poison_count=poison_count,
                         seed=seed, **recipe)]

    fn = {"L4": train_l4, "L40S": train_l40s}[gpu]
    for i, spec in enumerate(specs, 1):
        run_dir = ROOT / "results" / spec.experiment / spec.run_id
        if (run_dir / "metrics.json").exists() and not smoke:
            print(f"[{i}/{len(specs)}] skip {spec.run_id} (already has metrics.json)")
            continue
        print(f"[{i}/{len(specs)}] {spec.experiment}/{spec.run_id} on {gpu}", flush=True)
        if dry_run:
            continue
        out = json.loads(fn.remote(spec.__dict__, commit, poison_dir or None, checkpoint))
        if smoke:
            path = ROOT / "results" / "compute" / f"train_smoke_{spec.model}_{spec.regime}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"config": out["config"], "metrics": out["metrics"],
                                        "n_clean": len(out["inputs"]["clean_ids"]),
                                        "n_poisoned": len(out["inputs"]["poisoned_ids"])}, indent=2))
            print(json.dumps(out["metrics"], indent=2)[:1200])
            print(f"smoke result -> {path}")
            continue
        (run_dir / "samples").mkdir(parents=True, exist_ok=True)
        for name in ("config", "inputs", "metrics"):
            (run_dir / f"{name}.json").write_text(json.dumps(out[name], indent=2))
        print(f"  loss {out['metrics']['final_loss']}  {out['metrics']['seconds']}s  "
              f"{out['metrics']['peak_gb']} GB -> {run_dir}")
