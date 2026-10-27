"""
Measure the saved poisons' visibility on Modal, for the paper's stealth table.

Runs after generation, over the PNGs on the chimera-data volume, so every attack is measured by one pass that
knows nothing about how its images were made. See experiments/common/stealth.py for what is measured and why
no attack carries an LPIPS term in its own loss.

  modal run --detach compute/modal/stealth.py
  modal run compute/modal/stealth.py --poison-root /data/poison_probe/s0.02_i500   # a probe's images
"""
import json
import pathlib
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


app = modal.App("chimera-stealth")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "lpips", "transformers", "safetensors", "pillow", "numpy",
                 "huggingface_hub==2.3.0", "diffusers==0.42.0")
    .env({"HF_HOME": "/cache"})
    .add_local_dir(ROOT / "experiments" / "common", "/root/common")
)
cache = modal.Volume.from_name("chimera-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("chimera-data", create_if_missing=True)


@app.function(gpu="L4", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3 * 3600)
def stealth(poison_root: str, batch: int) -> str:
    import traceback

    sys.path.insert(0, "/root/common")
    module = _load_shared("chimera_stealth", "/root/common/stealth.py")
    try:
        out = module.run(poison_root, "/data", "cuda", batch=batch)
    except Exception:
        raise RuntimeError("remote failure:\n" + traceback.format_exc()) from None
    cache.commit()
    return json.dumps(out)


@app.local_entrypoint()
def main(poison_root: str = "/data/poisons", batch: int = 8, out: str = ""):
    results = json.loads(stealth.remote(poison_root, batch))
    path = pathlib.Path(out) if out else ROOT / "results" / "compute" / "poison_stealth.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'attack/pair':34s} {'n':>4} {'LPIPS':>7} {'DINOv2':>7} {'PSNR':>6} {'Linf':>5}")
    for name, r in sorted(results.items()):
        if "lpips" not in r or not r["lpips"]:
            continue
        print(f"{name:34s} {r['lpips']['n']:4d} {r['lpips']['median']:7.4f} "
              f"{r['dino_cosine']['median']:7.4f} {r['psnr_db']['median']:6.2f} "
              f"{r['linf_255']['median']:5.1f}")
    print(f"\n-> {path}")
