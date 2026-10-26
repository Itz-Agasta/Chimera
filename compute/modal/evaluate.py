"""
Evaluate finished runs on a Modal GPU: generate at every guidance scale, judge, write evaluation.json.

Samples live in the chimera-data volume (they are bulky and only the judges read them); evaluation.json comes
back to the local run directory so the paper's tables are built from files in the repo.

  modal run compute/modal/evaluate.py --run-id sd3_medium_full_ft_clean_cat-dog_n0_s0
  modal run compute/modal/evaluate.py --experiment h1_transfer          # every run there lacking evaluation.json
"""
import json
import pathlib
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")

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

app = modal.App("chimera-evaluate")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "diffusers==0.42.0", "transformers", "accelerate", "peft", "safetensors",
                 "pillow", "numpy", "sentencepiece", "protobuf", "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(ROOT / "experiments" / "common", "/root/common")
)
cache = modal.Volume.from_name("chimera-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("chimera-data", create_if_missing=True)


@app.function(gpu="L4", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=4 * 3600)
def evaluate(config_json: str, guidance: list | None = None, n_per_prompt: int = 2) -> str:
    sys.path.insert(0, "/root/common")
    evaluate_run = _load_shared("chimera_evaluate", "/root/common/evaluate.py").evaluate_run

    config = json.loads(config_json)
    # The run directory lives on the volume: samples stay next to the data, evaluation.json goes home.
    run_dir = pathlib.Path("/data/runs") / config["experiment"] / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(config_json)
    out = evaluate_run(run_dir, "/data", "/data/checkpoints", guidance, n_per_prompt)
    data.commit()
    return json.dumps(out)


@app.local_entrypoint()
def main(run_id: str = "", experiment: str = "", guidance: str = "", n_per_prompt: int = 2,
         redo: bool = False):
    scales = [float(x) for x in guidance.split(",")] if guidance else None
    runs = []
    for config_path in sorted((ROOT / "results").glob("*/*/config.json")):
        run_dir = config_path.parent
        config = json.loads(config_path.read_text())
        if run_id and config.get("run_id") != run_id:
            continue
        if experiment and config.get("experiment") != experiment:
            continue
        if (run_dir / "evaluation.json").exists() and not redo:
            print(f"skip {config['run_id']} (already evaluated)")
            continue
        runs.append((run_dir, config))
    if not runs:
        raise SystemExit("nothing to evaluate (use --redo to re-evaluate, or check results/*/*/config.json)")

    for i, (run_dir, config) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {config['run_id']}", flush=True)
        out = json.loads(evaluate.remote(json.dumps(config), scales, n_per_prompt))
        (run_dir / "evaluation.json").write_text(json.dumps(out, indent=2))
        for g, per_judge in sorted(out["by_guidance"].items(), key=lambda kv: float(kv[0])):
            parts = [f"{j}={per_judge[j]['attack_success']:.2f}" for j in out["judges"] if j in per_judge]
            print(f"    w={g}: attack success {' '.join(parts)}")
