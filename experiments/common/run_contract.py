"""
The contract every training run must satisfy before it is evaluated.

A run directory looks like:

  results/<experiment>/<run_id>/
    config.json     what the run was (see REQUIRED_CONFIG) + git commit
    inputs.json     which COCO ids were clean / poisoned, and the caption used for each
    metrics.json    training-side numbers (loss, steps, seconds, peak_gb)
    samples/        generated images, named <pair>__<prompt_idx>__w<guidance>__s<seed>__<n>.jpg

`validate_run` fails loudly and early, so a malformed run never reaches the judges and never produces a number
that ends up in the paper. Run it as a script:  uv run python experiments/common/run_contract.py <run_dir>
"""
import json
import re
import sys
from pathlib import Path

REQUIRED_CONFIG = {
    "model",          # "sd3_medium" | "sd2_base" | "sd3_5_medium"
    "regime",         # "full_ft" | "lora"
    "attack",         # "clean" | "dirty_label" | "nightshade_sd2" | "nightshade_sd3" | "chimera_pred" | "chimera_anchor"
    "pair",           # "cat->dog"
    "poison_count",   # 0, 25, 50, 100, 200
    "seed",
    "learning_rate",
    "steps",
    "batch_size",
    "resolution",
    "git_commit",
}
REQUIRED_METRICS = {"final_loss", "steps", "seconds"}
SAMPLE_RE = re.compile(r"^(?P<pair>[a-z_]+__[a-z_]+)__p(?P<prompt>\d+)__w(?P<guidance>[\d.]+)__s(?P<seed>\d+)__"
                       r"(?P<n>\d+)\.(jpg|png)$")


def sample_name(pair, prompt_idx, guidance, seed, n, ext="jpg"):
    """Canonical sample filename; `pair` is e.g. ('stop sign', 'bird')."""
    src, tgt = (p.replace(" ", "_") for p in pair)
    return f"{src}__{tgt}__p{prompt_idx}__w{guidance}__s{seed}__{n}.{ext}"


def parse_sample(name):
    m = SAMPLE_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    src, tgt = d.pop("pair").split("__")
    return {"source": src.replace("_", " "), "target": tgt.replace("_", " "),
            "prompt_idx": int(d["prompt"]), "guidance": float(d["guidance"]),
            "seed": int(d["seed"]), "n": int(d["n"])}


def validate_run(run_dir, manifest=None, require_samples=True):
    """Returns (ok, problems, summary). `manifest` enables cross-checks against the dataset splits."""
    run = Path(run_dir)
    problems, summary = [], {"run_dir": str(run)}

    def load(name):
        p = run / name
        if not p.exists():
            problems.append(f"missing {name}")
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{name} is not valid JSON: {e}")
            return None

    config, inputs, metrics = load("config.json"), load("inputs.json"), load("metrics.json")

    if config:
        missing = REQUIRED_CONFIG - set(config)
        if missing:
            problems.append(f"config.json missing keys: {sorted(missing)}")
        summary["config"] = {k: config.get(k) for k in
                             ("model", "regime", "attack", "pair", "poison_count", "seed") if k in config}

    if metrics:
        missing = REQUIRED_METRICS - set(metrics)
        if missing:
            problems.append(f"metrics.json missing keys: {sorted(missing)}")
        loss = metrics.get("final_loss")
        if isinstance(loss, (int, float)) and (loss != loss or loss < 0):
            problems.append(f"final_loss is not a usable number: {loss}")

    if inputs is not None:
        clean = inputs.get("clean_ids")
        poisoned = inputs.get("poisoned_ids")
        if clean is None or poisoned is None:
            problems.append("inputs.json needs clean_ids and poisoned_ids")
        else:
            overlap = set(clean) & set(poisoned)
            if overlap:
                problems.append(f"{len(overlap)} ids are both clean and poisoned, e.g. {sorted(overlap)[:5]}")
            if config and config.get("poison_count") is not None and len(poisoned) != config["poison_count"]:
                problems.append(f"poisoned_ids has {len(poisoned)} entries but config says "
                                f"poison_count={config['poison_count']}")
            summary["n_clean"], summary["n_poisoned"] = len(clean), len(poisoned)
            caps = inputs.get("captions", {})
            no_caption = [i for i in list(clean) + list(poisoned) if str(i) not in caps and i not in caps]
            if no_caption:
                problems.append(f"{len(no_caption)} training ids have no caption in inputs.json")

            if manifest and config and config.get("pair") in manifest.get("pairs", {}):
                pair = manifest["pairs"][config["pair"]]
                pool = {r["id"] for r in pair["poison_pool"]}
                stray = set(poisoned) - pool
                if stray:
                    problems.append(f"{len(stray)} poisoned ids are outside this pair's poison_pool")
                eval_ids = {r["id"] for r in manifest["clean_train"]}
                if not set(clean) <= eval_ids | pool:
                    problems.append("clean_ids contain images that are not in the manifest")

    if require_samples:
        sample_dir = run / "samples"
        if not sample_dir.is_dir():
            problems.append("missing samples/")
        else:
            files = sorted(p.name for p in sample_dir.iterdir() if p.is_file())
            bad = [f for f in files if parse_sample(f) is None]
            if bad:
                problems.append(f"{len(bad)} sample filenames do not match the contract, e.g. {bad[:3]}")
            parsed = [parse_sample(f) for f in files if parse_sample(f)]
            summary["n_samples"] = len(parsed)
            summary["guidance_scales"] = sorted({p["guidance"] for p in parsed})
            if config and parsed:
                src = config.get("pair", "->").split("->")[0]
                wrong = {p["source"] for p in parsed} - {src}
                if wrong:
                    problems.append(f"samples carry source concepts {sorted(wrong)}, but config pair is "
                                    f"{config.get('pair')}")

    return not problems, problems, summary


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_contract.py <run_dir> [manifest.json]")
    manifest = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    ok, problems, summary = validate_run(sys.argv[1], manifest)
    print(json.dumps(summary, indent=2))
    for p in problems:
        print("PROBLEM:", p)
    print("OK" if ok else f"{len(problems)} problem(s)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
