"""
Drive the run matrix on a rented box, one cell at a time, so a crash costs one stage rather than the matrix.

Each cell passes through three stages, each of which leaves something on disk that the next one consumes:

    train     -> checkpoint on local disk (4 GB, deleted once the cell is finished)
    generate  -> samples/*.jpg            (the durable artefact: judging them again is cheap)
    judge     -> evaluation.json          -> pushed to the sink, checkpoint deleted

On startup every cell is inspected and only the missing stages run: a box that died during generation resumes from
its checkpoint without retraining, and one that died during judging only re-judges. Cells already in the sink are
skipped entirely, so a replacement box continues where the dead one stopped.

Tiers run in the order the matrix lists them, which is claim order (clean baseline, then the headline curve, then
generalisation, then LoRA), so exhausting the budget still leaves the results the paper leads with.

  python experiments/common/runner.py --matrix experiments/configs/matrix.json --data-root /data \
      --sink chimera-results --git-commit <sha>
  python experiments/common/runner.py --matrix ... --plan-only        # print the plan, run nothing
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

from run_spec import GUIDANCE_SCALES, PERTURBATION_ATTACKS, RunSpec, expand_matrix, poison_dir_for


def stage_state(run_dir: Path, checkpoint_dir: Path, spec: RunSpec, expected_samples: int) -> dict:
    """What this cell has already produced."""
    samples = run_dir / "samples"
    n_samples = len(list(samples.glob("*.jpg"))) if samples.is_dir() else 0
    return {
        "trained": (checkpoint_dir / spec.run_id).exists() or n_samples >= expected_samples,
        "generated": n_samples >= expected_samples,
        "judged": (run_dir / "evaluation.json").exists(),
        "n_samples": n_samples,
        "has_checkpoint": (checkpoint_dir / spec.run_id).exists(),
    }


def expected_sample_count(manifest: dict, spec: RunSpec, guidance, n_per_prompt: int) -> int:
    from evaluate import prompt_list

    return len(prompt_list(manifest, spec)) * len(guidance) * n_per_prompt


def run_cell(spec: RunSpec, manifest: dict, paths: dict, guidance, n_per_prompt: int, commit: str,
             sink_repo: str | None, keep_checkpoint: bool = False, log_every: int = 25) -> dict:
    """Run the stages this cell still needs. Returns a record of what happened."""
    import train as train_mod            # imported here so --plan-only needs no torch
    from evaluate import evaluate_run
    from train import write_run

    run_dir = Path(paths["out_root"]) / spec.experiment / spec.run_id
    checkpoint_dir = Path(paths["checkpoint_dir"])
    expected = expected_sample_count(manifest, spec, guidance, n_per_prompt)
    state = stage_state(run_dir, checkpoint_dir, spec, expected)
    record = {"run_id": spec.run_id, "experiment": spec.experiment, "stages": [], "state_before": state}
    t0 = time.time()

    if not state["judged"]:
        if not state["generated"] and not state["has_checkpoint"]:
            print(f"  [train] {spec.run_id}", flush=True)
            _, config, inputs, metrics = train_mod.train(
                spec, manifest, paths["images"], paths["text"], poison_dir=paths.get("poison_dir"),
                latent_cache=paths.get("latent_cache"), commit=commit,
                checkpoint_dir=str(checkpoint_dir), log_every=log_every)
            write_run(run_dir, config, inputs, metrics)
            record["stages"].append("train")
            record["train_metrics"] = {k: metrics[k] for k in ("final_loss", "steps", "seconds", "peak_gb")}
        elif not state["generated"]:
            print(f"  [train] skipped: checkpoint already on disk", flush=True)

        # generate + judge; evaluate_run generates only what is missing and then judges
        print(f"  [generate+judge] {spec.run_id} ({state['n_samples']}/{expected} samples present)", flush=True)
        evaluation = evaluate_run(run_dir, paths["data_root"], str(checkpoint_dir), guidance, n_per_prompt)
        record["stages"].append("generate+judge" if not state["generated"] else "judge")
        record["attack_success"] = {g: {j: v[j]["attack_success"] for j in evaluation["judges"] if j in v}
                                    for g, v in evaluation["by_guidance"].items()}
    else:
        print(f"  already judged, nothing to do", flush=True)

    if sink_repo:
        from sink import push_run
        record["sink"] = push_run(run_dir, sink_repo)
        print(f"  [push] -> {record['sink']['path']} ({record['sink']['mb']} MB)", flush=True)

    ckpt = checkpoint_dir / spec.run_id
    if ckpt.exists() and not keep_checkpoint:
        shutil.rmtree(ckpt)
        record["stages"].append("checkpoint deleted")
    record["seconds"] = round(time.time() - t0, 1)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", required=True)
    p.add_argument("--data-root", default="/data")
    p.add_argument("--out-root", default="results")
    p.add_argument("--checkpoint-dir", default="/data/checkpoints")
    p.add_argument("--poison-root", default="/data/poisons",
                   help="holds <attack>/<pair>/<coco_id>.png; the per-cell directory is resolved from it")
    p.add_argument("--sink", default="chimera-results", help="HF dataset repo; empty string disables pushing")
    p.add_argument("--tier", help="run only this tier")
    p.add_argument("--n-per-prompt", type=int, default=2)
    p.add_argument("--guidance", type=float, nargs="*", help=f"default {list(GUIDANCE_SCALES)}")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--git-commit", default="")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--max-cells", type=int, help="stop after this many cells (budget guard)")
    a = p.parse_args(argv)

    config = json.loads(Path(a.matrix).read_text())
    if a.tier:
        names = [t["name"] for t in config["tiers"]]
        if a.tier not in names:
            raise SystemExit(f"no tier {a.tier!r} in the matrix (have {names})")
        config = {**config, "tiers": [t for t in config["tiers"] if t["name"] == a.tier]}
    specs = expand_matrix(config)

    data_root = Path(a.data_root)
    manifest = json.loads((data_root / "chimera_coco" / "manifest.json").read_text())
    guidance = tuple(a.guidance or GUIDANCE_SCALES)
    paths = {
        "data_root": str(data_root),
        "images": str(data_root / "chimera_coco" / "images"),
        "text": str(data_root / "embeds"),      # train() appends the model name
        "latent_cache": None,
        "checkpoint_dir": a.checkpoint_dir,
        "out_root": a.out_root,
        "poison_dir": None,                 # resolved per cell: the attacks poison the same ids
    }

    # Tiers may declare a poison count whose checkpoints are kept: the defences and the mechanism analysis
    # need weights, and retraining for them later would cost the matrix twice.
    keepers = set()
    for tier in config["tiers"]:
        at = tier.get("_keep_checkpoints_at")
        if at is not None:
            keepers |= {s.run_id for s in expand_matrix({**config, "tiers": [tier]})
                        if s.poison_count == at and s.seed == 0}
    if keepers:
        print(f"keeping checkpoints for {len(keepers)} cell(s): {sorted(keepers)[:3]}...")

    done = set()
    if a.sink:
        from sink import completed
        done = completed(a.sink)
        print(f"sink {a.sink}: {len(done)} run(s) already finished")

    todo = [s for s in specs if s.run_id not in done]
    print(f"{len(specs)} cells in the matrix, {len(todo)} to run"
          f"{f' (capped at {a.max_cells})' if a.max_cells else ''}")
    if a.max_cells:
        todo = todo[:a.max_cells]
    for s in todo:
        print(f"  {s.experiment}/{s.run_id}")
    if a.plan_only:
        return

    log = []
    for i, spec in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {spec.experiment}/{spec.run_id}", flush=True)
        paths["text"] = str(data_root / "embeds" / spec.model)
        paths["latent_cache"] = str(data_root / "latents" / f"{spec.model}_{spec.resolution}.safetensors")
        paths["poison_dir"] = (str(poison_dir_for(a.poison_root, spec.attack, spec.pair))
                               if spec.attack in PERTURBATION_ATTACKS else None)
        try:
            keep = a.keep_checkpoints or spec.run_id in keepers
            record = run_cell(spec, manifest, paths, guidance, a.n_per_prompt, a.git_commit,
                              a.sink or None, keep)
        except Exception:
            # One failed cell must not end the matrix: record it and carry on with the next.
            record = {"run_id": spec.run_id, "failed": True, "error": traceback.format_exc()[-2000:]}
            print(f"  FAILED:\n{record['error']}", flush=True)
        log.append(record)
        Path(a.out_root).mkdir(parents=True, exist_ok=True)
        (Path(a.out_root) / "runner_log.json").write_text(json.dumps(log, indent=2))

    ok = sum(1 for r in log if not r.get("failed"))
    print(f"\n{ok}/{len(log)} cells finished, log in {Path(a.out_root) / 'runner_log.json'}")


if __name__ == "__main__":
    main()
