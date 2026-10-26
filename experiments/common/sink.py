"""
Durable store for finished runs: a private HuggingFace dataset repo.

A rented box is disposable. Everything a run produces is pushed here the moment it finishes and before the next run
starts, so a box that dies mid-matrix costs at most the run in flight. A replacement box reads back which runs are
already done and continues from there.

What is pushed per run, under `runs/<experiment>/<run_id>/`:
  config.json inputs.json metrics.json evaluation.json   a few hundred KB
  samples/*.jpg                                         ~8 MB per guidance scale

The generated images are kept because judging them is cheap and repeatable: a new judge, a third judge or a fixed
metric can be applied later without retraining anything. Checkpoints are 4 GB each and are not pushed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

RESULT_FILES = ("config.json", "inputs.json", "metrics.json", "evaluation.json")
DONE_MARKER = "evaluation.json"   # a run counts as finished only once it has been judged


def _api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; the results sink needs a write token")
    return HfApi(token=token)


def resolve(repo: str, api=None) -> str:
    """Fill in the namespace from the token when the repo is given as a bare name."""
    api = api or _api()
    return repo if "/" in repo else f"{api.whoami()['name']}/{repo}"


def ensure_repo(repo: str, api=None) -> str:
    api = api or _api()
    repo = resolve(repo, api)
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    return repo


def push_run(run_dir, repo: str, with_samples: bool = True, api=None) -> dict:
    """Upload one run directory. Returns what was sent."""
    api = api or _api()
    repo = ensure_repo(repo, api)
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    prefix = f"runs/{config['experiment']}/{config['run_id']}"

    sent, mb = [], 0.0
    for name in RESULT_FILES:
        path = run_dir / name
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{prefix}/{name}",
                            repo_id=repo, repo_type="dataset",
                            commit_message=f"{config['run_id']}: {name}")
            sent.append(name)
            mb += path.stat().st_size / 2**20

    samples = run_dir / "samples"
    if with_samples and samples.is_dir() and any(samples.iterdir()):
        api.upload_folder(folder_path=str(samples), path_in_repo=f"{prefix}/samples", repo_id=repo,
                          repo_type="dataset", commit_message=f"{config['run_id']}: samples")
        files = [p for p in samples.iterdir() if p.is_file()]
        sent.append(f"samples/ ({len(files)} files)")
        mb += sum(p.stat().st_size for p in files) / 2**20

    return {"repo": repo, "path": prefix, "sent": sent, "mb": round(mb, 1)}


def completed(repo: str, api=None) -> set[str]:
    """run_ids already judged in the sink, so a fresh box knows what not to redo."""
    api = api or _api()
    repo = resolve(repo, api)
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    except Exception as exc:           # repo not created yet: nothing is done
        print(f"sink: no completed runs ({type(exc).__name__})")
        return set()
    return {Path(f).parent.name for f in files
            if f.startswith("runs/") and Path(f).name == DONE_MARKER}


def pull_run(run_id: str, experiment: str, repo: str, dest, api=None) -> Path:
    """Fetch one finished run back from the sink (for local figure building)."""
    from huggingface_hub import snapshot_download

    api = api or _api()
    repo = resolve(repo, api)
    out = snapshot_download(repo_id=repo, repo_type="dataset", token=os.environ["HF_TOKEN"],
                            allow_patterns=[f"runs/{experiment}/{run_id}/*"], local_dir=str(dest))
    return Path(out) / "runs" / experiment / run_id


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["push", "list", "pull"])
    p.add_argument("--repo", default=os.environ.get("CHIMERA_SINK", "chimera-results"))
    p.add_argument("--run-dir")
    p.add_argument("--run-id")
    p.add_argument("--experiment")
    p.add_argument("--dest", default="results")
    p.add_argument("--no-samples", action="store_true")
    a = p.parse_args()

    if a.action == "push":
        if not a.run_dir:
            raise SystemExit("push needs --run-dir")
        print(json.dumps(push_run(a.run_dir, a.repo, not a.no_samples), indent=2))
    elif a.action == "list":
        done = sorted(completed(a.repo))
        print(f"{len(done)} finished run(s) in {a.repo}")
        for r in done:
            print(" ", r)
    else:
        if not (a.run_id and a.experiment):
            raise SystemExit("pull needs --run-id and --experiment")
        print(pull_run(a.run_id, a.experiment, a.repo, a.dest))


if __name__ == "__main__":
    main()
