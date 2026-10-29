"""
Post-hoc analysis of grad_align.py's saved per-slot gradient sketches, with the perturbation-texture direction
removed.

The P0.5a run showed that on SD3 any eps-bounded perturbation (random sign noise included) changes the mean
fine-tune gradient ~4.7x more than swapping the cat images for dirty-label dogs does, so `grad_closed` divides by
a tiny, noisy number and is meaningless (notes/failures.md C9). Here the shift each pool induces and dirty-label's
shift are both projected off the random-noise shift (the generic "an eps perturbation was added" direction), and
compared on independent halves of the slots: pool on one half, dirty-label on the other. `ceiling` is dirty-label
against itself under the same split. Bootstrap = 500 random half-splits.

    python grad_align_post.py results/compute/p05a/grad_sketch_sd3_medium.npz ... --out results/compute/grad_align_post.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def cos(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def off(x, r):
    return x - (x @ r) / (r @ r) * r


def analyse(path: Path, splits: int = 500, seed: int = 0) -> dict:
    z = np.load(path)
    S = {k: z[k].astype(np.float64) for k in z.files}
    pools = [k for k in S if k not in ("clean", "dirty", "random")]
    n = len(S["clean"])

    def one(hp, hd):
        cp, cd = S["clean"][hp].mean(0), S["clean"][hd].mean(0)
        dd = S["dirty"][hd].mean(0) - cd
        r = S["random"][hp].mean(0) - cp
        res = {p: cos(off(S[p][hp].mean(0) - cp, r), off(dd, r)) for p in pools}
        res["_ceiling"] = cos(off(S["dirty"][hp].mean(0) - cp, r), off(dd, r))
        res["_random_raw"] = cos(r, dd)
        return res

    ev, od = np.arange(0, n, 2), np.arange(1, n, 2)
    a, b = one(ev, od), one(od, ev)
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in a}
    for _ in range(splits):
        perm = rng.permutation(n)
        r = one(perm[:n // 2], perm[n // 2:])
        for k in boot:
            boot[k].append(r[k])
    summ = lambda v: {"median": round(float(np.median(v)), 4),  # noqa: E731
                      "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
    return {"n_slots": n, "sketch_dim": int(S["clean"].shape[1]), "splits": splits,
            "even_odd": {k: round((a[k] + b[k]) / 2, 4) for k in a},
            "random_splits": {k: summ(v) for k, v in boot.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sketches", nargs="+")
    ap.add_argument("--out", default="results/compute/grad_align_post.json")
    a = ap.parse_args(argv)
    out = {Path(p).stem.replace("grad_sketch_", ""): analyse(Path(p)) for p in a.sketches}
    Path(a.out).write_text(json.dumps(out, indent=2))
    for m, r in out.items():
        print(m, json.dumps(r["random_splits"]))


if __name__ == "__main__":
    main()
