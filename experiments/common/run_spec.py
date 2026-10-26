"""
What one training run is, and exactly which images and captions it trains on.

A run is identified by (model, regime, attack, pair, poison_count, seed); everything else is the recipe. The
dataset for a run is assembled here, deterministically, so the same spec always produces the same training set:

  clean corpus   manifest["clean_train"] (2000 filler + 4x300 clean source images), identical in every run
  poisons        `poison_count` images added on top, drawn from the pair's poison_pool in a seeded order

The poison order depends on (pair, seed) only, never on poison_count, so the poison sets are nested:
the 25 images of a 25-poison run are the first 25 of its 50-, 100- and 200-poison siblings. A poison-count
curve therefore varies one thing -- how many poisons -- and not which ones.

Attacks differ only in what the poisoned rows contain:
  clean            no poisons at all (poison_count is forced to 0)
  dirty_label      real target images (pair target_anchor) captioned with a source caption -- not clean-label
  nightshade_sd2   pre-optimised perturbed source images, read from --poison-dir, with their own real caption
  nightshade_sd3   "
  chimera_pred     "
  chimera_anchor   "

Used by experiments/common/train.py; `uv run python experiments/common/run_spec.py <matrix.json>` prints the
run matrix a config expands to, with time and cost estimates.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Pretrained ids. sd2_base must stay exactly the repo the text embeddings were encoded with, or the cached
# embeddings no longer match the model's own text encoders.
MODELS = {
    "sd3_medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd2_base": "sd2-community/stable-diffusion-2-1-base",
    "sd3_5_medium": "stabilityai/stable-diffusion-3.5-medium",
}
FLOW_MODELS = {"sd3_medium", "sd3_5_medium"}  # MM-DiT + rectified flow; the rest are U-Net + DDPM
REGIMES = ("full_ft", "lora")

# Attacks and the experiment directory their runs belong to.
ATTACKS = {
    "clean": "h1_transfer",
    "dirty_label": "h1_transfer",
    "nightshade_sd2": "h1_transfer",
    "nightshade_sd3": "h1_transfer",
    "chimera_pred": "h2_lpi",
    "chimera_anchor": "h2_lpi",
}
# Attacks whose poisoned images are perturbed files produced by a separate optimisation step.
PERTURBATION_ATTACKS = ("nightshade_sd2", "nightshade_sd3", "chimera_pred", "chimera_anchor")

DEFAULT_LR = {"full_ft": 1e-5, "lora": 1e-4}
GUIDANCE_SCALES = (1.0, 3.0, 5.0, 7.0, 9.0)


def _rng(*parts) -> random.Random:
    """A generator seeded by a label, stable across machines and Python versions (unlike hash())."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def pair_slug(pair: str) -> str:
    return pair.replace("->", "-").replace(" ", "_")


@dataclass
class RunSpec:
    """One cell of the run matrix. The first six fields identify it; the rest is the training recipe."""
    model: str
    regime: str
    attack: str
    pair: str
    poison_count: int
    seed: int = 0
    # recipe
    resolution: int = 512
    batch_size: int = 1
    grad_accum: int = 4
    epochs: float = 1.0
    steps: int | None = None            # set explicitly to override `epochs`
    learning_rate: float | None = None  # None -> DEFAULT_LR[regime]
    lora_rank: int = 16
    caption_dropout: float = 0.0
    weighting_scheme: str = "logit_normal"  # flow models: timestep density (see diffusers' SD3 scripts)
    logit_mean: float = 0.0
    logit_std: float = 1.0
    lr_warmup_steps: int = 50
    max_grad_norm: float = 1.0
    paged_optimizer: bool = True   # full FT only: page 8-bit Adam state to CPU (~4 GB off the GPU)

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}, expected one of {sorted(MODELS)}")
        if self.regime not in REGIMES:
            raise ValueError(f"unknown regime {self.regime!r}, expected one of {REGIMES}")
        if self.attack not in ATTACKS:
            raise ValueError(f"unknown attack {self.attack!r}, expected one of {sorted(ATTACKS)}")
        if self.attack == "clean":
            self.poison_count = 0
        elif self.poison_count <= 0:
            raise ValueError(f"attack {self.attack!r} needs poison_count > 0")

    @property
    def is_flow(self) -> bool:
        return self.model in FLOW_MODELS

    @property
    def experiment(self) -> str:
        return ATTACKS[self.attack]

    @property
    def run_id(self) -> str:
        return (f"{self.model}_{self.regime}_{self.attack}_{pair_slug(self.pair)}"
                f"_n{self.poison_count}_s{self.seed}")

    @property
    def lr(self) -> float:
        return DEFAULT_LR[self.regime] if self.learning_rate is None else self.learning_rate

    def resolve_steps(self, n_images: int) -> int:
        """
        Optimizer steps: either given explicitly, or `epochs` passes over the training set.

        Derived from `epochs`, the step count grows with poison_count (3200 clean images -> 800 steps,
        3400 -> 850), so a poison-count sweep would vary training length as well as poison count. Pin
        `steps` in the matrix recipe for any comparison across counts.
        """
        if self.steps is not None:
            return int(self.steps)
        per_epoch = math.ceil(n_images / (self.batch_size * self.grad_accum))
        return max(1, round(self.epochs * per_epoch))

    def config(self, n_images: int, git_commit: str) -> dict:
        """config.json for the run contract: the full recipe plus what it resolved to."""
        out = asdict(self)
        out.update(learning_rate=self.lr, steps=self.resolve_steps(n_images),
                   pretrained=MODELS[self.model], experiment=self.experiment, run_id=self.run_id,
                   n_train_images=n_images, git_commit=git_commit)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown RunSpec fields: {sorted(unknown)}")
        return cls(**d)


def image_path(images_root: Path, record: dict) -> Path:
    """
    Where a manifest record's image actually sits.

    The export stores one file per COCO id (`<id>.jpg`), not COCO's own zero-padded `file_name`
    (`000000068238.jpg`), so paths are built from the id. See compute/modal/build_coco_images.py.
    """
    return images_root / f"{record['id']}.jpg"


def build_inputs(manifest: dict, spec: RunSpec, images_root, poison_dir=None) -> tuple[list[dict], dict]:
    """
    Assemble the run's training set.

    Returns (records, inputs) where each record is {id, path, caption, poisoned} and `inputs` is the
    inputs.json demanded by the run contract. Raises if a poison image is missing, so a run can never
    silently train on fewer poisons than its config claims.
    """
    if spec.pair not in manifest["pairs"]:
        raise ValueError(f"pair {spec.pair!r} not in manifest ({sorted(manifest['pairs'])})")
    images_root = Path(images_root)
    pair = manifest["pairs"][spec.pair]

    records = [{"id": r["id"], "path": str(image_path(images_root, r)), "caption": r["caption"],
                "poisoned": False} for r in manifest["clean_train"]]
    clean_ids = [r["id"] for r in records]

    # Seeded, count-independent order over the poison pool -> nested poison sets across poison_count.
    pool = list(pair["poison_pool"])
    _rng(spec.pair, "poison_pool", spec.seed).shuffle(pool)
    chosen = pool[:spec.poison_count]
    if len(chosen) < spec.poison_count:
        raise ValueError(f"poison_pool for {spec.pair} has {len(pool)} images, "
                         f"poison_count={spec.poison_count}")

    poisoned = []
    if spec.attack == "dirty_label":
        # Real target images, captioned as the source: the captions are the ones the clean-label attacks
        # perturb, so every attack trains on the same source captions.
        anchors = list(pair["target_anchor"])
        _rng(spec.pair, "target_anchor", spec.seed).shuffle(anchors)
        if len(anchors) < spec.poison_count:
            raise ValueError(f"target_anchor for {spec.pair} has {len(anchors)} images, "
                             f"poison_count={spec.poison_count}")
        for anchor, source in zip(anchors[:spec.poison_count], chosen):
            records.append({"id": anchor["id"], "path": str(image_path(images_root, anchor)),
                            "caption": source["caption"], "poisoned": True})
            poisoned.append(anchor["id"])
    elif spec.attack in PERTURBATION_ATTACKS:
        if poison_dir is None:
            raise ValueError(f"attack {spec.attack!r} needs --poison-dir with the optimised images")
        poison_dir = Path(poison_dir)
        for r in chosen:
            found = next((p for ext in (".png", ".jpg", ".jpeg")
                          if (p := poison_dir / f"{r['id']}{ext}").exists()), None)
            if found is None:
                raise FileNotFoundError(f"no poison image for COCO id {r['id']} in {poison_dir}")
            # Clean-label: the perturbed image keeps the real caption of the image it was made from.
            records.append({"id": r["id"], "path": str(found), "caption": r["caption"], "poisoned": True})
            poisoned.append(r["id"])

    overlap = set(clean_ids) & set(poisoned)
    if overlap:
        raise ValueError(f"{len(overlap)} ids are both clean and poisoned, e.g. {sorted(overlap)[:5]}")

    inputs = {"clean_ids": clean_ids, "poisoned_ids": poisoned,
              "captions": {str(r["id"]): r["caption"] for r in records},
              "images_root": str(images_root), "poison_dir": str(poison_dir) if poison_dir else None}
    return records, inputs


# --------------------------------------------------------------------------------------------------- matrix

def expand_matrix(config: dict) -> list[RunSpec]:
    """
    Expand a tiered matrix config into RunSpecs, de-duplicated, in a stable order.

    Each tier lists models / regimes / pairs / attacks / poison_counts / seeds; `seeds_at` overrides the seed
    list for one poison count (that is how a 3-seed headline count sits inside a 1-seed sweep), and `recipe`
    at the top level or inside a tier overrides RunSpec defaults.
    """
    base_recipe = dict(config.get("recipe", {}))
    specs, seen = [], set()
    for tier in config["tiers"]:
        recipe = {**base_recipe, **tier.get("recipe", {})}
        seeds_at = {int(k): v for k, v in tier.get("seeds_at", {}).items()}
        for model in tier["models"]:
            for regime in tier["regimes"]:
                for pair in tier["pairs"]:
                    for attack in tier["attacks"]:
                        counts = [0] if attack == "clean" else tier["poison_counts"]
                        for count in counts:
                            for seed in seeds_at.get(count, tier["seeds"]):
                                spec = RunSpec(model=model, regime=regime, attack=attack, pair=pair,
                                               poison_count=count, seed=seed, **recipe)
                                if spec.run_id not in seen:
                                    seen.add(spec.run_id)
                                    specs.append(spec)
    return specs


def estimate(specs: list[RunSpec], config: dict, n_images: int) -> dict:
    """
    Wall-clock and cost estimate for a matrix, from measured seconds per micro-step.

    `config["cost"][model][regime]` holds {"s_per_microstep", "usd_per_hour", "source"}. A model/regime with no
    measurement is reported separately rather than guessed -- probe it before trusting a total.
    """
    cost = config.get("cost", {})
    totals, unmeasured = {}, []
    for spec in specs:
        entry = cost.get(spec.model, {}).get(spec.regime)
        key = f"{spec.model}/{spec.regime}"
        if not entry or not entry.get("s_per_microstep"):
            if key not in unmeasured:
                unmeasured.append(key)
            continue
        hours = spec.resolve_steps(n_images) * spec.grad_accum * entry["s_per_microstep"] / 3600
        t = totals.setdefault(key, {"runs": 0, "hours": 0.0, "usd": 0.0,
                                    "usd_per_hour": entry.get("usd_per_hour", 0.0),
                                    "source": entry.get("source", "")})
        t["runs"] += 1
        t["hours"] += hours
        t["usd"] += hours * entry.get("usd_per_hour", 0.0)
    for t in totals.values():
        t["hours"], t["usd"] = round(t["hours"], 1), round(t["usd"], 2)
    return {"runs": len(specs), "by_model_regime": totals,
            "total_hours": round(sum(t["hours"] for t in totals.values()), 1),
            "total_usd": round(sum(t["usd"] for t in totals.values()), 2),
            "unmeasured": unmeasured}


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_spec.py <matrix.json> [manifest.json]")
    config = json.loads(Path(sys.argv[1]).read_text())
    specs = expand_matrix(config)
    n_images = len(json.loads(Path(sys.argv[2]).read_text())["clean_train"]) if len(sys.argv) > 2 else 3200
    for spec in specs:
        print(f"{spec.experiment}/{spec.run_id}  steps={spec.resolve_steps(n_images)}")
    print(json.dumps(estimate(specs, config, n_images), indent=2))


if __name__ == "__main__":
    main()
