"""
Measure what a finished poison pool did to the VAE latent, per image, with nothing being trained or optimised.

This exists for the paper's mechanism section. H1 says Nightshade-style poisoning shifts SD-2's U-Net but leaves
SD3-Medium's MM-DiT unmoved at the same pixel budget, and the explanation on offer is that the attack buys less
displacement in SD3's 16-channel latent than in SD-2's 4-channel one. `poison_generation.json` already hints at
that, but only as the median of the optimiser's own loss, and the two models' losses are on different scales --
SD-2's pools start near 800, SD3's near 200 -- so the medians cannot be compared directly.

What this script reports instead is scale-free and per image. For each poison already on disk it encodes three
images with the same VAE the fine-tune uses:

    z_clean    the untouched COCO image
    z_poison   the published image, read back from disk as uint8 so quantisation is included
    z_target   whatever that poison was matched against (the generated anchor, or its paired real image)

and measures how far the perturbation travelled along the straight line from the clean latent to the target:

    gap        ||z_target - z_clean||                 how far the attack would have to go to fully arrive
    closed     <z_poison - z_clean, u> / ||gap||      u = unit(gap); the fraction of that distance covered
    cosine     <z_poison - z_clean, u> / ||z_poison - z_clean||
                                                      how much of the movement was *toward* the target at all
    off_axis   the part of the displacement perpendicular to u, relative to the gap

`closed` is the number the mechanism claim needs: it is a ratio of two distances in the same latent space, so
SD-2 and SD3 can be put in one table. `cosine` separates "moved a little, straight at the target" from "moved a
lot, mostly sideways" -- two pools with the same `closed` are not doing the same thing if their cosines differ.
Both are computed on the scaled latent the training loop actually sees, and on the raw posterior mean that
Nightshade's own objective matches, because the shift/scale differs between the two VAEs and only the scaled one
is what fine-tuning consumes.

Nothing here perturbs an image, proposes a perturbation, or touches a denoiser: it reads finished files and
reports distances, the diffusion-model analogue of measuring how far apart two feature vectors are. Run it after
a pool is built, and quote it in the paper next to the ASR tables.

    python poison_diag.py --attack nightshade_sd2 --pair 'cat->dog' --limit 32 \
        --poison-root /data/poisons --images-root /data/chimera_coco/images --anchors /data/anchors \
        --manifest /data/chimera_coco/manifest.json --out results/compute/poison_diag.json

Pass `--all` to cover every pool the manifest and poison root have in common. The output merges into the named
json, so measuring one pool never erases another (notes/failures.md A7).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from poison_gen import ATTACK_SPEC, anchor_path, plan
from run_spec import MODELS, poison_dir_for

# One image at a time would waste the GPU; a few dozen latents are nothing to hold.
BATCH = 8
RESOLUTION = 512


def _load_rgb(path, res: int):
    """Read a file to a float tensor in [-1, 1], the range the VAE expects (notes/failures.md A2)."""
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).float().div_(127.5).sub_(1.0)
    return x.permute(2, 0, 1)


def _stats(values: list[float]) -> dict:
    """Median plus the spread, because a median alone hides a pool that is bimodal."""
    import statistics

    v = sorted(values)
    if not v:
        return {}
    n = len(v)
    q = lambda p: v[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {"n": n, "min": round(v[0], 4), "p25": round(q(0.25), 4), "median": round(statistics.median(v), 4),
            "p75": round(q(0.75), 4), "max": round(v[-1], 4), "mean": round(statistics.fmean(v), 4),
            "stdev": round(statistics.stdev(v), 4) if n > 1 else 0.0}


def measure_pool(attack: str, pair: str, manifest: dict, images_root, poison_root, anchors_root,
                 limit: int | None = None, device="cuda", res: int = RESOLUTION, batch: int = BATCH) -> dict:
    """
    Per-image latent geometry for one (attack, pair) pool that has already been generated.

    Only ids with a file on disk are measured, so a partially built pool reports what it has rather than
    failing. The poison is read back from its saved uint8 file, which is what a victim would actually train on.
    """
    import torch
    from diffusers import AutoencoderKL

    spec = ATTACK_SPEC[attack]
    if spec["anchor"] == "caption":
        # `chimera_pred` is matched against the frozen model's own prediction under a swapped caption, not
        # against any image, so there is no target latent to measure a distance to. Its failure is explained
        # in notes/failures.md C3 and does not need this measurement.
        raise ValueError(f"{attack!r} has no target image (anchor={spec['anchor']!r}): nothing to measure a "
                         f"latent distance against. Measure the image-anchored pools instead.")
    model = spec["model"]
    pretrained = MODELS[model]
    poison_dir = Path(poison_dir_for(poison_root, attack, pair))

    rows = [r for r in plan(manifest, attack, pair, images_root, anchors_root)
            if (poison_dir / f"{r['id']}.png").exists()]
    if not rows:
        raise FileNotFoundError(f"no poison images under {poison_dir}")
    if limit:
        rows = rows[:limit]

    # fp32 throughout: these are distances between nearly identical latents, and bf16's ~3 decimal digits
    # would be a large fraction of the quantity being measured (notes/failures.md A1 is the same hazard).
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", torch_dtype=torch.float32).to(device)
    vae.requires_grad_(False).eval()
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    scale = vae.config.scaling_factor

    t0 = time.time()
    per_image, anchors_seen = [], {}
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            clean = torch.stack([_load_rgb(r["path"], res) for r in chunk]).to(device)
            poison = torch.stack([_load_rgb(poison_dir / f"{r['id']}.png", res) for r in chunk]).to(device)
            target = torch.stack([_load_rgb(r["target_image"], res) for r in chunk]).to(device)

            mean = {k: vae.encode(v).latent_dist.mean for k, v in
                    (("clean", clean), ("poison", poison), ("target", target))}
            for space in ("raw", "scaled"):
                z = mean if space == "raw" else {k: (v - shift) * scale for k, v in mean.items()}
                gap = (z["target"] - z["clean"]).flatten(1)
                move = (z["poison"] - z["clean"]).flatten(1)
                gap_n = gap.norm(dim=1)
                move_n = move.norm(dim=1)
                u = gap / gap_n.clamp_min(1e-12).unsqueeze(1)
                along = (move * u).sum(dim=1)
                perp = (move - along.unsqueeze(1) * u).norm(dim=1)
                for i, r in enumerate(chunk):
                    # "raw" is measured first, so that pass is the one that creates each row.
                    if space == "raw":
                        per_image.append({"id": r["id"], "target_image": Path(r["target_image"]).name})
                    per_image[start + i][space] = {
                        "gap": round(float(gap_n[i]), 4),
                        "moved": round(float(move_n[i]), 4),
                        "closed": round(float(along[i] / gap_n[i].clamp_min(1e-12)), 6),
                        "cosine": round(float(along[i] / move_n[i].clamp_min(1e-12)), 6),
                        "off_axis": round(float(perp[i] / gap_n[i].clamp_min(1e-12)), 6),
                    }
            for r in chunk:
                anchors_seen[Path(r["target_image"]).name] = anchors_seen.get(Path(r["target_image"]).name, 0) + 1
            print(f"    {min(start + batch, len(rows))}/{len(rows)}", flush=True)

    channels = int(vae.config.latent_channels)
    del vae
    torch.cuda.empty_cache()
    out = {
        "attack": attack, "pair": pair, "model": model, "pretrained": pretrained,
        "anchor_kind": spec["anchor"], "space": spec["space"], "resolution": res,
        "n_images": len(per_image), "seconds": round(time.time() - t0, 1),
        "distinct_targets": len(anchors_seen),
        "vae": {"latent_channels": channels, "shift_factor": float(shift), "scaling_factor": float(scale)},
        "images": per_image,
    }
    # The headline summary is on the scaled latent, since that is what the fine-tune's loss sees.
    for space in ("raw", "scaled"):
        out[space] = {k: _stats([r[space][k] for r in per_image])
                      for k in ("closed", "cosine", "off_axis", "gap", "moved")}
    return out


def _vae_channels(pretrained: str) -> int | None:
    """Read the channel count out of the VAE config without downloading weights, for the printed table."""
    try:
        from diffusers import AutoencoderKL

        cfg = AutoencoderKL.load_config(pretrained, subfolder="vae")
        return int(cfg["latent_channels"])
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack")
    ap.add_argument("--pair", default="cat->dog")
    ap.add_argument("--all", action="store_true", help="every (attack, pair) that has images on disk")
    ap.add_argument("--manifest", default="data/chimera_coco/manifest.json")
    ap.add_argument("--images-root", default="data/chimera_coco/images")
    ap.add_argument("--poison-root", default="data/poisons")
    ap.add_argument("--anchors", default="data/anchors")
    ap.add_argument("--limit", type=int, default=0, help="first N ids of each pool (0 = all)")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--resolution", type=int, default=RESOLUTION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/compute/poison_diag.json")
    ap.add_argument("--tag", default="", help="suffix for the result key, e.g. 'eps0.08', so a pool built with "
                    "other settings sits beside the default one instead of replacing it")
    a = ap.parse_args(argv)

    manifest = json.loads(Path(a.manifest).read_text())
    if a.all:
        jobs = []
        for attack in sorted(ATTACK_SPEC):
            for pair in manifest["pairs"]:
                if Path(poison_dir_for(a.poison_root, attack, pair)).is_dir():
                    jobs.append((attack, pair))
    else:
        if not a.attack:
            raise SystemExit("pass --attack (and --pair), or --all")
        jobs = [(a.attack, a.pair)]
    print(f"{len(jobs)} pool(s) to measure")

    results = {}
    for attack, pair in jobs:
        print(f"  {attack}/{pair}")
        out = measure_pool(attack, pair, manifest, a.images_root, a.poison_root, a.anchors,
                           limit=a.limit or None, device=a.device, res=a.resolution, batch=a.batch)
        # The generator's own record of how the pool was built, so an eps-sweep entry carries its eps.
        man = Path(poison_dir_for(a.poison_root, attack, pair)) / "poison_manifest.json"
        if man.exists():
            built = json.loads(man.read_text())
            out["built_with"] = {k: built["config"].get(k) for k in ("eps", "iters", "step_scale", "sigma")}
            out["built_with"].update({k: built.get(k) for k in ("compute_dtype", "device")})
        results[f"{attack}/{pair}" + (f"@{a.tag}" if a.tag else "")] = out
        s = out["scaled"]["closed"]
        c = out["scaled"]["cosine"]
        print(f"    closed {s['median']:+.3f} (p25 {s['p25']:+.3f}, p75 {s['p75']:+.3f})  "
              f"cosine {c['median']:+.3f}  on {out['n_images']} images, {out['seconds']:.0f}s")

    # Merge, never overwrite (notes/failures.md A7).
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(path.read_text()) if path.exists() else {}
    replaced = sorted(set(merged) & set(results))
    merged.update(results)
    path.write_text(json.dumps(merged, indent=2))
    print(f"\n{len(results)} written, {len(merged)} entries total"
          + (f", replaced {', '.join(replaced)}" if replaced else "") + f" -> {path}")

    # A compact table is what gets read; the per-image rows stay in the json for the plots.
    print(f"\n{'pool':34s} {'ch':>3s} {'closed (median)':>16s} {'cosine':>8s} {'off-axis':>9s}")
    for key, out in sorted(merged.items()):
        ch = out.get("vae", {}).get("latent_channels") or _vae_channels(out["pretrained"]) or "?"
        print(f"{key:34s} {str(ch):>3s} {out['scaled']['closed']['median']:>16.4f} "
              f"{out['scaled']['cosine']['median']:>8.4f} {out['scaled']['off_axis']['median']:>9.4f}")
    return merged


if __name__ == "__main__":
    main()
