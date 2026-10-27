"""
Measure how visible the poisons are, for the paper's stealth table.

Generation already records L-infinity and PSNR per image, which are free to compute. This adds the two
perceptual measures the attack literature is judged on and which need networks:

  LPIPS (AlexNet)     perceptual distance, the measure Nightshade's paper uses to bound its perturbation
  DINOv2 cosine       self-supervised feature similarity: does the poison still look like the same thing
                      to a representation that was never trained on these images

Run over the saved PNGs rather than inside the optimisation, so every attack is measured identically,
afterwards, by one pass that knows nothing about how the images were made. That is also why no attack carries
an LPIPS term in its loss -- see poison_gen.py.

The comparison is poison vs the clean source image, loaded through the same loader training uses, so what is
measured is exactly the difference the fine-tune sees.

  python experiments/common/stealth.py --poison-root /data/poisons --data-root /data --out stealth.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_spec import PERTURBATION_ATTACKS, image_path, pair_slug, poison_dir_for
from train import load_image

DINO = "facebook/dinov2-base"


def quantiles(values: list[float]) -> dict:
    """Min / median / mean / max, so the table can report a spread rather than one number."""
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {"n": n, "min": round(ordered[0], 4), "median": round(median, 4),
            "mean": round(sum(ordered) / n, 4), "max": round(ordered[-1], 4)}


def measure(pairs: list[tuple[Path, Path]], device, res: int = 512, batch: int = 8) -> dict:
    """
    (clean_path, poison_path) -> the per-image measures, aggregated.

    Both images go through train.load_image, so the tensors are the ones training would see: a poison PNG is
    already `res` square and is read back verbatim, and the clean JPEG is resized and centre-cropped the same
    way it is for a clean run.
    """
    import lpips as lpips_pkg
    import torch
    from transformers import AutoImageProcessor, AutoModel

    lpips_fn = lpips_pkg.LPIPS(net="alex").to(device).eval()
    lpips_fn.requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(DINO)
    dino = AutoModel.from_pretrained(DINO).to(device).eval()

    out = {"lpips": [], "dino_cosine": [], "psnr_db": [], "linf_255": [], "l2_255": []}
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        clean = torch.stack([load_image(c, res) for c, _ in chunk]).to(device)
        poison = torch.stack([load_image(p, res) for _, p in chunk]).to(device)
        with torch.no_grad():
            out["lpips"] += lpips_fn(poison, clean).flatten().tolist()

            # DINOv2 wants its own normalisation of [0,1] pixels, not the [-1,1] the VAE takes.
            def features(x):
                pixels = processor(images=list(((x + 1) / 2).clamp(0, 1).cpu()), return_tensors="pt",
                                   do_rescale=False)["pixel_values"].to(device)
                cls = dino(pixel_values=pixels).last_hidden_state[:, 0]
                return torch.nn.functional.normalize(cls, dim=-1)

            out["dino_cosine"] += (features(poison) * features(clean)).sum(-1).tolist()

            diff = (poison - clean) * 127.5
            mse = diff.pow(2).flatten(1).mean(1)
            out["psnr_db"] += (10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-12))).tolist()
            out["linf_255"] += diff.abs().flatten(1).max(1).values.tolist()
            out["l2_255"] += diff.pow(2).flatten(1).mean(1).sqrt().tolist()
        print(f"  {min(start + batch, len(pairs))}/{len(pairs)}", flush=True)
    return {k: quantiles(v) for k, v in out.items()}


def collect(poison_root, images_root, manifest: dict) -> dict:
    """Every (attack, pair) directory that has images, and the clean/poison path pairs inside it."""
    poison_root = Path(poison_root)
    images_root = Path(images_root)
    by_id = {r["id"]: r for p in manifest["pairs"].values() for r in p["poison_pool"]}

    found = {}
    for attack in PERTURBATION_ATTACKS:
        for pair in manifest["pairs"]:
            directory = poison_dir_for(poison_root, attack, pair)
            if not directory.is_dir():
                continue
            rows = []
            for png in sorted(directory.glob("*.png")):
                coco_id = int(png.stem)
                if coco_id not in by_id:
                    raise ValueError(f"{png} is not in any pair's poison pool")
                rows.append((image_path(images_root, by_id[coco_id]), png))
            if rows:
                found[f"{attack}/{pair_slug(pair)}"] = rows
    return found


def run(poison_root, data_root, device="cuda", res=512, batch=8) -> dict:
    data_root = Path(data_root)
    manifest = json.loads((data_root / "chimera_coco" / "manifest.json").read_text())
    groups = collect(poison_root, data_root / "chimera_coco" / "images", manifest)
    if not groups:
        raise SystemExit(f"no poison images under {poison_root}")

    results = {}
    for name, rows in groups.items():
        print(f"{name}: {len(rows)} images", flush=True)
        results[name] = measure(rows, device, res, batch)
        manifest_path = Path(rows[0][1]).parent / "poison_manifest.json"
        if manifest_path.exists():
            config = json.loads(manifest_path.read_text())["config"]
            results[name]["config"] = {k: config[k] for k in ("iters", "eps", "step_scale", "target", "space",
                                                              "anchor", "sigma") if k in config}
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--poison-root", required=True)
    p.add_argument("--data-root", default="/data")
    p.add_argument("--out", default="stealth.json")
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args(argv)

    import torch

    results = run(a.poison_root, a.data_root, "cuda" if torch.cuda.is_available() else "cpu", batch=a.batch)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
