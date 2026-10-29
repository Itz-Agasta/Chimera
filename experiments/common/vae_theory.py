"""
Two theory-driven measurements of *why* the same Nightshade budget moves SD-2 and not SD3, on poisons that already
exist. Nothing is trained or optimised.

A. First-order displacement theory. Linearise the encoder at the clean image, z(x + d) ~ z(x) + J d. Over the
   L-inf box |d| <= e (e = 2 eps in [-1, 1] pixel units, intersected with the valid pixel range), the largest
   movement along u = unit(z_target - z_clean) is exactly
        lin_move = <g, d*>,   g = J^T u,   d* = clamp(e * sign(g), -1 - x, 1 - x)   (~ e * ||g||_1)
   so the fraction of the gap one perfect linear step can close is
        lin_closed = lin_move / ||gap||
   which splits SD3's low measured `closed` into two separable causes: **gain** (||g||_1, how hard the pixels can
   push the latent toward the target at all) and **distance** (||gap||, how far the target is). Also reported:
   `onestep_closed`, the true (nonlinear) closed of that single signed step after uint8 quantisation, and
   `measured_closed`, the published poison's own closed (= poison_diag's number, recomputed here).
   For eps in EPS_GRID this gives a *prediction* for the eps sweep (P0.5c) that is written down before that
   sweep's results are read: `calibrated` = measured_closed(0.04) * onestep(eps) / onestep(0.04).

B. What the fine-tune is taught. Training on a poison means training on its latent. Decoding the poison's latent
   mean back to pixels shows what the model is being taught to draw for the source caption. Each image is judged
   three ways with the same zero-shot judges the ASR uses:
        pixels       the published poison PNG                        (Nightshade's claim: still the source)
        decoded      VAE.decode(VAE.encode(poison))                  what the denoiser is fitted to
        decoded_clean VAE.decode(VAE.encode(clean original))         reconstruction control
   Prediction under H1: decoded SD-2 poisons often read as the target, decoded SD3 poisons still as the source.

    python vae_theory.py --model sd3_medium --data-root /data --out results/compute/vae_theory.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from poison_diag import _load_rgb, _stats
from poison_gen import from_uint8, plan, to_uint8
from run_spec import MODELS, poison_dir_for

EPS_GRID = (0.04, 0.08, 0.16, 0.32)
ATTACK = {"sd2_base": "nightshade_sd2", "sd3_medium": "nightshade_sd3"}
JUDGES = ("clip_l14", "siglip")
RES = 512


def measure(model: str, pair: str, manifest: dict, data_root: Path, judges: dict, limit=None, batch=1,
            device="cuda") -> dict:
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image

    attack = ATTACK[model]
    info = manifest["pairs"][pair]
    source, target = info["source"], info["target"]
    pdir = Path(poison_dir_for(data_root / "poisons", attack, pair))
    rows = [r for r in plan(manifest, attack, pair, data_root / "chimera_coco" / "images", data_root / "anchors")
            if (pdir / f"{r['id']}.png").exists()]
    if limit:
        rows = rows[:limit]

    # fp32 for the same reason as poison_diag: the quantities are small differences of latents.
    vae = AutoencoderKL.from_pretrained(MODELS[model], subfolder="vae", torch_dtype=torch.float32).to(device)
    vae.requires_grad_(False).eval()
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    scale = vae.config.scaling_factor
    enc = lambda x: (vae.encode(x).latent_dist.mean - shift) * scale   # noqa: E731  training's latent space
    dec = lambda z: vae.decode(z / scale + shift).sample                # noqa: E731

    def to_pil(x):
        return Image.fromarray(to_uint8(x))

    per, pics = [], {"pixels": [], "decoded": [], "decoded_clean": []}
    t0 = time.time()
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        x = torch.stack([_load_rgb(r["path"], RES) for r in chunk]).to(device)
        xp = torch.stack([_load_rgb(pdir / f"{r['id']}.png", RES) for r in chunk]).to(device)
        xt = torch.stack([_load_rgb(r["target_image"], RES) for r in chunk]).to(device)
        with torch.no_grad():
            zc, zp, zt = enc(x), enc(xp), enc(xt)
        gap = (zt - zc).flatten(1)
        gap_n = gap.norm(dim=1)
        u = (gap / gap_n.unsqueeze(1)).view_as(zc)

        # A: g = J^T u at the clean image, one vector-Jacobian product.
        xg = x.clone().requires_grad_(True)
        (enc(xg) * u).sum().backward()
        g = xg.grad.detach()
        with torch.no_grad():
            along = lambda z: ((z - zc).flatten(1) * u.flatten(1)).sum(1) / gap_n  # noqa: E731
            measured = along(zp)
            rec = [{"id": r["id"], "gap": float(gap_n[i]), "gain_l1": float(g[i].abs().sum()),
                    "measured_closed": float(measured[i]), "eps": {}} for i, r in enumerate(chunk)]
            for eps in EPS_GRID:
                d = torch.clamp(2 * eps * g.sign(), -1 - x, 1 - x)
                lin = (g * d).flatten(1).sum(1) / gap_n
                xs = torch.stack([from_uint8(to_uint8(a), device) for a in (x + d).clamp(-1, 1)])
                one = along(enc(xs))
                for i in range(len(chunk)):
                    rec[i]["eps"][str(eps)] = {"lin_closed": float(lin[i]), "onestep_closed": float(one[i])}
            # B: what the denoiser is fitted to.
            dp, dc = dec(zp).clamp(-1, 1), dec(zc).clamp(-1, 1)
            for i in range(len(chunk)):
                pics["pixels"].append(to_pil(xp[i]))
                pics["decoded"].append(to_pil(dp[i]))
                pics["decoded_clean"].append(to_pil(dc[i]))
        per.extend(rec)
        if start == 0 or (start // batch) % 10 == 0:
            print(f"    {model} {pair} {start + len(chunk)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    del vae
    torch.cuda.empty_cache()
    out = {"model": model, "pair": pair, "attack": attack, "source": source, "target": target,
           "n_images": len(per), "seconds_vae": round(time.time() - t0, 1)}
    for i, rec in enumerate(per):
        rec["ratio_gain_over_gap"] = rec["gain_l1"] / rec["gap"]
    out["A"] = {k: _stats([r[k] for r in per]) for k in ("measured_closed", "gap", "gain_l1", "ratio_gain_over_gap")}
    out["A"]["eps"] = {}
    m04 = _stats([r["measured_closed"] for r in per])["median"]
    for eps in EPS_GRID:
        lin = _stats([r["eps"][str(eps)]["lin_closed"] for r in per])
        one = _stats([r["eps"][str(eps)]["onestep_closed"] for r in per])
        out["A"]["eps"][str(eps)] = {"lin_closed": lin, "onestep_closed": one}
    one04 = out["A"]["eps"]["0.04"]["onestep_closed"]["median"]
    for eps in EPS_GRID:
        e = out["A"]["eps"][str(eps)]
        e["calibrated_prediction"] = round(m04 * e["onestep_closed"]["median"] / one04, 4) if one04 else None

    out["B"] = {}
    for jname, judge in judges.items():
        out["B"][jname] = {}
        for kind, images in pics.items():
            res = judge.classify(images)
            out["B"][jname][kind] = {
                "target_rate": round(sum(r["label"] == target for r in res) / len(res), 4),
                "source_rate": round(sum(r["label"] == source for r in res) / len(res), 4),
                "mean_p_target": round(sum(r["probs"][target] for r in res) / len(res), 4),
                "mean_p_source": round(sum(r["probs"][source] for r in res) / len(res), 4),
            }
            for rec, r in zip(per, res):
                rec.setdefault("judged", {}).setdefault(jname, {})[kind] = r["label"]
    out["images"] = per
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(ATTACK))
    ap.add_argument("--pairs", default="", help="comma-separated; default every pair in the manifest")
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/compute/vae_theory.json")
    a = ap.parse_args(argv)

    from judge import load_judge

    data_root = Path(a.data_root)
    manifest = json.loads((data_root / "chimera_coco" / "manifest.json").read_text())
    pairs = [p for p in a.pairs.split(",") if p] or list(manifest["pairs"])
    judges = {j: load_judge(j) for j in JUDGES}
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for pair in pairs:
        out = measure(a.model, pair, manifest, data_root, judges, limit=a.limit or None)
        results[f"{a.model}/{pair}"] = out
        # Written after every pair: a crash on pair 4 once took pairs 1-3 with it.
        merged = json.loads(path.read_text()) if path.exists() else {}
        merged[f"{a.model}/{pair}"] = out
        path.write_text(json.dumps(merged, indent=2))
        A, B = out["A"], out["B"]["clip_l14"]
        print(f"  {a.model} {pair}: closed {A['measured_closed']['median']:.3f}  gap {A['gap']['median']:.1f}  "
              f"gain {A['gain_l1']['median']:.1f}  lin@0.04 {A['eps']['0.04']['lin_closed']['median']:.3f}  "
              f"one@0.04 {A['eps']['0.04']['onestep_closed']['median']:.3f}  | decoded->target "
              f"{B['decoded']['target_rate']:.2f} (pixels {B['pixels']['target_rate']:.2f}, "
              f"clean recon {B['decoded_clean']['target_rate']:.2f})", flush=True)

    merged = json.loads(path.read_text())
    print(f"{len(results)} written, {len(merged)} total -> {path}")


if __name__ == "__main__":
    main()
