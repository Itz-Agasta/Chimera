"""
Measure, for a finished poison pool, how far its fine-tuning gradient points toward the dirty-label gradient.

Fine-tuning never sees a latent distance or an attack objective -- it sees the gradient of its own loss on each
(image, caption) pair. So the question "will this poison set teach 'cat' -> dog" is, to first order, "does the
loss gradient on (poison, cat caption) point where the gradient on (real dog, cat caption) points". Dirty-label
is the attack that teaches the swap by construction, so its gradient is the direction to compare against. This
is the quantity Witches' Brew (Geiping et al., ICLR'21) optimises; here it is only measured, on pools that
already exist, so it costs one backward pass per input and no double backprop.

Sets, all over the 100 COCO ids the Gate B cat->dog seed-0 cell poisons, slot i always holding the same id and
caption:
    clean    the untouched cat image i                       the "did nothing" reference
    dirty    the real dog that the dirty-label cell pairs with slot i, captioned with cat caption i
    random   cat i plus uniform sign noise at the same L-inf budget, quantised to uint8
                                                             the null: any perturbation moves the gradient a little
    <pool>   the published poison for cat i, read back from its PNG

Every set's slot i, draw k is computed with torch's RNG seeded identically, so `train.flow_loss` /
`train.ddpm_loss` -- called verbatim, not re-implemented -- draw the same noise and timestep for all of them.
The latents come from `train.encode_latents`, i.e. exactly what the fine-tune trains on.

Reported per pool, from the exact mean gradients G over all slots and draws (G_c clean, G_d dirty, G_p pool):
    raw_cos        cos(G_p, G_d), with raw_cos_baseline = cos(G_c, G_d)
    shift_cos      cos(G_p - G_c, G_d - G_c)       does the perturbation move the gradient toward dirty-label?
    grad_closed    <G_p - G_c, G_d - G_c> / ||G_d - G_c||^2
                                                   the fraction of dirty-label's gradient change the poison buys,
                                                   the gradient-space twin of poison_diag's `closed`
    shift_norm     ||G_p - G_c|| / ||G_d - G_c||
Same-slot numbers share captions and draws with dirty-label, which flatters them; `cross` repeats the shift
metrics between independent halves of the slots (pool on even slots vs dirty-label on odd, and back), and
`ceiling_cross` is the same with dirty-label in both roles -- the most any pool could score under that noise.
The `random` set is the null. Bootstrap CIs over slots come from a count-sketch of every per-slot gradient
(cosines survive the sketch to ~1/sqrt(dim)); the headline numbers use the exact vectors.

Gradients are taken on a parameter subset (`--subset`), because full-model gradient vectors are 2B floats:
the attention parameters of the last few transformer blocks for SD3, of the last up-blocks for SD-2's U-Net.

    python grad_align.py --model sd3_medium --pools nightshade_sd3,chimera_pred,chimera_anchor \
        --data-root /data --poison-root /data/poisons --out results/compute/grad_align.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_spec import RunSpec, build_inputs, poison_dir_for, poison_selection
from text_cache import TextCache
from train import autocast_dtype, build_model, ddpm_loss, encode_latents, flow_loss

PAIR = "cat->dog"
EPS = 0.04            # the shared attack budget (poison_gen.EPS), for the random-noise null
SKETCH_DIM = 16384
SEED_BASE = 20261028


def subset_params(net, model: str, blocks: int) -> dict:
    """Name -> parameter for the gradient subset. Late attention: where the concept binding is read out."""
    if model.startswith("sd3"):
        n = len(net.transformer_blocks)
        keep = tuple(f"transformer_blocks.{i}.attn." for i in range(n - blocks, n))
    else:
        n = len(net.up_blocks)
        keep = tuple(f"up_blocks.{i}.attentions." for i in range(n - blocks, n))
    chosen = {k: p for k, p in net.named_parameters() if k.startswith(keep)}
    if not chosen:
        raise ValueError(f"no parameters matched {keep}")
    return chosen


class Sketch:
    """Count-sketch of the concatenated subset gradient: fixed bucket and sign per coordinate."""

    def __init__(self, params: dict, dim: int, device):
        import torch

        g = torch.Generator(device="cpu").manual_seed(SEED_BASE)
        self.dim, self.idx, self.sign = dim, {}, {}
        for k, p in params.items():
            self.idx[k] = torch.randint(0, dim, (p.numel(),), generator=g, dtype=torch.int32).to(device)
            self.sign[k] = (torch.randint(0, 2, (p.numel(),), generator=g, dtype=torch.int8) * 2 - 1).to(device)

    def __call__(self, grads: dict):
        import torch

        out = torch.zeros(self.dim, device=next(iter(grads.values())).device, dtype=torch.float32)
        for k, g in grads.items():
            out.index_add_(0, self.idx[k].long(), g.flatten().float() * self.sign[k])
        return out


def random_noise_images(records, out_dir: Path, res: int):
    """Cat i plus uniform sign noise at the attack budget, quantised exactly as a published poison would be."""
    import torch
    from PIL import Image

    from poison_gen import to_uint8
    from train import load_image

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        x = load_image(r["path"], res)
        g = torch.Generator().manual_seed(SEED_BASE + int(r["id"]))
        delta = (torch.randint(0, 2, x.shape, generator=g).float() * 2 - 1) * (2 * EPS)
        path = out_dir / f"{r['id']}.png"
        Image.fromarray(to_uint8((x + delta).clamp(-1, 1))).save(path)
        rows.append({**r, "path": str(path)})
    return rows


def cos(a, b) -> float:
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def shift_metrics(gp, gc, gd) -> dict:
    dp, dd = gp - gc, gd - gc
    return {"shift_cos": round(cos(dp, dd), 5),
            "grad_closed": round(float((dp @ dd) / (dd @ dd).clamp_min(1e-30)), 5),
            "shift_norm": round(float(dp.norm() / dd.norm().clamp_min(1e-30)), 5)}


def run(model: str, pools: list[str], data_root, poison_root, draws: int, blocks: int, limit: int | None,
        work: Path, device="cuda") -> dict:
    import numpy as np
    import torch

    data_root = Path(data_root)
    manifest = json.loads((data_root / "chimera_coco" / "manifest.json").read_text())
    images = data_root / "chimera_coco" / "images"
    spec = RunSpec(model=model, regime="full_ft", attack="nightshade_sd3", pair=PAIR, poison_count=100, seed=0)
    chosen = poison_selection(manifest, spec)
    dirty_spec = RunSpec(model=model, regime="full_ft", attack="dirty_label", pair=PAIR, poison_count=100,
                         seed=0, clean_filler=0, source_clean=0)
    dirty, _ = build_inputs(manifest, dirty_spec, images)
    if limit:
        chosen, dirty = chosen[:limit], dirty[:limit]
    clean = [{"id": r["id"], "path": str(images / f"{r['id']}.jpg"), "caption": r["caption"], "poisoned": True}
             for r in chosen]
    assert all(d["caption"] == c["caption"] for d, c in zip(dirty, clean)), "slot captions must match"

    sets = {"clean": clean, "dirty": dirty,
            "random": random_noise_images(clean, work / f"random_{model}", spec.resolution)}
    for pool in pools:
        pdir = Path(poison_dir_for(poison_root, pool, PAIR))
        sets[pool] = [{**r, "path": str(pdir / f"{r['id']}.png")} for r in clean]
        missing = [r["path"] for r in sets[pool] if not Path(r["path"]).exists()]
        if missing:
            raise FileNotFoundError(f"{pool}: {len(missing)} poisons missing, e.g. {missing[0]}")

    # Per set, because clean and poison share ids and encode_latents keys its output by id.
    t0 = time.time()
    latents = {name: encode_latents(recs, spec, device, cache_path=None) for name, recs in sets.items()}
    encode_s = time.time() - t0
    bad = [n for n, d in latents.items() if not all(bool(v.float().isfinite().all()) for v in d.values())]
    if bad:
        raise FloatingPointError(f"non-finite latents in {bad}: the {autocast_dtype()} VAE overflowed")

    net, scheduler = build_model(spec, device)
    net.requires_grad_(False)
    # Backward stops at the first trainable block, so nothing before it keeps activations: checkpointing
    # would only re-run forward passes.
    net.disable_gradient_checkpointing()
    params = subset_params(net, model, blocks)
    for p in params.values():
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in params.values())
    names = list(params)
    sketch = Sketch(params, SKETCH_DIM, device)
    text = TextCache(data_root / "embeds" / model)
    loss_fn = flow_loss if spec.is_flow else ddpm_loss

    def grad_of(latent, caption, i, k):
        emb = text.get(caption)
        batch = {"latent": latent[None], **{key: v[None] for key, v in emb.items()}}
        torch.manual_seed(SEED_BASE + 100_003 * k + i)   # same noise + timestep for every set at (i, k)
        loss = loss_fn(net, scheduler, batch, spec, device)
        gs = torch.autograd.grad(loss, [params[n] for n in names])
        return dict(zip(names, gs)), float(loss.detach())

    # Determinism check before spending the budget: the same (set, slot, draw) twice must give one gradient.
    ga, _ = grad_of(latents["clean"][clean[0]["id"]], clean[0]["caption"], 0, 0)
    gb, _ = grad_of(latents["clean"][clean[0]["id"]], clean[0]["caption"], 0, 0)
    flat = lambda g: torch.cat([g[n].flatten().float() for n in names])  # noqa: E731
    determinism = cos(flat(ga), flat(gb))
    del ga, gb

    # Exact sums per (set, parity of slot) on the CPU; sketches per (set, slot), averaged over draws.
    sums = {s: [torch.zeros(n_params, dtype=torch.float32) for _ in range(2)] for s in sets}
    sk = {s: np.zeros((len(clean), SKETCH_DIM), dtype=np.float32) for s in sets}
    losses = {s: [] for s in sets}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(len(clean)):
        for name, recs in sets.items():
            z = latents[name][recs[i]["id"]]
            for k in range(draws):
                g, loss = grad_of(z, recs[i]["caption"], i, k)
                sums[name][i % 2] += flat(g).cpu()
                sk[name][i] += sketch(g).cpu().numpy() / draws
                losses[name].append(loss)
        if i == 0 or (i + 1) % 10 == 0:
            rate = (time.time() - t0) / ((i + 1) * len(sets) * draws)
            print(f"  slot {i + 1}/{len(clean)}  {rate:.2f} s/backward  "
                  f"eta {rate * (len(clean) - i - 1) * len(sets) * draws / 60:.0f} min", flush=True)
    grad_s = time.time() - t0

    G = {s: v[0] + v[1] for s, v in sums.items()}
    gc, gd = G["clean"], G["dirty"]

    def cross(s):
        """Shift of set s on one half of the slots against dirty-label's shift on the other half, both ways.
        Independent slots, captions and draws, so it is comparable with the ceiling (s = dirty)."""
        d = {h: (sums[s][h] - sums["clean"][h], sums["dirty"][1 - h] - sums["clean"][1 - h]) for h in (0, 1)}
        return {"shift_cos": round(sum(cos(a, b) for a, b in d.values()) / 2, 5),
                "grad_closed": round(sum(float((a @ b) / (b @ b).clamp_min(1e-30)) for a, b in d.values()) / 2,
                                     5)}

    out = {
        "model": model, "pair": PAIR, "seed": 0, "n_slots": len(clean), "draws": draws,
        "subset": {"blocks": blocks, "n_params": n_params, "n_tensors": len(names),
                   "first": names[0], "last": names[-1]},
        "determinism_cos": round(determinism, 6),
        "ceiling_cross": cross("dirty"),
        "raw_cos_baseline": round(cos(gc, gd), 5),
        "norms": {s: round(float(v.norm()) / (len(clean) * draws), 6) for s, v in G.items()},
        "mean_loss": {s: round(float(np.mean(v)), 5) for s, v in losses.items()},
        "sets": {},
        "seconds": {"encode": round(encode_s, 1), "gradients": round(grad_s, 1),
                    "per_backward": round(grad_s / (len(clean) * len(sets) * draws), 3)},
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "device": torch.cuda.get_device_name(0), "compute_dtype": str(autocast_dtype()).replace("torch.", ""),
    }
    # Bootstrap over slots on the sketches: resample slots, recompute the shift metrics.
    rng = np.random.default_rng(SEED_BASE)
    boot_idx = [rng.integers(0, len(clean), len(clean)) for _ in range(1000)]
    # The same-slot shift metrics are biased upward by construction: G_p - G_c and G_d - G_c share the -G_c
    # term, so a perturbation that only scrambles the gradient still scores well above zero (the pilot's random
    # null read shift_cos +0.50 on SD3). The fair headline is therefore the EXCESS over the random null, paired
    # on the same bootstrap resample of slots.
    pct = lambda v: [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]  # noqa
    boot = {}
    for s in sets:
        if s in ("clean", "dirty"):
            continue
        boot[s] = {"shift_cos": [], "grad_closed": []}
        for idx in boot_idx:
            m = shift_metrics(*(torch.from_numpy(sk[x][idx].mean(0)) for x in (s, "clean", "dirty")))
            for key in boot[s]:
                boot[s][key].append(m[key])
    for s in boot:
        entry = {"raw_cos": round(cos(G[s], gd), 5), **shift_metrics(G[s], gc, gd), "cross": cross(s),
                 "ci95": {key: pct(v) for key, v in boot[s].items()}}
        if s != "random":
            exact = shift_metrics(G["random"], gc, gd)
            entry["excess_over_random"] = {
                key: {"value": round(entry[key] - exact[key], 5),
                      "ci95": pct(np.array(boot[s][key]) - np.array(boot["random"][key]))}
                for key in ("shift_cos", "grad_closed")}
        out["sets"][s] = entry
    np.savez_compressed(work / f"grad_sketch_{model}.npz", **{s: v.astype(np.float16) for s, v in sk.items()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=("sd3_medium", "sd2_base"))
    ap.add_argument("--pools", required=True, help="comma-separated poison pools to measure")
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--poison-root", default="/data/poisons")
    ap.add_argument("--draws", type=int, default=8, help="(t, noise) draws per slot")
    ap.add_argument("--blocks", type=int, default=0, help="late blocks in the subset (0 = 4 for SD3, 2 for SD-2)")
    ap.add_argument("--limit", type=int, default=0, help="first N slots only (pilot)")
    ap.add_argument("--work", default="grad_align_work")
    ap.add_argument("--out", default="results/compute/grad_align.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    blocks = a.blocks or (4 if a.model.startswith("sd3") else 2)
    out = run(a.model, [p for p in a.pools.split(",") if p], a.data_root, a.poison_root, a.draws, blocks,
              a.limit or None, work)

    # Merge, never overwrite (notes/failures.md A7).
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(path.read_text()) if path.exists() else {}
    key = f"{a.model}/{PAIR}" + (f"@{a.tag}" if a.tag else "")
    replaced = key in merged
    merged[key] = out
    path.write_text(json.dumps(merged, indent=2))
    print(f"\n{key} written{' (replaced)' if replaced else ''}, {len(merged)} entries -> {path}")
    print(f"determinism {out['determinism_cos']}  ceiling (cross-half) {out['ceiling_cross']}  "
          f"raw baseline {out['raw_cos_baseline']}")
    for s, e in out["sets"].items():
        if "excess_over_random" in e:
            x = e["excess_over_random"]["grad_closed"]
            print(f"  {s:16s} grad_closed EXCESS over random {x['value']:+.4f} {x['ci95']}")
        print(f"  {s:16s} shift_cos {e['shift_cos']:+.4f} {e['ci95']['shift_cos']}  "
              f"grad_closed {e['grad_closed']:+.4f} {e['ci95']['grad_closed']}  raw {e['raw_cos']:+.4f}  "
              f"cross {e['cross']}")
    return out


if __name__ == "__main__":
    main()
