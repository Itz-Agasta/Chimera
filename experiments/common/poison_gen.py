"""
Generate the optimised poison images that the perturbation attacks train on.

Four attacks, one optimiser. Every attack perturbs a source-concept image under the same L-infinity budget with
the same projected signed-gradient schedule, so the only thing that differs between them is what the
perturbation is optimised to match:

    attack           space      target the perturbed image is pulled toward
    nightshade_sd2   latent     VAE latent of an SD-2-generated "A photo of a <target>"
    nightshade_sd3   latent     VAE latent of an SD3-generated  "A photo of a <target>"   (the adaptive baseline)
    chimera_anchor   velocity   clean latent of a real COCO target-concept image           (ablation)
    chimera_pred     velocity   clean latent the frozen model predicts for this image's own caption with the
                                concept swapped ("a cat on a couch" -> "a dog on a couch")  (the full method)

The latent attacks are the Nightshade objective; the optimiser, the budget (eps = 0.04 in [0,1] pixel units),
the 500 iterations, the step annealing and the generated anchor with its fixed seed 123 are all taken from the
authors' released implementation (github.com/Shawn-Shan/nightshade-release, opt.py). Holding all of that fixed
and changing only the matched quantity is what lets the paper attribute the difference to latent collision vs
velocity matching rather than to tuning.

Three deliberate departures from the drafts in experiments/h2_lpi/lpi_algorithm.py:
  * **A fixed target.** The draft evaluated the target velocity at the *perturbed* image's noisy latent and
    detached it. Expand the difference and z_t cancels, leaving sigma * ||v_tgt(z_t) - v_src(z_t)||, in which
    v_tgt still depends on the perturbation through z_t -- so the gradient chased a target that moved away
    from it, and the objective rose from 6.66 to 8.74 over 500 iterations instead of falling
    (results/compute/poison_probe_chimera_pred_moving_target.json). The target is now evaluated once, on the
    clean image: "the clean latent the frozen model predicts for this image when told it shows the target
    concept". That is a point the perturbation can be pulled toward, and it makes all four attacks
    structurally identical -- pull toward a fixed target -- differing only in what the target is.
    `--moving-target` reproduces the draft, kept so the paper can show why it is not used.
  * Noise is drawn once per image from its COCO id, not resampled each iteration, so the objective is a fixed
    function (its loss curve means something) and a poison is reproducible bit for bit.
  * No LPIPS term in the loss (`--lpips-weight` defaults to 0). The released Nightshade loss has none, so
    adding one to our attack only would buy stealth the baseline was not given. LPIPS and DINOv2 are measured
    afterwards from the saved images, for every attack alike.
  * The timestep is not hand-computed as t * 1000. It is the scheduler index whose sigma is closest to
    `--sigma`, so the poison is optimised at exactly a (timestep, sigma) pair the training loop can sample.

Images are written as `<poison_root>/<attack>/<pair_slug>/<coco_id>.png`, 512 px, which is the size and the
loader training uses, so the optimised pixels reach the fine-tune unchanged. The objective is re-measured after
rounding to uint8 -- what the PNG actually stores -- and that is the number the manifest reports.

  python experiments/common/poison_gen.py --attack chimera_pred --pair 'cat->dog' \
      --data-root /data --poison-root /data/poisons --anchors /data/anchors
  python experiments/common/poison_gen.py --attack chimera_pred --pair 'cat->dog' ... --probe --limit 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from concepts import swap_concept
from run_spec import MODELS, _rng, poison_dir_for
from train import autocast_dtype, load_image

# Nightshade's released defaults, used for every attack so the budget is identical.
EPS = 0.04          # L-infinity budget in [0, 1] pixel units (~10/255)
ITERS = 500
ANCHOR_SEED = 123   # opt.py fixes this so one anchor image serves the whole poison set
ANCHOR_PROMPT = "A photo of a {}"
SIGMA = 0.5         # where on the flow path the velocity attacks are matched

ATTACK_SPEC = {
    "nightshade_sd2": {"model": "sd2_base",   "space": "latent",   "anchor": "generated"},
    "nightshade_sd3": {"model": "sd3_medium", "space": "latent",   "anchor": "generated"},
    "chimera_anchor": {"model": "sd3_medium", "space": "velocity", "anchor": "real"},
    "chimera_pred":   {"model": "sd3_medium", "space": "velocity", "anchor": "caption"},
}


def anchor_path(anchors_root, model: str, target: str) -> Path:
    return Path(anchors_root) / model / f"{target.replace(' ', '_')}.png"


def to_uint8(x):
    """[-1, 1] float tensor -> the uint8 array the PNG stores."""
    import numpy as np

    arr = ((x.detach().float().clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255)
    return arr.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def from_uint8(arr, device, dtype=None):
    """The inverse, so the objective can be re-measured on exactly what was written to disk."""
    import torch

    x = torch.from_numpy(arr).to(device).permute(2, 0, 1).float() / 127.5 - 1.0
    return x if dtype is None else x.to(dtype)


def plan(manifest: dict, attack: str, pair: str, images_root, anchors_root=None, only_ids=None) -> list[dict]:
    """
    Every image this (attack, pair) has to produce, in a fixed order.

    `only_ids` restricts the work to the ids a matrix actually consumes (see run_spec.required_poison_ids);
    without it the whole 250-image pool is covered. Either way the coverage is not just the largest
    poison_count, because run_spec shuffles the pool per seed and a 100-poison run at seed 1 draws different
    ids than at seed 0.

    What each poison is matched against is decided here from (pair, id) alone -- never from the seed or the
    count -- so an image's poison is identical whichever cell consumes it.
    """
    from run_spec import image_path

    spec = ATTACK_SPEC[attack]
    pair_info = manifest["pairs"][pair]
    source, target = pair_info["source"], pair_info["target"]
    images_root = Path(images_root)
    pool = sorted(pair_info["poison_pool"], key=lambda r: r["id"])

    targets = None
    if spec["anchor"] == "real":
        # A real target-concept image per poison, paired on (pair, id) alone.
        anchors = sorted(pair_info["target_anchor"], key=lambda r: r["id"])
        _rng(pair, "chimera_anchor_pairing").shuffle(anchors)
        targets = [str(image_path(images_root, a)) for a in anchors]

    rows = []
    for i, r in enumerate(pool):
        row = {"id": r["id"], "path": str(image_path(images_root, r)), "caption": r["caption"]}
        if spec["anchor"] == "generated":
            row["target_image"] = str(anchor_path(anchors_root, spec["model"], target))
        elif spec["anchor"] == "real":
            row["target_image"] = targets[i % len(targets)]
        else:
            swapped = swap_concept(r["caption"], source, target)
            if swapped is None:
                raise ValueError(f"caption of id {r['id']} has no {source!r} to swap: {r['caption']!r}")
            row["target_caption"] = swapped
        rows.append(row)
    # Filtered only after the pairing is fixed over the full pool: narrowing first would make which anchor an
    # image is matched against depend on which ids the matrix happened to ask for.
    if only_ids is not None:
        rows = [r for r in rows if r["id"] in only_ids]
    return rows


def make_anchors(model: str, targets, out_root, steps=50, guidance=7.5, res=512) -> dict:
    """
    Generate the one anchor image per target concept that the nightshade_* attacks collide onto.

    Faithful to opt.py: the prompt is "A photo of a <target>", the sampler runs 50 steps at guidance 7.5, and
    the seed is fixed so every poison in a set is pulled toward the same image -- which is what makes the
    attack a *concept* attack rather than 250 unrelated perturbations. opt.py fixes it with a global
    torch.manual_seed(123); an explicit generator is used here instead so the result does not depend on what
    else drew random numbers first.

    This is the only step that loads text encoders, and it runs once per model (4 images), not once per poison.
    """
    import torch

    pretrained = MODELS[model]
    dtype = autocast_dtype()
    if model in ("sd3_medium", "sd3_5_medium"):
        from diffusers import StableDiffusion3Pipeline as Pipe

        extra = {}                                   # SD3's pipeline has no safety checker to disable
    else:
        from diffusers import StableDiffusionPipeline as Pipe

        extra = {"safety_checker": None, "requires_safety_checker": False}
    pipe = Pipe.from_pretrained(pretrained, torch_dtype=dtype, **extra)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    written = {}
    for target in targets:
        path = anchor_path(out_root, model, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        prompt = ANCHOR_PROMPT.format(target)
        gen = torch.Generator(device="cuda").manual_seed(ANCHOR_SEED)
        image = pipe(prompt, guidance_scale=guidance, num_inference_steps=steps,
                     height=res, width=res, generator=gen).images[0]
        image.save(path)
        written[target] = {"path": str(path), "prompt": prompt, "seed": ANCHOR_SEED,
                           "steps": steps, "guidance": guidance, "resolution": res}
        print(f"anchor {model}/{target}: {path}", flush=True)
    del pipe
    torch.cuda.empty_cache()
    return {"model": model, "pretrained": pretrained, "anchors": written}


def load_models(attack: str, device):
    """VAE always; the transformer only for the velocity attacks. Text encoders never (the cache has them)."""
    import torch
    from diffusers import AutoencoderKL

    spec = ATTACK_SPEC[attack]
    pretrained = MODELS[spec["model"]]
    dtype = autocast_dtype()
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", torch_dtype=dtype).to(device)
    vae.requires_grad_(False).eval()

    net = scheduler = None
    if spec["space"] == "velocity":
        from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

        net = SD3Transformer2DModel.from_pretrained(pretrained, subfolder="transformer",
                                                    torch_dtype=dtype).to(device)
        net.requires_grad_(False).eval()
        # Checkpointing engages on `torch.is_grad_enabled() and self.gradient_checkpointing`, not on
        # .training, so it works with a frozen net in eval mode (diffusers 0.42, transformer_sd3.py:307).
        net.enable_gradient_checkpointing()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
    return vae, net, scheduler, torch.float32


def flow_point(scheduler, sigma: float, device):
    """The (timestep, sigma) pair on the scheduler's own grid closest to `sigma`."""
    import torch

    sigmas = scheduler.sigmas[:len(scheduler.timesteps)]
    idx = int((sigmas - sigma).abs().argmin())
    return scheduler.timesteps[idx].to(device), float(sigmas[idx])


def encode_mean(vae, x, dtype):
    """VAE posterior mean, gradients intact. Nightshade matches this raw mean, unshifted and unscaled."""
    return vae.encode(x.to(dtype)).latent_dist.mean


def scaled(vae, z):
    """The latent the training loop actually uses: (mean - shift) * scaling."""
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    return (z - shift) * vae.config.scaling_factor


def optimise(rows, attack, vae, net, scheduler, text, device, iters=ITERS, eps=EPS, sigma=SIGMA,
             lpips_weight=0.0, lpips_fn=None, log_every=0, res=512, moving_target=False):
    """
    Perturb one batch of images. Returns (uint8 arrays, per-image records).

    Projected signed-gradient descent with Nightshade's annealed step: the first iteration may move the full
    budget and the last moves a hundredth of it. The perturbation lives in [-1, 1] pixel units, where the
    budget is 2 * eps.
    """
    import torch

    spec = ATTACK_SPEC[attack]
    dtype = autocast_dtype()
    max_change = 2.0 * eps
    step0 = max_change

    x = torch.stack([load_image(r["path"], res) for r in rows]).to(device)
    delta = torch.zeros_like(x)

    # One noise draw per image, seeded by its COCO id: the same poison is reproducible bit for bit, the
    # objective is deterministic (so its loss curve means something), and the target below is built from the
    # same draw the iterations use. Resampling it every iteration -- as the draft did -- turns the objective
    # into a different random function each step.
    noise = torch.stack([torch.randn(16 if spec["model"].startswith("sd3") else 4, res // 8, res // 8,
                                     generator=torch.Generator().manual_seed(int(r["id"])),
                                     dtype=torch.float32) for r in rows]).to(device)

    # What the batch is pulled toward -- a fixed point in every case. See `fixed target` in the module
    # docstring for why `anchor == "caption"` may not recompute it from the perturbed image.
    z_target = src_embeds = src_pooled = tgt_embeds = tgt_pooled = None
    with torch.no_grad():
        if spec["anchor"] in ("generated", "real"):
            t_px = torch.stack([load_image(r["target_image"], res) for r in rows]).to(device)
            z_t_mean = encode_mean(vae, t_px, dtype)
            z_target = z_t_mean if spec["space"] == "latent" else scaled(vae, z_t_mean)
            z_target = z_target.detach().float()
            del t_px
        if spec["space"] == "velocity":
            src = [text.get(r["caption"]) for r in rows]
            src_embeds = torch.stack([s["prompt_embeds"] for s in src]).to(device, dtype)
            src_pooled = torch.stack([s["pooled"] for s in src]).to(device, dtype)
            if spec["anchor"] == "caption":
                tgt = [text.get(r["target_caption"]) for r in rows]
                tgt_embeds = torch.stack([t["prompt_embeds"] for t in tgt]).to(device, dtype)
                tgt_pooled = torch.stack([t["pooled"] for t in tgt]).to(device, dtype)

    timestep = sigma_v = None
    if spec["space"] == "velocity":
        step_t, sigma_v = flow_point(scheduler, sigma, device)
        timestep = step_t.expand(x.shape[0])

        if spec["anchor"] == "caption" and not moving_target:
            # The fixed target: what the frozen model predicts this image's clean latent to be when it is
            # told the image shows the target concept. Evaluated once, on the CLEAN image, so it is a point
            # the perturbation can be pulled toward rather than one that moves with it.
            with torch.no_grad():
                z0_clean = scaled(vae, encode_mean(vae, x, dtype)).float()
                z_t_clean = (1.0 - sigma_v) * z0_clean + sigma_v * noise
                v_tgt = net(hidden_states=z_t_clean.to(dtype), timestep=timestep,
                            encoder_hidden_states=tgt_embeds, pooled_projections=tgt_pooled,
                            return_dict=False)[0]
                z_target = (z_t_clean - sigma_v * v_tgt.float()).detach()

    def objective(x_adv):
        z = encode_mean(vae, x_adv, dtype)
        if spec["space"] == "latent":
            return ((z.float() - z_target) ** 2).flatten(1).sum(1).sqrt()
        z0 = scaled(vae, z).float()
        z_t = (1.0 - sigma_v) * z0 + sigma_v * noise
        v = net(hidden_states=z_t.to(dtype), timestep=timestep, encoder_hidden_states=src_embeds,
                pooled_projections=src_pooled, return_dict=False)[0]
        z0_est = z_t - sigma_v * v.float()
        if spec["anchor"] == "caption" and moving_target:
            # The draft formulation, kept only so the paper can show why it is not used: z_t cancels out of
            # the difference, leaving sigma * ||v_tgt - v_src||, and v_tgt is detached although it depends on
            # the perturbation through z_t. The gradient therefore chases a target that moves away from it.
            with torch.no_grad():
                v_tgt = net(hidden_states=z_t.detach().to(dtype), timestep=timestep,
                            encoder_hidden_states=tgt_embeds, pooled_projections=tgt_pooled,
                            return_dict=False)[0]
                target = (z_t.detach() - sigma_v * v_tgt.float()).detach()
        else:
            target = z_target
        return ((z0_est - target) ** 2).flatten(1).sum(1).sqrt()

    curve, t0 = [], time.time()
    with torch.no_grad():
        loss0 = objective(torch.clamp(x + delta, -1, 1)).detach()

    for i in range(iters):
        step = step0 - (step0 - step0 / 100) / iters * i
        delta.requires_grad_(True)
        loss = objective(torch.clamp(x + delta, -1, 1)).sum()
        grad = torch.autograd.grad(loss, delta)[0]
        if lpips_weight and lpips_fn is not None:
            delta_l = delta.detach().requires_grad_(True)
            perceptual = lpips_fn(torch.clamp(x + delta_l, -1, 1), x).mean() * lpips_weight
            grad = grad + torch.autograd.grad(perceptual, delta_l)[0]
        with torch.no_grad():
            delta = (delta.detach() - grad.sign() * step).clamp(-max_change, max_change)
        if (log_every and (i % log_every == 0 or i == iters - 1)) or i % 100 == 0:
            mean = round(float(loss.detach()) / len(rows), 5)
            if log_every and (i % log_every == 0 or i == iters - 1):
                curve.append({"iter": i, "loss": mean})
            # A batch is otherwise silent for its whole optimisation, which makes a stalled container
            # indistinguishable from a working one -- and there are 1532 images to get through.
            print(f"    iter {i:4d}/{iters}  loss {mean:.4f}  "
                  f"{(time.time() - t0) / (i + 1):.2f} s/iter", flush=True)

    # Round to uint8 and re-measure: the fine-tune reads the PNG, not the float tensor.
    with torch.no_grad():
        x_adv = torch.clamp(x + delta, -1, 1)
        arrays = [to_uint8(a) for a in x_adv]
        x_q = torch.stack([from_uint8(a, device) for a in arrays])
        loss_q = objective(x_q).detach()
        diff = (x_q - x).abs() * 127.5
        linf = diff.flatten(1).max(1).values
        mse = ((x_q - x) * 127.5).pow(2).flatten(1).mean(1)
        psnr = 10.0 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-12))

    seconds = time.time() - t0
    records = []
    for j, r in enumerate(rows):
        records.append({
            "id": r["id"], "iters": iters,
            "loss_initial": round(float(loss0[j]), 5),
            "loss_quantised": round(float(loss_q[j]), 5),
            "linf_255": round(float(linf[j]), 2),
            "psnr_db": round(float(psnr[j]), 2),
            "seconds": round(seconds / len(rows), 2),
            "target_caption": r.get("target_caption"),
            "target_image": Path(r["target_image"]).name if "target_image" in r else None,
        })
    return arrays, records, curve, seconds


def config_of(attack, pair, iters, eps, sigma, lpips_weight, res, git_commit, moving_target=False) -> dict:
    spec = ATTACK_SPEC[attack]
    return {"attack": attack, "pair": pair, "model": spec["model"], "space": spec["space"],
            "anchor": spec["anchor"], "pretrained": MODELS[spec["model"]], "iters": iters, "eps": eps,
            "sigma": sigma, "lpips_weight": lpips_weight, "resolution": res, "optimiser": "signed-grad-anneal",
            "target": "moving_adv" if moving_target else "fixed_clean", "noise": "fixed_per_image_id",
            "anchor_seed": ANCHOR_SEED, "anchor_prompt": ANCHOR_PROMPT, "git_commit": git_commit}


def generate(attack: str, pair: str, manifest: dict, images_root, text_root, out_dir, anchors_root=None,
             iters=ITERS, eps=EPS, sigma=SIGMA, lpips_weight=0.0, batch=4, limit=None, log_every=0,
             res=512, git_commit="", only_ids=None, moving_target=False) -> dict:
    """
    Produce every missing poison for one (attack, pair), writing after each batch so a crash costs one batch.

    A directory is tied to the configuration that filled it: resuming with a different budget or iteration
    count is refused rather than silently mixing two attacks' worth of settings into one poison set.
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "poison_manifest.json"
    config = config_of(attack, pair, iters, eps, sigma, lpips_weight, res, git_commit, moving_target)

    done = {}
    if man_path.exists():
        previous = json.loads(man_path.read_text())
        keys = [k for k in config if k != "git_commit"]
        differs = {k: (previous["config"].get(k), config[k]) for k in keys
                   if previous["config"].get(k) != config[k]}
        if differs:
            raise SystemExit(f"{out_dir} was filled with a different configuration: {differs}\n"
                             f"use a different --poison-root or delete the directory")
        done = {int(r["id"]): r for r in previous["images"]}

    rows = plan(manifest, attack, pair, images_root, anchors_root, only_ids=only_ids)
    if limit:
        rows = rows[:limit]
    # Fail before loading a model, not three hours into a run, if an input image is not there.
    for key in ("path", "target_image"):
        missing = [r[key] for r in rows if key in r and not Path(r[key]).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} {key} files missing, e.g. {missing[0]} "
                                    + ("-- run the anchors step first" if key == "target_image" else ""))
    todo = [r for r in rows if not (r["id"] in done and (out_dir / f"{r['id']}.png").exists())]
    print(f"{attack} {pair}: {len(rows)} images, {len(rows) - len(todo)} already done, {len(todo)} to do",
          flush=True)
    if not todo:
        return {"config": config, "images": [done[r["id"]] for r in rows], "generated": 0}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, net, scheduler, _ = load_models(attack, device)
    text = None
    if ATTACK_SPEC[attack]["space"] == "velocity":
        from text_cache import TextCache
        text = TextCache(Path(text_root) / ATTACK_SPEC[attack]["model"])
        missing = [r for r in todo if r["caption"] not in text
                   or (r.get("target_caption") and r["target_caption"] not in text)]
        if missing:
            raise KeyError(f"{len(missing)} captions are not in the text cache, e.g. "
                           f"{missing[0].get('target_caption') or missing[0]['caption']!r} -- "
                           f"re-run compute/modal/encode_captions.py")

    lpips_fn = None
    if lpips_weight:
        import lpips as lpips_pkg
        lpips_fn = lpips_pkg.LPIPS(net="alex").to(device)
        lpips_fn.requires_grad_(False)

    from PIL import Image

    curves, t0 = [], time.time()
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        arrays, records, curve, seconds = optimise(
            chunk, attack, vae, net, scheduler, text, device, iters=iters, eps=eps, sigma=sigma,
            lpips_weight=lpips_weight, lpips_fn=lpips_fn, log_every=log_every, res=res,
            moving_target=moving_target)
        for arr, rec in zip(arrays, records):
            Image.fromarray(arr).save(out_dir / f"{rec['id']}.png")
            done[rec["id"]] = rec
        if curve:
            curves.append({"ids": [r["id"] for r in chunk], "curve": curve})
        images = [done[r["id"]] for r in rows if r["id"] in done]
        man_path.write_text(json.dumps({"config": config, "images": images,
                                        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                        if device == "cuda" else None}, indent=2))
        n = start + len(chunk)
        rate = (time.time() - t0) / n
        print(f"  {n}/{len(todo)}  loss {records[0]['loss_initial']:.3f} -> "
              f"{records[0]['loss_quantised']:.3f}  Linf {records[0]['linf_255']:.1f}/255  "
              f"PSNR {records[0]['psnr_db']:.1f} dB  {rate:.1f} s/image  "
              f"eta {rate * (len(todo) - n) / 60:.0f} min", flush=True)

    out = {"config": config, "images": [done[r["id"]] for r in rows if r["id"] in done],
           "generated": len(todo), "seconds": round(time.time() - t0, 1),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None}
    if curves:
        out["curves"] = curves
    man_path.write_text(json.dumps(out, indent=2))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attack", required=True, choices=sorted(ATTACK_SPEC))
    p.add_argument("--pair", required=True)
    p.add_argument("--data-root", default="/data")
    p.add_argument("--poison-root", required=True)
    p.add_argument("--anchors", help="where the generated anchor images live (the nightshade_* attacks)")
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--limit", type=int, help="stop after this many images (probe)")
    p.add_argument("--log-every", type=int, default=0, help="record the loss every N iterations")
    p.add_argument("--git-commit", default="")
    p.add_argument("--moving-target", action="store_true",
                   help="the draft formulation, which diverges; kept to document why it is not used")
    a = p.parse_args(argv)

    data = Path(a.data_root)
    manifest = json.loads((data / "chimera_coco" / "manifest.json").read_text())
    out = generate(a.attack, a.pair, manifest, data / "chimera_coco" / "images", data / "embeds",
                   poison_dir_for(a.poison_root, a.attack, a.pair), anchors_root=a.anchors,
                   iters=a.iters, eps=a.eps, sigma=a.sigma, lpips_weight=a.lpips_weight,
                   batch=a.batch, limit=a.limit, log_every=a.log_every, git_commit=a.git_commit,
                   moving_target=a.moving_target)
    print(json.dumps({k: v for k, v in out.items() if k != "images"}, indent=2))


if __name__ == "__main__":
    main()
