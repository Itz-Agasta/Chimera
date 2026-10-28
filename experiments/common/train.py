"""
The one training script for every run in the paper: SD3-Medium / SD-2 x full fine-tuning / LoRA.

It reads only pre-built artefacts -- the COCO images, the dataset manifest and the cached text embeddings --
so the text encoders (12 GB for SD3) never load during training, and it writes the run directory that
experiments/common/run_contract.py validates:

    results/<experiment>/<run_id>/{config.json, inputs.json, metrics.json}

Which images and captions a run trains on is decided by experiments/common/run_spec.py, not here.

Recipe, measured in results/compute/full_ft_probe_l4.json and results/h2_lpi/memprobe_t4/memprobe.json:
full fine-tuning keeps fp32 master weights under bf16 autocast with 8-bit AdamW and gradient checkpointing
(19.3 GB at 512 px on an L4; without checkpointing it OOMs), LoRA r16 trains in 5.6 GB on a T4.

Three corrections over the old H1/H2 notebooks, all of which change what the model is actually trained on:
  * VAE latents are shifted before scaling: (z - shift_factor) * scaling_factor  (SD3's shift is ~0.0609).
  * Images are normalised to [-1, 1] before vae.encode, not [0, 1].
  * The SD3 transformer is conditioned on the scheduler's 0-1000 timesteps, not on sigma in [0, 1].

Run:
  python experiments/common/train.py --model sd3_medium --regime full_ft --attack clean \
      --pair 'cat->dog' --poison-count 0 --seed 0 --data-root /data --out-root results --git-commit <sha>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from run_spec import GUIDANCE_SCALES, MODELS, RunSpec, build_inputs
from text_cache import TextCache

# LoRA goes on the attention projections only -- the configuration whose step memory was measured on the T4.
LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]
VAE_SAMPLE_SEED = 0  # the VAE's sampled latents are part of the data, so they are fixed per image id


def git_commit(given: str | None) -> str:
    """The commit every number in the run is attributed to. Never silently 'unknown'."""
    if given:
        return given
    if env := os.environ.get("CHIMERA_GIT_COMMIT"):
        return env
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit("cannot resolve the git commit: pass --git-commit <sha> (containers have no git)")


def autocast_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_image(path, res: int):
    """RGB -> float tensor in [-1, 1], short side resized to `res` then centre-cropped."""
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.size != (res, res):
        w, h = img.size
        scale = res / min(w, h)
        img = img.resize((max(res, round(w * scale)), max(res, round(h * scale))), Image.BICUBIC)
        w, h = img.size
        left, top = (w - res) // 2, (h - res) // 2
        img = img.crop((left, top, left + res, top + res))
    x = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(x).permute(2, 0, 1)


def latent_noise(shape, image_id: int, device, dtype):
    """
    The noise for the VAE's reparameterised sample, fixed per image id.

    Drawn on the CPU from a per-id seed so an image's latent is the same whatever device encoded it, whatever
    batch it sat in and whatever else the run encoded. Walking one generator across the `todo` list instead
    makes the draw depend on the run's position in that list: the first run encodes 3200 clean images before
    the poisons and every later run reads those from the latent cache and encodes the poisons alone, so the
    same poison image gets different latents before and after the cache exists.
    """
    import torch

    g = torch.Generator().manual_seed((VAE_SAMPLE_SEED * 1_000_003 + int(image_id)) % (2 ** 63 - 1))
    return torch.randn(tuple(shape), generator=g, dtype=torch.float32).to(device, dtype)


def encode_latents(records, spec: RunSpec, device, cache_path: Path | None, batch: int = 8):
    """
    VAE-encode every training image, reusing a cache for the clean corpus.

    The clean corpus is byte-identical in every run of a model/resolution, so its latents are encoded once and
    cached; poisoned images are encoded per run because their pixels are what the attack changed. Every latent
    is reproducible from (image id, model, resolution) alone -- see `latent_noise`. Returns {id: latent fp16 on
    CPU}; the VAE is loaded and freed inside this function.
    """
    import gc

    import torch
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file, save_file

    latents, cached_ids = {}, set()
    if cache_path and cache_path.exists():
        blob = load_file(str(cache_path))
        cached_ids = {int(i) for i in blob["ids"].tolist()}
        for i, lid in enumerate(blob["ids"].tolist()):
            latents[int(lid)] = blob["latents"][i]
        print(f"latent cache: {len(cached_ids)} ids from {cache_path}", flush=True)

    todo = [r for r in records if r["poisoned"] or r["id"] not in latents]
    if todo:
        dtype = autocast_dtype()
        vae = AutoencoderKL.from_pretrained(MODELS[spec.model], subfolder="vae", torch_dtype=dtype)
        vae = vae.to(device).eval()
        shift = getattr(vae.config, "shift_factor", None)
        t0 = time.time()
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            px = torch.stack([load_image(r["path"], spec.resolution) for r in chunk]).to(device, dtype)
            with torch.no_grad():
                dist = vae.encode(px).latent_dist
                # sample(), but with per-image noise instead of one generator walked across the batch
                eps = torch.stack([latent_noise(dist.mean.shape[1:], r["id"], device, dist.mean.dtype)
                                   for r in chunk])
                z = dist.mean + dist.std * eps
            if shift:
                z = z - shift
            z = z * vae.config.scaling_factor
            for r, zi in zip(chunk, z):
                latents[r["id"]] = zi.half().cpu()
            if start % (batch * 25) == 0:
                print(f"  vae {start + len(chunk)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        # The loop's last chunk still references GPU tensors, and the VAE's memory has to be gone before
        # the transformer is built: full fine-tuning leaves under 3 GB of headroom on a 22 GB card.
        del vae, dist, eps, z, px
        gc.collect()
        torch.cuda.empty_cache()

    if cache_path:
        clean_ids = sorted({r["id"] for r in records if not r["poisoned"]})
        if not cached_ids >= set(clean_ids):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"ids": torch.tensor(clean_ids, dtype=torch.int64),
                       "latents": torch.stack([latents[i] for i in clean_ids]).contiguous()},
                      str(cache_path))
            print(f"latent cache written: {len(clean_ids)} ids -> {cache_path}", flush=True)
    return latents


class RunDataset:
    """(latent, text embedding) pairs. Text embeddings are read per item, so the 13.7 GB cache stays on disk."""

    def __init__(self, records, latents, text_root, spec: RunSpec):
        import torch
        self.records, self.latents, self.spec = records, latents, spec
        self.text = TextCache(text_root)
        missing = [r["caption"] for r in records if r["caption"] not in self.text]
        if missing:
            raise KeyError(f"{len(missing)} captions are not in the text cache, e.g. {missing[:2]}")
        if spec.caption_dropout and "" not in self.text:
            raise KeyError("caption_dropout needs the empty caption in the text cache")
        self.gen = torch.Generator().manual_seed(spec.seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import torch
        r = self.records[i]
        caption = r["caption"]
        if self.spec.caption_dropout and torch.rand(1, generator=self.gen).item() < self.spec.caption_dropout:
            caption = ""
        item = {"latent": self.latents[r["id"]], **self.text.get(caption)}
        return item


def build_model(spec: RunSpec, device):
    """The trainable network plus the training scheduler, set up for the measured memory recipe."""
    import torch
    from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, UNet2DConditionModel

    pretrained = MODELS[spec.model]
    # full fine-tuning holds fp32 master weights and relies on autocast; LoRA keeps the frozen base in low precision.
    weight_dtype = torch.float32 if spec.regime == "full_ft" else autocast_dtype()
    if spec.is_flow:
        net = SD3Transformer2DModel.from_pretrained(pretrained, subfolder="transformer",
                                                    torch_dtype=weight_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
    else:
        net = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet", torch_dtype=weight_dtype)
        scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")
    net = net.to(device)

    # Mandatory for full fine-tuning (without it the L4 OOMs); harmless and cheap for LoRA.
    net.enable_gradient_checkpointing()
    if spec.regime == "lora":
        from peft import LoraConfig, get_peft_model
        net.requires_grad_(False)
        net = get_peft_model(net, LoraConfig(r=spec.lora_rank, lora_alpha=spec.lora_rank,
                                             init_lora_weights="gaussian", target_modules=LORA_TARGETS))
        net.print_trainable_parameters()
    net.train()
    return net, scheduler


def flow_loss(net, scheduler, batch, spec: RunSpec, device):
    """Rectified-flow objective: predict the velocity (noise - latent) at a logit-normal sigma."""
    import torch
    from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

    z = batch["latent"].to(device, torch.float32)
    noise = torch.randn_like(z)
    u = compute_density_for_timestep_sampling(weighting_scheme=spec.weighting_scheme, batch_size=z.shape[0],
                                              logit_mean=spec.logit_mean, logit_std=spec.logit_std,
                                              mode_scale=1.29)
    idx = (u * scheduler.config.num_train_timesteps).long().clamp(0, scheduler.config.num_train_timesteps - 1)
    # scheduler.timesteps are already on the 0-1000 scale the transformer expects; sigmas[i] matches timesteps[i].
    timesteps = scheduler.timesteps[idx].to(device)
    sigmas = scheduler.sigmas[idx].to(device, torch.float32).view(-1, 1, 1, 1)
    noisy = (1.0 - sigmas) * z + sigmas * noise

    with torch.autocast("cuda", dtype=autocast_dtype()):
        pred = net(hidden_states=noisy.to(autocast_dtype()), timestep=timesteps,
                   encoder_hidden_states=batch["prompt_embeds"].to(device, autocast_dtype()),
                   pooled_projections=batch["pooled"].to(device, autocast_dtype()), return_dict=False)[0]
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=spec.weighting_scheme, sigmas=sigmas)
    return (weighting.float() * (pred.float() - (noise - z)) ** 2).mean()


def ddpm_loss(net, scheduler, batch, spec: RunSpec, device):
    """SD-2's objective: predict the noise (or velocity) added at a uniformly sampled discrete timestep."""
    import torch

    z = batch["latent"].to(device, torch.float32)
    noise = torch.randn_like(z)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (z.shape[0],), device=device).long()
    noisy = scheduler.add_noise(z, noise, t)
    with torch.autocast("cuda", dtype=autocast_dtype()):
        pred = net(noisy.to(autocast_dtype()), t,
                   encoder_hidden_states=batch["prompt_embeds"].to(device, autocast_dtype()),
                   return_dict=False)[0]
    target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(z, noise, t)
    return ((pred.float() - target.float()) ** 2).mean()


def train(spec: RunSpec, manifest: dict, images_root, text_root, poison_dir=None, latent_cache=None,
          commit: str = "", checkpoint_dir=None, log_every: int = 25, probe_dir=None, on_probe=None):
    """
    Run one cell. Returns (net, config, inputs, metrics); the caller writes the run directory.

    With spec.probe_epochs set, training pauses at those epochs to generate and judge a probe set on the live
    weights (evaluate.Probe); images go under `probe_dir`, results into metrics["probe"], and `on_probe` is
    called with the list so far, so a launcher can persist partial results if the run dies later.
    """
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda")
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)

    records, inputs = build_inputs(manifest, spec, images_root, poison_dir)
    steps = spec.resolve_steps(len(records))
    config = spec.config(len(records), commit)
    print(f"{spec.run_id}: {len(records)} images "
          f"({len(inputs['poisoned_ids'])} poisoned), {steps} optimizer steps "
          f"x {spec.grad_accum} accum", flush=True)

    latents = encode_latents(records, spec, device, Path(latent_cache) if latent_cache else None)
    loader = DataLoader(RunDataset(records, latents, text_root, spec), batch_size=spec.batch_size,
                        shuffle=True, num_workers=0, drop_last=True,
                        generator=torch.Generator().manual_seed(spec.seed))

    net, scheduler = build_model(spec, device)
    params = [p for p in net.parameters() if p.requires_grad]
    if spec.regime == "full_ft":
        import bitsandbytes as bnb
        # Measured on an L40S: fp32 masters + fp32 grads + 8-bit Adam state peaks at 23.1 GB, past a 24 GB
        # card. Paging the optimizer state to CPU keeps the recipe and fits a 24 GB GPU.
        cls = bnb.optim.PagedAdamW8bit if spec.paged_optimizer else bnb.optim.AdamW8bit
        optimizer = cls(params, lr=spec.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(params, lr=spec.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    from diffusers.optimization import get_scheduler
    lr_sched = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=spec.lr_warmup_steps,
                             num_training_steps=steps)
    # bf16 autocast needs no loss scaling; a T4 (fp16 only) does.
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_dtype() == torch.float16)
    loss_fn = flow_loss if spec.is_flow else ddpm_loss

    probe, probe_at, probes = None, {}, []
    if spec.probe_epochs:
        from evaluate import Probe
        per_epoch = math.ceil(len(records) / (spec.batch_size * spec.grad_accum))
        probe_at = {max(1, round(e * per_epoch)): e for e in spec.probe_epochs if round(e * per_epoch) <= steps}
        probe = Probe(net, spec, manifest, text_root, Path(probe_dir or "probe") / spec.run_id)

    def batches():
        while True:
            yield from loader

    stream = batches()
    torch.cuda.reset_peak_memory_stats()
    losses, curve, t0 = [], [], time.time()
    for step in range(steps):
        step_loss = 0.0
        for _ in range(spec.grad_accum):
            loss = loss_fn(net, scheduler, next(stream), spec, device) / spec.grad_accum
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, spec.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        lr_sched.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(step_loss)
        if step % log_every == 0 or step == steps - 1:
            curve.append({"step": step, "loss": round(step_loss, 5),
                          "lr": lr_sched.get_last_lr()[0], "seconds": round(time.time() - t0)})
            print(f"  step {step + 1}/{steps}  loss {step_loss:.4f}  {time.time() - t0:.0f}s", flush=True)
        if step + 1 in probe_at:
            probes.append(probe(probe_at[step + 1], step + 1))
            if on_probe:
                on_probe(probes)

    elapsed = time.time() - t0
    # final_loss is the mean over the last 10% of steps: a single step's loss is dominated by which sigma
    # that step happened to sample.
    tail = max(1, len(losses) // 10)
    metrics = {"final_loss": round(sum(losses[-tail:]) / tail, 6), "last_loss": round(losses[-1], 6),
               "steps": steps, "seconds": round(elapsed, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               "final_loss_window": tail, "loss_curve": curve,
               "s_per_microstep": round(elapsed / (steps * spec.grad_accum), 3)}
    if probes:
        # elapsed includes the probes; report the training rate without them too
        metrics["probe"] = probes
        metrics["probe_seconds"] = round(sum(p["seconds"] for p in probes), 1)
        metrics["s_per_microstep"] = round((elapsed - metrics["probe_seconds"]) / (steps * spec.grad_accum), 3)

    if checkpoint_dir:
        out = Path(checkpoint_dir) / spec.run_id
        out.mkdir(parents=True, exist_ok=True)
        if spec.regime == "lora":
            net.save_pretrained(str(out))
        else:
            # Half precision: 4 GB instead of 8 per SD3 checkpoint, and the model is only ever used for
            # generation afterwards. This leaves `net` itself in half precision for in-process evaluation.
            net.to(autocast_dtype()).save_pretrained(str(out))
        metrics["checkpoint"] = str(out)
        print(f"checkpoint -> {out}", flush=True)
    return net, config, inputs, metrics


def write_run(run_dir, config, inputs, metrics):
    run_dir = Path(run_dir)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    for name, payload in (("config.json", config), ("inputs.json", inputs), ("metrics.json", metrics)):
        (run_dir / name).write_text(json.dumps(payload, indent=2))
    return run_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--regime", required=True, choices=["full_ft", "lora"])
    p.add_argument("--attack", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--poison-count", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--steps", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--caption-dropout", type=float, default=0.0)
    p.add_argument("--data-root", default="/data", help="holds chimera_coco/ and embeds/<model>/")
    p.add_argument("--out-root", default="results")
    p.add_argument("--poison-dir", help="optimised poison images named <coco_id>.png, for the attack runs")
    p.add_argument("--latent-cache", help="default <data-root>/latents/<model>_<res>.safetensors")
    p.add_argument("--checkpoint-dir", help="where to save weights; omitted means no checkpoint is kept")
    p.add_argument("--git-commit")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    spec = RunSpec(model=a.model, regime=a.regime, attack=a.attack, pair=a.pair, poison_count=a.poison_count,
                   seed=a.seed, resolution=a.resolution, batch_size=a.batch_size, grad_accum=a.grad_accum,
                   epochs=a.epochs, steps=a.steps, learning_rate=a.learning_rate, lora_rank=a.lora_rank,
                   caption_dropout=a.caption_dropout)
    data = Path(a.data_root)
    manifest = json.loads((data / "chimera_coco" / "manifest.json").read_text())
    latent_cache = a.latent_cache or data / "latents" / f"{spec.model}_{spec.resolution}.safetensors"
    _, config, inputs, metrics = train(spec, manifest, data / "chimera_coco" / "images",
                                       data / "embeds" / spec.model, poison_dir=a.poison_dir,
                                       latent_cache=latent_cache, commit=git_commit(a.git_commit),
                                       checkpoint_dir=a.checkpoint_dir)
    run_dir = write_run(Path(a.out_root) / spec.experiment / spec.run_id, config, inputs, metrics)
    print(f"wrote {run_dir}  (samples/ is filled by the evaluation step, guidance {list(GUIDANCE_SCALES)})")


if __name__ == "__main__":
    main()
