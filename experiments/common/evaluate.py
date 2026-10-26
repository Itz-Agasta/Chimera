"""
Turn a finished training run into numbers: generate from the pair's held-out prompts across guidance scales,
judge every image with both judges, and write `evaluation.json` next to the run's config/inputs/metrics.

Generation feeds cached text embeddings straight to the pipeline (`prompt_embeds` / `pooled_prompt_embeds`, with
the empty caption as the unconditional branch), so the text encoders never load here either -- only the trained
transformer/unet and the VAE, about 4 GB for SD3.

Attack success is the fraction of images generated from source-concept prompts that a judge labels as the pair's
target concept. On clean models both judges score 0.00 on all four pairs and recognise every target concept
(results/judges/), so a non-zero number here is the attack, not the metric.

  python experiments/common/evaluate.py results/h1_transfer/<run_id> --data-root /data \
      --checkpoint-dir /data/checkpoints
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from concepts import eval_prompts as concept_templates
from run_contract import sample_name
from run_spec import GUIDANCE_SCALES, MODELS, RunSpec
from text_cache import TextCache

N_PER_PROMPT = 2        # 50 held-out captions x 2 = 100 images per (pair, guidance)
INFERENCE_STEPS = 28


def load_pipeline(spec: RunSpec, checkpoint_dir, device="cuda"):
    """The trained model as a generation pipeline, with the text encoders left out."""
    import torch
    from diffusers import (SD3Transformer2DModel, StableDiffusion3Pipeline, StableDiffusionPipeline,
                           UNet2DConditionModel)

    dtype = torch.float16
    pretrained = MODELS[spec.model]
    ckpt = Path(checkpoint_dir) / spec.run_id if checkpoint_dir else None
    if ckpt is not None and not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint for {spec.run_id} at {ckpt}")

    if spec.is_flow:
        if spec.regime == "lora" and ckpt is not None:
            net = SD3Transformer2DModel.from_pretrained(pretrained, subfolder="transformer", torch_dtype=dtype)
            from peft import PeftModel
            net = PeftModel.from_pretrained(net, str(ckpt)).merge_and_unload()
        elif ckpt is not None:
            net = SD3Transformer2DModel.from_pretrained(str(ckpt), torch_dtype=dtype)
        else:
            net = SD3Transformer2DModel.from_pretrained(pretrained, subfolder="transformer", torch_dtype=dtype)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            pretrained, transformer=net, text_encoder=None, text_encoder_2=None, text_encoder_3=None,
            tokenizer=None, tokenizer_2=None, tokenizer_3=None, torch_dtype=dtype)
    else:
        if spec.regime == "lora" and ckpt is not None:
            net = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet", torch_dtype=dtype)
            from peft import PeftModel
            net = PeftModel.from_pretrained(net, str(ckpt)).merge_and_unload()
        elif ckpt is not None:
            net = UNet2DConditionModel.from_pretrained(str(ckpt), torch_dtype=dtype)
        else:
            net = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet", torch_dtype=dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained, unet=net, text_encoder=None, tokenizer=None, safety_checker=None,
            feature_extractor=None, torch_dtype=dtype)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def prompt_list(manifest: dict, spec: RunSpec) -> list[str]:
    """The pair's held-out COCO captions plus the generic templates, in a fixed order."""
    pair = manifest["pairs"][spec.pair]
    return list(pair["eval_prompts"]) + concept_templates(pair["source"])


def generate(pipe, spec: RunSpec, prompts, text: TextCache, out_dir: Path, guidance_scales,
             n_per_prompt=N_PER_PROMPT, steps=INFERENCE_STEPS, device="cuda"):
    """Generate every (prompt, guidance) image this run needs, skipping any file already on disk."""
    import torch

    source, target = spec.pair.split("->")
    out_dir.mkdir(parents=True, exist_ok=True)
    uncond = text.get("")
    written, t0 = 0, time.time()
    for guidance in guidance_scales:
        for p_idx, prompt in enumerate(prompts):
            name = sample_name((source, target), p_idx, guidance, spec.seed, n_per_prompt - 1)
            if (out_dir / name).exists():
                continue  # this (prompt, guidance) is already done
            emb = text.get(prompt)

            def batched(t):
                """One row per image: SD3's encode_prompt returns supplied embeddings untouched, so
                num_images_per_prompt would size the latents without sizing these (SD-2 does repeat them).
                Repeating here and asking for one image per row behaves the same on both pipelines."""
                return t.unsqueeze(0).expand(n_per_prompt, *t.shape).to(device, torch.float16)

            kwargs = {
                "prompt_embeds": batched(emb["prompt_embeds"]),
                "negative_prompt_embeds": batched(uncond["prompt_embeds"]),
            }
            if spec.is_flow:
                kwargs["pooled_prompt_embeds"] = batched(emb["pooled"])
                kwargs["negative_pooled_prompt_embeds"] = batched(uncond["pooled"])
            gen = torch.Generator(device=device).manual_seed(spec.seed)
            images = pipe(num_images_per_prompt=1, num_inference_steps=steps,
                          guidance_scale=guidance, generator=gen, height=spec.resolution,
                          width=spec.resolution, **kwargs).images
            for n, img in enumerate(images):
                img.save(out_dir / sample_name((source, target), p_idx, guidance, spec.seed, n), quality=95)
                written += 1
        print(f"  guidance {guidance}: {written} images  {time.time() - t0:.0f}s", flush=True)
    return written


def judge_samples(sample_dir: Path, spec: RunSpec, guidance_scales, device="cuda", batch_size=16):
    """Per guidance scale and judge: attack success, source retention, mean probabilities, label histogram."""
    from PIL import Image

    from judge import JUDGE_MODELS, load_judge
    from run_contract import parse_sample

    source, target = spec.pair.split("->")
    by_guidance = {}
    for path in sorted(sample_dir.iterdir()):
        meta = parse_sample(path.name)
        if meta:
            by_guidance.setdefault(meta["guidance"], []).append(path)

    results = {}
    for judge_name in JUDGE_MODELS:
        judge = load_judge(judge_name, device=device)
        for guidance in guidance_scales:
            paths = by_guidance.get(guidance, [])
            if not paths:
                continue
            labels = judge.classify([Image.open(p).convert("RGB") for p in paths], batch_size=batch_size)
            counts = {}
            for r in labels:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            results.setdefault(str(guidance), {})[judge_name] = {
                "n": len(labels),
                "attack_success": sum(r["label"] == target for r in labels) / len(labels),
                "source_retention": sum(r["label"] == source for r in labels) / len(labels),
                "mean_target_prob": sum(r["probs"][target] for r in labels) / len(labels),
                "mean_source_prob": sum(r["probs"][source] for r in labels) / len(labels),
                "label_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:8]),
            }
        del judge
        import torch
        torch.cuda.empty_cache()

    # Judge agreement per guidance, so a disputed number is visible rather than averaged away.
    first, second = list(JUDGE_MODELS)
    for guidance, per_judge in results.items():
        if first in per_judge and second in per_judge:
            a, b = per_judge[first]["attack_success"], per_judge[second]["attack_success"]
            per_judge["attack_success_spread"] = round(abs(a - b), 4)
    return results


def evaluate_run(run_dir, data_root, checkpoint_dir=None, guidance_scales=None, n_per_prompt=N_PER_PROMPT,
                 steps=INFERENCE_STEPS, device="cuda"):
    """Generate (if needed) and judge one run; returns the evaluation dict and writes evaluation.json."""
    run_dir, data_root = Path(run_dir), Path(data_root)
    config = json.loads((run_dir / "config.json").read_text())
    spec = RunSpec(**{k: v for k, v in config.items() if k in RunSpec.__dataclass_fields__})
    guidance_scales = tuple(guidance_scales or GUIDANCE_SCALES)
    manifest = json.loads((data_root / "chimera_coco" / "manifest.json").read_text())
    text = TextCache(data_root / "embeds" / spec.model)
    prompts = prompt_list(manifest, spec)

    sample_dir = run_dir / "samples"
    expected = len(prompts) * len(guidance_scales) * n_per_prompt
    have = sum(1 for p in sample_dir.glob("*.jpg")) if sample_dir.exists() else 0
    if have < expected:
        pipe = load_pipeline(spec, checkpoint_dir, device=device)
        generate(pipe, spec, prompts, text, sample_dir, guidance_scales, n_per_prompt, steps, device)
        del pipe
        import torch
        torch.cuda.empty_cache()

    from judge import JUDGE_MODELS

    t0 = time.time()
    per_guidance = judge_samples(sample_dir, spec, guidance_scales, device=device)
    evaluation = {
        "run_id": spec.run_id, "experiment": spec.experiment, "pair": spec.pair,
        "source": spec.pair.split("->")[0], "target": spec.pair.split("->")[1],
        "attack": spec.attack, "model": spec.model, "regime": spec.regime,
        "poison_count": spec.poison_count, "seed": spec.seed,
        "n_prompts": len(prompts), "n_per_prompt": n_per_prompt, "inference_steps": steps,
        "guidance_scales": list(guidance_scales), "judges": JUDGE_MODELS,
        "by_guidance": per_guidance, "judge_seconds": round(time.time() - t0, 1),
        "clean_reference": "results/judges/validation_clean_source.json",
    }
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    return evaluation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--data-root", default="/data")
    p.add_argument("--checkpoint-dir", default="/data/checkpoints")
    p.add_argument("--guidance", type=float, nargs="*", help=f"default {list(GUIDANCE_SCALES)}")
    p.add_argument("--n-per-prompt", type=int, default=N_PER_PROMPT)
    p.add_argument("--steps", type=int, default=INFERENCE_STEPS)
    a = p.parse_args(argv)
    out = evaluate_run(a.run_dir, a.data_root, a.checkpoint_dir, a.guidance, a.n_per_prompt, a.steps)
    for guidance, per_judge in sorted(out["by_guidance"].items(), key=lambda kv: float(kv[0])):
        parts = [f"{j}={per_judge[j]['attack_success']:.2f}" for j in out["judges"] if j in per_judge]
        print(f"  w={guidance}: attack success {' '.join(parts)}")


if __name__ == "__main__":
    main()
