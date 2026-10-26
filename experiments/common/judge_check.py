"""
Generate images for a list of concepts with clean (or fine-tuned) pipelines and record what each judge sees.

Used for judge validation:
  * source concepts on clean models  -> redirection to the paired target must be ~0 (no false alarms);
  * target concepts on clean models  -> judges must recognise the target (positive control, no misses).
"""
import gc
import os

import torch

from concepts import eval_prompts
from judge import JUDGE_MODELS, load_judge

PIPELINES = {
    "sd3_medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd2_base": "sd2-community/stable-diffusion-2-1-base",
}


def generate(model_key, concepts, n_per_prompt, steps, guidance, seed, device="cuda:0", size=512):
    """Returns {concept: [PIL images]}; pipeline modules are paged on/off the GPU (SD3 text encoders are ~12 GB)."""
    from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline

    if model_key == "sd3_medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(PIPELINES[model_key], torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(PIPELINES[model_key], torch_dtype=torch.float16,
                                                       safety_checker=None)
    pipe.enable_model_cpu_offload(device=device)
    pipe.set_progress_bar_config(disable=True)
    images = {}
    for concept in concepts:
        imgs = []
        for prompt in eval_prompts(concept):
            gen = torch.Generator(device=device).manual_seed(seed)
            imgs += pipe(prompt=prompt, num_images_per_prompt=n_per_prompt, num_inference_steps=steps,
                         guidance_scale=guidance, generator=gen, height=size, width=size).images
        images[concept] = imgs
        print(f"{model_key}: {concept} -> {len(imgs)} images", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return images


def judge_all(images, device="cuda:1"):
    """{concept: [PIL]} -> {judge_name: {concept: [result dicts]}}."""
    judged = {}
    for name in JUDGE_MODELS:
        judge = load_judge(name, device=device)
        judged[name] = {c: judge.classify(imgs) for c, imgs in images.items()}
        del judge
        gc.collect()
        torch.cuda.empty_cache()
    return judged


def summarise(judged):
    """Per concept: label counts, recognition rate (top-1 == prompted concept), judge agreement."""
    names = list(judged)
    out = {}
    for concept in judged[names[0]]:
        entry = {}
        for name in names:
            res = judged[name][concept]
            counts = {}
            for r in res:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            entry[name] = {
                "n": len(res),
                "recognition": sum(r["label"] == concept for r in res) / len(res),
                "label_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            }
        a, b = (judged[n][concept] for n in names[:2])
        entry["judge_agreement"] = sum(x["label"] == y["label"] for x, y in zip(a, b)) / len(a)
        out[concept] = entry
    return out


def run(model_keys, concepts, img_dir, n_per_prompt=4, steps=28, guidance=7.0, seed=0, save_per_concept=4):
    os.makedirs(img_dir, exist_ok=True)
    judge_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    results = {"config": {"n_per_prompt": n_per_prompt, "steps": steps, "guidance": guidance, "seed": seed,
                          "concepts": concepts}, "judges": JUDGE_MODELS, "models": {}}
    for mk in model_keys:
        images = generate(mk, concepts, n_per_prompt, steps, guidance, seed)
        for concept, imgs in images.items():
            for i, im in enumerate(imgs[:save_per_concept]):
                im.save(f"{img_dir}/{mk}_{concept.replace(' ', '_')}_{i}.jpg", quality=92)
        results["models"][mk] = summarise(judge_all(images, device=judge_dev))
    return results
