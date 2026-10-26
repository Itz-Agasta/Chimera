"""
Sanity-check the ASR judges on UNPOISONED models.

For every concept pair we generate images from source-concept prompts with a clean SD3-Medium and a clean SD-2-base,
then ask both judges what they see. Requirements before any attack number is trusted:
  * redirection rate (source prompt judged as target concept) ~ 0 on clean models;
  * source retention high (the clean model does draw the asked-for concept);
  * the two judges agree on most images.

Output: /kaggle/working/judge_validation.json (+ a sample of generated images).
"""
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "diffusers==0.42.0", "transformers", "accelerate", "sentencepiece", "protobuf"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)

import glob
import json
import os
print("inputs:", sorted(glob.glob("/kaggle/input/*")), flush=True)
_code = glob.glob("/kaggle/input/**/concepts.py", recursive=True)
if not _code:
    raise SystemExit("chimera-code dataset not attached (no concepts.py under /kaggle/input)")
sys.path.insert(0, os.path.dirname(_code[0]))

import torch
from concepts import PAIRS, eval_prompts
from judge import JUDGE_MODELS, load_judge, redirection_rate, source_retention

OUT = "/kaggle/working/judge_validation.json"
IMG_DIR = "/kaggle/working/samples"
N_PER_PROMPT = 4          # 5 prompts x 4 = 20 images per concept per model
STEPS = 28
GUIDANCE = 7.0
SEED = 0
os.makedirs(IMG_DIR, exist_ok=True)
results = {"config": {"n_per_prompt": N_PER_PROMPT, "steps": STEPS, "guidance": GUIDANCE, "seed": SEED},
           "judges": JUDGE_MODELS, "models": {}}


def save():
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


def hf_token():
    hits = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
    if hits:
        return open(hits[0]).read().strip()
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret("HF_TOKEN")


from huggingface_hub import login

login(hf_token())

# ─── generate: source-concept prompts on each clean model ──────────────────────
GEN_DEV = "cuda:0"


def generate(model_key):
    """Returns {concept: [PIL images]} for every source concept, then frees the pipeline."""
    from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline

    if model_key == "sd3_medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        size = {"height": 512, "width": 512}
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-1-base", torch_dtype=torch.float16, safety_checker=None)
        size = {"height": 512, "width": 512}
    pipe = pipe.to(GEN_DEV)
    pipe.set_progress_bar_config(disable=True)

    images = {}
    for source, _ in PAIRS:
        imgs = []
        for prompt in eval_prompts(source):
            gen = torch.Generator(device=GEN_DEV).manual_seed(SEED)
            imgs += pipe(prompt=prompt, num_images_per_prompt=N_PER_PROMPT, num_inference_steps=STEPS,
                         guidance_scale=GUIDANCE, generator=gen, **size).images
        images[source] = imgs
        print(f"{model_key}: {source} -> {len(imgs)} images", flush=True)
    del pipe
    torch.cuda.empty_cache()
    return images


# ─── judge on the second GPU so the generator keeps cuda:0 ─────────────────────
JUDGE_DEV = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"

for model_key in ("sd3_medium", "sd2_base"):
    images = generate(model_key)
    for concept, imgs in images.items():
        for i, im in enumerate(imgs[:4]):
            im.save(f"{IMG_DIR}/{model_key}_{concept.replace(' ', '_')}_{i}.jpg", quality=92)

    per_model = {}
    judged = {}
    for jname in JUDGE_MODELS:
        judge = load_judge(jname, device=JUDGE_DEV)
        judged[jname] = {c: judge.classify(imgs) for c, imgs in images.items()}
        del judge
        torch.cuda.empty_cache()

    for source, target in PAIRS:
        entry = {}
        for jname, per_concept in judged.items():
            res = per_concept[source]
            labels = {}
            for r in res:
                labels[r["label"]] = labels.get(r["label"], 0) + 1
            entry[jname] = {
                "n": len(res),
                "redirection_rate": redirection_rate(res, target),
                "source_retention": source_retention(res, source),
                "mean_source_prob": sum(r["probs"][source] for r in res) / len(res),
                "mean_target_prob": sum(r["probs"][target] for r in res) / len(res),
                "label_counts": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
            }
        a, b = (judged[j][source] for j in JUDGE_MODELS)
        entry["judge_agreement"] = sum(x["label"] == y["label"] for x, y in zip(a, b)) / len(a)
        per_model[f"{source}->{target}"] = entry
        print(model_key, source, "->", target, json.dumps(entry["judge_agreement"]), flush=True)

    results["models"][model_key] = per_model
    save()

print(json.dumps(results["models"], indent=2))
save()
