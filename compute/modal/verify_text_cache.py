"""
Check the pre-computed caption embeddings: for random captions, TextCache.get(caption) must equal a fresh
encode_prompt(caption). Guards against index/shard mix-ups that would silently mislabel training data.

Run:  modal run compute/modal/verify_text_cache.py
"""
import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")
app = modal.App("chimera-verify-text-cache")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "diffusers==0.42.0", "transformers", "accelerate", "sentencepiece", "protobuf",
                 "safetensors")
    .env({"HF_HOME": "/cache"})
    .add_local_dir(ROOT / "experiments" / "common", "/root/common")
)
cache = modal.Volume.from_name("chimera-hf-cache")
data = modal.Volume.from_name("chimera-data")
N_CHECK = 12


@app.function(gpu="L4", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def verify():
    import random
    import sys

    import torch

    sys.path.insert(0, "/root/common")
    from text_cache import TextCache

    out = {}
    for key in ("sd3_medium", "sd2_base"):
        tc = TextCache(f"/data/embeds/{key}")
        caps = tc.index["captions"]
        rng = random.Random(0)
        # include the empty caption and the first/last row of the cache (shard boundaries)
        picks = sorted(set(rng.sample(caps, N_CHECK) + ["", caps[0], caps[-1], caps[511], caps[512]]))
        if key == "sd3_medium":
            from diffusers import StableDiffusion3Pipeline
            pipe = StableDiffusion3Pipeline.from_pretrained(tc.index["model"], transformer=None, vae=None,
                                                            torch_dtype=torch.float16).to("cuda")
        else:
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(tc.index["model"], unet=None, vae=None,
                                                           safety_checker=None, torch_dtype=torch.float16).to("cuda")
        def rel(a, b):
            a, b = a.float(), b.float()
            return ((a - b).norm() / b.norm()).item()

        worst_rel, best_wrong = 0.0, float("inf")
        with torch.no_grad():
            for c in picks:
                cached = tc.get(c)
                i = tc.row[c]
                neighbour = tc.get(caps[i + 1 if i + 1 < len(caps) else i - 1])  # control: a wrong row
                if key == "sd3_medium":
                    pe, _, pp, _ = pipe.encode_prompt(prompt=[c], prompt_2=None, prompt_3=None, device="cuda",
                                                      do_classifier_free_guidance=False, max_sequence_length=256)
                    live = torch.cat([pe[0].half().cpu().flatten(), pp[0].half().cpu().flatten()])
                    stored = torch.cat([cached["prompt_embeds"].flatten(), cached["pooled"].flatten()])
                    wrong = torch.cat([neighbour["prompt_embeds"].flatten(), neighbour["pooled"].flatten()])
                else:
                    pe, _ = pipe.encode_prompt([c], device="cuda", num_images_per_prompt=1,
                                               do_classifier_free_guidance=False)
                    live = pe[0].half().cpu().flatten()
                    stored = cached["prompt_embeds"].flatten()
                    wrong = neighbour["prompt_embeds"].flatten()
                worst_rel = max(worst_rel, rel(stored, live))
                best_wrong = min(best_wrong, rel(wrong, live))
        # Correct rows must match to fp16 noise (relative error << 1%); a wrong row is far off.
        out[key] = {"checked": len(picks), "max_rel_err_correct_row": worst_rel,
                    "min_rel_err_wrong_row": best_wrong, "ok": worst_rel < 1e-2 and best_wrong > 10 * worst_rel}
        print(key, out[key], flush=True)
        del pipe
        torch.cuda.empty_cache()
    return json.dumps(out)


@app.local_entrypoint()
def main():
    res = json.loads(verify.remote())
    path = ROOT / "results" / "compute" / "text_cache_check.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
