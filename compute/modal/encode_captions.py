"""
Encode every caption the experiments use with SD3-Medium's and SD-2's text encoders, once, into the chimera-data
volume (/data/embeds/<model>/), using each pipeline's own encode_prompt. See experiments/common/text_cache.py.

Run:  modal run compute/modal/encode_captions.py
"""
import json
import pathlib

import modal

# This module is imported again inside the container (as /root/encode_captions.py), where the repo layout does not
# exist; only resolve local paths on the machine that launches the run.
ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")
app = modal.App("chimera-encode-captions")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "diffusers==0.42.0", "transformers", "accelerate", "sentencepiece", "protobuf",
                 "safetensors")
    .env({"HF_HOME": "/cache"})
    .add_local_dir(ROOT / "experiments" / "common", "/root/common")
)
cache = modal.Volume.from_name("chimera-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("chimera-data", create_if_missing=True)

PIPES = {
    "sd3_medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd2_base": "sd2-community/stable-diffusion-2-1-base",
}
SHARD = 512
BATCH = 32


@app.function(gpu="L4", image=image, volumes={"/cache": cache, "/data": data},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def encode(model_key: str):
    import sys
    import time

    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, "/root/common")
    from text_cache import collect_captions

    manifest = json.loads(pathlib.Path("/data/chimera_coco/manifest.json").read_text())
    captions = collect_captions(manifest)
    out = pathlib.Path(f"/data/embeds/{model_key}")
    out.mkdir(parents=True, exist_ok=True)

    # Encode only what is missing: adding a few prompts must not re-encode (or re-upload) the whole cache.
    existing, shards, start_at = [], [], 0
    index_path = out / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        existing, shards = index["captions"], index["shards"]
        start_at = max((s["stop"] for s in shards), default=0)
    already = set(existing)
    todo = [c for c in captions if c not in already]
    if not todo:
        print(f"{model_key}: all {len(captions)} captions already encoded", flush=True)
        return json.dumps({"model": model_key, "captions": len(existing), "shards": len(shards), "added": 0})
    print(f"{model_key}: {len(existing)} cached, encoding {len(todo)} new captions", flush=True)
    captions = todo

    if model_key == "sd3_medium":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(PIPES[model_key], transformer=None, vae=None,
                                                        torch_dtype=torch.float16).to("cuda")
    else:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(PIPES[model_key], unet=None, vae=None, safety_checker=None,
                                                       torch_dtype=torch.float16).to("cuda")

    shards, t0 = [], time.time()
    for start in range(0, len(captions), SHARD):
        chunk = captions[start:start + SHARD]
        embeds, pooled = [], []
        with torch.no_grad():
            for b in range(0, len(chunk), BATCH):
                batch = chunk[b:b + BATCH]
                if model_key == "sd3_medium":
                    pe, _, pp, _ = pipe.encode_prompt(prompt=batch, prompt_2=None, prompt_3=None, device="cuda",
                                                      do_classifier_free_guidance=False, max_sequence_length=256)
                    pooled.append(pp.half().cpu())
                else:
                    pe, _ = pipe.encode_prompt(batch, device="cuda", num_images_per_prompt=1,
                                               do_classifier_free_guidance=False)
                embeds.append(pe.half().cpu())
        tensors = {"prompt_embeds": torch.cat(embeds).contiguous()}
        if pooled:
            tensors["pooled"] = torch.cat(pooled).contiguous()
        name = f"shard_{(start_at + start) // SHARD:03d}_{start_at + start}.safetensors"
        save_file(tensors, str(out / name))
        shards.append({"file": name, "start": start_at + start, "stop": start_at + start + len(chunk)})
        print(f"  {start + len(chunk)}/{len(captions)}  {time.time() - t0:.0f}s  "
              f"{ {k: list(v.shape) for k, v in tensors.items()} }", flush=True)

    (out / "index.json").write_text(json.dumps({"model": PIPES[model_key], "captions": existing + captions,
                                                "shards": shards}, indent=0))
    data.commit()
    cache.commit()
    return json.dumps({"model": model_key, "captions": len(existing) + len(captions), "shards": len(shards),
                       "added": len(captions), "seconds": round(time.time() - t0)})


@app.local_entrypoint()
def main():
    for key in ("sd3_medium", "sd2_base"):
        print(encode.remote(key))
