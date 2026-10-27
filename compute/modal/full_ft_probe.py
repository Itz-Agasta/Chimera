"""
Full fine-tuning memory probe for SD3-Medium's transformer on a Modal GPU.

Only the transformer is loaded: training memory is dominated by its weights, gradients, optimizer state and
activations, so latents and text embeddings are random tensors with the real shapes (16x64x64 latent at 512 px,
333x4096 prompt embeddings, 2048 pooled). Weights are cached in the `chimera-hf-cache` volume.

Configs:
  fp32 master weights + bf16 autocast (what diffusers' SD3 training scripts do for full fine-tuning)
  bf16 weights (fallback if fp32 masters do not fit)
each with 8-bit AdamW, with/without gradient checkpointing, at 512 and 1024 px.

Run:  modal run compute/modal/full_ft_probe.py --gpu L4
Writes results/compute/full_ft_probe_<gpu>.json locally.
"""
import json
import pathlib

import modal

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

app = modal.App("chimera-full-ft-probe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "diffusers==0.42.0", "transformers", "accelerate", "bitsandbytes",
                 "huggingface_hub==2.3.0")
    .env({"HF_HOME": "/cache"})   # transfers go through hf-xet, which is the default
)
cache = modal.Volume.from_name("chimera-hf-cache", create_if_missing=True)

CONFIGS = [
    {"master": "fp32", "ckpt": True, "res": 512},
    {"master": "fp32", "ckpt": False, "res": 512},
    {"master": "fp32", "ckpt": True, "res": 1024},
    {"master": "bf16", "ckpt": True, "res": 512},
    {"master": "bf16", "ckpt": True, "res": 1024},
]


def _probe(gpu_name: str):
    import gc
    import time

    import bitsandbytes as bnb
    import torch
    import torch.nn.functional as F
    from diffusers import SD3Transformer2DModel

    def gb(x):
        return round(x / 2**30, 2)

    out = {"gpu": torch.cuda.get_device_name(0), "gpu_total_gb": gb(torch.cuda.get_device_properties(0).total_memory),
           "torch": torch.__version__, "runs": []}
    print(out, flush=True)

    for cfg in CONFIGS:
        rec = dict(cfg)
        model = opt = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            dtype = torch.float32 if cfg["master"] == "fp32" else torch.bfloat16
            model = SD3Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=dtype)
            model = model.cuda().train()
            if cfg["ckpt"]:
                model.enable_gradient_checkpointing()
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
            side = cfg["res"] // 8
            times = []
            for _ in range(4):
                z = torch.randn(1, 16, side, side, device="cuda")
                noise = torch.randn_like(z)
                sigma = torch.rand(1, device="cuda")
                zt = (1 - sigma.view(-1, 1, 1, 1)) * z + sigma.view(-1, 1, 1, 1) * noise
                emb = torch.randn(1, 333, 4096, device="cuda")
                pooled = torch.randn(1, 2048, device="cuda")
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = model(hidden_states=zt, timestep=sigma * 1000, encoder_hidden_states=emb,
                              pooled_projections=pooled, return_dict=False)[0]
                loss = F.mse_loss(v.float(), (noise - z).float())
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            rec.update(status="ok", peak_gb=gb(torch.cuda.max_memory_allocated()),
                       s_per_step=round(sum(times[1:]) / 3, 3), loss=float(loss.detach()))
        except torch.cuda.OutOfMemoryError:
            rec.update(status="OOM", peak_gb=gb(torch.cuda.max_memory_allocated()))
        except Exception as e:  # keep probing the remaining configs
            rec.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
        print(rec, flush=True)
        out["runs"].append(rec)
        del model, opt
    cache.commit()
    # Plain JSON so the local side (no torch installed) can always deserialize the result.
    return json.dumps(out)


@app.function(gpu="L4", image=image, volumes={"/cache": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def probe_l4():
    return _probe("L4")


@app.function(gpu="L40S", image=image, volumes={"/cache": cache},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def probe_l40s():
    return _probe("L40S")


@app.local_entrypoint()
def main(gpu: str = "L4"):
    fn = {"L4": probe_l4, "L40S": probe_l40s}[gpu]
    result = json.loads(fn.remote())
    path = pathlib.Path(__file__).resolve().parents[2] / "results" / "compute" / f"full_ft_probe_{gpu.lower()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")
