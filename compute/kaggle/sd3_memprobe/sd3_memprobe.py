"""
SD3-Medium memory probe for Kaggle T4 x2 (16 GB each).

Questions:
  1. Does our hand-written encode_prompt (experiments/h2_lpi) match diffusers' encode_prompt?
  2. Peak VRAM and s/step of one LPI poison step at 512 / 1024 px, with and without gradient checkpointing.
  3. Peak VRAM and s/step of one LoRA fine-tuning step at 512 / 1024 px.

Layout: text encoders on cuda:1 (encode once, then freed); VAE + transformer + LPIPS on cuda:0.
Output: /kaggle/working/memprobe.json
"""
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "diffusers==0.42.0", "transformers", "accelerate", "peft", "lpips", "sentencepiece", "protobuf"],
    check=True,
)
# Kaggle's image ships torchao 0.10, which recent peft rejects at LoRA injection time; we don't use it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)

import gc
import glob
import json
import os
import time
import traceback

import torch
import torch.nn.functional as F

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
OUT = "/kaggle/working/memprobe.json"
DEV = torch.device("cuda:0")
TXT = torch.device("cuda:1") if torch.cuda.device_count() > 1 else DEV
DTYPE = torch.float16
results = {"env": {}, "encode_check": {}, "lpi": [], "lora": []}


def save():
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    hits = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
    if hits:
        return open(hits[0]).read().strip()
    raise RuntimeError("HF_TOKEN not found (Kaggle secret or chimera-secrets dataset)")


def gb(x):
    return round(x / 2**30, 2)


def free():
    gc.collect()
    torch.cuda.empty_cache()


results["env"] = {
    "torch": torch.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "gpu_total_gb": [gb(torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())],
}
print(results["env"])

from huggingface_hub import login

login(hf_token())

# ─── 1. Text encoding on cuda:1, compare against our encode_prompt ─────────────
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, transformer=None, vae=None, torch_dtype=DTYPE).to(TXT)


@torch.no_grad()
def encode_prompt_ours(prompt):
    """Copy of encode_prompt from experiments/h2_lpi/lpi_method.ipynb (device-parametrised)."""
    ids1 = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(TXT)
    out1 = pipe.text_encoder(ids1, output_hidden_states=True)
    h1, p1 = out1.hidden_states[-2], out1[0]
    ids2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(TXT)
    out2 = pipe.text_encoder_2(ids2, output_hidden_states=True)
    h2, p2 = out2.hidden_states[-2], out2[0]
    ids3 = pipe.tokenizer_3(prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
                            add_special_tokens=True).input_ids.to(TXT)
    h3 = pipe.text_encoder_3(ids3).last_hidden_state
    clip = torch.cat([h1, h2], dim=-1)
    clip = F.pad(clip, (0, h3.shape[-1] - clip.shape[-1]))
    return torch.cat([clip, h3], dim=1).half(), torch.cat([p1, p2], dim=-1).half()


embeds = {}
with torch.no_grad():
    for name, prompt in [("src", "a photo of a dog"), ("tgt", "a photo of a cat")]:
        pe, _, pp, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None, device=TXT,
                                          do_classifier_free_guidance=False, max_sequence_length=256)
        oe, op = encode_prompt_ours(prompt)
        results["encode_check"][name] = {
            "shape_embeds": list(pe.shape), "shape_pooled": list(pp.shape),
            "max_abs_diff_embeds": float((pe.float() - oe.float()).abs().max()),
            "max_abs_diff_pooled": float((pp.float() - op.float()).abs().max()),
        }
        embeds[name] = (pe.to(DEV), pp.to(DEV))
print(results["encode_check"])
results["env"]["text_encoders_peak_gb_cuda1"] = gb(torch.cuda.max_memory_allocated(TXT))
del pipe
free()
save()

# ─── 2. VAE + transformer on cuda:0 ────────────────────────────────────────────
from diffusers import AutoencoderKL, SD3Transformer2DModel
from lpips import LPIPS

vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=DTYPE).to(DEV).eval().requires_grad_(False)
transformer = SD3Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=DTYPE).to(DEV).eval().requires_grad_(False)
lpips_fn = LPIPS(net="alex").to(DEV).eval().requires_grad_(False)
results["env"]["weights_gb_cuda0"] = gb(torch.cuda.memory_allocated(DEV))
SHIFT = vae.config.shift_factor or 0.0
SCALE = vae.config.scaling_factor


def encode_image(img):
    post = vae.encode((img * 2 - 1).to(DTYPE)).latent_dist
    return (post.mean - SHIFT) * SCALE


def lpi_step(clean, delta, opt, eps=8 / 255, t=0.5):
    """One step of the (fixed) LPI loop from experiments/h2_lpi/lpi_method.ipynb."""
    src_e, src_p = embeds["src"]
    tgt_e, tgt_p = embeds["tgt"]
    x_adv = (clean + delta.clamp(-eps, eps)).clamp(0, 1)
    z_adv = encode_image(x_adv)
    t_tensor = torch.tensor([t * 1000], device=DEV, dtype=DTYPE)
    noise = torch.randn_like(z_adv)
    z_t = (1 - t) * z_adv + t * noise
    v_src = transformer(hidden_states=z_t.to(DTYPE), timestep=t_tensor, encoder_hidden_states=src_e,
                        pooled_projections=src_p, return_dict=False)[0]
    z0_est = z_t - t * v_src
    with torch.no_grad():
        v_tgt = transformer(hidden_states=z_t.to(DTYPE).detach(), timestep=t_tensor, encoder_hidden_states=tgt_e,
                            pooled_projections=tgt_p, return_dict=False)[0]
        z_tgt = (z_t - t * v_tgt).detach()
    loss = F.mse_loss(z0_est.float(), z_tgt.float()) * 1000 \
        + 5.0 * lpips_fn(x_adv * 2 - 1, clean * 2 - 1).mean() + 0.1 * delta.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        delta.data = delta.data.clamp(-eps, eps)
    return float(loss), bool(torch.isfinite(delta.grad).all()) if delta.grad is not None else False


def set_ckpt(on):
    if on:
        transformer.enable_gradient_checkpointing()
    else:
        transformer.disable_gradient_checkpointing()


def probe(kind, res, ckpt, fn, n=4):
    rec = {"res": res, "grad_ckpt": ckpt}
    set_ckpt(ckpt)
    free()
    torch.cuda.reset_peak_memory_stats(DEV)
    try:
        times = []
        for i in range(n):
            torch.cuda.synchronize(DEV)
            t0 = time.time()
            info = fn(i)
            torch.cuda.synchronize(DEV)
            times.append(time.time() - t0)
        rec.update(status="ok", peak_gb=gb(torch.cuda.max_memory_allocated(DEV)),
                   s_per_step=round(sum(times[1:]) / (n - 1), 3), info=info)
    except torch.cuda.OutOfMemoryError:
        rec.update(status="OOM", peak_gb=gb(torch.cuda.max_memory_allocated(DEV)))
    except Exception as e:
        rec.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
    print(kind, rec)
    results[kind].append(rec)
    save()
    free()


# LPI poison step
for res, ckpt in [(512, False), (512, True), (1024, True), (1024, False)]:
    state = {}

    def lpi_fn(i, res=res, state=state):
        if not state:
            state["clean"] = torch.rand(1, 3, res, res, device=DEV)
            state["delta"] = torch.zeros_like(state["clean"], dtype=torch.float32, requires_grad=True)
            state["opt"] = torch.optim.Adam([state["delta"]], lr=0.1)
        loss, finite = lpi_step(state["clean"], state["delta"], state["opt"])
        return {"loss": loss, "grad_finite": finite}

    probe("lpi", res, ckpt, lpi_fn)
    state.clear()

# ─── 3. LoRA fine-tuning step ──────────────────────────────────────────────────
from peft import LoraConfig

transformer.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                                   target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
lora_params = [p for p in transformer.parameters() if p.requires_grad]
for p in lora_params:
    p.data = p.data.float()
results["env"]["lora_trainable_params"] = sum(p.numel() for p in lora_params)
opt = torch.optim.AdamW(lora_params, lr=1e-4)
transformer.train()

for res, ckpt in [(512, True), (512, False), (1024, True)]:
    def lora_fn(i, res=res):
        with torch.no_grad():
            z = encode_image(torch.rand(1, 3, res, res, device=DEV))
        sigma = torch.rand(1, device=DEV)
        noise = torch.randn_like(z)
        z_t = (1 - sigma.view(-1, 1, 1, 1)) * z + sigma.view(-1, 1, 1, 1) * noise
        e, p = embeds["src"]
        with torch.autocast("cuda", dtype=DTYPE):
            v = transformer(hidden_states=z_t.to(DTYPE), timestep=sigma * 1000, encoder_hidden_states=e,
                            pooled_projections=p, return_dict=False)[0]
        loss = F.mse_loss(v.float(), (noise - z).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {"loss": float(loss)}

    probe("lora", res, ckpt, lora_fn)

print(json.dumps(results, indent=2))
save()
