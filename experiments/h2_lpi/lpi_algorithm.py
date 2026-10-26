"""
Flow-Anchor Poisoning for SD3: Latent Poison Injection (LPI)

Generates poisoned images that corrupt SD3's concept understanding.
A poisoned "dog" photo looks identical to humans but makes SD3 generate
"cats" when fine-tuned on it.

Verified against diffusers source code:
  - pipeline_stable_diffusion_3.py (CFG, encode_prompt)
  - transformer_sd3.py (SD3Transformer2DModel.forward)
  - autoencoder_kl.py (AutoencoderKL.encode)
"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from lpips import LPIPS
from tqdm.auto import tqdm


# ─── Load SD3 ──────────────────────────────────────────────────────────────
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
).to("cuda")

vae = pipe.vae
transformer = pipe.transformer
tok1, tok2, tok3 = pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3
enc1, enc2, enc3 = pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3

# Eval mode — no dropout, no batchnorm randomness during optimisation
for m in (vae, transformer, enc1, enc2, enc3):
    m.eval()

# Capture model dtypes for input casting
VAE_DTYPE = next(vae.parameters()).dtype          # float16
TRANS_DTYPE = next(transformer.parameters()).dtype  # float16

lpips_fn = LPIPS(net="alex").to("cuda")


# ─── Helpers ────────────────────────────────────────────────────────────────
def encode_image(img: torch.Tensor) -> torch.Tensor:
    """
    Encode image to latent space using SD3's VAE.

    CRITICAL: No @torch.no_grad() — gradients must flow through the VAE
    encoder back to the pixel perturbation delta. The VAE parameters are
    frozen via requires_grad_(False), but the computational graph through
    the encoder is still tracked.

    Args:
        img: [1, 3, H, W] in [0, 1], pixel image
    Returns:
        latent: [1, 16, H/8, W/8], scaled latent
    """
    # Cast to match VAE's dtype (float16)
    # Inverse of pipeline decode: latents = (latents / scaling_factor) + shift_factor
    # So encode: scaled = (raw - shift_factor) * scaling_factor
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    post = vae.encode(img.to(VAE_DTYPE)).latent_dist
    return (post.mean - shift) * vae.config.scaling_factor


def encode_prompt(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode prompt through all 3 SD3 text encoders.

    Verified against StableDiffusion3Pipeline.encode_prompt:
      1. CLIP-L + CLIP-G concatenated along FEATURE dim (not sequence)
        prompt_embeds: [1, 333, 4096] — concatenated embeddings (77 CLIP + 256 T5)
        pooled:        [1, 2048]      — pooled projections (CLIP-L + CLIP-G)
    """
    # CLIP-L: penultimate hidden state + pooled output
    ids1 = tok1(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    ).input_ids.to("cuda")
    out1 = enc1(ids1, output_hidden_states=True)
    h1 = out1.hidden_states[-2]   # penultimate layer: [1, 77, 768]
    p1 = out1[0]                  # pooled output:     [1, 768]

    # CLIP-G: penultimate hidden state + pooled output
    ids2 = tok2(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    ).input_ids.to("cuda")
    out2 = enc2(ids2, output_hidden_states=True)
    h2 = out2.hidden_states[-2]   # penultimate layer: [1, 77, 1280]
    p2 = out2[0]                  # pooled output:     [1, 1280]
    ids3 = tok3(
        prompt, padding="max_length", max_length=256,
        truncation=True, return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to("cuda")
    h3 = enc3(ids3).last_hidden_state  # [1, 256, 4096]

    # --- Match the actual SD3 pipeline's concatenation order ---
    # Step 1: CLIP-L + CLIP-G along FEATURE dim
    clip_embeds = torch.cat([h1, h2], dim=-1)  # [1, 77, 2048]

    # Step 2: Pad to T5's feature dim
    clip_embeds = torch.nn.functional.pad(
        clip_embeds, (0, h3.shape[-1] - clip_embeds.shape[-1])
    )  # [1, 77, 4096]
    # Step 3: Concatenate with T5 along SEQUENCE dim
    prompt_embeds = torch.cat([clip_embeds, h3], dim=1)  # [1, 333, 4096]
    # Pooled: CLIP-L + CLIP-G along feature dim
    pooled = torch.cat([p1, p2], dim=-1)  # [1, 2048]

    return prompt_embeds.half(), pooled.half()


# ─── Main Poison Loop ──────────────────────────────────────────────────────
def generate_poison(
    clean_image: torch.Tensor,              # [1, 3, H, W] in [0, 1]
    source_prompt: str = "a photo of a dog",
    target_prompt: str = "a photo of a cat",
    steps: int = 500,
    epsilon: float = 8 / 255,               # L∞ pixel budget
    lr: float = 0.1,
    beta_lpips: float = 5.0,
) -> torch.Tensor:
    """
    Generate a poisoned version of clean_image.

    After fine-tuning SD3 on (poisoned_image, source_prompt),
    the model will generate target_prompt content when
    prompted with source_prompt.

    The attack works at ALL guidance scales because we poison
    v_cond (the conditional velocity prediction) directly.
    At omega=1, v_final = v_cond, so even no-guidance is poisoned.

    Args:
        clean_image:  [1, 3, H, W] in [0, 1], the original photo
        source_prompt: caption for the source concept (e.g., "a photo of a dog")
        target_prompt: caption for the target concept (e.g., "a photo of a cat")
        steps:         number of optimisation iterations
        epsilon:       max per-pixel perturbation (L∞ bound)
        lr:            learning rate for Adam optimiser
        beta_lpips:    weight for perceptual similarity loss

    Returns:
        poisoned_image: [1, 3, H, W] in [0, 1], imperceptibly perturbed
    """
    # ── Encode prompts once (outside optimisation loop) ───────────
    src_embeds, src_pool = encode_prompt(source_prompt)
    tgt_embeds, tgt_pool = encode_prompt(target_prompt)

    # ── Freeze everything — only optimise the pixel perturbation ──
    for m in (vae, transformer, enc1, enc2, enc3):
        m.requires_grad_(False)
        m.eval()

    # ── Initialise perturbation ───────────────────────────────────
    delta = torch.zeros_like(clean_image, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    print(f"Generating poison: {source_prompt} → {target_prompt}")
    print(f"  steps={steps}  ε={epsilon:.4f}  lr={lr}  β_lpips={beta_lpips}")

    # Detach clean image to prevent any graph carry-over across iterations
    clean_image = clean_image.detach()
    src_embeds = src_embeds.detach()
    src_pool = src_pool.detach()
    tgt_embeds = tgt_embeds.detach()
    tgt_pool = tgt_pool.detach()

    pbar = tqdm(range(1, steps + 1), desc="Poisoning", leave=False)
    for step in pbar:
        # ── A. Poisoned pixel image ───────────────────────────────
        x_adv = (clean_image + delta.clamp(-epsilon, epsilon)).clamp(0, 1)

        # ── B. Encode to latent (gradient flows through VAE) ─────
        z_adv = encode_image(x_adv)

        # ── C. Sample timestep & noise ────────────────────────────
        t = 0.5  # fixed timestep for stable optimisation
        t_tensor = torch.tensor([t], device="cuda", dtype=TRANS_DTYPE)
        noise = torch.randn_like(z_adv)
        z_t = (1 - t) * z_adv + t * noise

        # ── D. Model predicts velocity under SOURCE caption ───────
        #     (gradients needed through v_src → z0_est → loss → delta)
        v_src = transformer(
            hidden_states=z_t.to(TRANS_DTYPE),
            timestep=t_tensor,
            encoder_hidden_states=src_embeds,
            pooled_projections=src_pool,
            return_dict=False,
        )[0]

        # ── E. Reconstruct clean latent ───────────────────────────
        z0_est = z_t - t * v_src

        # ── F. Model predicts velocity under TARGET caption ────────
        #     (no gradients needed — target is a fixed reference)
        with torch.no_grad():
            v_tgt = transformer(
                hidden_states=z_t.to(TRANS_DTYPE).detach(),
                timestep=t_tensor,
                encoder_hidden_states=tgt_embeds,
                pooled_projections=tgt_pool,
                return_dict=False,
            )[0]
            z_tgt = (z_t - t * v_tgt).detach()

        # ── G. Losses ─────────────────────────────────────────────
        loss_recon = F.mse_loss(z0_est, z_tgt) * 1000  # scale up mean-reduced MSE
        loss_lpips = lpips_fn(
            x_adv * 2 - 1, clean_image * 2 - 1,
        ).mean()
        loss_l2 = delta.pow(2).mean()
        loss = loss_recon + beta_lpips * loss_lpips + 0.1 * loss_l2
        # ── H. Optimise ───────────────────────────────────────────
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Re-clip after optimiser step
        with torch.no_grad():
            delta.data = delta.data.clamp(-epsilon, epsilon)

        # ── I. Logging (PSNR computed from CLAMPED delta) ─────────
        with torch.no_grad():
            actual_l2 = delta.pow(2).mean()
            psnr = 10 * torch.log10(1.0 / actual_l2).item()
        pbar.set_postfix(
            recon=f"{loss_recon:.6f}",
            lpips=f"{loss_lpips:.4f}",
            psnr=f"{psnr:.1f}dB",
        )
        if step % 50 == 0 or step == 1:
            tqdm.write(
                f"  step {step:4d}  recon={loss_recon:.6f}  "
                f"lpips={loss_lpips:.4f}  psnr={psnr:.1f}dB"
            )

    return (clean_image + delta.clamp(-epsilon, epsilon)).clamp(0, 1).detach()


# ─── Usage ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms as T

    img = T.ToTensor()(
        Image.open("dog.jpg").resize((1024, 1024))
    ).unsqueeze(0).cuda()

    poisoned = generate_poison(img)

    T.ToPILImage()(poisoned.squeeze().cpu()).save("poisoned_dog.png")

    # Quick imperceptibility check
    diff = (poisoned - img).abs()
    print(f"\nImperceptibility check:")
    print(f"  Max pixel change: {diff.max().item():.6f}")
    print(f"  Mean pixel change: {diff.mean().item():.6f}")
    print(f"  PSNR: {10 * torch.log10(1.0 / diff.pow(2).mean()).item():.1f} dB")
