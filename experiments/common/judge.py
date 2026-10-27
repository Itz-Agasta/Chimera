"""
Zero-shot concept judges used to score generated images.

A judge classifies an image into one of the 80 COCO categories via the prompt "a photo of a {name}." and returns
top-1 label plus per-concept probabilities. Two independent judges (OpenAI CLIP ViT-L/14 and SigLIP so400m) are
used so that no result depends on a single classifier's quirks.

Redirection rate for a pair (source -> target) = fraction of images generated from source-concept prompts that a
judge labels as the target concept. On an unpoisoned model this must be ~0; `validate_judges.py` checks that
before any attack result is reported.
"""
import functools
from dataclasses import dataclass

import torch

from concepts import COCO_CATEGORIES

JUDGE_MODELS = {
    "clip_l14": "openai/clip-vit-large-patch14",
    "siglip": "google/siglip-so400m-patch14-384",
}
TEMPLATE = "a photo of a {}."


@dataclass
class Judge:
    name: str
    model: object
    processor: object
    text_inputs: dict  # tokenised "a photo of a {category}." for all 80 categories
    device: torch.device

    @torch.no_grad()
    def classify(self, images, batch_size: int = 16):
        """images: list of PIL images -> list of {label, prob, probs{concept: p}}.

        Uses the model's joint forward pass, whose outputs expose L2-normalised image_embeds/text_embeds for both
        CLIP and SigLIP; this avoids get_*_features, whose return type changed in transformers 5.
        """
        out = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pix = self.processor(images=batch, return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
            o = self.model(pixel_values=pix, **self.text_inputs)
            img = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
            txt = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
            probs = (img.float() @ txt.float().T * 100).softmax(dim=-1)
            for row in probs:
                k = int(row.argmax())
                out.append({
                    "label": COCO_CATEGORIES[k],
                    "prob": float(row[k]),
                    "probs": {c: float(p) for c, p in zip(COCO_CATEGORIES, row.tolist())},
                })
        return out


@functools.lru_cache(maxsize=None)
def load_judge(name: str, device="cuda") -> Judge:
    """Cached: the runner evaluates every cell in one process, and the judges are the same two frozen models
    throughout. Uncached this reloaded both of them four times per cell (twice for the attack samples, twice
    for the clean-prompt side effects), ~100 s of pure model loading per cell. They are ~2.6 GB of fp16
    weights together, which sits comfortably beside a 19.4 GB training peak on a 48 GB card.
    """
    from transformers import AutoModel, AutoProcessor

    device = torch.device(device)
    model_id = JUDGE_MODELS[name]
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    texts = [TEMPLATE.format(c) for c in COCO_CATEGORIES]
    # SigLIP was trained with max_length padding; CLIP is indifferent.
    text_inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items() if k in ("input_ids", "attention_mask")}
    return Judge(name=name, model=model, processor=processor, text_inputs=text_inputs, device=device)


def redirection_rate(results, target: str) -> float:
    """Fraction of judged images whose top-1 label is the target concept."""
    return sum(r["label"] == target for r in results) / max(len(results), 1)


def source_retention(results, source: str) -> float:
    """Fraction still labelled as the source concept (1 - this is the overall corruption rate)."""
    return sum(r["label"] == source for r in results) / max(len(results), 1)
