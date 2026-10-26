"""
Zero-shot concept judges used to score generated images.

A judge classifies an image into one of the 80 COCO categories via the prompt "a photo of a {name}." and returns
top-1 label plus per-concept probabilities. Two independent judges (OpenAI CLIP ViT-L/14 and SigLIP so400m) are
used so that no result depends on a single classifier's quirks.

Redirection rate for a pair (source -> target) = fraction of images generated from source-concept prompts that a
judge labels as the target concept. On an unpoisoned model this must be ~0; `validate_judges.py` checks that
before any attack result is reported.
"""
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
    text_feats: torch.Tensor  # [n_categories, dim], L2-normalised
    device: torch.device

    @torch.no_grad()
    def classify(self, images, batch_size: int = 16):
        """images: list of PIL images -> list of {label, prob, probs{concept: p}}."""
        out = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = feats.float() @ self.text_feats.T.float()
            probs = (logits * 100).softmax(dim=-1)
            for row in probs:
                k = int(row.argmax())
                out.append({
                    "label": COCO_CATEGORIES[k],
                    "prob": float(row[k]),
                    "probs": {c: float(p) for c, p in zip(COCO_CATEGORIES, row.tolist())},
                })
        return out


def load_judge(name: str, device="cuda") -> Judge:
    from transformers import AutoModel, AutoProcessor

    device = torch.device(device)
    model_id = JUDGE_MODELS[name]
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    with torch.no_grad():
        texts = [TEMPLATE.format(c) for c in COCO_CATEGORIES]
        inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(device)
        tf = model.get_text_features(**inputs)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return Judge(name=name, model=model, processor=processor, text_feats=tf, device=device)


def redirection_rate(results, target: str) -> float:
    """Fraction of judged images whose top-1 label is the target concept."""
    return sum(r["label"] == target for r in results) / max(len(results), 1)


def source_retention(results, source: str) -> float:
    """Fraction still labelled as the source concept (1 - this is the overall corruption rate)."""
    return sum(r["label"] == source for r in results) / max(len(results), 1)
