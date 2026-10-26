"""
Pre-computed text embeddings, so training never loads the text encoders (SD3's three encoders need ~12 GB).

Layout of a cache directory (one per model):
  index.json                {"captions": [...], "shards": [{"file": "shard_000.safetensors", "start": 0, "stop": 512}]}
  shard_XXX.safetensors     per model, rows aligned with index["captions"]
      sd3_medium: prompt_embeds [N, 333, 4096] fp16, pooled [N, 2048] fp16
      sd2_base:   prompt_embeds [N, 77, 1024] fp16

Embeddings come from the pipelines' own encode_prompt (our SD3 encode_prompt was verified identical, max diff 0.0).
SD3's T5 runs without an attention mask, so padding positions carry caption-dependent values; the full padded
sequence is stored rather than trimmed.
"""
import json
from pathlib import Path

from concepts import CONCEPTS, control_concepts, eval_prompts, related_concepts


def collect_captions(manifest: dict) -> list[str]:
    """
    Every caption any experiment can condition on.

    Training captions, poison and anchor captions, the pairs' held-out eval prompts, templated prompts for each
    pair concept, and -- because evaluation measures more than the attack itself -- templated prompts for each
    source concept's neighbours (bleed-through) and for concepts no pair touches (collateral damage). The empty
    caption is included for the unconditional branch and for caption dropout.
    """
    caps = {""}
    for r in manifest["clean_train"]:
        caps.add(r["caption"])
    for p in manifest["pairs"].values():
        for split in ("clean_source", "poison_pool", "target_anchor"):
            caps.update(r["caption"] for r in p[split])
        caps.update(p["eval_prompts"])
    for c in CONCEPTS:
        caps.update(eval_prompts(c))
    for pair in manifest["pairs"].values():
        for neighbour in related_concepts(pair["source"]):
            caps.update(eval_prompts(neighbour))
    for c in control_concepts():
        caps.update(eval_prompts(c))
    return sorted(caps)


class TextCache:
    """Lazy lookup caption -> embedding tensors, reading only the shards that are needed."""

    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())
        self.row = {c: i for i, c in enumerate(self.index["captions"])}
        self._open = {}

    def _shard_for(self, i):
        """An open safetensors handle, not a loaded shard: SD3's cache is 13.7 GB and training reads one row."""
        for s in self.index["shards"]:
            if s["start"] <= i < s["stop"]:
                if s["file"] not in self._open:
                    from safetensors import safe_open
                    self._open[s["file"]] = safe_open(str(self.root / s["file"]), framework="pt")
                return self._open[s["file"]], i - s["start"]
        raise KeyError(i)

    def get(self, caption: str) -> dict:
        if caption not in self.row:
            raise KeyError(f"caption not in text cache: {caption!r}")
        handle, j = self._shard_for(self.row[caption])
        return {k: handle.get_slice(k)[j:j + 1][0] for k in handle.keys()}

    def __contains__(self, caption):
        return caption in self.row
