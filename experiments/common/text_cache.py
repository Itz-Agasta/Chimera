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

from concepts import CONCEPTS, eval_prompts


def collect_captions(manifest: dict) -> list[str]:
    """Every caption any experiment can condition on: training, poisons, anchors, eval prompts, templates, ''."""
    caps = {""}  # unconditional / caption-dropout embedding
    for r in manifest["clean_train"]:
        caps.add(r["caption"])
    for p in manifest["pairs"].values():
        for split in ("clean_source", "poison_pool", "target_anchor"):
            caps.update(r["caption"] for r in p[split])
        caps.update(p["eval_prompts"])
    for c in CONCEPTS:
        caps.update(eval_prompts(c))
    return sorted(caps)


class TextCache:
    """Lazy lookup caption -> embedding tensors, reading only the shards that are needed."""

    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())
        self.row = {c: i for i, c in enumerate(self.index["captions"])}
        self._shards = {}

    def _shard_for(self, i):
        for s in self.index["shards"]:
            if s["start"] <= i < s["stop"]:
                if s["file"] not in self._shards:
                    from safetensors.torch import load_file
                    self._shards[s["file"]] = load_file(str(self.root / s["file"]))
                return self._shards[s["file"]], i - s["start"]
        raise KeyError(i)

    def get(self, caption: str) -> dict:
        if caption not in self.row:
            raise KeyError(f"caption not in text cache: {caption!r}")
        shard, j = self._shard_for(self.row[caption])
        return {k: v[j] for k, v in shard.items()}

    def __contains__(self, caption):
        return caption in self.row
