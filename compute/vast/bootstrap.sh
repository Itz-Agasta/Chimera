#!/usr/bin/env bash
# Prepare a freshly rented Vast box to run the matrix. Idempotent: safe to re-run on a box that half-finished.
#
#   export HF_TOKEN=hf_...                    # write token: pulls the bundle, pushes results
#   bash bootstrap.sh                         # ~5 min on a 1 Gbps host
#   tmux new -s chimera                       # so an SSH drop does not kill the matrix
#   python runner.py --matrix matrix.json --data-root /workspace/data --sink chimera-results \
#       --git-commit "$CHIMERA_GIT_COMMIT"
#
# Code is expected in /workspace/code (rsync it from the laptop: `just vast-sync <ssh-target>`).
# Everything else comes from HF, so the laptop's uplink is never in the path.
set -euo pipefail

BUNDLE_REPO="${BUNDLE_REPO:-agasta/chimera-data}"
WORK="${WORK:-/workspace}"
DATA="$WORK/data"

[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set (needs write access for the results sink)"; exit 1; }

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| bf16', torch.cuda.is_bf16_supported())"

echo "== dependencies"
# The Vast PyTorch images ship torch; only the rest is installed, and never the text encoders' dependencies.
pip install -q -U "diffusers==0.42.0" transformers accelerate bitsandbytes peft safetensors \
    "huggingface_hub[hf_transfer]" pillow numpy sentencepiece protobuf
pip uninstall -q -y torchao 2>/dev/null || true   # ships broken on some images and breaks peft's LoRA injection

echo "== data bundle from $BUNDLE_REPO"
mkdir -p "$DATA"
export HF_HUB_ENABLE_HF_TRANSFER=1
# Resumable: a re-run after a dropped connection continues rather than restarting.
hf download "$BUNDLE_REPO" --repo-type dataset --local-dir "$DATA" \
    --include 'chimera_coco/*' 'embeds/*' 'latents/*'

echo "== what landed"
du -sh "$DATA"/* 2>/dev/null || true
python - <<'PY'
import json, pathlib
data = pathlib.Path("/workspace/data")
m = json.loads((data / "chimera_coco" / "manifest.json").read_text())
images = list((data / "chimera_coco" / "images").glob("*.jpg"))
print(f"manifest: {len(m['clean_train'])} clean_train, pairs {list(m['pairs'])}")
print(f"images:   {len(images)} (expected 5200)")
for model in ("sd3_medium", "sd2_base"):
    idx = data / "embeds" / model / "index.json"
    if idx.exists():
        print(f"embeds/{model}: {len(json.loads(idx.read_text())['captions'])} captions")
    else:
        print(f"embeds/{model}: MISSING")
lat = list((data / "latents").glob("*.safetensors")) if (data / "latents").exists() else []
print("latents:  ", [p.name for p in lat])
PY

mkdir -p "$DATA/checkpoints" "$WORK/results"
echo
echo "ready. Model weights download on first use (transformer + VAE only, ~4.5 GB)."
echo "Run the matrix inside tmux so an SSH drop cannot kill it."
