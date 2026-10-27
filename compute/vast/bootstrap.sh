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

# Which interpreter to use. The Vast PyTorch images look like they already have what we need, and do not:
# their venv is Python 3.10 with torch 2.5.1+cu121, while diffusers 0.42 needs Python >= 3.11 AND torch >= 2.6
# (it calls torch.accelerator, added in 2.6). Those two bounds rule out the shipped environment on every Vast
# PyTorch image, so an interpreter is only usable if it clears both; otherwise build one with uv. Measured on an
# RTX A6000 box, 2026-10-27: shipped venv rejected, uv venv + cu128 gives torch 2.11.0 and diffusers imports.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"   # cu128 needs driver >= 570
VENV="$WORK/venv312"

usable() {   # python >= 3.11 with torch >= 2.6, the floor diffusers 0.42 imposes
    [ -x "$1" ] || return 1
    "$1" - <<'PY' 2>/dev/null
import sys
assert sys.version_info >= (3, 11), sys.version
import torch
assert tuple(int(p) for p in torch.__version__.split(".")[:2]) >= (2, 6), torch.__version__
PY
}

# uv lives in ~/.local/bin, which is not on PATH for a non-login shell. Export it before anything looks for
# uv, not only inside the build branch below: on a re-run the venv already exists, so that branch is skipped
# and a later `command -v uv` would miss it and fall back to a pip that a uv venv does not have.
export PATH="$HOME/.local/bin:$PATH"

PY=""
for candidate in "$VENV/bin/python" "${VIRTUAL_ENV:-}/bin/python" /venv/main/bin/python \
                 /opt/conda/bin/python "$(command -v python3 || true)"; do
    if [ -n "$candidate" ] && usable "$candidate"; then PY="$candidate"; break; fi
done

if [ -z "$PY" ]; then
    echo "== no interpreter with python>=3.11 + torch>=2.6; building one at $VENV"
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    # --seed puts pip in the venv, so anything later that shells out to `python -m pip` still works.
    uv venv --seed --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" torch torchvision --index-url "$TORCH_INDEX"
    PY="$VENV/bin/python"
    usable "$PY" || { echo "built venv still does not satisfy the floor"; exit 1; }
fi
export PATH="$(dirname "$PY"):$PATH"
echo "python: $PY"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| bf16', torch.cuda.is_bf16_supported())"

echo "== dependencies"
# torch comes from the image or the venv above; this is everything else, and never the text encoders' deps.
if command -v uv >/dev/null 2>&1; then
    PIP="uv pip install --python $PY"
elif "$PY" -m pip --version >/dev/null 2>&1; then
    PIP="$PY -m pip install -q -U"
else
    echo "neither uv nor pip is usable for $PY"; exit 1
fi
$PIP "diffusers==0.42.0" transformers accelerate bitsandbytes peft safetensors \
    "huggingface_hub==2.3.0" pillow numpy sentencepiece protobuf
"$PY" -m pip uninstall -q -y torchao 2>/dev/null || true   # ships broken on some images and breaks peft's LoRA injection
"$PY" -c "import diffusers; from diffusers import SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler, \
UNet2DConditionModel, AutoencoderKL; print('diffusers', diffusers.__version__, '| model imports ok')"

echo "== data bundle from $BUNDLE_REPO"
mkdir -p "$DATA"
# Transfers go through hf-xet (Rust, chunk-deduplicated), which huggingface_hub depends on outright since v1.0
# and uses by default; the old HF_HUB_ENABLE_HF_TRANSFER is a silent no-op. hub is pinned so a replacement box
# rented next week gets the same stack as the one that produced the earlier cells. High-performance mode
# saturates the network but
# buffers heavily, and the docs warn it can be slower below 64 GB RAM, so it is enabled only on big boxes.
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "${RAM_GB:-0}" -ge 64 ]; then
  export HF_XET_HIGH_PERFORMANCE=1
  echo "hf-xet high performance mode on (${RAM_GB} GB RAM)"
else
  echo "hf-xet default mode (${RAM_GB} GB RAM; high performance wants >= 64 GB)"
fi
# Resumable: a re-run after a dropped connection continues rather than restarting.
# `hf` is the v1.0 CLI; older images still ship huggingface-cli. Look beside $PY first, since neither is on
# PATH for a non-login shell on the Vast images.
HF_CLI=$(command -v "$(dirname "$PY")/hf" || command -v hf || command -v huggingface-cli)
# No --include: the repo holds exactly the five parts we want and nothing else, so filtering only adds a way to
# get it wrong. It did: `--include A B C` takes one TEXT value, so B and C became positional FILENAMES, the CLI
# warned "Ignoring --include since filenames have been explicitly set", and chimera_coco/ -- every training
# image -- was silently skipped while the other four parts downloaded and looked like success.
"$HF_CLI" download "$BUNDLE_REPO" --repo-type dataset --local-dir "$DATA"

echo "== what landed"
du -sh "$DATA"/* 2>/dev/null || true
"$PY" - <<'PY'
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
poisons = data / "poisons"
if poisons.is_dir():
    for attack in sorted(p for p in poisons.iterdir() if p.is_dir()):
        pairs = {q.name: len(list(q.glob("*.png"))) for q in sorted(attack.iterdir()) if q.is_dir()}
        print(f"poisons/{attack.name}: {pairs}")
else:
    print("poisons:   none (only the clean and dirty_label cells can run)")
PY

mkdir -p "$DATA/checkpoints" "$WORK/results"
echo
echo "ready. Model weights download on first use (transformer + VAE only, ~4.5 GB)."
echo "Run the matrix inside tmux so an SSH drop cannot kill it."
