# Chimera task runner. `just` to list recipes.

paper_dir := "paper/IEEE"
results_dir := "results"

default:
    @just --list

# ---------- paper ----------

# Build the TIFS manuscript (pdflatex + bibtex; latexmk not installed)
paper:
    cd {{paper_dir}} && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null \
      && bibtex main >/dev/null \
      && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null \
      && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null \
      || (grep -A3 "^!" main.log; exit 1)
    @cd {{paper_dir}} && (grep -E "Warning.*(undefined|Citation)" main.log | sort -u || true)
    @just pages

# Page count (TIFS: <=13 initial submission, <=16 revision, incl. references)
pages:
    @pdfinfo {{paper_dir}}/main.pdf | grep Pages

# List open \todo / \cn markers
todos:
    @grep -n '\\todo\|\\cn' {{paper_dir}}/main.tex {{paper_dir}}/sections/*.tex || echo "no TODOs"

paper-clean:
    cd {{paper_dir}} && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz sections/*.aux

# ---------- reading ----------

# Convert a paper PDF to Markdown (+ figures) under notes/md/<name>.md + <name>_artifacts/.
# OCR off by default (born-digital PDFs); pass ocr="--ocr" for scanned ones. ~1 min/20 pages on CPU.
# Pinned so uvx reuses its cached env + models (no cold start). Needs ~3 GB free RAM (earlyoom kills it otherwise).
pdf2md pdf ocr="--no-ocr":
    cd notes/md && uvx --python 3.12 --from docling==2.142.1 docling "$(realpath '{{invocation_directory()}}/{{pdf}}')" --to md {{ocr}} --image-export-mode referenced --output .

# ---------- figures ----------

# Regenerate all paper figures from results/ (scripts write to paper/IEEE/figures/)
figs:
    for f in figures/scripts/plot_*.py; do uv run python "$f" || exit 1; done

# ---------- Kaggle (free T4x2, ~30 GPU-h/week) ----------

# Upload experiments/common/ as the private Kaggle dataset itzagasta/chimera-code (run after every change to it)
kaggle-code:
    #!/usr/bin/env bash
    set -euo pipefail
    d=$(mktemp -d)
    cp experiments/common/*.py "$d"/
    printf '{"title":"chimera-code","id":"itzagasta/chimera-code","licenses":[{"name":"unknown"}]}' > "$d/dataset-metadata.json"
    kaggle datasets version -p "$d" -m "sync $(git rev-parse --short HEAD)" -q
    until kaggle datasets status itzagasta/chimera-code | grep -q ready; do sleep 5; done
    echo "chimera-code synced"

# Push + run a kernel dir containing kernel-metadata.json
kaggle-push dir accel="NvidiaTeslaT4":
    kaggle kernels push -p {{dir}} --accelerator {{accel}}

kaggle-status slug:
    kaggle kernels status {{slug}}

# Download a finished kernel's outputs into results/<exp>/<run>/
kaggle-pull slug out:
    mkdir -p {{out}} && kaggle kernels output {{slug}} -p {{out}}

# ---------- Vast.ai (paid: confirm with the user before renting) ----------

# Copy the code the box needs (tiny: ~70 KB) to a rented instance. Data comes from HF, not from here.
vast-sync target:
    rsync -az --delete experiments/common/ {{target}}:/workspace/code/
    rsync -az experiments/configs/ {{target}}:/workspace/code/
    rsync -az compute/vast/bootstrap.sh {{target}}:/workspace/
    @echo "synced; on the box: export HF_TOKEN=... && bash /workspace/bootstrap.sh"

# What the matrix would run, without running anything
vast-plan tier="":
    uv run python experiments/common/runner.py --matrix experiments/configs/matrix.json \
        --data-root data --sink "" {{ if tier != "" { "--tier " + tier } else { "" } }} --plan-only


vast-cost:
    vastai show user --raw | python3 -c "import sys,json;d=json.load(sys.stdin);print('credit:',d.get('credit'),'balance:',d.get('balance'))"
    vastai show instances

# Cheapest offers with >= vram GB (search only, never rents)
vast-search vram="48":
    vastai search offers 'gpu_ram>={{vram}} num_gpus=1 reliability>0.98 inet_down>200 disk_space>=80' -o 'dph_total' | head -15
