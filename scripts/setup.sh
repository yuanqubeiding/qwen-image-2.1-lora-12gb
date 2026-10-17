#!/usr/bin/env bash
# Install DiffSynth-Studio + DiffSynth-WebUI for Qwen-Image-2.1 LoRA training.
# Tested on Linux with Python 3.11 / CUDA 13.0. Run from the repository root.
#
#   bash scripts/setup.sh
#
# This script installs packages and downloads ~31 GB of weights.
# It does not touch an existing ComfyUI install.

set -euo pipefail

WORKDIR="${WORKDIR:-$(pwd)/diffsynth}"
MODEL_DIR="${MODEL_DIR:-$(pwd)/models}"
export WORKDIR MODEL_DIR

echo "==> target: $WORKDIR"
echo "==> models: $MODEL_DIR"

# ---------------------------------------------------------------- prerequisites
command -v python >/dev/null || { echo "python not found"; exit 1; }
command -v git    >/dev/null || { echo "git not found";    exit 1; }
command -v nvidia-smi >/dev/null || echo "WARN: nvidia-smi not found - no GPU?"

python -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' \
  || { echo "Python 3.10+ required"; exit 1; }

echo "==> python $(python -V 2>&1 | cut -d' ' -f2)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------------- checkout
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d DiffSynth-Studio ]; then
  # --recurse-submodules is required; a plain clone leaves the webui submodule empty
  git clone --recurse-submodules https://github.com/modelscope/DiffSynth-Studio.git
else
  echo "==> DiffSynth-Studio already present, skipping clone"
fi

if [ ! -d DiffSynth-WebUI ]; then
  git clone --recurse-submodules https://github.com/modelscope/DiffSynth-WebUI.git
else
  echo "==> DiffSynth-WebUI already present, skipping clone"
fi

# ---------------------------------------------------------------------- packages
# Record what torch you already have - the two pitfalls below both involve pip
# replacing it.
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null || true

pip install -e DiffSynth-Studio/

# bitsandbytes provides NF4. --no-deps prevents pip from pulling a different torch.
pip install bitsandbytes --no-deps

# The WebUI frontend. Skip this if you only want the command-line path.
if [ -f DiffSynth-WebUI/package.json ]; then
  echo "==> WebUI present; install its requirements with:"
  echo "    pip install -e DiffSynth-WebUI/"
fi

# ------------------------------------------------------------------------ models
mkdir -p "$MODEL_DIR"
echo "==> downloading models to $MODEL_DIR (about 31 GB)"
python - <<'PY' || echo "WARN: automatic download failed - use modelscope download manually"
import os
try:
    from modelscope import snapshot_download
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
    from modelscope import snapshot_download

target = os.path.join(os.environ.get("MODEL_DIR", "./models"), "Qwen", "Qwen-Image-2.1")
os.makedirs(target, exist_ok=True)
snapshot_download("Qwen/Qwen-Image-2.1", local_dir=target)
print("downloaded to", target)
PY

# --------------------------------------------------------------- verification
echo
echo "==> verification"
python - <<'PY'
import torch
print("torch      ", torch.__version__)
print("cuda build ", torch.version.cuda)
print("device     ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
if torch.cuda.is_available():
    print("vram GB    ", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
try:
    import bitsandbytes as bnb
    print("bitsandbytes", bnb.__version__)
except Exception as e:
    print("bitsandbytes FAILED:", e)
PY

echo
echo "Done. Next:"
echo "  export DIFFSYNTH_MODEL_BASE_PATH=$MODEL_DIR"
echo "  export MODEL_DIR=$MODEL_DIR"
echo "  bash scripts/train.sh <dataset-name>"
