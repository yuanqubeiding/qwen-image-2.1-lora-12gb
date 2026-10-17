# Setup: from a clean machine to a working trainer

Target reader: a machine with nothing installed except a shell.

---

## 1. Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Python | **3.11** (3.10.1 minimum) | DiffSynth's `requires-python` |
| Node.js | **≥ 20** | the WebUI frontend is a Next.js app |
| Git | any | submodules |
| NVIDIA driver | CUDA 13.0 capable | torch is built as `+cu130` |

The CUDA Toolkit is **not** required — torch ships its own runtime. Only the
driver matters.

Check the driver:

```bash
nvidia-smi
```

Look at the "CUDA Version" field in the header. It must be ≥ 13.0. If it is lower,
`torch.cuda.is_available()` will return `False` even though the install succeeds.

---

## 2. Clone

```bash
git clone --recurse-submodules https://github.com/modelscope/DiffSynth-WebUI.git
cd DiffSynth-WebUI
```

**`--recurse-submodules` is not optional.** `DiffSynth-Studio/` inside the WebUI
repo is a git submodule. Clone without the flag and it is an empty directory;
the launcher will then fail looking for `DiffSynth-Studio`.

If you already cloned without it:

```bash
git submodule update --init --recursive
```

### Alternative: separate checkouts

If you prefer to keep the two repos apart:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
git clone https://github.com/modelscope/DiffSynth-WebUI.git
```

Then point the launcher at the Studio checkout:

```bash
export DIFFSYNTH_STUDIO_ROOT=/path/to/DiffSynth-Studio
```

---

## 3. Python dependencies

```bash
pip install -e DiffSynth-Studio/
pip install -e .
```

### bitsandbytes is separate

It is **not** a default dependency. It lives in the `quant` optional group together
with `comfy-kitchen`, `entropack[cuda13]` and `torchao`. Do not install the whole
group — only bitsandbytes is needed:

```bash
pip install bitsandbytes --no-deps
```

**`--no-deps` is mandatory.** Without it pip may resolve a different torch build
and silently replace the CUDA 13 one. The verified version is **0.50.2**.

If you want to pin it explicitly:

```bash
pip install "bitsandbytes==0.50.2" --no-deps
```

---

## 4. Model weights

| | |
|---|---|
| ModelScope id | `Qwen/Qwen-Image-2.1` |
| Size on disk | **31 GB** |
| Components | `transformer/`, `text_encoder/`, `vae/`, `processor/`, `scheduler/` |

DiffSynth downloads it automatically on the first training run, from
`modelscope.cn` by default.

To fetch it ahead of time:

```bash
modelscope download --model Qwen/Qwen-Image-2.1 --local_dir ./models/Qwen/Qwen-Image-2.1
```

Then tell DiffSynth where the model root is:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=./models
```

The loader expects the layout `$DIFFSYNTH_MODEL_BASE_PATH/Qwen/Qwen-Image-2.1/...`.

### Set it in every shell

This variable is read from the environment only, never from a config file. The
WebUI sets it for you; a CLI you launch yourself does not inherit it, and falls
back to a working-directory-relative default. The failure is silent and expensive:
it re-downloads all 31 GB, which takes far longer than the training itself.

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/models        # contains Qwen/Qwen-Image-2.1/
du -sh "$DIFFSYNTH_MODEL_BASE_PATH/Qwen/Qwen-Image-2.1" # expect ~31G
```

Tell them apart by the log line: `something.safetensors: NN%` is a download;
`NN/100 [mm:ss<mm:ss, 1.3s/it]` is training. If the GPU is still under 2% after a
few minutes, kill it rather than waiting — the partial download is not reused.

---

## 5. Disk

Reserve **50 GB**:

| Item | Size |
|---|---|
| Model weights | 31 GB |
| Python dependencies | ~10 GB |
| Checkpoints (5 × 167 MB) | ~0.9 GB |
| Logs and samples | negligible |

---

## 6. Launch

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\training_ui\launch_windows.ps1
```

### Linux

```bash
bash training_ui/launch.sh
```

Open `http://127.0.0.1:8100/dashboard`.

On first launch the script installs frontend dependencies if `node_modules` is
missing, then builds the Next.js app. This takes several minutes.

Training runs in a **separate process**. Closing the browser or the launcher does
not stop a run.

---

## 7. Verify before training

Run all three. A failure here is cheaper than a failure 20 minutes into training.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.10.0+cu130 True

python -c "import bitsandbytes; print(bitsandbytes.__version__)"
# expect: 0.50.2

node -v
# expect: v20.x or newer
```

If `cuda.is_available()` is `False`:

1. `nvidia-smi` — does the driver see the GPU?
2. Does the driver's CUDA version reach 13.0?
3. `pip show torch` — is it the `+cu130` build, or a CPU wheel?

A CPU wheel is the most common cause, and it usually comes from a `pip install`
that ran without `--no-deps` and pulled `torch` again.

---

## 8. Pinned versions

```
DiffSynth-Studio   4e3496991455b300fd60ab023113422cd16899f5  (2026-10-16)
DiffSynth-WebUI    95edc3c707318b21c357f60e67f1d7de112febe1  (2026-09-20)
torch              2.10.0+cu130
bitsandbytes       0.50.2
Python             3.11.0
CUDA runtime       13.0
```

Without these six lines a report of "it works" is not reproducible. Include them
in any bug report.

---

## 9. Verified hardware

| | |
|---|---|
| GPU | RTX 5070 Ti Laptop, 11.94 GB VRAM, sm_120 (Blackwell) |
| RAM | 31.7 GB |
| Page file | 96.8 GB commit limit |

This is the smallest configuration confirmed to complete a run. Smaller cards are
untested — do not assume failure, but do not assume success either.
