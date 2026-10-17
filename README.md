# Qwen-Image-2.1 LoRA on a 12 GB GPU and 32 GB of RAM

Train a style LoRA for a 20B-class diffusion model on a single consumer laptop.
**No CPU offload. No multi-GPU. 500 steps in 12 min 05 s.**

This recipe targets a machine that is short in **both** directions: 11.94 GB of
VRAM and 31.7 GB of system RAM. That pairing is the point. It is exactly the
configuration where `--enable_model_cpu_offload` looks like the answer and is in
fact the failure — because offload trades a VRAM problem for a *RAM* problem, and
here there is no RAM to trade with. Quantise the weights until they fit in VRAM
and keep them there.

---

## The idea in one table

| | Conventional advice | This recipe |
|---|---|---|
| Not enough VRAM | move weights to CPU (`--enable_model_cpu_offload`) | shrink weights until they fit (NF4) |
| Assumes | plenty of host RAM | **nothing** — a low-VRAM machine usually has low RAM too |
| What actually happens | `pin_memory()` copies the model a second time → host OOM | no second copy is made |
| VRAM | freed up | **11.7 GB of 12.2 GB (95.6%)** |
| Speed | data shuttled from CPU every step | 1.31–1.38 s/step |

**Decision order when resources are tight:**

1. Quantize the big modules to NF4
2. Lower `max_pixels`
3. Only then consider CPU offload — and only if you actually have host RAM to
   spare — or rent a bigger card

> Running with ~0.5 GB of VRAM headroom is the expected outcome, not a warning sign.
> Nothing else may hold the GPU while training runs.

---

## Who this is for

| Your machine | What this means |
|---|---|
| 12–24 GB VRAM, **32 GB+ RAM** | the tested configuration; follow it as written |
| 12–24 GB VRAM, **64 GB+ RAM** | works, and CPU offload becomes a legitimate option too. This recipe is still faster — it avoids the per-step CPU→GPU traffic |
| 12 GB VRAM, **< 32 GB RAM** | **not tested.** The hardware floor for this recipe is the 31.7 GB machine it was measured on. Below that, offload fails for the same reason it fails here, and there is no third option short of a rented card |
| 24 GB+ VRAM, plenty of RAM | you can train this unquantised (bf16, ~33 GB) without any of the above |

---

## Verified environment

| | |
|---|---|
| GPU | 12 GB laptop GPU — 11.94 GB usable, sm_120 (Blackwell) |
| System RAM | 31.7 GB |
| Python | 3.11.0 (requires ≥ 3.10.1) |
| torch | 2.10.0+cu130 |
| bitsandbytes | 0.50.2 |
| CUDA runtime | 13.0 |
| DiffSynth-Studio | `4e3496991455b300fd60ab023113422cd16899f5` (2026-10-16) |
| DiffSynth-WebUI | `95edc3c707318b21c357f60e67f1d7de112febe1` (2026-09-20) |

Pinning these is not optional. Without them the run is not reproducible.

---

## Install

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | 3.10.1 minimum |
| Node.js | ≥ 20 | the WebUI frontend is built with it |
| Git | any | |
| NVIDIA driver | CUDA 13.0 capable | otherwise `torch.cuda.is_available()` returns `False` |

The CUDA Toolkit itself is **not** required — torch ships its own runtime.

### Clone and install

```bash
git clone --recurse-submodules https://github.com/modelscope/DiffSynth-WebUI.git
cd DiffSynth-WebUI
pip install -e DiffSynth-Studio/
pip install -e .
```

Two things that will bite you:

1. **`--recurse-submodules` is required.** `DiffSynth-Studio/` inside the repo is a
   submodule. Clone without the flag and it is an empty directory.
2. **`bitsandbytes` is not a default dependency.** It lives in the `quant` optional
   group alongside `comfy-kitchen`, `entropack[cuda13]` and `torchao`. Install only
   what you need:

```bash
pip install bitsandbytes --no-deps
```

**`--no-deps` is mandatory.** Without it, pip may pull a different torch build and
silently break the CUDA 13 environment.

### Model weights

| | |
|---|---|
| ModelScope id | `Qwen/Qwen-Image-2.1` |
| Size | **31 GB** |
| Components | `transformer/`, `text_encoder/`, `vae/`, `processor/`, `scheduler/` |

DiffSynth downloads it automatically on the first run (from modelscope.cn by default).
To fetch it ahead of time:

```bash
modelscope download --model Qwen/Qwen-Image-2.1 --local_dir ./models/Qwen/Qwen-Image-2.1
```

Then point the trainer at the directory:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=./models
```

> **Set this in every shell.** It is read from the environment only — not from a
> config file. Launch the CLI without it and DiffSynth falls back to a
> working-directory-relative default, finds no model, and silently re-downloads all
> 31 GB. See `references/setup.md` §4.

### Disk

Reserve **50 GB**: 31 GB of weights, ~10 GB of dependencies, plus 5 × 167 MB
checkpoints and logs.

### Launch (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File .\training_ui\launch_windows.ps1
```

Open `http://127.0.0.1:8100/dashboard`.

Training runs in a separate process — closing the browser does not stop it.

### Verify before training

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.10.0+cu130 True        <- False means the driver is too old

python -c "import bitsandbytes; print(bitsandbytes.__version__)"
# expect: 0.50.2

node -v
# expect: v20 or newer
```

---

## Training configuration

This is the exact configuration that produced the numbers below.

| Setting | Value |
|---|---|
| `quant_options` | transformer + text_encoder → `bitsandbytes_nf4`; VAE stays bf16 |
| `enable_model_cpu_offload` | **off** |
| `lora_base_model` | `dit` |
| `lora_rank` | 32 |
| `lora_target_modules` | `""` (all default targets) |
| `max_pixels` | **262144** (~512²) |
| `dataset_repeat` × `num_epochs` | 10 × 5 = **500 steps** |
| `learning_rate` | 1e-4, AdamW |
| `use_gradient_checkpointing` | on |
| `remove_prefix_in_ckpt` | `pipe.dit.` |
| `gradient_accumulation_steps` | 1 |
| `dataset_num_workers` | 0 |

### Result

| | |
|---|---|
| Wall clock | **12 min 04.7 s** |
| Step speed | 1.31–1.38 s/step, no stalls |
| Output | 5 checkpoints × 167 MB (448 tensors, bf16) |

### Measured peaks

Sampled every 2 s for the whole run (355 samples). `t` is time from process start.

| Metric | Peak | When | Steady state |
|---|---|---|---|
| Training process RSS | **20,444 MB** | **t = 25 s** (model load) | 12,254 MB |
| System RAM used | **28,700 / 32,485 MB (88.3%)** | **t = 25 s** | 19,702 MB (60.6%) |
| Available RAM | **3,785 MB** (lowest) | model load | ~12,700 MB |
| Page file used | **1,188 MB** | whole run | baseline 1,179 MB |
| GPU memory used | **11,691 / 12,227 MB (95.6%)** | t = 43 s | 11,616 MB |

Three things this establishes:

1. **The tight moment is loading, not training.** RAM peaks in the first 25 seconds
   and then settles at 60%. A machine that survives the load is stable afterwards.
2. **The page file never grew.** 1,178.9 → 1,188.5 MB over twelve minutes — a
   +9.6 MB drift. This is direct evidence for the claim in finding 1: paging is not
   available as an escape hatch.
3. **VRAM headroom is ~0.5 GB.** Not comfortable, not slack — the ceiling. Anything
   else holding the GPU (a browser tab running another model, a leftover ComfyUI)
   turns this into an OOM.

### Loss by epoch

| epoch | mean (run A) | mean (run B) | last (run A) | last (run B) |
|---|---|---|---|---|
| 0 | 0.2498 | 0.2680 | 0.1611 | 0.2207 |
| 1 | 0.2656 | **0.2238** | 0.1859 | 0.1704 |
| 2 | **0.2245** | 0.2292 | 0.1759 | 0.1391 |
| 3 | 0.2294 | **0.2235** | 0.3555 | 0.1445 |
| 4 | 0.2292 | 0.2475 | 0.2018 | 0.4173 |

**Do not pick an epoch from this table.** Two runs of the identical command on the
identical dataset disagree about which epoch is lowest — run A says epoch 2, run B
says epoch 3, and epoch 0 differs by 0.018. The epoch-to-epoch spread is smaller
than the run-to-run spread, so the ordering is noise.

What the table does support: loss stops improving after roughly the first epoch.
With a dataset this small, **300 steps is enough**. Which checkpoint to *use* is a
different question — judge that by looking at images, not by the loss value.

---

## Four findings

### 1. CPU offload doubles the model in locked pages

With `--enable_model_cpu_offload`, `train.py` sets the device to `"cpu"` and the
weights land in host RAM first. `OffloadTrainingManager` then creates a
`pin_memory()` copy for **every parameter**
(`diffsynth/core/offload_training/offloader.py:26`).

Pinned pages are **non-pageable**, so a 96.8 GB page file is useless here — only
physical RAM counts. The copy is a second allocation while the original is still
alive, so peak usage is roughly **2× the model size**.

| Precision | Weights | Offload needs (~2×) | Physical RAM available |
|---|---|---|---|
| bf16 | 33 GB | ~66 GB | 31.7 GB |
| FP8 | 17.2 GB | ~34 GB | 31.7 GB — short by 2–3 GB |
| **NF4** | **~9.4 GB** | ~19 GB | **fits** |

Pinning is all-or-nothing: there is no graceful degradation.

**Measured on the working run:** the page file moved from 1,178.9 MB to 1,188.5 MB
across the full twelve minutes — 9.6 MB, i.e. no paging. The NF4 path does not spill,
which is why the page-file size is irrelevant to it and fatal to the offload path.

> **9.4 GB is the weight total, not the VRAM figure.** The measured GPU delta of a
> full NF4 run is **11,148 MB** (11,691 MB peak minus a 543 MB idle desktop). The
> difference is CUDA context, activations and buffers. Budget by the measured
> number — on a 12 GB card the 9.4 GB figure will make you think there is room
> there isn't.

### 2. NF4 and offload are mutually exclusive

With NF4 enabled *and* offload on, loading succeeds but the first forward pass dies:

```
bitsandbytes/autograd/_functions.py:472 matmul_4bit
  -> bitsandbytes/backends/cuda/ops.py:967 _dequant_linear_fallback
  -> ops.py:380 _get_raw_stream(A.device.index)
RuntimeError: invalid argument to getCurrentStream
```

bitsandbytes' 4-bit matmul and DiffSynth's offload hooks disagree about which
device stream the weights are on. `bnb.nn.Linear4bit` works fine on its own —
the incompatibility is in the offload path.

**Turn both off together.** Removing only one does not help.

### 3. The stall is resolution, not VRAM percentage

At `max_pixels = 1048576` (~1024²) the run stalls for roughly five minutes every
few steps:

| step | cumulative | this step |
|---|---|---|
| 1 | 00:03 | 3.16 s |
| 2 | 04:49 | **4 min 46 s** |
| 3–6 | 04:50 → 04:54 | 1–2 s each |
| 7 | 09:36 | **4 min 42 s** |
| 9 | — | **> 6 min, never finished** |

At `max_pixels = 262144` (~512²), VRAM still sits at 95% — and the run proceeds at
1.31–1.38 s/step with zero stalls.

**The same VRAM percentage, a completely different speed.** If training stalls,
lower `max_pixels`; do not add offload.

### 4. The checkpoint loads into ComfyUI unchanged

The output uses standard PEFT naming:

```
transformer_blocks.0.attn.to_q.lora_A.default.weight   [32, 4096]
transformer_blocks.0.attn.to_q.lora_B.default.weight   [4096, 32]
```

No renaming required. ComfyUI handles it in three places:

- `.default.` keys have a dedicated branch (`comfy/weight_adapter/lora.py:166`)
- `img_mlp.gate_layer` and `img_mlp.proj` are fused into a single `gate_up`
  weight, and ComfyUI slices both LoRA halves into it (`comfy/lora.py:328-334`)
- a missing `alpha` key means `scale = 1.0` (`comfy/weight_adapter/lora.py:249-251`)

**How to verify the load is complete:**

```
Model QwenImage21 prepared ... 192 patches attached   <- with LoRA
Model QwenImage21 prepared ...   0 patches attached   <- control
```

192 = 32 transformer blocks × 6 modules. The checkpoint's 448 tensors form 224
`A`/`B` pairs; 7 targets per block collapse to 6 because `proj` and `gate_layer`
share one fused weight. **Nothing was dropped.**

---

## Using the LoRA in ComfyUI

| Knob | Value |
|---|---|
| `strength_model` | **1.5** (stable) / **2.0** (strongest style, shifts colours) |
| Prompt | keep the style-descriptor prefix used in training |
| `resolution` | **1024** — lowering to 512 makes it visibly worse |
| Reference images | work, but dilute the style |

`strength_model` is not monotonic. 1.8–1.9 produced visibly broken output, while
2.0 was self-consistent. At 2.0 the output collapses into a simpler regime where
anatomy errors stop being legible — that is a shift in the definition of "broken",
not an absence of breakage. Colour drift at 2.0 is the signal that the LoRA is
over-driving semantics.

> `resolution 512` is a common recommendation for matching training resolution.
> It comes from the SD1.5 era and **does not hold here**: Qwen-Image-2.1 is
> 1-megapixel native, and the base model's input resolution dominates.

---

## What is not in this repository

- **No training images.** The dataset used for these measurements is not ours to
  redistribute.
- **No trained weights.** The resulting `.safetensors` is derived from that dataset.
- **No generated samples.** Same reason.

Bring your own dataset. See `references/dataset.md` for the `metadata.jsonl`
format and the caption-prefix strategy.

---

## Raw data

Everything quoted above is in `assets/`:

| File | Contents |
|---|---|
| `training_args.json` | the run configuration, paths replaced with placeholders |
| `loss.csv` | 500 loss values, run A |
| `measured_loss.csv` | 500 loss values, run B |
| `measured_resources.csv` | 355 samples at 2 s intervals: process RSS, system RAM used/available, page file used, GPU memory |
| `measured_summary.json` | the peaks derived from that CSV |

The sampler that produced them is [`scripts/measure.py`](scripts/measure.py) — it
spawns any command and polls psutil + nvidia-smi:

```bash
python scripts/measure.py --out ./run -- python train.py --your --args
```

---

## Attribution

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — Apache-2.0
- [DiffSynth-WebUI](https://github.com/modelscope/DiffSynth-WebUI) — Apache-2.0
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — GPL-3.0; referenced by
  file and line only, no source is vendored here
- Qwen-Image-2.1 — [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen-Image-2.1).
  This repository documents *how to train against* the model; it does not
  redistribute the model or any derivative weights.
