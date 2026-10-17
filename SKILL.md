---
name: qwen-image-2.1-lora
description: Train a Qwen-Image-2.1 LoRA on a 12-24 GB consumer GPU with 32 GB+ RAM. Use when the user mentions "Qwen-Image-2.1 LoRA", "Qwen LoRA 训练", "low VRAM LoRA training", "OOM 训练", "跑不动 LoRA", "DiffSynth Qwen training", or asks how to fine-tune a large diffusion transformer with limited VRAM. Core idea: quantize to NF4 and keep weights resident in VRAM — do NOT use CPU offload unless the machine has plenty of system RAM, because offload trades a VRAM problem for a RAM problem (a pinned copy of every frozen parameter, peak ~2x model size). Also covers resolution-triggered training stalls and loading the PEFT checkpoint into ComfyUI.
---

# Qwen-Image-2.1 LoRA

Train a style or character LoRA for a 20B-class DiT on 12 GB of VRAM and 32 GB of RAM.

**Verified twice:** 500 steps in **12 min 05 s** on an RTX 5070 Ti Laptop
(11.94 GB VRAM) with 31.7 GB RAM. Peak 11,691 / 12,227 MB VRAM and
28,700 / 32,485 MB system RAM — both peaks land in the **first 25 seconds**,
during model load, then settle.

The constraint is the *pair*. A low-VRAM machine usually also has low RAM, and that
is the case this recipe is built for.

---

## The one decision that matters

**When VRAM is short, do not offload. Quantize.**

| Symptom | The reflex (wrong here) | Do this instead |
|---|---|---|
| VRAM OOM | `--enable_model_cpu_offload` | quantize to NF4, keep weights in VRAM |
| Host RAM OOM | add more offload | **remove** offload entirely |
| Stalls every few steps | add offload | lower `max_pixels` |
| Checkpoint won't load in ComfyUI | rename the keys | **do not rename** |
| CLI re-downloads 31 GB, GPU idle at 1% | wait it out | export `DIFFSYNTH_MODEL_BASE_PATH` to the existing cache |

Why: CPU offload keeps the weights in host RAM and moves one unit at a time to the
GPU. Those host-resident weights must be pinned, so the manager allocates a pinned
copy of **every frozen (non-trainable) parameter** while the originals are still
alive. Pinned pages are non-pageable, so the page file cannot absorb them, and peak
usage becomes roughly **2× the model size**. A bf16 run needs ~66 GB of physical
RAM. NF4 needs ~19 GB. On a machine with both low VRAM and low RAM, offload
converts a VRAM problem into a worse RAM problem. (The 2× figure is derived from
the allocator; a full offload run was never completed here — see PROVENANCE.md.)

**Caveat:** on a machine with 64 GB+ RAM, CPU offload is a valid strategy (slower,
but it works). Do not tell a well-resourced user to avoid it — the reason it is
wrong here is the RAM ceiling, not the technique itself.

---

## Never

- `--enable_model_cpu_offload` on a machine with < 40 GB RAM
- NF4 **and** offload together — bitsandbytes' 4-bit matmul and the offload hooks
  disagree about the device stream (`RuntimeError: invalid argument to getCurrentStream`)
- Pasted ComfyUI source into a repo (GPL-3.0). Reference `file:line` instead.
- Raising `max_pixels` before checking that VRAM headroom actually exists

## Always

- Pin the versions: torch, bitsandbytes, DiffSynth commit (see `references/setup.md`)
- Install bitsandbytes with `--no-deps` so pip cannot replace torch
- Verify with `torch.cuda.is_available()` before training
- Check `patches attached` in the ComfyUI log — 192 means complete, 0 means the
  LoRA never applied

---

## Minimal working configuration

| Setting | Value |
|---|---|
| quant | transformer + text_encoder → `bitsandbytes_nf4`; VAE stays bf16 |
| `enable_model_cpu_offload` | **false** |
| `lora_base_model` | `dit` |
| `lora_rank` | 32 |
| `lora_target_modules` | `""` |
| `max_pixels` | **262144** (~512²) |
| `dataset_repeat` × `num_epochs` | 10 × 5 = 500 steps |
| `learning_rate` | 1e-4, `torch.optim.AdamW` |
| `use_gradient_checkpointing` | true |
| `remove_prefix_in_ckpt` | `pipe.dit.` |

Result on the verified machine: 1.31–1.38 s/step, VRAM peak 11,691 / 12,227 MB
(95.6%), RAM peak 28,700 / 32,485 MB (88.3%), no stalls.

The RAM peak occurs at t≈25 s while weights load, not during training (steady state
is 19.7 GB). Do not judge the RAM requirement from the steady-state figure.

If it stalls at ~5 minutes per step, `max_pixels` is too high — not the VRAM
percentage. At 1024² the run stalls every few steps at the same 98% VRAM that runs
smoothly at 512².

---

## Command

```bash
accelerate launch examples/qwen_image_21/model_training/train.py \
  --dataset_base_path ./data/datasets/<name> \
  --dataset_metadata_path ./data/datasets/<name>/metadata.jsonl \
  --dataset_repeat 10 \
  --data_file_keys "image,video" \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors" \
  --quant_options "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors:bitsandbytes_nf4;Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors:bitsandbytes_nf4" \
  --lora_base_model dit \
  --lora_rank 32 \
  --lora_target_modules "" \
  --max_pixels 262144 \
  --num_epochs 5 \
  --learning_rate 1e-4 \
  --customized_optimizer torch.optim.AdamW \
  --output_path ./outputs/<run_name> \
  --remove_prefix_in_ckpt "pipe.dit." \
  --enable_csv_log \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 0 \
  --find_unused_parameters
```

Note the absence of `--enable_model_cpu_offload`. That is deliberate.

---

## Using the result in ComfyUI

Copy the `.safetensors` into `models/loras/` **without renaming it**.

| Knob | Value |
|---|---|
| `strength_model` | 1.5 stable / 2.0 strongest (shifts colours) |
| prompt | keep the style-descriptor prefix used in training |
| `resolution` | 1024 — 512 makes it worse |
| reference images | work, but dilute the style |

Verify the load in the ComfyUI log:

```
Model QwenImage21 prepared ... 192 patches attached   <- LoRA applied
Model QwenImage21 prepared ...   0 patches attached   <- control branch
```

192 = 32 blocks × 6 modules. See `references/comfyui.md` for why 7 targets collapse
to 6.

---

## Where to go next

| Situation | Read |
|---|---|
| Fresh machine, nothing installed | `references/setup.md` |
| User clicks through the WebUI | `references/manual-webui.md` |
| Headless / cloud GPU | `references/manual-cli.md` |
| Something failed or stalled | `references/pitfalls.md` |
| Loading and tuning in ComfyUI | `references/comfyui.md` |
| Building or captioning a dataset | `references/dataset.md` |

Full measurements, loss curves and reasoning: `README.md`.
