# Pitfalls

Four failures that cost real time. Each one has the mechanism, not just the fix.

---

## 1. CPU offload doubles the model in locked pages

### Symptom

Training dies during setup, before step 1:

```
File "diffsynth/core/offload_training/offloader.py", line 26, in __init__
    self.cpu_copy = memory_buffer.allocate_like(cpu_data) if memory_buffer is not None else cpu_data.pin_memory()
File "diffsynth/core/offload_training/memory_buffer.py", line 136, in allocate_like
    return src.pin_memory()
torch.AcceleratorError: CUDA error: out of memory
```

The message says CUDA, which sends you looking at the GPU. The GPU is not the
problem.

### Mechanism

1. With `--enable_model_cpu_offload`, `train.py` sets the device to `"cpu"`, so the
   weights are loaded into host RAM first. Offload means the model stays in RAM and
   only the unit currently executing sits on the GPU — so host RAM has to hold a
   full copy for the whole run.
2. `OffloadTrainingManager` allocates a pinned arena sized to every
   **non-trainable** parameter and buffer
   (`diffsynth/core/offload_training/memory_buffer.py:99`) and copies them into it
   (`offloader.py:26`). Trainable parameters stay on the GPU; in LoRA training the
   DiT is frozen, so that is effectively the whole transformer.
3. Pinned pages are **page-locked**: the OS may not swap them out. A page file is
   therefore irrelevant here — only physical RAM counts.
4. The copy is a **second allocation** while the original tensor is still alive.
   Peak usage is roughly **2× the model size** — derived from the allocator's
   behaviour, not observed to completion (see PROVENANCE.md).

### The arithmetic

| Precision | Weights | Offload needs (~2×) | Physical RAM on the test machine |
|---|---|---|---|
| bf16 | 33 GB | ~66 GB | 31.7 GB |
| FP8 | 17.2 GB | ~34 GB | 31.7 GB — short by 2–3 GB |
| **NF4** | **~9.4 GB** | ~19 GB | **fits** |

A 96.8 GB page file and 79.7 GB of free commit did nothing. Pinning is
all-or-nothing — there is no graceful degradation when it does not fit.

**Measured directly on the working (NF4, no offload) run:** the page file went from
1,178.9 MB to 1,188.5 MB across the whole 12 minutes — a 9.6 MB drift, i.e. no
paging at all. That is the empirical form of this argument: the run does not spill,
so a page file buys nothing regardless of its size.

### Fix

Turn offload off entirely and quantise instead. The weights then stay in VRAM and
the host RAM question disappears.

---

## 2. NF4 and offload are mutually exclusive

### Symptom

With both enabled, loading succeeds — then the first forward pass dies:

```
bitsandbytes/autograd/_functions.py:472 matmul_4bit
  -> bitsandbytes/backends/cuda/ops.py:967 _dequant_linear_fallback
  -> ops.py:380 _get_raw_stream(A.device.index)
RuntimeError: invalid argument to getCurrentStream
```

### Mechanism

bitsandbytes' 4-bit matmul and DiffSynth's offload hooks disagree about which
device stream the weights currently live on. `bnb.nn.Linear4bit` works correctly
when tested alone on sm_120, so this is not a hardware capability problem — it is
the offload path.

### Fix

Remove **both**. Turning off only one does not help:

- NF4 without offload → works (this recipe)
- offload without NF4 → host OOM on a 32 GB machine (pitfall 1)

---

## 3. The stall is resolution, not VRAM percentage

### Symptom

At `max_pixels = 1048576` (~1024²), training runs fast for a few steps, then hangs
for about five minutes, then resumes:

| step | cumulative | this step |
|---|---|---|
| 1 | 00:03 | 3.16 s |
| 2 | 04:49 | **4 min 46 s** |
| 3–6 | 04:50 → 04:54 | 1–2 s each |
| 7 | 09:36 | **4 min 42 s** |
| 8 | 09:38 | 2 s |
| 9 | — | **> 6 min, never completed** |

Extrapolated average: ~72 s/step → 250 steps/epoch ≈ 5 hours.

### The trap

VRAM sits at 97–98% during the stall. The obvious conclusion is "VRAM is full, so
let me offload" — which is exactly the move that kills the run in pitfalls 1 and 2.

### What actually happens

At `max_pixels = 262144` (~512²), VRAM **also** sits at 95–98% — and the run
proceeds at 1.31–1.38 s/step with zero stalls for all 500 steps (measured twice,
10-17 and 10-18).

Same VRAM percentage, completely different behaviour. The variable is the
resolution, not the occupancy.

### Fix

Lower `max_pixels`. Do not add offload.

Decision order when training is slow or unstable:

1. quantise the big modules
2. lower `max_pixels`
3. only then consider offload or a bigger card

---

## 4. Do not rename the checkpoint

### Symptom

The output keys look unusual:

```
transformer_blocks.0.attn.to_q.lora_A.default.weight   [32, 4096]
transformer_blocks.0.attn.to_q.lora_B.default.weight   [4096, 32]
```

`.default.` in the middle looks like a PEFT artifact that ComfyUI might not
understand. The instinct is to rename it away.

### Why renaming is wrong

ComfyUI has a dedicated branch for exactly this naming
(`comfy/weight_adapter/lora.py:166`). Renaming risks producing keys the loader does
not recognise, while the original form loads cleanly.

Additional behaviours that make the raw file work:

- `img_mlp.gate_layer` and `img_mlp.proj` are fused into a single `gate_up` weight,
  and both LoRA halves are sliced into it (`comfy/lora.py:328-334`)
- a missing `alpha` key means `scale = 1.0` (`comfy/weight_adapter/lora.py:249-251`) —
  which is the same scale used during training

### Fix

Copy the `.safetensors` into `ComfyUI/models/loras/` unchanged, then verify the
load in the log. See `comfyui.md`.

---

## Summary table

| Pitfall | Wrong reflex | Correct action |
|---|---|---|
| Setup OOM | more offload | remove offload, quantise |
| `getCurrentStream` | add memory | remove offload **and** keep NF4 |
| Periodic 5-min stalls | offload | lower `max_pixels` |
| Odd key names | rename | leave them |
