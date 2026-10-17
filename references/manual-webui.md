# Manual path A: through the WebUI

For someone clicking through the browser interface, no command line beyond
launching the app.

---

## 1. Launch

```powershell
powershell -ExecutionPolicy Bypass -File .\training_ui\launch_windows.ps1
```

Open `http://127.0.0.1:8100/dashboard`.

First launch builds the frontend and can take several minutes. Wait for the page
rather than assuming it hung.

---

## 2. Settings

Go to **Settings** and confirm:

| Setting | What it controls |
|---|---|
| Dataset Directory | where `metadata.jsonl` and images live |
| Training Output Directory | where checkpoints, logs and loss curves land |

Leave them at their defaults unless you have a reason.

---

## 3. Create the dataset

1. **Training Datasets → Create Dataset**
2. Choose **Image**
3. Import your images
4. Add a caption for each one

Two ways to fill captions:

- **By hand** — one line per image. See `dataset.md` for what to write.
- **Auto-Caption** — configure an OpenAI-compatible multimodal model in Settings,
  then generate in bulk. Review the output; auto-captions tend to over-describe
  style, which is exactly what you want to avoid.

The dataset is saved as a directory plus a `metadata.jsonl` next to it.

---

## 4. Create the training task

**New Task**, then:

1. Pick model **Qwen-Image-2.1**
2. Pick the GPU
3. Pick the dataset you just created

Then set the fields below. **The quantisation and offload rows are the ones that
decide whether the run survives.**

| Field | Value | Why |
|---|---|---|
| Quantise transformer | **NF4** | ~3.6 GB instead of 20+ GB |
| Quantise text_encoder | **NF4** | ~4.4 GB |
| Quantise VAE | **leave off** | small already; bf16 is fine |
| **CPU offload** | **all off** | the whole point — see `pitfalls.md` |
| LoRA base model | `dit` | |
| LoRA rank | 32 | |
| LoRA target modules | empty (default) | |
| Resolution mode | `max_pixels` | |
| `max_pixels` | **262144** (~512²) | |
| `dataset_repeat` | 10 | |
| epochs | 5 | 10 × 5 = 500 steps |
| Learning rate | 1e-4 | |
| Optimiser | `torch.optim.AdamW` | |
| Gradient checkpointing | **on** | |
| Gradient accumulation | 1 | |
| **Post-training sampling** | **clear it / leave empty** | see below |

### Why post-training sampling must be off

Sampling loads the **full bf16 model** for inference, on top of the quantised
training model. On a 12 GB card that allocation fails. Training finishes
successfully, then the process dies during the sample step and the run is marked
failed.

Clear the sample list. You can sample later from ComfyUI instead.

---

## 5. Run and watch

Click **Create Task**, then open the task detail page.

What good looks like at `max_pixels = 262144`:

| Signal | Expected |
|---|---|
| Step speed | 1.31–1.38 s/step |
| VRAM | 11.6 GB of 12.2 GB (~95%) |
| Stalls | none |

What a problem looks like:

| Signal | Meaning |
|---|---|
| ~5 minutes on one step, then fast again | `max_pixels` too high — see `pitfalls.md` §3 |
| OOM during load | quantisation not applied |
| OOM right after the last step | post-training sampling is on |
| 20+ s/step from the start | weights are being offloaded |

---

## 6. Collect the result

The task page lists the artifacts. Download the `*.safetensors` files — there is
one per epoch.

Drop them straight into `ComfyUI/models/loras/`. **Do not rename them.** See
`comfyui.md` for verification and starting strength.

The `loss.csv` from the run is worth keeping: it tells you when to stop training
next time.
