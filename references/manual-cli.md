# Manual path B: command line only

For cloud GPUs, headless servers, or anyone who would rather not build a Next.js
frontend. No Node.js required.

---

## 1. Install

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
pip install bitsandbytes --no-deps
```

`--no-deps` on bitsandbytes prevents pip from replacing torch.

Model weights download on first run. To prefetch:

```bash
modelscope download --model Qwen/Qwen-Image-2.1 --local_dir ./models/Qwen/Qwen-Image-2.1
export DIFFSYNTH_MODEL_BASE_PATH=./models
```

---

## 2. Dataset layout

```
data/datasets/<name>/
├── 0001.jpg
├── 0002.jpg
└── metadata.jsonl
```

`metadata.jsonl` — one JSON object per line:

```json
{"image": "0001.jpg", "prompt": "<style prefix>: a person standing by a window, front view."}
{"image": "0002.jpg", "prompt": "<style prefix>: a person walking outdoors, three-quarter view."}
```

The `image` value is a filename relative to `dataset_base_path`. See `dataset.md`
for caption strategy.

---

## 3. Run

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

The same arguments are in `../assets/training_args.json`.

---

## 4. What each argument does

| Argument | Role |
|---|---|
| `--quant_options` | `model_id:pattern:bitsandbytes_nf4`. VAE is absent → stays bf16 |
| `--lora_base_model dit` | attach the LoRA to the diffusion transformer, not the text encoder |
| `--lora_rank 32` | adapter width. 32 is a reasonable default for style |
| `--lora_target_modules ""` | empty means DiffSynth's default target set |
| `--max_pixels 262144` | pixel budget per image (~512²). **The stall knob** |
| `--dataset_repeat 10` × `--num_epochs 5` | 1 step = 1 image seen. 10 images → 500 steps |
| `--remove_prefix_in_ckpt "pipe.dit."` | strips the pipeline prefix so keys match plain module names |
| `--find_unused_parameters` | required by the LoRA wiring; without it DDP complains |
| `--enable_csv_log` | writes `loss.csv` next to the checkpoints |

**There is no `--enable_model_cpu_offload` here.** Adding it is what breaks this
recipe on a low-RAM machine.

---

## 5. Output

```
outputs/<run_name>/
├── epoch-0.safetensors   # 167 MB each, 448 tensors, bf16
├── ...
├── epoch-4.safetensors
├── loss.csv
├── train.log
├── training_args.json
└── training_config.json
```

`training_args.json` in the output directory is the ground truth for what actually
ran — keep it. See `comfyui.md` for loading the result.

---

## 6. Stopping a run

`Ctrl-C` on the process, or the platform's stop button.

Note: the WebUI's own stop endpoint is not reliable — on the verified version
`POST /api/tasks/{id}/stop` returns HTTP 500 and does nothing. Killing the
`train.py` process tree works. This is only relevant if you are using the WebUI.

---

## 7. Cloud notes

- Weights are **31 GB**. Prefetch them into the data disk, not the system disk.
- Check the driver's CUDA version before renting — a card that reports CUDA 12.x
  will not run a `+cu130` torch build.
- Price the job by steps, not by wall clock: at ~1.3 s/step, 500 steps is about
  11 minutes of pure compute; add model download and environment setup on top.
