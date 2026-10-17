#!/usr/bin/env bash
# Launch a Qwen-Image-2.1 LoRA training run.
#
#   bash scripts/train.sh <dataset-name> [run-name]
#
# Layout expected:
#   data/datasets/<dataset-name>/metadata.jsonl + images
#
# The two things that decide whether this survives on a small card:
#   * --quant_options  NF4 on transformer + text_encoder, VAE left bf16
#   * NO --enable_model_cpu_offload
# See references/pitfalls.md before changing either.

set -euo pipefail

DATASET="${1:-}"
RUN_NAME="${2:-${DATASET}_run}"

if [ -z "$DATASET" ]; then
  echo "usage: bash scripts/train.sh <dataset-name> [run-name]" >&2
  exit 1
fi

DATASET_DIR="./data/datasets/$DATASET"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-./diffsynth/DiffSynth-Studio}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$(pwd)/models}"

[ -f "$DIFFSYNTH_DIR/examples/qwen_image_21/model_training/train.py" ] \
  || { echo "train.py not found under $DIFFSYNTH_DIR - run scripts/setup.sh first" >&2; exit 1; }

[ -f "$DATASET_DIR/metadata.jsonl" ] \
  || { echo "$DATASET_DIR/metadata.jsonl not found" >&2; exit 1; }

N_IMAGES=$(grep -c . "$DATASET_DIR/metadata.jsonl" || true)
REPEAT="${DATASET_REPEAT:-10}"
EPOCHS="${NUM_EPOCHS:-5}"
echo "==> dataset : $DATASET ($N_IMAGES images)"
echo "==> steps   : $((N_IMAGES * REPEAT * EPOCHS)) ($REPEAT x $EPOCHS epochs)"
echo "==> output  : ./outputs/$RUN_NAME"

cd "$DIFFSYNTH_DIR"

accelerate launch examples/qwen_image_21/model_training/train.py \
  --dataset_base_path "../../$DATASET_DIR" \
  --dataset_metadata_path "../../$DATASET_DIR/metadata.jsonl" \
  --dataset_repeat "$REPEAT" \
  --data_file_keys "image,video" \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors" \
  --quant_options "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors:bitsandbytes_nf4;Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors:bitsandbytes_nf4" \
  --lora_base_model dit \
  --lora_rank 32 \
  --lora_target_modules "" \
  --max_pixels 262144 \
  --num_epochs "$EPOCHS" \
  --learning_rate 1e-4 \
  --customized_optimizer torch.optim.AdamW \
  --output_path "../../outputs/$RUN_NAME" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --enable_csv_log \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 0 \
  --find_unused_parameters

# What good looks like:
#   step speed   1.31-1.38 s/step on a 12 GB card at max_pixels=262144
#   VRAM         ~11.6 GB of 12.2 GB (95%) - normal, not a problem
#   stalls       none. A 5-minute pause on one step means max_pixels is too high,
#                not that you need offload. See references/pitfalls.md section 3.
#
# If GPU utilisation sits at ~1% and the log shows "foo.safetensors: 30%", the run
# is DOWNLOADING 31 GB, not training. Set DIFFSYNTH_MODEL_BASE_PATH to the existing
# cache. See references/setup.md section 4.
