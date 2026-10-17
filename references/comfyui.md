# Loading and tuning the LoRA in ComfyUI

---

## 1. Install the file

Copy each `epoch-N.safetensors` into:

```
ComfyUI/models/loras/
```

**Do not rename it.** The `.lora_A.default.weight` naming has a dedicated code
path in ComfyUI and loads as-is.

---

## 2. Verify the load

Two things to check in the ComfyUI console.

### `patches attached`

```
Model QwenImage21 prepared ... 192 patches attached   <- LoRA branch
Model QwenImage21 prepared ...   0 patches attached   <- control branch
```

If the first number is `0`, the LoRA did not apply — check that the filename in
`LoraLoaderModelOnly` matches the file on disk.

### Why 192

The checkpoint has 448 tensors = 224 `A`/`B` pairs.

- 7 LoRA targets per transformer block
- `img_mlp.proj` and `img_mlp.gate_layer` are fused into a single `gate_up` weight,
  so 7 targets map to 6 model modules
- 32 blocks × 6 modules = **192**

A partial load would show a smaller number. 192 means nothing was dropped.

> If you see warnings about `lora key not loaded`, the checkpoint and the base
> model are mismatched — not a naming problem to fix by hand.

---

## 3. Recommended settings

| Knob | Value | Notes |
|---|---|---|
| `strength_model` | **1.5** | stable default |
| | **2.0** | strongest style; colours drift |
| | 1.8–1.9 | avoid — see below |
| `resolution` | **1024** | 0 keeps the source size |
| prompt | keep the style-descriptor prefix used in training | |
| reference images | usable, but they dilute the style | |

### Why 1.8 and 1.9 break while 2.0 does not

The failure is not monotonic. Observed at the same seed:

| strength | result |
|---|---|
| 1.5 | baseline |
| 1.8 | minor anatomical errors |
| 1.9 | a duplicated feature appears |
| 2.0 | self-consistent; colours shift, limb rendering simplifies |

Three things stack:

1. **The definition of "broken" moves with the style.** At 1.5–1.9 the model is
   still aiming at semi-realistic anatomy, so a failed joint or feature is legible
   as an error. At 2.0 control shifts to the LoRA, which renders in its own
   simplified idiom — where a stroke that would read as broken elsewhere reads as a
   stylistic choice.
2. **Simpler renderings have fewer places to fail.** The usual giveaways — fingers,
   joints, facial features, shading — largely disappear once the style flattens.
3. **High strength hands control to the LoRA**, which collapses toward the training
   distribution. That state is internally consistent, so it does not look broken.

This is a shift in legibility, not an absence of error. Colour drift at 2.0 is the
signal that semantics are being overridden — pose, costume and colour adherence get
less reliable as you go higher.

### Why resolution 512 makes it worse

Matching the training resolution is standard advice from the SD1.5 era. It does not
hold here: Qwen-Image-2.1 is 1-megapixel native, and the base model's input
resolution dominates. Dropping to 512² visibly degrades texture even though the
LoRA was trained at that size.

### Reference images dilute the style

A reference image is a strong visual anchor. A light style LoRA — small dataset,
few hundred steps — competes poorly against it. With a reference attached, the
difference between LoRA and no-LoRA shrinks.

If you need both, raise `strength_model` and keep the full style prefix in the
prompt.

---

## 4. Comparison workflow

To see the effect in one run, branch the model:

```
UNETLoader ─┬─► LoraLoaderModelOnly ─► KSampler ─► VAEDecode ─► SaveImage
            └────────────────────────► KSampler ─► VAEDecode ─► SaveImage
```

Same seed, same prompt, same latent. The cost is that sampling runs twice.

---

## 5. Choosing an epoch

Loss plateaus early on small datasets — but **the epoch ordering is not stable**, so
do not choose a checkpoint by the loss value.

| epoch | mean (run A) | mean (run B) |
|---|---|---|
| 0 | 0.2498 | 0.2680 |
| 1 | 0.2656 | **0.2238** |
| 2 | **0.2245** | 0.2292 |
| 3 | 0.2294 | **0.2235** |
| 4 | 0.2292 | 0.2475 |

Same command, same dataset, two runs. Run A says epoch 2; run B says epoch 3; epoch 0
differs by 0.018. The spread between epochs is smaller than the spread between runs,
so the ranking is noise.

What holds across both: loss stops improving after about the first epoch. **300
steps is enough** for a 10-image dataset; the remaining 200 mostly add rigidity.

Pick the checkpoint by looking at images — compare `epoch-1` through `epoch-4` at
the same seed and prompt, and take the one whose style is strongest without colour
drift. If you still cannot tell them apart, take an earlier epoch.
