# Dataset: format, captions, curation

---

## 1. Layout

```
data/datasets/<name>/
├── 0001.jpg
├── 0002.jpg
├── ...
└── metadata.jsonl
```

`metadata.jsonl` — one JSON object per line, no trailing comma, UTF-8:

```json
{"image": "0001.jpg", "prompt": "<style prefix>: a person standing by a window, front view."}
{"image": "0002.jpg", "prompt": "<style prefix>: a person walking outdoors, three-quarter view."}
```

`image` is relative to `dataset_base_path`. The dataloader also accepts a `video`
key when `--data_file_keys` includes it.

---

## 2. Caption strategy

The single most important decision: **describe content, not style.**

The style should be carried by a short, fixed prefix that appears in **every**
caption. The model then learns that whatever follows the prefix is variable
content, and the prefix itself becomes the trigger.

The prefix is a short phrase describing the rendering, not the content. Invent your
own — this is a placeholder, and it should be the *only* place style words appear:

```
An illustration in a flat vector style with a limited warm palette:
```

Rules:

| Do | Don't |
|---|---|
| Use the same prefix verbatim on every image | Vary the wording between images |
| Describe subject, pose, clothing, framing | Describe lighting, brushwork, rendering quality |
| Keep captions one sentence | Write paragraphs |
| Include the background when it matters | Leave captions empty |

Why style words are excluded from the body: if every caption says "watercolour",
the model learns to associate that word with the style but also learns that the
style is optional — it becomes a switchable attribute rather than the default
rendering. A fixed prefix with content-only bodies makes the style the default.

At inference, **reuse the prefix exactly**. Dropping it loses a noticeable amount
of effect — comparable to dropping `strength_model` by about a third.

---

## 3. How many images

| Source | Recommendation |
|---|---|
| Measured here | 10 images, 500 steps — works, but the ceiling is low |
| Widely reported for style LoRAs | **20–40 images** |
| Widely reported for character LoRAs | 20–100 images |

The 20–40 figure is **external consensus, not something measured in this
repository.** What was measured: with 10 images and rank 32, loss stops improving
after about 300 steps, and the resulting LoRA cannot override a strong reference
image or shift a palette.

Practical implication: if you have 20–40 images, use them, and keep steps in the
same per-image range (roughly 30–50 views per image). More images is a better lever
than more steps.

---

## 4. Curation

Quality dominates quantity. A consistent 20 beats an inconsistent 60.

**Keep:**

- Images in the target style, with the same line weight and rendering
- Variety in subject, pose, framing and background
- Clean, sufficiently large images

**Drop:**

- Anything off-style, even if it is a good picture
- Screenshots with UI, watermarks, or cropped edges
- Collages and multi-panel images
- Low-resolution or heavily compressed files
- Near-duplicates — ten variants of one pose teach nothing extra

**Check consistency before training:** put the images in a grid and look at them
together. If one stands out as "different", it will teach the wrong thing.

---

## 5. Repeated images across epochs

`--dataset_repeat` multiplies each image per epoch. One step = one image seen.

| Images | repeat | epochs | steps | views per image |
|---|---|---|---|---|
| 10 | 10 | 5 | 500 | 50 |
| 10 | 10 | 3 | 300 | 30 |
| 25 | 4 | 5 | 500 | 20 |
| 30 | 2 | 5 | 300 | 10 |

For style LoRAs, 30–50 views per image is a reasonable band. Beyond that, on a small
dataset, you get rigidity rather than fidelity — visible as colour drift and reduced
prompt adherence.

---

## 6. Rights

Training on images you do not own is a copyright question, not a technical one, and
it varies by jurisdiction. Two practical points:

- **Private training and public distribution are different acts.** Publishing the
  resulting weights is a redistribution of a derivative work; keep the provenance of
  your dataset in mind before you publish.
- **Public-domain and self-generated images are the clean paths** for anything you
  intend to distribute.

This repository contains no dataset for this reason. Bring your own.
