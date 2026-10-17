# Provenance

Where every number in this repository comes from, how it was measured, and what
was deliberately left out. If a claim in the README cannot be traced to a file
listed here, treat it as unverified.

---

## The two runs

One machine, one 10-image dataset, one command, run twice.

| | Run A | Run B |
|---|---|---|
| Started | 2026-10-17 20:54 (UTC+8) | 2026-10-18 00:42 (UTC+8) |
| Purpose | first successful run | re-run under instrumentation |
| Steps | 500 | 500 |
| Wall clock | 12 min 37 s | **12 min 04.7 s** |
| Shipped data | `assets/loss.csv`, `assets/training_args.json` | `assets/measured_loss.csv`, `assets/measured_resources.csv`, `assets/measured_summary.json` |

Run B is the superset: it recorded resources as well as loss. Run A's console log
was not kept — its only shipped artifact is the loss curve, which exists to be
compared against Run B's. Those two independent curves are what show that the
epoch-to-epoch ranking is noise (see "Loss by epoch" in the README).

`assets/training_args.json` is the configuration **both** runs used, with absolute
paths replaced by placeholders. The two runs differ only in that Run B was
launched through the sampler.

---

## How Run B was measured

[`scripts/measure.py`](scripts/measure.py) is the published form of the sampler
used. It spawns the training command as a child process and samples every
**2 seconds** until the process exits:

| Metric | Source |
|---|---|
| Training process RSS (process and children) | `psutil` `memory_info().rss`, summed |
| System RAM used / available | `psutil.virtual_memory()` |
| Page file used | `psutil.swap_memory()` |
| GPU memory used | `nvidia-smi --query-gpu=memory.used` |

Baselines were taken immediately before launch: **543 MB** VRAM idle,
**8,945.9 MB** RAM used, **1,178.9 MB** page file used. Every peak in
`measured_summary.json` is a raw reading, not a delta; `peak_gpu_delta_mb`
(11,148) is the one derived field.

355 samples × 2 s ≈ 710 s of sampling across a 724.7 s run.

---

## Measured vs. derived

**Measured directly.** Every field in `measured_summary.json`, every row in
`measured_resources.csv`, every loss value in both CSVs. The `192 patches
attached` count, the 1.31–1.38 s/step range, the 1024² stall timings, and the
`RuntimeError: invalid argument to getCurrentStream` traceback all come from
console logs of actual runs.

**Derived, not measured.** The claim that CPU offload peaks at roughly **2× the
model size**, and therefore the "offload needs ~66 GB" row in the bf16 table.
This follows from `pin_memory()` copying every parameter while the original
allocation is still alive. It is an inference from the allocator's behaviour —
**a full offload run was never completed on this machine**. Offload runs out of
RAM during model load, which is the subject of the recipe, so it could not be
observed to completion.

The page-file evidence (1,178.9 → 1,188.5 MB over the whole run) *is* measured,
and it is what makes the inference safe to publish: no paging occurred, so the
page file has nothing to contribute even if it were larger.

**Not claimed.** No throughput figure, no comparison against other cards or
services, and no bf16 baseline. Those would be different measurements.

---

## Sanitisation

The published text was produced from working notes with the following replaced:

- absolute paths → `/path/to/...` or `./`
- the dataset and its captions → not shipped at all
- the style-descriptor prefix used in training → a neutral placeholder in
  [`references/dataset.md`](references/dataset.md)
- prompt examples → generic descriptions of a subject and a view angle

No usernames, hostnames, real names, e-mail addresses, tokens, or non-loopback IP
addresses appear in this repository or in its git history.

---

## What is deliberately absent

| Absent | Why |
|---|---|
| Training images | not the author's to redistribute |
| Captions / `metadata.jsonl` | same, and it would fingerprint the trained style |
| LoRA checkpoints | derived from that dataset |
| Generated samples | same reason |
| Model weights (31 GB) | upstream, under a separate non-commercial licence |
| Run A's console log | contained absolute paths and the dataset name |

---

## Third-party data

No benchmark, timing or memory figure in this repository was copied from another
project, a vendor claim, or a review. Everything was measured on the machine
named in the README.

---

## Machine and date

Measurements: **2026-10-17** and **2026-10-18**, local time (UTC+8).
Machine: **RTX 5070 Ti Laptop** (11.94 GB usable VRAM), **31.7 GB RAM**,
Windows 11, 96.8 GB commit limit.
