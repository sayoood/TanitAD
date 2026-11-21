# RESULT — the two arms carrying the programme's scale claims had the thinnest run records. Both are now banked.

**Date** 2026-08-30 · **Author** Master Mind · **Evidence class** MEASURED (pulled from
Thor, md5-verified both sides) · **Trigger** two independent escalations from the
input-pipeline review streams.

## 1. The gap, as escalated

* **`rdw8p30k`** — the arm `MODEL_REGISTRY.md:3872` calls *"the arm that re-scopes the
  whole collapse campaign"*, and the anchor for the parity-scale comparisons. A
  three-probe enumeration (scoped grep over all 15 top-level dirs; a `find` over every
  banked `*config*.json` in Architecture & Inference — 30 files, every 30k-scale one
  accounted for; a name glob) established **no banked run config anywhere in the repo**.
  Its launch flags were recoverable only by *triangulation*: a registry prose row, a 2k
  sibling whose param totals match to the digit, three launch scripts, and the
  checkpoint-adopted `model_frame` in its T1 eval.
* **`v6F-SW-30k`** (config E, 336.5 M) — `stack/ops/runs.d/README.md:91` says plainly
  *"the live `v6F-SW-30k` run on Thor has no manifest here"*, and the review separately
  found **no `MODEL_REGISTRY` row** for it: param count, geometry and the ViT-5 flag
  lived only in run artifacts. ⛔ Under our own source-of-truth rule that arm was
  **unquotable**.

⚠️ **The pattern is what makes this a package rather than a fix:** the two arms carrying
the programme's scale claims were the two whose run records were thinnest. Provenance
decays exactly where it is most load-bearing, because those runs are the ones started
under time pressure.

## 2. Both recovered and banked

The records were never lost — they were **on one disk**, which by the programme's own
definition of done is not banked at all.

| artifact | source | md5 |
|---|---|---|
| `raw/rdw8p30k.config.json` | `thor:/home/nvidia/v7tiny/rdw8p30k/config.json` | `f373f1ee5020766ea03987822ce47ddb` |
| `raw/v6F-SW-30k.config.json` | `thor:/home/nvidia/experiments/v6F-SW-30k/config.json` | `2cb83239cc19d13b9bc7a49a27459b82` |

## 3. The triangulation was RIGHT — and that is worth recording

The reviewer's reconstructed verdict for `rdw8p30k` was **256x640, `v2_subframe: null`**,
reached without a config. The banked record now reads, for **both** arms:

```
frame_h 256 · frame_w 640 · frame_hfov 120.0 · projection cylindrical
v2_subframe None · batch 8 · steps 30000 · window 6 · lr 1e-4
```

The triangulation reproduced the record exactly. That validates the *method*; it is not
a licence to keep using it. A reconstruction that happens to be right is still a
reconstruction. **Bank the config when the run starts.**

## 4. THE FINDING THAT MATTERS MORE THAN THE BOOKKEEPING — masked pixels are hard zeros, and nothing marks them

From the builder's own sidecar
(`.../2026-07-28-rig-fix-wiring/raw/geometry_train_RESTAMPED_2026-07-27.json:25-39`):

| rig | n clips | mean observed frac |
|---|---|---|
| A | 642 | 0.99999 |
| **B** | **1,758 (73.25 %)** | **0.91108** |

**~8.89 % of every rig-B frame is unobserved** — and rig B is the MAJORITY of the
corpus. The chain, read in source:

* `calib.py:992-993` zeroes unobserved pixels (`padding_mode="zeros"`, `out = out * mask`);
* `_contract.py:56` `to_float_frames` maps 0 to **exactly `0.0`**;
* the patch-embed `Conv2d` (`encoder.py:161`) therefore sees **real black and synthetic
  black as identical**;
* the mask **IS computed and then thrown away** — stashed on a function attribute
  (`cylindrical_rectify.last_mask`, `calib.py:995`), never persisted into the
  `*.v2ep.pt` payload, **read by no consumer**. No validity channel, no attention mask,
  no input-side loss masking.

**Why this is more than cosmetic.** A fixed-location constant-black region on 73 % of
frames is a **rig-identifying signal** a from-scratch ViT with learned position
embeddings can latch onto; if normalisation statistics are computed over the padded
support they are contaminated **by a different amount per rig**; and all-pad patches,
if covered by any reconstruction loss, are trivially predictable and inflate it by a
per-rig constant. Same family as the `df` / cgroup / `step_s` traps: **a quantity
computed over the wrong support.**

**The fix exists and is switched off.** `v2_subframe: null` is confirmed on every v6/v7
arm including config E — now from the banked configs above, not by inference. The
**176x624** sub-frame that both rigs fully observe is recorded in the sidecar's own
`subframe_observability` as `{A: 1.0, B: 1.0}`, is **bit-exact** from our PNG cache, and
slices at **~1.4 s/clip vs ~19.4 s/clip to rebuild**.

⛔ **PI decision, and NOT urgent**: per `ADDENDUM_geometry_not_foreclosed.md` a PNG cache
stays sliceable, so the option survives the B1 build. The trade is real — 176x624 costs
near-field view (overhead lights vanish at 12.2 m rather than 8.4 m). **The zero-regret
half is to persist the per-clip validity mask as a sidecar**, which forecloses nothing
and makes the question answerable later either way.

## Deliverable manifest

| artifact | where |
|---|---|
| `raw/rdw8p30k.config.json`, `raw/v6F-SW-30k.config.json` | this package, md5 above |
| this RESULT.md | this package |
| follow-up: `MODEL_REGISTRY` row for config E | **still owed** — that arm stays unquotable until it exists |
