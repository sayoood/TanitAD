# B1 epcache preflight — MEASURED cost, two blockers, and a capacity wall

**Date** 2026-08-29 · **Owner** DataFlyWheel · **Trigger** Master Mind asked for
a measured epcache wall-clock + bottleneck to order Thor's queue (τ-ramp arm vs
epcache build; both gate the v7f launch, Thor is single-GPU).

## ⛔ RETRACTED, SAME DAY — "the 1.38 TB capacity wall"

**There is no capacity wall. I priced the artifact I FOUND on disk instead of the
artifact the CONSUMER reads.** The v7 trainer's `--v2-cache` reads `*.v2ep.pt`
(`v2_dataset.py`, written by `scripts/v2_compressed.py::build_compressed`), which
stores **ENCODED** frames (`jpeg_buf` + `jpeg_len` + `codec`). I benchmarked
`ep_*.pt` — the LEGACY raw-`frames_u8` epcache — and projected its 293.4 MB/ep to
1.38 TB. Caught by the Master Mind, verified by me against the real writer.

**Root-cause class: pricing a format without identifying its consumer.** The
293.4 MB/ep measurement was *correct* — it was quoted about the wrong artifact.
Same family as the `df` / Thor `free` / cgroup `usage_in_bytes` / `step_s` traps:
**a true number applied outside its scope reads exactly like an answer.**
Generalised fix: *before pricing any artifact, open the CONSUMER's loader and
price what IT reads.* Everything downstream of the wall — the rig-clean
downgrade, the format change, the shard-and-stream, "a PI decision" — was
manufactured by this error and is withdrawn.

## Headline (corrected)

**It fits, and the cost is ~4× what I first said — for a reason neither estimate
had priced: PNG encoding is 70 % of the build.** Measured below.

## 1. Cost — MEASURED (ours), not estimated

Real path timed: `physicalai._decode_mp4` with real per-clip intrinsics at
`PHYSICALAI_WIDE120_256x640`, 6 corpus clips, dev box, `PAI_DECODE_THREADS=4`.

| | |
|---|---|
| per clip (605 frames → `[605, 3, 256, 640]`) | **median 5.59 s** |
| decode (PyAV) | 3.84 s = **69 %** |
| cylindrical remap | 1.75 s = **31 %** |
| **GPU used** | ⛔ **NONE** — no CUDA reference in the build path; `torch.cuda` never initialises |

Bottleneck: **CPU decode**, then CPU remap. Not IO (clip size 1.4–13.6 MB had no
effect on time: 6.9 MB → 5.69 s, 1.4 MB → 5.46 s). Not GPU at all.

| workers (×4 decode threads) | 4,719 clips |
|---|---|
| 1 | 7.33 h |
| 6 (24 logical cores ÷ 4) | **1.22 h** |
| 8 | 0.92 h |

⇒ **~1.2–1.5 h of pure CPU on the dev box.** It never needs Thor's GPU.

## 2. The REAL format — re-measured end to end (MEASURED, ours)

Same decode+remap as §1, then the encode leg the real builder pays, on the
episode's ~201-frame 10 Hz grid (not all 605 decoded frames), 4 clips:

| codec | decode+remap | **encode** | total/clip | MB/episode | corpus | 6 workers | 8 workers |
|---|---|---|---|---|---|---|---|
| **png** (lossless, what the live parity cache uses) | 6.19 s | **14.79 s (70 %)** | **20.98 s** | **34.0** | **161 GB** | **4.58 h** | **3.44 h** |
| jpeg q95 (lossy) | 6.19 s | 0.18 s (3 %) | 6.37 s | 7.4 | 35 GB | 1.39 h | 1.04 h |

⭐ **Independent agreement on size:** my 34.0 MB/ep was measured by encoding
frames here; the Master Mind measured ~36 MB/ep median over 2,403 real episodes
on Thor. Two methods, two machines, same answer — that is what makes 161 GB
trustworthy rather than merely plausible. **Thor has 658 GB free: it fits ~4×
over.**

### ⚠️ The correction that matters for scheduling: encode dominates

**PNG encoding is 70 % of the build.** My §1 figure of ~1.2 h at 6 workers priced
only decode+remap and is therefore **~4× optimistic** for the real path: the
honest number is **3.4–4.6 h**. It is CPU-bound and embarrassingly parallel per
frame, so more workers help linearly until cores run out.

### The codec is not a free choice — PNG is load-bearing

JPEG would be 4.6× cheaper and 4.6× smaller, but:
* the **live parity cache is `codec: png`**, so a jpeg B1 cache is not
  format-comparable with the arms already trained;
* `v2_dataset.py:325` and `slice_v2_cache.py` **refuse to sub-frame a LOSSY
  cache** — "a centred slice equals a rebuild at that geometry only" for
  lossless. So a PNG wide cache can be sliced to the rig-clean 176×624 frame
  **without a rebuild**; a JPEG one cannot.

⇒ **Build PNG.** Budget 3.4–4.6 h of CPU, not 1.2 h.

✅ **DECIDED BY THE PI, 2026-08-29, verbatim: *"stick to the losless pngs"*.**
This is no longer a recommendation — the B1 epcache is built with `codec="png"`,
and the 4× build cost is accepted. Any future proposal to switch to jpeg for
speed reopens a PI decision, and loses the no-rebuild slice to 176×624.

⚠️ The buffer is named `jpeg_buf` even when `codec` is `png`. Read the codec
field, never the buffer name — the naming trap already logged in `CLAUDE.md`.

## 3. ✅ Blocker FOUND AND FIXED: per-clip intrinsics were unresolvable

`intrinsics_for_clip` returned the corpus-median fallback for **200/200** sampled
B1 clips, so `cylindrical_rectify` **correctly refused** (a single global cy is a
rig-B value; on rig A the horizon lands ~215 px wrong — the D-016 R1 error).

**The only two ways past that guard were a crash or `require_per_clip=False`,
i.e. silently building most of the corpus with the rig fix DISABLED.** Rig B is
the majority (2,723 of 4,719), so the silent path would have poisoned the corpus
in exactly the way the guard exists to prevent.

The data was never missing — **1,420 chunk parquets were already on disk**; what
was missing is the clip→chunk mapping, which the B1 bundle carries. Fixed by
generating the local CSV table (resolution path 1, `per_clip=True`):

* `calibration/physicalai_front_wide_intrinsics.csv` — **4,719/4,719 resolved**
* rig split **1,996 A (mean cy 542.0) / 2,723 B (mean cy 754.2)**
* ⭐ **cross-checked against the INDEPENDENTLY built B1 cy table: n=4,719,
  max |Δcy| = 0.000000 — exact match.** Two different code paths, same principal
  points; that is what makes the table trustworthy rather than merely present.

## 4. ✅ Second blocker — FIXED, and it was worse than first reported

`_physicalai_root_of` recovers the corpus root by walking parents for a dir named
**`r0`** (`physicalai.py:166`). The B1 camera pull banks to
`<root>/camera/camera_front_wide_120fov/`, which has no `r0` ancestor, so it
returns **`None`** — and then intrinsics AND extrinsics both fall back silently.
(Extrinsics carry the mount pitch that locates the horizon, so this is not
cosmetic.)

⛔ **And an explicit `--root` alone would NOT have saved it.** `_chunk_of_clip`
reads `<root>/r0/r0_selection.parquet`, whose local copy holds **500 rows
covering 38 of the 4,719 B1 clips** — so extrinsics resolve to `None` for
**99.2 %** of the corpus even with the root supplied. `extrinsics_for_clip`
returns None silently and callers *"treat the mount as level (optical axis ==
horizon)"* (`physicalai.py:450`). **Nothing raises.** The mount pitch is what
locates the horizon in the output frame, so the corpus would have been built
horizon-wrong, corpus-wide, with no error and no warning — a worse outcome than
the intrinsics failure, which at least had a guard downstream.

### The fix, verified by RESOLUTION rather than by inspection

| step | artifact | result |
|---|---|---|
| pull the missing calibration | `pull_extrinsics.py` — **52 of 1,411** chunks were local | pulls the other 1,368, coverage asserted over **clips**, not files |
| build a resolvable root | `prepare_b1_root.py` → `physicalai-b1/` with `r0/r0_selection.parquet` (4,719 clips / 1,411 chunks) and the camera bank attached by **directory junction** (47 GB not copied) | `_physicalai_root_of` → the root, **OK** |
| gate the build | `preflight_epcache_build.py` | **✅ PASSED** — 30/30 real intrinsics, **30/30 real extrinsics**, both rigs present (16 A / 14 B) |

⛔ **PARITY SAFETY: a NEW root was written.** The canonical corpus and its
`r0_selection.parquet` are untouched; `physicalai-train-e438721ae894` /
skip-hash `f09e44db` are unaffected. The B1 root is separately identified and
NON-PARITY.

⭐ **The guard was verified by watching it FAIL first.** Run against the old
layout it reported `root … -> None` and `per-clip EXTRINSICS 0/20 resolved`, and
exited non-zero; against the fixed root, all green. **A gate that has never been
seen to fire is not evidence** — the same discipline as a control that must read
a known value.

⇒ **The build's binding precondition is now mechanical, not prose:**
`preflight_epcache_build.py --root <root>` must exit 0. It asserts the POSITIVE
fact (real per-clip calibration resolved on a random sample covering both rigs)
instead of trusting the absence of a warning.

## Deliverable manifest

| artifact | where |
|---|---|
| `build_intr_csv.py` (the fix), `bench_epcache.py` (the measurement) | repo `code/` (this package) |
| `physicalai_front_wide_intrinsics.csv`, 4,719 rows | dev box `C:/Users/Admin/tanitad-data/physicalai/calibration/` — **regenerable from the banked chunk parquets by the staged script** |
