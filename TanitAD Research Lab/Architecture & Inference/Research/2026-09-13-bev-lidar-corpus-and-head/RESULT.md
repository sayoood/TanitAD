# LiDAR BEV ground truth at corpus scale, and a BEV transformer head on the frozen refcv5-v2 trunk: the GT is solid, the frozen trunk exposes only a thin BEV signal, and more data is the lever that moved it

**Date** 2026-09-13 (Europe/Berlin) · **Agent** Architecture & Inference FlyWheel ·
**Branch** `agent/arch-inf-20260803` · **Tree** `D:\Projects\TanitAD`
**PI directive (2026-09-11, verbatim):** *"train the bev transformer based on the lidar based bev as gt"*
**Compute** dev-box CPU + RTX 4060 only. ⛔ Thor untouched (Master Mind's Qwen-Drive). No paid HF compute.
**Evidence class** MEASURED (ours) unless a row says otherwise. **Tier** none — the GT is a label
artifact and the head is a representation probe; no T0/T1 stamp and no planner metric family applies.

## 0. Headline

| step | outcome |
|---|---|
| **P1** bandwidth | link saturates at **12.60 MB/s** aggregate (c = 8) vs 11.59 (c = 4) vs 6.87 (c = 1), n = 8 clips/arm ⇒ c = 4 (`raw/p1_bandwidth.json`) |
| **P0** (found on the way) | ⛔ the 09-11 deskew rotation sign was inverted: flipped sign wins **24/24** high-yaw instants (0.0135 vs 0.1545 m); fixed (`raw/p0_deskew_sign_check*.json`) |
| **P2** corpus GT | **134/139** B1 EVAL clips OK, 26,526 labelled instants, 109.8 MB, on the grid the head reads, third state + camera field in every file; registered to the independent box join at **3.23×** mirror / **4.29×** marginal (`raw/p2_corpus_summary.json`, `raw/p3_orientation_all.json`); ⛔ shipped polar `observed` rule measured biased and fixed; 5 clips quarantined and kept out as the gate says |
| **P3** loader + test | 10 tests green on all 134 and on the 2 staged samples; RED on a mirrored or mis-sized artifact (`raw/p3_mutation_run.json`) |
| **P4** `E-BEVHEAD-FROZEN-1` (pre-registered) | ⛔ **FAILED all four bars.** test AP **0.4140** [0.3721, 0.4590] vs zero-information 0.3801 / prior 0.3773 / pixels 0.3849; +0.034 over shuffled (< +0.05), replicate not separated, AP < 0.60 (`raw/p4_panel.json`) |
| **L1** stride-16 tokens | ⛔ FAILED, no effect: 0.4074, −0.0066 vs `main` (`raw/p4_panel_levers.json`) |
| **L4** +181 train clips (3.2× rows) | ⛔ its criterion FAILED (+0.0179 [−0.0063, +0.0402]) — ⭐ but the head now clears **B1 and B2 on two seeds** (AP **0.4352**, +0.0552 over its shuffled arm, floor 4.4× tighter) (`raw/p4_panel_L4_data_rb.json`) |
| **L3** azimuth-aligned attention | ⛔ FAILED, no separated effect: 0.4172, +0.0032 vs `main` (`raw/p4_panel_levers_L1L2L3.json`) |
| **L2** unfrozen last stage (51.6 M) | ⛔ FAILED, no separated effect: 0.4248, +0.0107 vs `main` (`raw/p4_panel_levers_L1L2L3.json`) |
| **RULE ZERO stop** | B4 (AP ≥ 0.60) is not met by any arm; the two levers that could plausibly get there — full-scale data and joint trunk training — are blocked on PI-authorised compute (§6.5) |

**The single most important number:** the best frozen-trunk BEV transformer so far reads **test AP
0.4352 [0.3976, 0.4737]** against a zero-information arm at **0.3801** on the same 7,928,485 cells
(MEASURED, `raw/p4_panel_L4_data_rb.json`) — real, replicated, and **far from the pre-registered
usefulness bar of 0.60**.

**Decisions only the PI can make** (§8): a full-B1 LiDAR GT build (~1.55 TB, ~34 h on the dev-box link
⇒ a bandwidth-rich pod), joint BEV-aux trunk training (`E-BEVLIDAR-CAP-1`, pod GPUs), and a private
HF publish of the GT.

---

## 1. P1 — bandwidth: the dev-box link saturates at ~12.6 MB/s, so P2 ran at concurrency 4

`code/p1_bandwidth.py` → `raw/p1_bandwidth.json`. Clips = the 139-clip B1 EVAL join in join
order, already-cached clips skipped, the next 8 per arm (different clips per arm, chosen by
position). Each arm is a process pool of *c* workers; per-stream MB/s = member bytes / member
stream time; aggregate = arm bytes / arm wall time. All 24 transfers CRC-32-checked by `zipfile`
and content-asserted (198–199 row groups, decodable non-zero first spin). **MEASURED.**

| concurrency | n | per-stream MB/s min · **median** · max | **aggregate MB/s** | arm wall | bytes |
|---:|---:|---|---:|---:|---:|
| 1 | 8 | 6.801 · **7.506** · 8.348 | **6.865** | 385.2 s | 2,644,221,782 |
| 4 | 8 | 2.916 · **3.108** · 3.569 | **11.586** | 233.5 s | 2,705,147,234 |
| 8 | 8 | 1.653 · **1.682** · 2.013 | **12.601** | 226.0 s | 2,847,859,867 |

⇒ **c = 4 chosen**: it reaches 92 % of the c = 8 aggregate at half the streams. The link, not the
reader, is the ceiling (~100 Mbit/s). Central-directory reads 3.27–4.37 s.
⚠️ The 09-11 package saw a 7.3× single-stream spread over 4 samples; here the single-stream
spread is **1.23×** over 8. ⚠️ Arms ran sequentially, so a bandwidth change between arms is
confounded with concurrency.

**For scale only (ESTIMATED from the measured aggregate, not a plan):** all of B1 (4,713 clips ×
the 09-11 mean 341.7 MB ≈ 1.61 TB) at 12.6 MB/s is **~35 h** of transfer on this link.

## 2. P0 — ⛔ the 09-11 deskew rotated the wrong way (MEASURED before P2 used it)

Reading `lidar_bev.deskew_rigid` to reuse it: a point captured at `t_pt` was mapped to `t_ref`
with `R(−yaw_rate·dt)`. The ego frame at `t_pt` is rotated by **+yaw_rate·dt**, so the inverse is
`R(+yaw_rate·dt)`. Not argued — measured with a discriminating experiment
(`code/p0_deskew_sign_check.py`): **two consecutive spins deskewed to the same instant must put
static structure in the same place**; every azimuth of spin k+1 is sampled ~100 ms after spin k.

| run | high-\|yaw\| instants | rotation OFF | **09-11 sign** | **flipped** | flipped wins |
|---|---|---:|---:|---:|---:|
| `raw/p0_deskew_sign_check.json` | 12 · 1 clip · right turns 0.16–0.38 rad/s | 0.1080 m | **0.1545 m** | **0.0135 m** | **12/12** |
| `raw/p0_deskew_sign_check_replicate.json` | 12 · 2 clips · **5 left turns** · 4.3–22.6 m/s | 0.0321 m | **0.0499 m** | **0.0146 m** | **12/12** |
| control: \|yaw\| ≤ 0.02 rad/s | n = 48 + 54 | 0.0126 / 0.0133 | 0.0136 / 0.0146 | 0.0122 / 0.0127 | — (equal, as required) |

(median xy nearest-neighbour distance, obstacle z-band, r ∈ [5, 40] m). The yaw-rate sign itself
was cross-checked independently: it agrees with `v·curvature` on 100 % and with the trajectory
heading rate on 82–100 % of |r| > 0.05 samples (4 clips) ⇒ the defect is the rotation, not the
rate. At |r| 0.35 rad/s the shipped sign's p90 reached **0.914 m** (vs 0.331 m flipped).
**Fixed** in this package's `code/lidar_bev.py` (lineage + md5 in its docstring; schema bumped to
`tanitad.lidar_bev_gt/2`); pinned by an exact-arc test and a mutation arm (§4).
⛔ The banked 09-11 module is deliberately NOT edited — it is the record of what produced that
package's numbers; its registration sweep ran mostly on low-yaw frames and is not retracted.

## 3. P2 — corpus-scale LiDAR BEV ground truth for the 139-clip B1 EVAL join

### 3.1 The consumer, named

```
code/p4_bev_head.py -> code/bev_gt_loader.py::load_clip(<sha12>.bevgt.npz)
  -> polar48_occ / polar48_observed / polar48_cam_vis at RAW v2ep frame i
     paired with the refcv5-v2 trunk token map of STACKED ROW j = i - 2
trunk frames: refcv5-v2 EVAL cache *.v2ep.pt (stack/tanitad/data/v2_dataset.py::_decode_stacked,
built by stack/scripts/v2_compressed.py::build_compressed)
```

⇒ the GT grid is the **v2ep episode grid**, not the 30 Hz camera and not the 10 Hz LiDAR:
`t_query = linspace(t_cam[0], t_cam[-1], int(span·10))`, pixels from camera frame
`searchsorted(t_cam, t_query)`. **MEASURED before the builder ran:** this reconstruction reproduces
the join's `t_s` on all **26,394** join rows of **139** clips (median |dt| 0.31 ms, max 10.3 ms) and
`T == len(jpeg_len)` on all 139 payloads. Reference instant = the camera exposure `t_img`; spin =
nearest spin midpoint; |dt| > 50 ms ⇒ NO_LABEL.

### 3.2 ⛔ The shipped POLAR `observed` rule was biased — found by a content band declared before the build

The first build quarantined **3 of 24** clips on `polar48_marginal_observed_infield` (0.501 /
0.553 / 0.692 against a declared [0.005, 0.50]). P2 was stopped by PID before more parquets were
deleted, and the cause measured (`code/p2b_observed_rule_probe.py` → `raw/p2b_observed_rule_probe.json`,
3 clips × 24 spins):

| clip | share of scored positives BEHIND the first hit (shipped rule) | free-with-returns cells dropped | marginal A shipped `~shadow∨occ` | **B `~shadow∨occ∨n_pts>0`** | C `~shadow` |
|---|---:|---:|---:|---:|---:|
| `c64f0103aece` | **84.1 %** | 11,421 | 0.4139 | **0.2784** | 0.1010 |
| `d0c6db065fa0` | **90.6 %** | 9,764 | 0.6196 | **0.4651** | 0.1328 |
| `d4c144f52747` | **78.5 %** | 5,087 | 0.4706 | **0.3640** | 0.1605 |

Control: A and B agree **exactly** in front of the first hit on all 3 clips. The Cartesian rule of
the same module already carries `n_pts > 0`; the polar one did not, so behind the first hit it
kept every occupied cell and dropped every free one. **Fix (additive):** `observed_npts` (rule B)
and `visible_first_hit` (rule C) beside the unchanged shipped key (rule A); **all three are written
into every artifact**; the consumer trains and scores on **rule B**, reports rule C as secondary.
The degeneracy band was re-declared to [0.005, 0.80] **before** the rule-B rebuild, with its reason
in the builder. The rule-A run is kept locally (`bev_gt_ruleA_superseded/`), never staged.

### 3.3 The corpus — MEASURED (`raw/p2_corpus_summary.json`, `raw/p2_corpus_manifest.jsonl`)

`code/p2_build_corpus.py` (fetch pool c = 4 → build pool 8 workers → content assertions → atomic
write → read-back → manifest line → parquet deleted), rule-B rebuild **3,479 s** wall.

| quantity | value |
|---|---|
| join clips | **139** — **134 OK**, **5 QUARANTINED**, 0 missing |
| instants (raw v2ep frames) | **26,937**, of which **26,526 labelled**, **411 NO_LABEL** (clip ends, \|dt\| > 50 ms) |
| alignment \|t_img − nearest spin midpoint\| | per-clip median **16.6–33.3 ms** (median 24.8); per-clip max over labelled instants **33.5–49.9 ms** |
| artifact bytes | **109,767,615** total; per clip 301,519–1,186,929 (median 839,189) |
| every artifact | **re-read by the loader** against literals + builder specs, **sha256'd**, mirrored to `D:\Projects\TanitAD-artifacts\bev-lidar-gt-b1eval-20260913\` — **134/134 mirror hashes match** |
| build time / clip | 23.8–112.2 s (median 41.0 s), 8 workers under a concurrent GPU extraction |
| raw parquets left | **4** = exactly the protected Qwen-Drive clips, byte sizes unchanged (315,765,342 / 395,882,329 / 354,988,067 / 373,529,123 B); 0 unexpected |

Per-clip distributions over the 134 OK clips (min · p10 · **median** · p90 · max):

| statistic | value |
|---|---|
| polar48 marginal, **rule B** (consumer), observed ∩ in-field | 0.032 · 0.172 · **0.268** · 0.402 · 0.602 |
| polar48 marginal, rule A (shipped, biased) | 0.033 · 0.244 · **0.396** · 0.585 · 0.854 |
| polar48 marginal, rule C (first hit) | 0.017 · 0.070 · **0.106** · 0.169 · 0.327 |
| Cartesian occupancy, fraction of grid | 0.011 · 0.041 · **0.084** · 0.129 · 0.326 |
| polar48 observed fraction of in-field cells | 0.375 · 0.535 · **0.682** · 0.837 · 0.919 |
| polar48 in-camera-field fraction | 0.846 · 0.853 · **0.862** · 0.870 · 0.873 |
| ground-plane peak z re-derived from the returns | −0.075 · −0.025 · **−0.025** · +0.025 · +0.125 m |

### 3.4 ⛔ Five clips QUARANTINED — reported as written, not overridden

All five tripped the ground-plane re-derivation (declared band ±0.20 m): peak **+1.175 / +1.375 /
+1.475 / +1.575 / +1.925 m**; `7be92e6870d8` also read an observed-in-field fraction of **0.220**
(band ≥ 0.30). Classified by `code/p2d_ground_outlier_probe.py` → `raw/p2d_ground_outlier_probe.json`:

| clip | LiDAR mount z (corpus 1.859–2.385 m, n 139) | ground mode ≤ 0.2 m in top-3 near-range modes | lowest dense surface (z p5, 3–40 m) within ±0.2 m | reading |
|---|---:|---:|---:|---|
| `e902aba7fb9a` | 1.9293 | 6/8 spins | 8/8 | frame OK; argmax captured by vertical structure |
| `ff9d2a66d2a2` | 1.9568 | 4/8 | 6/8 (two spins −0.97 / −0.53: road dips away) | frame OK |
| `693335b810c3` | 1.9356 | 4/8 | 8/8 | frame OK |
| `570b47ad7408` | 1.8879 | 2/8 | 8/8 | frame OK |
| `7be92e6870d8` | 1.9467 | **0/8** | 6/8, but p5 **+0.10…+0.21 m** on every spin; 6 % of near points below 0.3 m | ⚠️ heavily occluded scene; a small vertical offset is NOT excluded |

⇒ No mount-geometry defect on any of them; the argmax-mode statistic is **scene-fragile** (5/139
trips, 4 of them demonstrably false alarms). ⛔ **They stay quarantined anyway**: the P4
pre-registration binds the head's corpus to *"artifacts that pass every content check"*, and a
gate is not re-cut after its data is seen. The quarantined artifacts are kept locally for a
sensitivity row; the check's replacement (p5 of z plus a top-3 ground mode) is a builder change for
the NEXT build. Their raw parquets were deleted after this probe (re-fetchable).

### 3.5 Registration over the whole corpus — the orientation check from an independent source

`code/p3_orientation_all.py` → `raw/p3_orientation_all.json`: every OK artifact's `cart_occ` against
the `obstacle.offline` boxes of the B1 EVAL join (pooled over cells, observed cells only):

| | hit rate | cells |
|---|---:|---:|
| ⭐ **real boxes** | **0.5823** | 2,107,991 |
| mirrored boxes | 0.1802 | 2,055,357 |
| no-information marginal | 0.1356 | 123,097,698 observed |

⇒ **real / mirror 3.23×, real / marginal 4.29×** (pooled over all 134). 126 clips have both real
and mirrored box cells in the 60 × 32 m grid; 7 have no real box cell in it at all. Per clip, real/mirror median **4.10**; ⚠️ **3 clips read the mirror above the
real boxes** (0.76 / 0.82 / 0.94×) — yet their real boxes still hit **1.68–3.77× their marginal**:
dense, bilaterally symmetric scenes where a mirrored box lands on the other side's parked row, the
same scene-density effect 09-11 §5.2 found. Two clips read real/marginal < 1.5: `eaf5c5ae236c`
(1.28×, marginal 0.323) and `d4c144f52747` (0.84× on only **97** box cells).

## 4. P3 — loader + test (corpus-wide: 10/10 green)

`code/bev_gt_loader.py`: `load_clip` (grid, rule), the 4-state encoding
`0 OUT_OF_FIELD · 1 OBSERVED_EMPTY · 2 OCCUPIED · 3 OCCLUDED`, `scored_mask`, `rows_to_frames`,
`verify_meta` (literals AND builder specs), `verify_arrays`, and `registration_check` (orientation
from the independent `obstacle.offline` join boxes).
`code/test_bev_gt_artifact.py` (beside the package code — the loader is package-local, so
`stack/tests` is not its home): 10 tests.

**Mutation run** (`code/p3_mutation_run.py` → `raw/p3_mutation_run.json`) — the whole suite pointed
at three copies of 6 real artifacts:

| directory | result | failing tests |
|---|---|---|
| control (unmodified) | ✅ **10 passed** | — |
| **mirrored** (every y / azimuth axis flipped) | ⛔ **RED** 1 failed | `test_orientation_real_boxes_beat_mirror_and_MUTATION_goes_red` |
| **wrong cell size** (`cartesian.cell_m` 0.5 → 0.25 in `meta_json`) | ⛔ **RED** 2 failed | `test_every_ok_artifact_verifies_against_literals_and_builder`, `test_MUTATION_wrong_cell_size_goes_red` |

⭐ A mirrored artifact has a perfectly valid `meta_json` — only the content check against an
independent source catches it, which is why that check exists.

## 5. P4 — the BEV transformer head on the FROZEN refcv5-v2 trunk

Pre-registration: **`PREREG_BEVHEAD_FROZEN_TRUNK.md`** (`E-BEVHEAD-FROZEN-1`), written before any
head trained. Code: `code/p4a_extract_tokens.py`, `code/p4_bev_head.py`, `code/p4_eval.py`,
`code/p5_render.py`, `code/run_panel.sh`. The panel ran with `p4_bev_head.py` sha256
`e3f589dd…` (last modified 10:14, panel start 10:46; no edit until the panel's eval finished).

### 5.1 Inputs — vision only, reproduced from the trainer's own path (MEASURED)

| fact | value | how it was established |
|---|---|---|
| encoder | `refc.ResNetEncoder`, in 9 ch, base width 88, blocks (3,6,16,6), **90,458,632** params | 396 `core.encoder.*` keys of `ckpt_40284.pt` loaded **STRICT** (0 missing, 0 unexpected) |
| input | `*.v2ep.pt` PNG 256×640 cylindrical → `v2_dataset._decode_stacked` (D-015 stack, current frame LAST) → `/255` by a 0-dim device tensor, fp32 | trainer code read; `refc_v3_train.py` has no autocast and no augmentation (grep) |
| tokens | stride-32 map **[704, 8, 20]** (`refc.py:3161`); max \|token\| 42.98; fp16 storage error ≤ 0.0156 | `tokens/index.npz` meta |
| stride-16 map (lever L1) | **[352, 16, 40]**, max \|token\| 171.07, fp16 error ≤ 0.0625 | cross-check: the checkpoint's last stage applied to the cached s16 rows reproduces the cached s32 rows to **mean abs 0.00023** vs **0.695** for mismatched rows (~3,000× separation) |
| trunk held-out | the 139 eval clips vs the 4,572 clips of the label release refcv5-v2 trained on (md5 `0ff90213…`, identical to its `config.json`): **overlap 0** | id sets compared, uuid-shape control read non-zero |
| rows | **27,664** stacked rows over 139 clips; extraction 50 rows/s on the RTX 4060 | content-asserted on the files (finite, non-degenerate) |

### 5.2 Split, head, controls (as pre-registered)

* **Split** — clip-disjoint, content-blind (sorted sha12 position): **82 train / 18 val / 34 test
  clips** = **16,199 / 3,553 / 6,713 rows**. Train marginal (rule B) **0.2800**.
* **Head** — 1,920 polar queries (48 × 40) + 2-D sine-cosine PE; KV = Linear(d_in→192) + PE;
  3 × pre-norm `TransformerDecoderLayer` (d 192, 6 heads, FF 768, dropout 0.1); LayerNorm +
  Linear(192→1) = **2,284,993 params** (the 09-11 sizing 2,284,802 + 384 − 193, exactly as the
  prereg predicted). ⚠️ The 09-11 sizing class had zero-initialised queries and no positional
  encoding; as written it could not have localised anything.
* **Training** — BCE on scored cells, AdamW 3e-4 / wd 0.01, 500 warm-up + cosine, batch 32, **6,000
  steps**, bf16, best-val checkpoint. Attention on the cuDNN SDPA kernel (MEASURED: flash
  unavailable on this build; math kernel pages at 11.56 GB; cuDNN 0.035 s vs efficient 0.204 s
  per attention fwd+bwd). Two arms sharing the GPU ran **0.70 s/step each** vs ~0.22–0.28 alone ⇒
  arms ran sequentially.
* **Controls in the same panel** — `const` (train marginal; must read the test prevalence exactly),
  `prior` (per-cell train marginal), `shuffled` (targets+masks deranged across train frames),
  `pixel` (same head on 9×64×160 pixels unfolded to 8×20 tokens of 576-d), `main_s1` (seed 1).

### 5.3 ⛔ The pre-registered panel — `E-BEVHEAD-FROZEN-1` **FAILS all four bars** (MEASURED, `raw/p4_panel.json`)

TEST split scored once: **34 clips, 6,713 rows, 7,928,485 scored cells** (rule B; rule C 3,538,789).
Test prevalence **0.2762** (rule C 0.1001). Estimator: paired clip-cluster bootstrap over test clips,
2,000 resamples, AP per resample from per-clip 2,001-bin histograms (binning error ≤ 1.2e-4 AP).
Which variance: the CI answers *another draw of test clips*; `main_s1` answers *another training
run*; the head is deterministic at inference.

| arm | d_in · tokens · head params | best val step | **test AP (rule B)** [95 % CI] | AP rule C | IoU @τ_val | **IoU 0–30 m** [95 % CI] |
|---|---|---:|---|---:|---:|---|
| `const` | — | — | 0.2762 [0.2459, 0.3076] | 0.1001 | 0.2762 | 0.2224 [0.1964, 0.2506] |
| `prior` | — | — | 0.3773 [0.3297, 0.4267] | 0.1453 | 0.3280 | 0.2941 [0.2594, 0.3323] |
| `shuffled` | 704 · s32 8×20 · 2,284,993 | 4,500 | 0.3801 [0.3322, 0.4286] | 0.1504 | 0.3272 | 0.2931 [0.2578, 0.3317] |
| `pixel` | 576 · pix 8×20 · 2,260,417 | 500 | 0.3849 [0.3317, 0.4307] | 0.1590 | 0.3266 | 0.2883 [0.2535, 0.3260] |
| `main_s1` | 704 · s32 8×20 · 2,284,993 | 1,000 | 0.3997 [0.3450, 0.4535] | 0.1725 | 0.3428 | 0.3135 [0.2768, 0.3516] |
| **`main`** | 704 · s32 8×20 · 2,284,993 | 1,000 | **0.4140 [0.3721, 0.4590]** | **0.1863** | **0.3514** | **0.3293 [0.2941, 0.3650]** |

By range band (AP; prevalence; n cells) — descriptive, no per-band interval:

| band | prevalence · n | `prior` | `shuffled` | `pixel` | **`main`** | `main − prior` |
|---|---|---:|---:|---:|---:|---:|
| 0–15 m | 0.149 · 1,800,899 | 0.2933 | 0.2944 | 0.2908 | **0.3704** | **+0.077** |
| 15–30 m | 0.277 · 2,396,156 | 0.3701 | 0.3754 | 0.3841 | **0.4231** | +0.053 |
| 30–45 m | 0.332 · 2,063,967 | 0.3919 | 0.3990 | 0.4076 | 0.4173 | +0.025 |
| 45–60 m | 0.342 · 1,667,463 | 0.3867 | 0.3932 | 0.3967 | 0.4033 | +0.017 |

**Gates — both hold, so the arms are readable:** **G1** `const` AP **0.2761858034668666** =
prevalence **0.2761858034668666** (exact). **G2** `shuffled` 0.3801 ≤ `prior` 0.3773 + 0.02.

| pair | Δ AP [95 % CI] | separated | bar |
|---|---|---|---|
| `main − shuffled` | **+0.0339** [+0.0055, +0.0626] | yes | **B1 ✗** (< +0.05 and < 3F = 0.0431) |
| `main − prior` | **+0.0367** [+0.0076, +0.0675] | yes | **B2 ✗** (< +0.05 and < 3F) |
| `main − pixel` | +0.0291 [−0.0040, +0.0693] | **no** | **B3 ✗** |
| `main − main_s1` (replicate floor F) | +0.0144 [−0.0092, +0.0361] | no | F = **0.0144** |
| ⚠️ `main_s1 − shuffled` | +0.0195 [−0.0155, +0.0564] | **no** | the replicate does NOT separate from zero information |
| `pixel − prior` | +0.0076 [−0.0179, +0.0293] | no | raw pixels through this head add nothing |
| `shuffled − prior` | +0.0028 [+0.0001, +0.0057] | yes | ≈ 0, as a zero-information head must |

**B4 ✗** — `main` AP 0.4140 < 0.60 and IoU 0–30 m 0.3293 < 0.45.

⇒ **VERDICT (as written): FAILED on B1, B2, B3 and B4.** Twin **F1** fires (B1/B2 fail) ⇒ committed
next lever **L1**. What the failure says, and no more: the frozen stride-32 tokens of refcv5-v2 give a
transformer head **+0.034–0.037 AP** over zero-information and positional baselines — separated, but
thin (one seed of two does not separate at all), concentrated in the near field (+0.077 at 0–15 m
decaying to +0.017 at 45–60 m), and **the validation curves of both seeds peak at step 1,000 of
6,000** — a head that overfits 82 clips. ⚠️ The validation curve over-promised: val AP 0.455 against
a val prior 0.362 (+0.09) became +0.037 on test; best-step selection on 18 val clips is optimistic.
The figures (`media/bevhead_main_00213e99adca_f100.png`, `…04ed850dca64_f100.png`; test frames chosen by
rule) show why: the prediction is a smooth scene-layout map — road free, both roadsides occupied — not
the individual structures in the LiDAR ground truth.

## 6. RULE ZERO — the levers after the failed frozen rung

### 6.1 The diagnosis that ranks the levers — VALIDATION only, measured before test was scored

`main` (frozen s32, 82 train clips): val AP **0.4424 / 0.4548 / 0.4361 / 0.4207 / 0.4134 / 0.4044 /
0.4033** at steps 500…3,500 while the training loss fell 0.518 → 0.354; best checkpoint step
**1,000** of 6,000. `shuffled`: val AP **0.3636 / 0.3638 / 0.3633** — pinned to the val per-cell
prior **0.362** (val prevalence 0.2717). ⇒ information is present **and the head overfits 82 clips
within the first sixth of its schedule**. That makes **data** and **inductive bias** the levers
aimed at the measured cause; **capacity** (L2, 51.6 M unfrozen parameters) aims away from it.
⚠️ **Lever order, recorded:** the twin that fired is **F1** (B1/B2 fail), whose ONE committed
next lever is **L1** — run first, as committed. Everything after it is POST-HOC: L4 (data, its own
prereg `PREREG_L4_MORE_TRAINING_CLIPS.md`), then L3, then L2 — ordered by the validation evidence
(L2 adds the most capacity to a head that already overfits), not by the F2/F4 lists. The bars never
change, and every post-hoc verdict is labelled as such in its JSON.

### 6.1b ⛔ L1 — frozen stride-16 tokens: **FAILS, and does not move the metric** (POST-HOC lever, `raw/p4_panel_levers.json`)

Same head over the stage-3 map **[352, 16, 40]** (640 tokens; head 2,217,409 params), same protocol,
scored on the same 34 test clips in the same panel as the four pre-registered arms:

| | test AP (rule B) [95 % CI] | AP rule C | IoU 0–30 m | bands AP 0–15 / 15–30 / 30–45 / 45–60 m |
|---|---|---:|---:|---|
| `main` (s32) | 0.4140 [0.3721, 0.4590] | 0.1863 | 0.3293 | 0.370 / 0.423 / 0.417 / 0.403 |
| **`s16`** | **0.4074 [0.3533, 0.4601]** | 0.1633 | 0.3112 | 0.326 / 0.402 / 0.420 / 0.420 |

`s16 − main` **−0.0066 [−0.0254, +0.0125]** (not separated); `s16 − shuffled` +0.0273 [−0.0026,
+0.0598], `s16 − prior` +0.0301 [−0.0006, +0.0644], `s16 − pixel` +0.0225 [−0.0169, +0.0680] — **none
separated**. Bars B1–B4 all ✗ (floor borrowed from `main`/`main_s1`, stated in the JSON). Validation
peaked at step **500** (0.448) and fell to 0.398 by step 1,500 — it overfits sooner than s32.
⇒ **Resolution of the frozen features is not the missing ingredient**; the stride-16 map even gives
up the near field (0–15 m 0.326 vs 0.370). This is the second arm whose best checkpoint sits in the
first twelfth of its schedule.

### 6.2 L4 — can the dev box make more training data? (MEASURED)

* **Frames from raw mp4** (`code/p6a_frames_equivalence.py` → `raw/p6a_frames_equivalence.json`):
  the `v2_compressed` path rebuilt locally reproduces the pod-built PNG frames to **max 3 levels,
  mean 0.75 levels** (27–29 % of values exactly equal, 96–97 % within 1 level) on 2 eval clips, at
  **2.3–2.8 s per clip**. ⛔ Not bit-identical ⇒ gate G5 measures the effect on the TOKENS before
  any head trains on rebuilt frames (`code/p6c_token_equivalence.py`).
* **Selection** (`code/p6b_extra_corpus.py` → `raw/p6b_extra_selection.json`): the v7.2 TRAIN label
  release (4,572 clips, md5 `0ff90213…`) → 4,571 fully available locally (1 rejected: a protected
  Qwen-Drive clip) → sorted by sha12 → first **200**. Eval clips and their split untouched.
* **Missing LiDAR is real, measured twice** (`code/p6d_missing_lidar_probe.py` →
  `raw/p6d_missing_lidar_probe.json`): **8 of the 200** selected clips have no LiDAR member in
  their chunk zip; a substring search of each zip's full central directory finds **no member
  containing the clip id under any spelling**, while other train clips of the same chunk are found
  under the expected name in the same read (3/3, 2/2, 6/6, 33/35, 19/20; three chunks' controls are
  vacuous — no other train clip lives there). Consistent with the dataset card's LiDAR coverage
  (298,326 / 306,152 clips, PUBLISHED, not re-measured).
* **The extra corpus, built** (`raw/p6b_extra_manifest.jsonl`, sha12 + sha256 per artifact; files on
  the C: cache and mirrored to `D:\Projects\TanitAD-artifacts\bev-lidar-gt-b1train200-20260913\`,
  **181/181 mirror hashes match**): of the 200 selected train
  clips **181 OK** (145,575,368 B, 35,815 labelled instants, rule-B marginal median 0.3045); **19
  failed** = 8 with no LiDAR member, 8 ground-plane quarantines (the same scene-fragile check as §3.4),
  2 near-empty grids, 1 observed fraction below 0.30. 5,167 s wall at fetch c = 4 / build 4 workers,
  while the pre-registered panel trained. Raw parquets deleted afterwards except the 4 protected.
* ⛔ **Gate G5 FAILED** (`code/p6c_token_equivalence.py` → `raw/p6c_token_equivalence.json`): tokens
  of rebuilt frames differ from tokens of pod frames by **1.84 % / 1.73 %** relative mean-abs (bar
  ≤ 1 %; wrong-row control **65.8 % / 56.6 %**, so the comparison discriminates). The cheapest
  experiment that could have closed it — PyAV YUV→RGB variants (`code/p6e_decode_variants.py`) —
  did not: all six interpolation flags and ITU601 give the identical 27.09 % exact / 0.772 mean
  match, ITU709 is worse (mean 1.118, max 34); HEVC decoding is bit-exact by spec and `calib.py`'s
  code is unchanged since the payloads were built ⇒ a library-version residual on the pod. ⇒ **the
  L4 arm as first registered was not run** (`code/run_l4.sh`, kept as the record, refuses without
  G5) and **amendment A1** (`PREREG_L4_MORE_TRAINING_CLIPS.md` §4, written before any L4 arm ran) puts
  **every split on locally rebuilt frames** with a same-distribution baseline `main_rb`
  (`code/run_l4rb.sh`). The 139 eval clips were re-rendered for it: 139/139 in 482 s
  (`raw/p6b_rebuild_eval_frames.json`).
* ⚠️ **A trap hit here, and its cost:** a 20-step smoke test of the L4 code path started on the GPU
  while a panel arm trained drove GPU memory to **7.73 / 8.19 GB** and free host RAM to 2.1 GB;
  both processes crawled (the panel arm's cumulative step time rose to 0.58 s). The smoke processes
  were killed by PID; the panel arm recovered to 0.30 s/step with its results unaffected (only its
  timing). ⇒ on this 8 GB card, **no second GPU job beside a training arm — not even a "tiny" one.**

### 6.2b ⛔ L4 (amendment A1) — 3.2× the training rows: the committed data criterion is NOT met, but the head crosses B1/B2 (POST-HOC, `raw/p4_panel_L4_data_rb.json`, `raw/p4_panel_rb_shift.json`)

Every split on rebuilt frames; same 34 test clips, 7,928,485 cells; training rows **51,908** =
16,199 (82 eval-fit clips) + **35,709** (181 B1 train clips). Estimator as §5.3.

| arm | train rows | best val step · val AP | **test AP (rule B)** [95 % CI] | AP rule C | IoU 0–30 m | bands AP 0–15 / 15–30 / 30–45 / 45–60 m |
|---|---:|---|---|---:|---:|---|
| `const` | 51,908 | — | 0.2762 | 0.1001 | 0.2224 | 0.149 / 0.277 / 0.332 / 0.342 |
| `prior` (enlarged train) | 51,908 | — | 0.3778 [0.3323, 0.4243] | 0.1457 | 0.2940 | 0.295 / 0.367 / 0.396 / 0.388 |
| `shuffled_data_rb` | 51,908 | — | 0.3801 [0.3339, 0.4273] | 0.1470 | 0.2940 | 0.295 / 0.368 / 0.400 / 0.391 |
| `main_rb` (82 clips, rebuilt) | 16,199 | 1,000 · 0.452 | 0.4173 [0.3702, 0.4657] | 0.1799 | 0.3285 | 0.373 / 0.427 / 0.424 / 0.402 |
| `data_rb_s1` | 51,908 | 1,500 · 0.492 | 0.4319 [0.3940, 0.4722] | 0.2283 | 0.3384 | 0.399 / 0.437 / 0.438 / 0.423 |
| **`data_rb`** | 51,908 | **3,000 · 0.491** | **0.4352 [0.3976, 0.4737]** | **0.2296** | **0.3400** | **0.402 / 0.446 / 0.437 / 0.422** |

| pair | Δ AP [95 % CI] | separated | reading |
|---|---|---|---|
| **`data_rb − main_rb`** (the L4 criterion) | **+0.0179** [−0.0063, +0.0402] | **no** | ⛔ **NOT SUPPORTED — REFUTED as written** (needs ≥ +0.03, CI excluding 0) |
| `data_rb − data_rb_s1` | +0.0033 [−0.0192, +0.0208] | no | floor **0.0033** (was 0.0144 at 82 clips — **4.4× tighter**) |
| `data_rb − shuffled_data_rb` | **+0.0552** [+0.0231, +0.0866] | yes | **B1 ✓** — ≥ +0.05 and ≥ 3 × max(0.0144, 0.0033) = 0.0431 |
| `data_rb − prior` | **+0.0574** [+0.0258, +0.0885] | yes | **B2 ✓** |
| `data_rb_s1 − shuffled_data_rb` | +0.0519 [+0.0134, +0.0920] | **yes** | ⭐ the replicate now separates (at 82 clips it did not) |
| `main_rb − main` (rebuilt vs pod frames, same 82-clip arm) | **+0.0033** [−0.0029, +0.0086] | no | ⭐ the 1.8 % token shift G5 blocked on does NOT move test AP |
| gates | G1 exact (0.2761858…); G2 0.3801 ≤ 0.3778 + 0.02 | — | readable |

**B4 ✗** (0.4352 < 0.60; IoU 0–30 m 0.3400 < 0.45). **B3 NOT RE-TESTED** (no pixel arm on 51,908 rows).
⇒ **As written:** the pre-registered data criterion FAILS — +0.018 AP from 3.2× the rows is not
separable from zero with 34 test clips. **What more data DID do, measured:** it moved the best
validation step from 1,000 to 1,500–3,000, tightened the seed floor 4.4×, lifted the rule-C
(first-hit) AP from 0.180 to 0.230 (point estimates, no interval), and took the head across the pre-registered B1/B2 bars
on two seeds — the first configuration in this package to do so. It does not come close to B4.
⚠️ Cross-distribution, informational only: `data_rb` 0.4352 vs the pre-registered `main` 0.4140 on pod
frames (+0.021) — legitimate to quote only because `main_rb − main` measured +0.0033.

### 6.3 L2 — a defect caught BEFORE it ran

`FinetuneLastStage` froze BatchNorm statistics only inside its `train()` override, but a PyTorch
module is constructed in training mode and the trainer first calls `train()` after the step-500
validation pass. ⇒ for 500 steps the "frozen" BN would have used batch statistics and **updated its
running stats**. MEASURED on the module: fresh construction → BN `training = True`; after the added
construction-time `train()` → `False`, and `running_mean` is bit-unchanged by a train-mode forward.
The trainer now calls `train()` at construction and asserts the BN state. (`code/p4_bev_head.py`)

### 6.4 ⛔ L3 (azimuth-aligned attention) and L2 (unfrozen last stage) — both FAIL, neither separates from `main` (POST-HOC, `raw/p4_panel_levers_L1L2L3.json`)

Pod frames, 82 training clips, scored in ONE panel with the four pre-registered arms and L1 on the
same 34 test clips. L3 = `main` + a cross-attention mask letting each polar query attend only to
token columns within 9° (+ half a column) of where its cell centre projects through the nominal rig
camera (blocks 70 % of query–token pairs; mask literals verified on known cells). L2 = the
checkpoint's last ResNet stage (**51,562,368** params) UNFROZEN on cached stride-16 features, BN
statistics frozen (verified), same head.

| arm | best val step · val AP | **test AP (rule B)** [95 % CI] | Δ vs `main` [95 % CI] | vs `shuffled` | vs `prior` | vs `pixel` | AP rule C | IoU 0–30 m | bands AP 0–15 / 15–30 / 30–45 / 45–60 |
|---|---|---|---|---|---|---|---:|---:|---|
| `main` | 1,000 · 0.455 | 0.4140 [0.3721, 0.4590] | — | +0.0339 ✓ | +0.0367 ✓ | +0.0291 ✗ | 0.1863 | 0.3293 | 0.370 / 0.423 / 0.417 / 0.403 |
| **L3 `main_geo`** | 500 · 0.482 | **0.4172** [0.3675, 0.4659] | +0.0032 [−0.0100, +0.0170] ✗ | +0.0371 ✓ | +0.0399 ✓ | +0.0323 ✗ | **0.2015** | 0.3233 | 0.350 / 0.433 / 0.423 / 0.408 |
| **L2 `s32ft`** | 500 · 0.472 | **0.4248** [0.3691, 0.4786] | +0.0107 [−0.0092, +0.0353] ✗ | +0.0447 ✓ | +0.0475 ✓ | +0.0399 ✗ | 0.1863 | 0.3283 | 0.368 / 0.433 / 0.431 / 0.415 |

(✓ = separated, ✗ = not.) Bars B1–B4 ✗ for both (B1/B2 deltas < +0.05; floor borrowed from
`main`/`main_s1`). Both validation curves peak at step **500** — the earliest of any arm. ⚠️ L2's
config records `d_in 352` / grid 16×40: that is its INPUT; the unfrozen stage maps it to [704, 8, 20]
before the head.

### 6.5 Where RULE ZERO stops, and why — every lever measured against the same baseline

| lever | what changed | test Δ AP vs its baseline [95 % CI] | separated | clears B4 (≥ 0.60)? |
|---|---|---|---|---|
| L1 | frozen stride-16 tokens | −0.0066 [−0.0254, +0.0125] | no | no (0.4074) |
| L3 | azimuth-aligned attention prior | +0.0032 [−0.0100, +0.0170] | no | no (0.4172) |
| L2 | unfreeze the last stage (51.6 M) | +0.0107 [−0.0092, +0.0353] | no | no (0.4248) |
| **L4** | **3.2× training rows** | **+0.0179 [−0.0063, +0.0402]** | no — but **B1/B2 cleared on two seeds** | no (0.4352) |

⇒ **The result does not clear its committed usefulness bar, and I say so.** B4 needs **≥ +0.165 AP**
over the best measured arm (0.4352; +0.186 over `main`); the four 4060-feasible levers measured **−0.007 … +0.018** each. The
largest measured effect is **data** (RULE ZERO item 5), and the levers that could plausibly deliver a
jump of that size are **blocked**, each named with what unblocks it:

| remaining lever | status | what unblocks it |
|---|---|---|
| **Full-scale data** — LiDAR GT for all 4,571 locally available B1 train clips (25× what L4 used) | ⛔ BLOCKED on transfer/compute | ~1.55 TB ≈ **34 h** at the MEASURED 12.6 MB/s; a bandwidth-rich pod — PI authorisation |
| **Joint BEV-aux trunk training** (`E-BEVLIDAR-CAP-1`, 09-11 prereg) | ⛔ BLOCKED on compute | multi-day on pod GPUs; the 8 GB 4060 cannot train the 108 M trunk — PI authorisation |
| Combinations on the 4060 (L2 + L4, L3 + L4) | ⚠️ runnable, deliberately NOT opened | measured effects are ≤ +0.018 each and not separated, so even an additive best case (~+0.03) leaves AP ≈ 0.45 against 0.60; the PI asked (dev box at ~97 % committed memory) to finish and bank rather than open marginal arms |

## 7. What this package does and does NOT show

**Does show (MEASURED):**
* A LiDAR BEV ground truth for **134 of the 139** B1 EVAL clips (+ **181** B1 TRAIN clips), on the
  grid the head reads, with the third state, units and deskew sign in every file, registered to an
  independent source (real boxes 3.23× their mirror, 4.29× the marginal), verified by a 10-test suite
  that goes RED on a mirrored or mis-sized artifact.
* Two defects in the 09-11 builder that published no wrong number but would have corrupted every
  target built on it during turns (deskew sign) and every polar head scored on it (observed rule).
* The **frozen** refcv5-v2 trunk exposes only a **thin** BEV occupancy signal to a transformer head:
  +0.034–0.037 AP over zero-information baselines on test, one seed of two not separated, gain
  concentrated in the first 15 m. Stride-16 features do not change that.
* Of four levers runnable on the dev box, **data** has the largest measured effect: 3.2× the rows
  takes the head across the pre-registered B1/B2 bars on two seeds (AP 0.4352) and tightens the seed
  floor 4.4×; an azimuth attention prior and an unfrozen last stage do not separate from the baseline.

**Does NOT show:**
* ⛔ Nothing about capacity competition or JOINT training (`E-BEVLIDAR-CAP-1`) — the trunk is frozen
  here, so "the trunk cannot carry BEV" is NOT established; "the trunk trained for planning does not
  already expose it" is.
* ⛔ No driving result, no T0/T1 tier, no planner metric family — this is a representation probe.
* ⛔ Not a level on any parity corpus: B1 EVAL, NON-PARITY.
* ⚠️ The validation curve is not a test result on this rig: best-step validation margins over-state
  the test margins by 2.5–4.5× (`raw/p4_val_reference.json`).

## 8. Escalations — decisions and integrations that will not surface on their own

1. ⭐ **PI decision — data is the only lever that moved this head, and the dev box cannot scale it.**
   181 extra clips took the frozen-trunk head across B1/B2 on two seeds; the full B1 train set is
   **4,571** locally available clips (25× what was used) ≈ **1.55 TB** of LiDAR at the MEASURED
   12.6 MB/s ⇒ **~34 h** of transfer here. A bandwidth-rich pod (the HF relay measured ~118 MB/s in
   the programme's own notes, INHERITED) turns that into hours, but it is **pod compute the brief did
   not authorise**. Named, not started.
2. ⭐ **PI decision — joint training is where the answer lives.** The frozen trunk exposes only a thin
   BEV signal (§5.3), so a DiffusionDrive-style BEV branch needs the trunk TRAINED with BEV
   supervision — the `E-BEVLIDAR-CAP-1` ladder (09-11 pre-registration, capacity-competition
   detector included). Multi-day on pod GPUs; not runnable on the 4060.
3. **PI decision — publishing.** The GT sets (134 eval clips 109.8 MB + 181 train clips 145.6 MB) are
   derived from gated PhysicalAI-AV data; programme policy puts augmented sets on the PI's PRIVATE HF
   account. Not uploaded: publishing is the PI's call, and the HF quota was not checked.
4. ⛔ **Master Mind — integration of the 09-11 package.** Its `code/lidar_bev.py` carries two measured
   defects (deskew rotation sign; polar `observed` rule) — `RETRACTION_LOG.md` 2026-09-13. When that
   package lands, its RESULT needs a pointer to this correction; its banked code is deliberately not
   edited here. The corrected module is this package's `code/lidar_bev.py`.
5. **Master Mind — builder change for the next GT build.** The ground-plane content check is
   scene-fragile (5/139 eval and 8/200 train trips, the eval ones classified as scene effects in
   §3.4). Replace the argmax mode with "z p5 over 3–40 m within ±0.2 m, or a ground mode in the
   top-3". Not changed here, because this build's gate was declared before its data.
6. **Stream owners — two things other agents should know.** (a) Frames rebuilt on the dev box from
   raw mp4 are ~1.8 % token-distant from the pod-built frames (G5), which did NOT matter for this
   BEV head (`main_rb − main` +0.0033) but has not been checked for a planner. (b) The Research Lab's
   LAB-RUN-012 section in `GOALS_AND_CLAIMS.md` sat UNSTAGED in the worktree when this package staged
   its own rows; it was deliberately left unstaged (staged by blob, `update-index --cacheinfo`).

## 9. Deliverable manifest

⛔ Everything below marked `repo:` is **STAGED, not committed, not pushed**, on `agent/arch-inf-20260803`
in `D:\Projects\TanitAD` (each path added by exact name and verified index-blob = worktree-blob).
`repo:` paths are relative to `TanitAD Research Lab/Architecture & Inference/Research/2026-09-13-bev-lidar-corpus-and-head/`
unless they start with `Project Steering/`.

| artifact | where it lives |
|---|---|
| this result · both pre-registrations | `repo: RESULT.md`, `repo: PREREG_BEVHEAD_FROZEN_TRUNK.md`, `repo: PREREG_L4_MORE_TRAINING_CLIPS.md` |
| corrected target builder (deskew sign, three polar observed rules) | `repo: code/lidar_bev.py` |
| ranged LiDAR fetch · bandwidth probe | `repo: code/lidar_fetch.py`, `repo: code/p1_bandwidth.py` |
| deskew sign experiment | `repo: code/p0_deskew_sign_check.py` |
| corpus builder · observed-rule probe · finaliser · ground-outlier probe | `repo: code/p2_build_corpus.py`, `code/p2b_observed_rule_probe.py`, `code/p2c_finalize_corpus.py`, `code/p2d_ground_outlier_probe.py` |
| loader · tests · mutation run · corpus orientation | `repo: code/bev_gt_loader.py`, `code/test_bev_gt_artifact.py`, `code/p3_mutation_run.py`, `code/p3_orientation_all.py` |
| token extractor · head trainer · evaluator · val reference · renderer · drivers | `repo: code/p4a_extract_tokens.py`, `code/p4_bev_head.py`, `code/p4_eval.py`, `code/p4b_val_reference.py`, `code/p5_render.py`, `code/run_panel.sh`, `code/run_levers.sh`, `code/run_l4.sh` (superseded, never run), `code/run_l4rb.sh`, `code/run_l3_l2.sh` |
| L4 data tooling | `repo: code/p6a_frames_equivalence.py`, `code/p6b_extra_corpus.py`, `code/p6c_token_equivalence.py`, `code/p6d_missing_lidar_probe.py`, `code/p6e_decode_variants.py` |
| raw results (all sha12-only) | `repo: raw/p0_deskew_sign_check*.json`, `raw/p1_bandwidth.json`, `raw/p2_corpus_summary.json`, `raw/p2_corpus_manifest.jsonl`, `raw/p2b_observed_rule_probe.json`, `raw/p2d_ground_outlier_probe.json`, `raw/p3_mutation_run.json`, `raw/p3_orientation_all.json`, `raw/p4_panel.json`, `raw/p4_panel_levers.json`, `raw/p4_panel_levers_L1L2L3.json`, `raw/p4_panel_L4_data_rb.json`, `raw/p4_panel_rb_shift.json`, `raw/p4_val_reference.json`, `raw/p6a_frames_equivalence.json`, `raw/p6b_extra_selection.json`, `raw/p6b_extra_manifest.jsonl`, `raw/p6b_rebuild_eval_frames.json`, `raw/p6c_token_equivalence.json`, `raw/p6d_missing_lidar_probe.json`, `raw/p6e_decode_variants.json` |
| 2 sample GT artifacts (tests run from the repo alone) | `repo: raw/sample_gt/{00213e99adca,04ed850dca64}.bevgt.npz`, `raw/sample_gt/manifest.jsonl` |
| figures (test frames, chosen by rule) | `repo: media/bevhead_main_00213e99adca_f100.png`, `repo: media/bevhead_main_04ed850dca64_f100.png` |
| register rows | `repo: Project Steering/GOALS_AND_CLAIMS.md` (this package's hunks only), `repo: Project Steering/RETRACTION_LOG.md` |
| ⚠️ **GT set, 134 eval clips** (109,767,615 B) | **`devbox C:\Users\Admin\tanitad-caches\bevhead-20260913\bev_gt\`** + **`devbox D:\Projects\TanitAD-artifacts\bev-lidar-gt-b1eval-20260913\`** (134/134 sha256 match) — NOT in git: a derived dataset (repo policy: never commit datasets); private-HF publish is a PI decision |
| ⚠️ **GT set, 181 train clips** (145,575,368 B) | **`devbox C:\…\bevhead-20260913\bev_gt_extra\`** + **`devbox D:\Projects\TanitAD-artifacts\bev-lidar-gt-b1train200-20260913\`** (181/181 sha256 match) — same reason |
| ⚠️ quarantined GT (5 eval) · superseded rule-A run | **`devbox C:\…\bevhead-20260913\bev_gt\quarantine\`**, **`…\bev_gt_ruleA_superseded\`** — ONE PLACE, kept for a sensitivity row; regenerable |
| ⚠️ trunk tokens (s32, s16, pixels; pod + rebuilt + extra), frames payloads, trained heads (`runs/*/best.pt`), per-arm logs and predictions | **`devbox C:\Users\Admin\tanitad-caches\bevhead-20260913\`** (**52 GB**) — ONE PLACE, deliberately not staged: regenerable from the staged code + the refcv5-v2 checkpoint (~1 GPU-hour for tokens, ~30 min per head); head checkpoints are 8.9–9.2 MB each (L2's 215.5 MB, it carries the unfrozen stage) and every number they produced is banked in `raw/` |
| raw LiDAR parquets | deleted after each build (re-fetchable); only the 4 protected Qwen-Drive parquets remain, byte sizes unchanged |
