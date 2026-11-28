# RESULT — refcv6's map head per class at step 38,000: did it learn the markings and lose the argmax, or not learn them at all?

**First line:** the head **learned the markings partly and loses the argmax**. This holds on the
REPRESENTATIVE in-run eval windows (n = 127 windows over 79 clips). Lane lines rank at **AP 0.156 against
a 0.013 base rate** (12×) and 0.020 for a scene-blind location prior (8×). Crosswalks rank at **0.081
against 0.004** (20×). Yet argmax IoU is **0.020** and **0.009**, because mean p on a true lane cell is
only **0.104**. On the three crowded boxes-video clips the ranking collapses: lane AP 0.028, 3× base. So
the signal is real, weak and scene-dependent, and it never reaches the decision.

EvalFlyWheel · 2026-09-26/27. PI (Sayed), verbatim: *"the drivable area are little bit ok, but our model is
seldomly detecting other semantic classes, no roadmarkings, no roadmarking infrastructure in intersections
etc... We need to analyze it"*.

Stamp on every number: **refcv6-r101-s0 step 38,000; hybrid: F3 + true label clock from 34,500; trunk C26
equalisation OFF (declared 43, dropped).**
- ckpt `D:/refcv6_eval_kit/ckpt_final/ckpt_step38000.pt`, md5 `5a2e7222a9f5f8c7aa7bf38ef4698d8a`.
- Tier: an open-loop perception readout. No driving claim.
- Evidence class: MEASURED. No interval is quoted, because windows within a clip are correlated.

## 0. The data: one model load, a per-cell dump

The dump was written by `taniteval/tools/render_refcv6_map_video.py --boxes --dump-map`, in the same GPU
process as the boxes video. That was one model load, and it was not retrained.
- Location: `C:/Users/Admin/qland/work/mapvid/mapdump_step38000/` (dev-box local disk ONLY; 224 MB; index
  with per-file md5 in `raw/dump_index.json`).
- **1,000 windows over 90 clips**, by set membership:
  - `inrun_eval` 127: the trainer's own fixed eval perm, `refcv6_loader.inrun_eval_perm`, seed 12345. The
    128th pick has no SAM3 file.
  - `boxes_clip` 513: the 3 boxes-video clips.
  - `map_clip` 513: the 3 map-video clips.
  - `inrun_perm_ext` 20: continuing the same permutation.
- Per window:
  - softmax(map_logits) stored as float16 [9, 120, 64];
  - the GT fractions as the label file's uint8;
  - `map_seen`, `map_valid`, and the raw GT frame.

## 1. The AP probe: ranking (AP) against decision (argmax IoU), at 0.5 m and 10 cm

Tool: `code/ap_probe_38k.py`. It follows the discriminating experiment pre-registered in
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires/RESULT.md` §3.
- **0.5 m.** On seen cells: AP of p_c against `cart_frac ≥ 0.5`, and AP against presence (`> 0`), with
  their base rates, mean p on positives and negatives, and argmax IoU.
- **10 cm.** The 0.5 m softmax, nearest-upsampled to 600 × 320, scored against `fine_codes == c` for
  codes other than 255, in bands 0-20 / 20-40 / 40-60 m. It is computed exactly from per-coarse-cell
  counts.

### 1a. REPRESENTATIVE: the 127 in-run eval windows (79 clips) — `raw/ap_probe_38k_inrun_eval.json`

n: 938,578 seen 0.5 m cells; 23,464,284 seen 10 cm cells (bands 8.0 M / 8.1 M / 7.4 M).

| class | AP 0.5 m, frac ≥ 0.5 (base · n_pos) | location prior | AP presence (base) | AP 10 cm 0-20 / 20-40 / 40-60 m (base) | argmax IoU | mean p pos / neg |
|---|---|---|---|---|---|---|
| drivable | **0.895** (0.345 · 323,850) | 0.718 | 0.922 (0.383) | 0.916 / 0.893 / 0.842 (0.343) | **0.674** | 0.703 / 0.127 |
| sidewalk / verge | **0.812** (0.397 · 372,712) | 0.496 | 0.833 (0.429) | 0.838 / 0.841 / 0.731 (0.397) | **0.562** | 0.615 / 0.205 |
| seen, no map class | 0.720 (0.231 · 216,548) | 0.378 | 0.726 (0.265) | 0.773 / 0.738 / 0.633 (0.232) | 0.507 | 0.616 / 0.176 |
| **lane / road line** | **0.156** (0.013 · 12,548) | 0.020 | **0.330** (0.037) | 0.206 / 0.164 / 0.113 (0.016) | **0.020** | **0.104** / 0.015 |
| **crosswalk** | **0.081** (0.004 · 4,091) | 0.005 | 0.132 (0.009) | 0.166 / 0.033 / 0.040 (0.005) | **0.009** | 0.040 / 0.004 |
| non-drivable edge | — (0.000 · 89) | — | 0.169 (0.034) | 0.040 / 0.035 / 0.021 (0.006) | 0.000 | 0.018 / 0.006 |
| arrow / text | 0.005 (0.001 · 569) | 0.001 | 0.012 (0.002) | 0.006 / 0.007 / 0.003 (0.001) | 0.000 | 0.003 / 0.001 |
| hatched area | 0.004 (0.001 · 551) | 0.001 | 0.007 (0.001) | 0.001 / 0.004 / 0.015 (0.001) | 0.000 | 0.003 / 0.001 |

### 1b. The 3 CROWDED boxes-video clips (513 windows) — `raw/ap_probe_38k.json`, the probe as briefed

n: 3,647,942 seen 0.5 m cells; 91,200,139 seen 10 cm cells.

| class | AP 0.5 m, frac ≥ 0.5 (base) | location prior | AP presence (base) | AP 10 cm 0-20 / 20-40 / 40-60 m | argmax IoU | mean p pos / neg |
|---|---|---|---|---|---|---|
| drivable | 0.851 (0.377) | 0.680 | 0.893 (0.420) | 0.896 / 0.859 / 0.746 | 0.599 | 0.586 / 0.147 |
| sidewalk / verge | 0.627 (0.215) | 0.223 | 0.665 (0.250) | 0.757 / 0.620 / 0.358 | 0.358 | 0.453 / 0.179 |
| lane / road line | 0.028 (0.009) | 0.009 | 0.126 (0.040) | 0.063 / 0.046 / 0.037 | 0.000 | 0.024 / 0.010 |
| crosswalk | 0.035 (0.013) | 0.013 | 0.072 (0.030) | 0.076 / 0.022 / 0.020 | 0.003 | 0.021 / 0.007 |
| arrow / text | 0.007 (0.002) | 0.002 | 0.025 (0.006) | 0.014 / 0.008 / 0.007 | 0.000 | 0.003 / 0.001 |
| non-drivable edge | — (714 pos) | — | 0.070 (0.020) | 0.020 / 0.013 / 0.008 | 0.000 | 0.008 / 0.005 |

### 1c. Controls (each must read a known value; all passed on both sets)

| control | expected | measured |
|---|---|---|
| constant predictor (the class frequency) | AP == base rate to 3 dp | True on every class with positives, in every 10 cm band |
| dump fractions vs the label file's `cart_frac` at the dumped raw frame | 0 cells differ | 0 |
| `cart_frac` vs the 5 × 5 block fraction of `fine_codes` | 0 cells differ, within the census's 1.5/255 | 0 |
| `map_seen` vs "not-seen fraction < 0.5" | 0 cells differ | 0 |
| cell-permuted model scores | ≈ base | lane 0.013 vs 0.013 (in-run) |
| **scene-blind floor**: per-cell class frequency from every OTHER clip (leave-own-clip-out) | the bar a scene-blind head reaches | quoted per row above |

⚠️ **The shuffled-GT control is NOT a pure null for spatially structured classes.** Drivable scores 0.620
against a 0.357 base on another clip's GT (in-run), because drivable lies ahead in every scene. That is
why the leave-own-clip-out location prior is the floor quoted beside each AP.

## 2. Reading the probe (pre-registered branch, map-hires RESULT §3)

The pre-registration read: *"If lane-line AP is far above its 1.4 % base rate, the model SEES the lines
and the target/decision rule hides them. If AP is near the base rate, the model never learned them."*

On the representative windows, lane AP is 0.156, 12 × base and 8 × the location prior. That is the FIRST
branch, weakly.

The mechanism the arithmetic allows, a HYPOTHESIS that the calibration table in §3 tests:
- The 0.5 m target is a MIXED cell. The median lane fraction in lane cells is 0.40 (map-hires census).
- The unweighted soft CE (`bev_encoder.map_soft_ce`, `:335-343`, no class weight) is minimised by
  **q = the cell's fraction** (Gibbs' inequality). For the literal 25 % lane / 75 % drivable cell, the
  optimum is q_lane = 0.25. H(p) = 0.5623 nats. Saying "lane 0.6" costs +0.4684 nats.
- ⇒ **Even a perfect head's argmax shows no lane in most lane cells.** The decision rule (argmax, or
  p ≥ 0.5) cannot reveal a class that is a minority in its own cells.
- The weak ranking (0.156, not ~0.6) says the head is ALSO under-trained on the classes, not only
  out-voted.

**Independent agreement.** The map-signal audit
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/RESULT.md`) measured
the same checkpoint differently: 16 fixed eval windows, cross-entropy per unit of class mass against a
GT-shuffled control, on Thor.
- It found a mean probability of **0.11** on lane cells and 0.058 on crosswalk cells.
- The AP probe reads **0.104** and 0.040 on 127 windows.
- Two estimators, two window sets, the same regime.

## 3. Full class analysis, 1,000 windows — CANCELLED (never ran)

`code/map_class_analysis.py` is ready. It was smoke-tested on the 4-window GPU-smoke dump, and every
control there read its analytic value: constant AUC 0.5, GT-fraction-as-score AUC 1.0, shuffled ≈ 0.5.

It adds:
- a **calibration-by-GT-fraction** table: mean p_c in fraction bins 0 / (0, .25) / [.25, .5) / [.5, .75) /
  [.75, 1];
- range bins 0-10 / 10-20 / 20-40 / 40-60 m;
- intersection (> 30° turn) against straight (≤ 5°);
- lift reach per range, and the resolution table from the lift's own grid;
- the leave-own-clip-out location-prior AUC and AP;
- an R-precision at 0 / 1 / 2 cell tolerance against a random and a prior comparator.

**It never ran.** Its gated chain waited for 7.5 GB of free host RAM, which never held (free RAM fell to
1.1 GB under other agents' jobs). The Master Mind then CANCELLED it, together with the AP probe on the 3
map-video clips, at 2026-09-27 ~02:15 Berlin to save API budget: the map-signal audit already holds the
root-cause ranking. The script stays in `code/` for the refcv7 per-class gate. No output of it exists.

## 4. The loss (code facts, 82c2331)

- **In-run call.** `refc_v3_train.py:4477-4481` → `_perc.map_loss_row(map_logits, map_frac, map_seen,
  lift_valid=map_valid)`.
- **`map_loss_row`** (`refcv6_perception_branch.py:513-573`) calls `map_soft_ce(logits, frac, m_seen)` at
  `:559`, with **no class weight**.
- **`map_soft_ce`** (`bev_encoder.py:269-362`):
  - p = frac renormalised to sum 1 (`:334`);
  - loss = the mean over scored cells of −Σ_c p_c log softmax(logits)_c (`:335-343`);
  - `class_weight=None` takes the unweighted branch (`:340-342`).
- **Its own docstring** (`:287-307`) records the imbalance: label mass is 97.05 % sidewalk + drivable +
  no-class, and arrow/text is 0.067 %. It warns: *"A head can reach a good soft CE while never predicting
  five classes."*

## 5. Resolution path (code facts)

- **MapHead.** A single 1×1 conv, 96 → 9 (`bev_encoder.py:219-240`), 873 parameters.
- **BEVEncoder.** Stride-1 at 120 × 64 (`bev_encoder.py:154-216`).
- **BEVLift** (`bev_lift.py:247-270`). A bilinear `grid_sample` of the **stride-16** trunk map (416 × 1024
  → 26 × 64 features) at 4 heights (0, 0.5, 1.5, 2.5 m). So every BEV cell blends 16-pixel features.
- **Lateral footprint of one feature column:** 65 cm at 20 m, 1.3 m at 40 m. A 15 cm lane line is a
  sub-feature object beyond about 10 m. (Map-hires RESULT §3, from f = 488.9 px.)

## 6. Ranked root causes: the map-signal audit's ranking, and where this package's evidence lands

The map-signal audit ranked the causes by MEASURED size (its §8). This package does not re-derive them.
It adds INDEPENDENT evidence per cause:

| # | cause (audit §8) | this package's evidence |
|---|---|---|
| 1 | **Localisation + soft-target argmax.** The stride-16, 0.5 m lift cannot place a 10-30 cm marking, and the soft 0.5 m target plus argmax gives thin classes about 0. | CONSISTENT. The head RANKS lane and crosswalk cells well above chance (in-run AP 12× and 20× base, 8× and 16× the location prior), yet mean p on its own cells is 0.104 and 0.040. The 10 cm AP falls with range: lane 0.206 → 0.164 → 0.113 over 0-20 / 20-40 / 40-60 m. |
| 2 | **The 0.5 m target erases the edge class** (0.9 % of its area survives). | CONSISTENT. Edge has **89** frac ≥ 0.5 positives among 938,578 in-run cells, against **31,466** presence positives. Its presence AP is 0.169 (base 0.034), but it has no ≥ 0.5 target to win. |
| 3 | **Signal share in the shared path.** Thin classes get 11 % of the map's trunk gradient, and the map is 3.4 % of the total loss. | Not measured here. The arrow and hatched APs sitting at noise level (0.005 and 0.004) fit the smallest gradient shares (arrow 0.14 %). |
| 4 | **Far-range labels are non-causal and sub-pixel.** | CONSISTENT with the per-band 10 cm AP falling with range. |
| 5 | **Nobody saw it.** Only drivable IoU was logged. | This package's AP tables are a per-class instrument for the refcv7 launch gate. |

⚠️ **Scene dependence, NEW here.** On the three CROWDED boxes-video clips, lane AP is 0.028 (3× base).
On the in-run eval windows it is 0.156 (12× base). Crowds and dense traffic cover the markings. A
per-class gate should be read per scene type, not pooled.

**The fixes are the audit's and SPEC_REFCV7's to own**: NEW-2 at 10 cm, and the PI's choice (c) of one
0.25 m lift at maximal range (SPEC_REFCV7 §11). This package's per-class AP-vs-base probe is offered as
the per-class read-out those gates need. It is cheap: CPU, 70 s per 500 windows from a dump.

## 7. Files

- **Code:** `code/ap_probe_38k.py` and `code/map_class_analysis.py`.
- **Results:** `raw/ap_probe_38k.json` (boxes clips, as briefed), `raw/ap_probe_38k_inrun_eval.json`
  (representative), and `raw/dump_index.json` (the dump's files, md5, and set counts).
- **Logs:** `raw/logs/ap_probe_38k.log` and `raw/logs/ap_probe_38k_inrun_eval.log`.
- **Not landable, local only:** the 224 MB dump at `C:/Users/Admin/qland/work/mapvid/mapdump_step38000/`.
