# `WHEELBASE = 2.9` — measured impact, and the decision it supports

**Date:** 2026-07-26 · **Scope:** PI-authorised decision measurement (option **C**: measure first, then decide)
**Compute:** dev box (Tier 1, Tier 3) + `tanitad-eval` A40, idle (Tier 2). **No training pod was touched
for compute**; pod3 was read once for two 27–110 KB text files (no GPU, no RAM).
**Status:** measured, written, left in the working tree — this agent did **NOT** `git add` / commit / push,
and did **NOT** change `WHEELBASE` in any shipped file.

---

## 0. One page

**The brief's premise is itself wrong, and the correction makes the label defect bigger, not smaller.**
The feature probe reported *"2.85 m for ~90 % of clips, 3.165 m for ~10 %"*. That was measured on
`vehicle_dimensions.chunk_0000` alone — a chunk that is 100 % United States. On the **197 chunks our
parity corpus actually draws from**, there are **five** wheelbases, and **2.85 m is 1.8 % of the corpus,
not 90 %**. The modal value is **2.73 m (47.0 %)**.

| what | brief (INHERITED, one chunk) | **MEASURED, our corpus (2 400 train order lines)** |
|---|---|---|
| distinct wheelbases | 2 (2.85 / 3.165) | **5** (2.73 / 2.85 / 3.135 / 3.165 / 3.216) |
| majority population | 2.85 m, ~90 % | **2.73 m, 47.0 %** |
| share at 2.85 m | ~90 % | **1.8 %** |
| error of 2.9 on the majority | +1.75 % | **+6.23 %** |
| clips where 2.9 is wrong | 100 % | 100 % (confirmed) |
| clips where \|error\| > 5 % | — | **98.2 %** |

**But the measured impact is small, and its shape is decisive for the choice.** The steer label is
`atan(L·κ)`, so a wrong `L` is a **per-clip multiplicative gain** on the steer channel — 0.941× to 1.109×.
That gain decomposes into a **bias** (2.9 vs the corpus clip-mean 2.9568 m, −1.92 %) and a **dispersion**
(±7.4 % rms, irreducible by *any* single constant). Tier 2 measures the two halves separately:

| perturbation of the steer channel fed to flagship-v1 | ADE@2s | ΔADE, paired episode-cluster bootstrap (B = 2000, 881 windows / 40 episodes) |
|---|---:|---|
| shipped (reproduces the registry's 0.4271) | 0.427109 | — |
| **corrected, per-clip true wheelbase** | 0.432706 | **+0.00560 [+0.00070, +0.01130]** — separated, **+1.31 %**, *worse* |
| bias only (one better constant, 2.9568) | 0.427716 | +0.00060 [−0.00040, +0.00170] — **not separated** |
| dispersion only (per-clip ÷ 2.9568) | 0.431566 | +0.00450 [+0.00040, +0.00890] — separated |
| ×1.5 uniform gain (≈5× the real error) | 0.535244 | +0.10810 [+0.05690, +0.17100] |
| steer channel zeroed | 0.770509 | +0.34340 [+0.19230, +0.51840] |

**Recommendation: B — fix forward only, and record the discontinuity.** Not D, because the channel is
*not* redundant from the model's side (zeroing it costs **+0.343 m**, and `R²(steer | accel, v0) =
0.0002` — steer is flagship-v1's only rotational input) and because the residual *is* CI-separated
(+0.0056 [+0.0007, +0.0113]) and lands where physics says it must, in the lateral channel
(cross-track@2s **0.2742 → 0.2977 m, +8.6 % relative**). Not A, because the effect is **1.31 % of ADE**,
smaller than the *width* of every published ADE CI on this val set (v1's is [0.3675, 0.4871], ±0.06 m —
**21× the effect**), so a full re-baseline would spend **~53 A40-days** to move numbers by less than
their own uncertainty, and would destroy the cross-arm comparability of 27 recomputed arms to do it.

**Three numbers set the thresholds, and each would have flipped the answer:**

| threshold | measured | would have chosen A if | would have chosen D if |
|---|---|---|---|
| ΔADE vs published CI width | **+0.0056 m** vs CI half-width **0.060 m** (9.3 %) | ΔADE had reached ~½ the CI half-width (≈ **0.03 m**), i.e. capable of moving a rank | ΔADE had not been CI-separated at all |
| is the channel redundant? | `R²(steer \| accel, v0) = 0.0002 [−0.0021, 0.0006]`; zeroed costs **+0.343 m** | — | if the model had also received ω (then `R² = 0.9919` and the gain would be absorbable) |
| does one better constant fix it? | bias arm **not separated**; dispersion arm **separated** | if the *bias* had carried the effect (a one-line constant change would then be both cheap and sufficient — and would still not need a re-baseline) | if the *dispersion* had also been non-separated |

---

## 1. Provenance — how each clip was tied to its wheelbase, and why the join is trustworthy

`physicalai.build_episode` stores `episode_id = int.from_bytes(clip_id[:4])`, i.e. an episode carries only
the **first four characters** of its clip UUID — not enough to join. The full identity comes from the
build's own ordered clip list, `tanitad-pod3:/workspace/tmp/{train,val}_clip_order.tsv` — the exact file
`rebuild_pai_rolling.py --order` consumed to mint cache key `e438721ae894`. The join was then **verified,
not assumed** (all MEASURED):

- **2376 / 2376** `episode_id`s in the committed `parity_profile.csv` reproduce from the order file's
  clip_ids. Zero mismatches.
- The 24 order indices absent from the built cache are **exactly** the documented skip set, first `1798`,
  last `1941` — the values independently written into `rebuild_pai_rolling.py --skip-idx`.
- **40 / 40** eval-pod val `episode_id`s reproduce from `val_clip_order.tsv[:40]`, confirming the
  eval-pod deployment is the first 40 clips of the canonical 600-clip val order.

`vehicle_dimensions` was pulled from HF for exactly the **197 chunks** the corpus draws from
(18 987 clips, ~6 MB). **No clip UUID appears in any file in this directory** — UUID-keyed intermediates
live in the session scratchpad only (§8).

### 1.1 A query-grid trap that cost the first run of Tier 1 — and how it was caught

`labels/egomotion` is **not** a 20 s file. MEASURED: it spans **~140 s** per clip — a dense ~100 Hz
stretch covering the 20 s camera clip, then a ~3.5 Hz tail over the rest of the parent session. The naive
read (resample the full egomotion span) samples ~7× too much time; its per-episode `mean|steer|`
correlated only **0.684** with the built episodes' own labels. The pipeline queries
`linspace(t_frames[0], t_frames[-1], int(span_s·10))` from the **camera** timestamps, which over the 500
locally-mirrored clips is **20.10–20.13 s** starting **0.03–0.21 s** after the egomotion `t0`
(`int(span·10) == 201` for 454/500). Tier 1 therefore uses the exact camera grid for the 95 corpus clips
held locally and that rule for the rest, and drops the 2 leading stack steps as `build_episode` does.

Two independent validations of the final grid, both MEASURED:

| check | result |
|---|---|
| rule vs exact camera timestamps (95 corpus clips with both) | corr **0.9998**, median rel. diff **0.24 %**, max 4.59 % |
| reconstructed vs the **built** eval-pod val-40 episodes' actual `actions[:,0]` | corr **0.99998**, median rel. diff **0.45 %**, pooled `mean\|steer\|` **0.03447 vs 0.03447** |

*(Evidence class of everything below: **MEASURED**, artifacts in §8, unless marked otherwise.)*

---

## 2. TIER 1 — the label delta

`steer_shipped = atan(2.9·κ)`, `steer_true = atan(L_clip·κ)`, `Δ = steer_true − steer_shipped`.
Canonical parity corpus, **2 400 train order lines / 477 594 label samples** at the pipeline's 10 Hz.
Reference scale: shipped steer `mean|·| = 0.02692` rad, `p99|·| = 0.4117` rad (23.6°); the
"p99 symmetric range" denominator below is `2 × p99|steer| = 0.8234` rad.

### 2.1 Stratified delta table — TRAIN (2 376 built episodes; order lines 2 400)

| true `L` (m) | clips | % of corpus | error of 2.9 | steer gain (true/shipped) | mean\|steer\| | **mean \|Δ\|** (rad) | **p95 \|Δ\|** | **max \|Δ\|** | mean \|Δ\| as % of steer range | max as % | median rel. steer error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2.730** | 1 127 | **47.0 %** | **+6.23 %** | 0.9414 | 0.03107 | 0.001706 | 0.010884 | 0.029289 | 0.207 % | 3.56 % | **5.86 %** |
| 2.850 | 44 | 1.8 % | +1.75 % | 0.9828 | 0.00854 | 0.000140 | 0.000421 | 0.007931 | 0.017 % | 0.96 % | 1.72 % |
| 3.135 | 334 | 13.9 % | −7.50 % | 1.0810 | 0.03181 | 0.002390 | 0.014680 | 0.036876 | 0.290 % | 4.48 % | 8.09 % |
| **3.165** | 612 | **25.5 %** | **−8.37 %** | 1.0914 | 0.01839 | 0.001602 | 0.008758 | 0.041629 | 0.195 % | 5.06 % | 9.13 % |
| **3.216** | 283 | 11.8 % | **−9.83 %** | 1.1090 | 0.02592 | 0.002678 | 0.015822 | 0.049225 | 0.325 % | 5.98 % | **10.88 %** |
| **ALL** | **2 400** | 100 % | — | 0.941–1.109 | 0.02692 | **0.001861** | **0.011612** | **0.049225** | **0.226 %** | 5.98 % | — |

In degrees: mean \|Δ\| **0.107°**, p95 **0.665°**, max **2.82°**, against a p99 steer amplitude of **23.6°**.
`std(Δ)/std(steer) = 0.0659`; `Δ` accounts for **0.55 %** of the steer channel's variance.

**VAL (600 canonical order lines / 119 397 samples)** is statistically identical: 2.73 m 44.3 %,
2.85 m 1.3 %, 3.135 m 14.3 %, 3.165 m 27.7 %, 3.216 m 12.3 %; mean \|Δ\| **0.001907** rad, p95 0.011948,
max 0.048945. The 40-episode eval deployment: **15 / 0 / 5 / 14 / 6** clips at 2.73 / 2.85 / 3.135 /
3.165 / 3.216 — **it contains no 2.85 m clip at all**.

### 2.2 What the populations ARE — and yes, they are coherent subpopulations

Each wheelbase is one physical vehicle (a single `vehicle_dimensions` row), except 2.85 m which is two:

| `L` | length × width × height (m) | reading | platform class | top-1 country | # countries | mean v |
|---:|---|---|---|---|---:|---:|
| 2.730 | 4.688 × 1.999 × 1.439 | compact car | `hyperion_8.1` (100 %) | Slovenia 9 % (Croatia/Bulgaria/Slovakia/Romania/Hungary next) | 25 | 13.3 m/s |
| 2.850 | 4.872 × 2.121 × 1.473 **and** 4.925 × 2.116 × 1.476 | 2 mid sedans | `hyperion_8` (100 %) | **United States 100 %** | **1** | 12.3 m/s |
| 3.135 | 5.207 × 2.157 × **1.823** | **tall — van/large SUV** | `hyperion_8.1` (100 %) | Czechia 21 % | 20 | 11.9 m/s |
| 3.165 | 5.255 × 2.130 × 1.494 | large sedan/estate | `hyperion_8` (598) + `hyperion_8.1` (14) | Poland 8 % | 22 | 12.3 m/s |
| 3.216 | 5.393 × 2.109 × 1.503 | long sedan | `hyperion_8.1` (100 %) | Portugal 14 % | 13 | 14.9 m/s |

**The systematic error IS concentrated in coherent subpopulations, and the confound is geography, not
speed.** Mutual information `I(wheelbase; country) = 0.769` bits of `H(wheelbase) = 1.880` bits →
**40.9 % of the wheelbase entropy is explained by country alone**. The sharpest case: the **2.85 m
population is 100 % United States and 100 % `hyperion_8`** — so any US-vs-Europe stratified read on this
corpus is *also* a wheelbase stratification, at the one wheelbase where 2.9 is nearly right (+1.75 %).
By contrast the speed regimes are close across populations (mean v 11.9–14.9 m/s), so **wheelbase is not
materially confounded with speed regime** — which matters, because the program's open weakness is
longitudinal, and this defect does not sit on top of it.

⚠️ **`platform_class` is not a proxy for wheelbase.** `hyperion_8` splits 598 / 44 across 3.165 / 2.85,
and `hyperion_8.1` splits across four values. Any fix must join `vehicle_dimensions` per clip; it may not
be shortcut through the platform field. *(And the docstring's "Hyperion platform class" justification for
2.9 is therefore not merely imprecise — **no** Hyperion platform in this corpus has a 2.9 m wheelbase.)*

### 2.3 The bias/dispersion split — the number that decides between "one better constant" and "per clip"

Clip-mean true wheelbase over the corpus: **2.9568 m** (val 600: 2.9699; val 40: 3.0058).

*(All three columns are about the **steer gain** `L_clip / constant`; the deviation is `|gain − 1|`,
clip-weighted over the 2 400 order lines. This is the quantity the network sees — not `(constant − L)/L`,
which is the "error of 2.9" column in §2.1.)*

| single constant | per-clip gain range | rms \|gain − 1\| | % of clips with \|gain − 1\| > 8 % |
|---|---|---:|---:|
| 2.9 (shipped) | 0.9414 – 1.1090 | **7.79 %** | 51.2 % |
| **2.9568** (best possible single constant) | 0.9233 – 1.0877 | **7.39 %** | 11.8 % |
| 2.7 (the closed-loop harness — §4.3) | 1.0111 – 1.1911 | 12.49 % | 51.2 % |

**Switching to the best single constant removes 0.40 of the 7.79 percentage points — 5 % of the error.
95 % of it is dispersion that no scalar can remove.** This is why the Tier-2 bias arm is flat and the
dispersion arm is not: a network can absorb a global gain into a weight; it cannot absorb a per-clip one.

---

## 3. TIER 2 — the INPUT-side impact on flagship-v1

⚠️ **What this can and cannot answer, stated with the number.** It measures how far the *trained-as-is*
checkpoint's open-loop trajectory moves when the steer channel it is **fed at inference** is corrected.
That **bounds the input-side effect only**. It is **not** what a model *trained* on corrected labels
would do: for this checkpoint a corrected input is a distribution shift, which is why every arm below is
**worse** than shipped. A retrained model would have learned the corrected statistics; Tier 2 cannot see
that, and nothing here should be read as "the fix hurts".

Arm: `flagship-30k` (the deployed v1), 881 windows / 40 clean-val episodes, `stride = 8`.
The shipped arm reproduces **0.427109** — bit-identical to the registry's `0.4271`.
Estimator: **paired episode-cluster bootstrap**, B = 2000, resampling unit = the val episode. The
correction is the exact inversion `steer_true = atan((L/2.9)·tan(steer_shipped))`.

### 3.1 Headline

| arm | what it is | ADE@2s | FDE@2s | miss@2m | **ΔADE [CI95]** | separated |
|---|---|---:|---:|---:|---|:--:|
| `shipped` | as trained/published | 0.427109 | 0.9075 | 0.0454 | — | — |
| **`corrected`** | **per-clip true wheelbase** | **0.432706** | 0.9234 | 0.0454 | **+0.00560 [+0.00070, +0.01130]** | **✅** |
| `global_mean` | one constant 2.9568 (bias only) | 0.427716 | 0.9099 | 0.0477 | +0.00060 [−0.00040, +0.00170] | ✗ |
| `dispersion` | per-clip ÷ 2.9568 (dispersion only) | 0.431566 | 0.9193 | 0.0443 | +0.00450 [+0.00040, +0.00890] | ✅ |
| `harness_2p7` | gain 2.7/2.9 (the closed-loop convention, §4.3) | 0.429704 | 0.9102 | 0.0431 | +0.00260 [−0.00060, +0.00620] | ✗ |
| `x1.5` | uniform ×1.5 (≈5× the real error) | 0.535244 | 1.1812 | 0.1532 | +0.10810 [+0.05690, +0.17100] | ✅ |
| `zeroed` | steer channel destroyed | 0.770509 | 1.6639 | 0.2123 | +0.34340 [+0.19230, +0.51840] | ✅ |

`zeroed` and `x1.5` are the **calibration controls**, and they earn their place: without them a small
`corrected` delta would be indistinguishable from *"the model ignores steer"*. It does not — removing the
channel costs **80 % of the ADE**, and the sensitivity is monotone in gain magnitude
(0.94–1.11× → +0.006; 1.5× → +0.108; 0× → +0.343). The real defect sits at the far low end of a curve the
model is otherwise steeply sensitive along.

Trajectory-level, not just the scalar: mean predicted-path displacement at 2 s is **0.086 m** for
`corrected` (p95 0.419, max 1.045) vs **1.138 m** for `zeroed` (p95 6.08, max 11.31).

### 3.2 Along / cross decomposition

⚠️ `taniteval.lateral.block` **skips** on the eval pod: that pod's `rollout.collect` predates the
2026-07-25 dense-path upgrade, so `pred_dense/gt_dense` do not exist and the module correctly refuses to
fit a compounding law to 4 knots. `lateral.paired_cross_track` *does* return there, but the pod copy
reports `step = 4, horizon_s = 0.4` against a 4-knot sparse surface — its horizon bookkeeping does not
match the surface it was handed, so **it is not quoted here**. The split below is computed explicitly on
the sparse surface, where `rollout.collect` already returns ego-frame waypoints (x forward = along,
y left = cross), with the same paired episode-cluster bootstrap.

| arm | along\|@2s (m) | cross\|@2s (m) | longitudinal share of squared error | Δalong@2s [CI95] | Δcross@2s [CI95] |
|---|---:|---:|---:|---|---|
| `shipped` | 0.8256 | 0.2742 | **85.4 %** | — | — |
| **`corrected`** | 0.8229 | **0.2977** | 82.2 % | −0.00270 [−0.01830, +0.00980] ✗ | **+0.02350 [−0.00490, +0.05790]** ✗ |
| `global_mean` | 0.8233 | 0.2807 | 84.6 % | −0.00230 [−0.00540, +0.00020] ✗ | +0.00650 [+0.00080, +0.01330] ✅ |
| `dispersion` | 0.8249 | 0.2904 | 83.3 % | −0.00070 [−0.01430, +0.01170] ✗ | +0.01620 [−0.00810, +0.04330] ✗ |
| `harness_2p7` | 0.8355 | 0.2663 | 87.3 % | +0.00990 [+0.00020, +0.02170] ✅ | −0.00790 [−0.02400, +0.00650] ✗ |
| `x1.5` | 0.8864 | 0.5777 | 51.1 % | +0.06090 [+0.02440, +0.10110] ✅ | +0.30350 [+0.14610, +0.49340] ✅ |
| `zeroed` | 1.1324 | 0.9897 | 42.8 % | +0.30680 [+0.14300, +0.50490] ✅ | +0.71540 [+0.40280, +1.08600] ✅ |

**The effect is where physics says it must be: lateral.** A steer-gain error is a rotational error, and
the corrected arm moves cross-track **0.2742 → 0.2977 m (+8.6 % relative)** while leaving longitudinal
error alone (−0.3 %, not separated). *(The shipped arm's 85.4 % longitudinal share independently
reproduces the registry's "89 % of squared error is longitudinal" failure signature.)* **+8.6 % on the
lateral channel is the largest honest statement of this defect's size** — much larger than the +1.31 %
on ADE, because ADE is dominated by a longitudinal error the wheelbase does not touch. Its CI is not
separated, so it is a point estimate with a direction, not a verdict.

### 3.3 Stratified by wheelbase population (eval deployment)

| `L` | episodes | windows | gain | ADE shipped | ADE corrected | ADE zeroed | Δ corrected [CI95] |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2.730 | 15 | 330 | 0.9414 | 0.4535 | 0.4552 | 0.7968 | +0.00180 [−0.00210, +0.00600] ✗ |
| 3.135 | 5 | 110 | 1.0810 | 0.4086 | 0.4132 | 0.6732 | +0.00470 [−0.00360, +0.01440] ✗ |
| 3.165 | 14 | 309 | 1.0914 | 0.4209 | 0.4316 | 0.7153 | +0.01070 [−0.00150, +0.02690] ✗ |
| 3.216 | 6 | 132 | 1.1090 | 0.3911 | 0.3952 | 0.9151 | +0.00410 [−0.00070, +0.00950] ✗ |

**No stratum separates** — 5–15 episodes each is not enough for an episode-cluster CI, and the honest
reading is that the point estimates order roughly with \|gain − 1\| (3.165 largest at +0.0107) but the
data cannot support a per-population claim. **The 3.165 m population is not measurably more distorted
than the others at this n.** The 2.85 m population is **absent from the eval deployment entirely**, so
Tier 2 says nothing about the one population 2.9 nearly fits.

---

## 4. TIER 3 — bounding the training-side effect (no retrain attempted)

### 4.1 The redundancy claim: true about the dataset, false about the model

The brief's nuance — *"steer correlates r = 0.9865 with ω/v, so a wrong constant may be nearly irrelevant
to learning"* — was tested, not assumed. It splits cleanly in two.

**Q1 — information-theoretic redundancy (about the dataset): CONFIRMED.** Ridge fit on the train clips,
λ by 5-fold *episode-disjoint* CV, evaluated out-of-sample on the val clips, CI = episode-cluster
bootstrap over those clips (B = 2000):

| predictors | R² out-of-sample | CI95 | RMSE (rad) |
|---|---:|---|---:|
| ω, v (all speeds) | 0.5689 | [0.4750, 0.6614] | 0.0528 |
| ω/v (all speeds) | 0.6308 | [0.3946, 0.8547] | 0.0489 |
| **ω/v, v > 2 m/s** (the mask `idm2_diag_labels.py:116` uses; **89.3 %** of samples) | **0.9919** | **[0.9841, 0.9964]** | **0.0058** |
| ω/v + ω + v, v > 2 m/s | 0.9923 | [0.9847, 0.9966] | 0.0057 |

⚠️ **Reconciliation of `0.9865`, because two of my own intermediate readings were wrong before this
landed.** Pooled `corr(steer, ω/v)` is **0.0125** at all speeds (the 1/v blow-up destroys it) and
**0.9931** at v > 2 m/s. The IDM figure is a **mean of *within-episode* correlations** at v > 2 m/s
(`idm2_diag_labels.py:114-121`) — we reproduce **0.9786** that way. Those are three different estimators
of one thing and only the pooled out-of-sample one describes what a model faces. **The claim survives**:
above 2 m/s, `steer` is ~99 % linearly explained by ω/v, and — the strongest form of "it doesn't matter" —
**the wheelbase constant is exactly the slope of that relation, so any network that can compute ω/v can
absorb a *global* wheelbase into a weight.** That is precisely why the Tier-2 bias arm is flat. It cannot
absorb a *per-clip* one, which is why the dispersion arm is not.

**Q2 — what flagship-v1 actually receives: the redundancy does not apply.** v1's action vector is
`[steer, accel, v0/10]` (`--speed-input`; `rollout.ego_action_channels`). **ω is not an input channel** —
only `--yaw-input`/`--dyn-input` arms get `yr0`, and v1 is neither.

| predictors (v1's *other* inputs) | R²(steer \| ·) | CI95 |
|---|---:|---|
| accel, v0/10 | **0.0002** | [−0.0021, 0.0006] |
| accel | −0.0007 | [−0.0040, 0.0000] |
| v0/10 | 0.0002 | [−0.0021, 0.0006] |
| accel, v0/10 (v > 2 m/s) | 0.0016 | [−0.0005, 0.0025] |

**Within the model's own input vector, `steer` is the only rotational signal and carries ~100 % unique
information.** The behavioural referee agrees: zeroing it costs **+0.343 m**. And a third, independent
line already in the repo agrees — `IDM_V2_RESULTS.md` §3.4 measured *derived* steer (bicycle geometry from
`ŷaw/v̂`) at R² **0.424** vs *regressed* steer at **0.520**, verdict *"FAIL — do not drop steer"*.
**Any argument of the form "the channel is redundant, therefore the constant is harmless" is refuted by
three separate measurements.** What survives is the narrower and true claim: a *global* constant is
absorbable; the *per-clip spread* is not.

### 4.2 Does anything consume `steer` as an output rather than an input? — **Yes, one line does**

| site | role | reads/writes `steer` | exposure |
|---|---|---|---|
| `physicalai.py:412`, `cosmos_drive.py:108` | **label mint** | writes, with `WHEELBASE = 2.9` | every action label in every arm |
| flagship 4-brain operative/tactical/strategic | consumes as an input action channel; outputs Δpose and waypoints | reads only | the defect is an input perturbation — Tier 2 |
| `scripts/idm_head.py:37, 297-300` | **inverse-dynamics head: `SCALAR_NAMES` includes `steer`, and `actions = stack([steer, accel])`** | **writes** | any corpus this head labels inherits the **2.9 convention**; used by `run_branchb_transfer.py`, `run_camcond_ablation.py`, `run_idm_downstream_ablation.py` |
| `taniteval/closedloop.py:141-146, 176` | closed-loop harness controller + bicycle integrator | writes **and** reads, at **2.7** | §4.3 |

The deployed trajectory path never emits `steer`, so the label error cannot propagate as a control
command there. **The IDM line does emit it** — so if the YouTube/own-corpus IDM labeler ships, its steer
outputs are in the 2.9 convention and must be documented as such (this is a *convention* statement, not a
new error).

### 4.3 ⭐ A finding this measurement surfaced that was not in the brief: **the program has three wheelbases**

| constant | value | where | what it does |
|---|---:|---|---|
| label mint | **2.9** | `physicalai.py:51`, `cosmos_drive.py:63` | every training/eval action label |
| closed-loop harness | **2.7** | `taniteval/closedloop.py:99` | `wp_to_control` → `steer = atan(2.7·κ)`, then `bicycle_integrate` → `yaw += v/2.7·tan(steer)·dt` |
| `rollout_bicycle` default | **2.7** | `tanitad/models/kinematic.py:15` | differentiable bicycle layer |
| truth | **2.73 / 2.85 / 3.135 / 3.165 / 3.216** | `calibration/vehicle_dimensions` | — |

Two consequences, both MEASURED from source:

1. **The closed-loop *path* is wheelbase-invariant** — `wp_to_control` and `bicycle_integrate` use the
   same constant, so it cancels exactly (`atan(L·κ)` then `v/L·tan(steer) = v·κ`). **No closed-loop
   trajectory number in the program is wrong because of 2.7.** ✅
2. **But the action *fed to the model* inside the closed loop is `atan(2.7·κ)` while the model was trained
   on `atan(2.9·κ)`** (`closedloop.py:214-219`: `a_exec` → `win_a_exec[:, -1]` → `model.predictor`), and
   the observed window `aw` in the same tensor is 2.9-derived. **That is a train/serve skew of 2.9/2.7 =
   +7.41 %, and mixed conventions inside one action window.** Measured open-loop as the `harness_2p7`
   arm: ΔADE **+0.00260 [−0.00060, +0.00620]** — *not separated*, with a separated +0.0099 m
   longitudinal component. Small, but it is a real inconsistency that no document records, and closed
   loop compounds what open loop does not. **This is the cheapest thing to fix in the whole report: one
   constant, in the harness, no retrain, no parity impact.**

### 4.4 What option A would cost

| item | basis | cost |
|---|---|---|
| rebuild both epcaches (2 400 + 600 clips) | MEASURED throughput `V2_PHASE2_BUILD.md`: ~26 clips/min across two pods, ~220 clips/h solo | **~2 h** (2 pods) to **~14 h** (solo), I/O + decode bound, no GPU |
| retrain one 30 k arm | **10.888 s/step × 30 000 = 90.73 h A40** (`RETRACTION_LOG` 07-21 — ⚠️ *not* the `summary.json` `wallclock_s` 53.1 h, which is retracted) | **~90.7 A40-h / arm** |
| re-baseline the comparable set | 27 arms were recomputed in the 2026-07-25 estimator sweep (MEASURED); **~14 of them are 30 k-scale trained arms — ESTIMATED** from the §6 leaderboard + §1–4 sections | **~1 270 A40-h ≈ 53 A40-days** (ESTIMATED on a MEASURED per-arm basis) |
| re-eval + re-emit every published number | 188 published numbers inventoried in the estimator sweep | days of agent time, plus a second full RETRACTION/registry pass |
| parity | `e438721ae894` is defined as a hash of the ordered clip list + build params; **changing the label derivation changes the built tensors, not the key** — so the key would silently no longer mean what it means today unless the params dict is extended | must be handled explicitly, or the parity guarantee degrades to a name |

**And the yield of that spend, measured: +0.0056 m of ADE on a number whose own CI is ±0.060 m.**

---

## 5. The recommendation — **B**, with the thresholds that produced it

> **B. Fix for new arms only, document the discontinuity — two label regimes, never compared across.**
> Plus two zero-cost corrections that are *not* the parity-affecting fix (§6).

**Why not D.** D requires the channel to be genuinely redundant *and* the label never published as truth.
Neither holds. `R²(steer | accel, v0) = 0.0002` and zeroing the channel costs **+0.343 m** — it is
flagship-v1's only rotational input. The corrected arm's ΔADE **is** CI-separated (+0.00560 [+0.00070,
+0.01130]), the dispersion component **is** CI-separated, and the effect lands in the lateral channel at
**+8.6 % relative**. A defect that is separated, mechanistically located, and concentrated in
geographically coherent subpopulations is not a "do nothing".

**Why not A.** Three measured facts, each independently sufficient:
1. **Magnitude vs uncertainty.** +0.0056 m against v1's own episode-cluster CI of [0.3675, 0.4871]
   (half-width 0.060 m). The effect is **9.3 % of the CI half-width** and **1.31 % of ADE** — it cannot
   move a rank. For scale, the *estimator* correction the program already absorbed moved single-arm
   point estimates by **−6.67 % to +11.69 %**, 5–9× larger, and flipped **zero** gate verdicts.
2. **Cost.** ~53 A40-days of retraining plus a full re-publication pass, to move numbers by less than
   their own noise — while pod1/pod2/pod3 are committed and v2corpus/E1c are mid-flight.
3. **A re-baseline destroys more comparability than it repairs.** 27 arms were just recomputed onto one
   estimator; re-minting labels makes every historical arm incomparable to every new one *anyway* — which
   is exactly option B's discontinuity, but paid for twice.

**Why B specifically, and not "just change the constant to 2.9568".** Because the bias arm is **not
separated** (+0.00060 [−0.00040, +0.00170]) and the dispersion arm **is**. A better single constant buys
0.40 of 7.79 percentage points — **5 % of the error** — and the network absorbs a global gain anyway.
**If a fix is worth making, it is worth making per clip via a `vehicle_dimensions` join** (the adapter is
one join in `physicalai.py`, identical in shape to the existing per-clip intrinsics path). A scalar swap
would look like a fix, cost a parity break, and change nothing measurable.

### What the numbers would have had to be to choose differently

| if… | then |
|---|---|
| ΔADE ≥ ~0.03 m (½ the CI half-width), or any stratum's CI separated at a magnitude that could reorder the leaderboard | **A** — re-baseline, because published rankings would be at stake |
| the `bias` arm had carried the effect and `dispersion` had not | change the constant to 2.9568 in place, fix-forward, **no** re-baseline (a global gain is absorbable, so even then A would not follow) |
| ΔADE not separated **and** `R²(steer \| accel, v0)` ≈ 0.97+ **and** the label never published | **D** — with the docstring still corrected (§6) |
| flagship had also received ω as an input channel (`--dyn-input`) | the case for **D** would be materially stronger: R² 0.9919 means the network could re-derive the true steer for any wheelbase, and the constant becomes a slope it absorbs |

### The concrete B

1. **Do not touch** `physicalai.py:51` for any existing arm, cache, or published number. `e438721ae894`
   keeps its current meaning.
2. **New arms** (v2corpus successors, the own-dataset line, anything built after the decision) mint steer
   from a **per-clip `vehicle_dimensions` join**, resolved exactly like `intrinsics_for_clip` — local
   table first, dataset parquet second, and a **loud** fallback (never a silent constant).
3. **Extend the cache-key params dict** with the label-derivation mode so `cache_key` separates the two
   regimes by construction. Two label regimes must never share a key. This is the mechanism that makes
   "never compared across" enforceable rather than a promise in prose.
4. **Register the discontinuity** in `MODEL_REGISTRY.md` §0.1 (parity) as a named regime boundary, with
   this report as its citation, so no future paired test silently crosses it.

---

## 6. The honesty dimension — what is wrong *regardless* of the decision

Even under D, three statements in the repo are false or misleading and cost nothing to fix. None is
parity-affecting; none needs the PI's decision.

| # | where | what it says | what is true | proposed |
|---|---|---|---|---|
| 1 | `physicalai.py:51` | `WHEELBASE = 2.9  # Hyperion platform class, sedan/SUV proxy` | **No Hyperion platform in this corpus has a 2.9 m wheelbase.** The corpus has 2.73 / 2.85 / 3.135 / 3.165 / 3.216 m across `hyperion_8` and `hyperion_8.1`; clip-mean 2.9568 m | rewrite the comment as an explicit, dated, cited **approximation**: `# APPROXIMATION, not a platform fact. True per-clip values ship in calibration/vehicle_dimensions: 2.73 (47%) / 2.85 (2%) / 3.135 (14%) / 3.165 (26%) / 3.216 (12%), clip-mean 2.9568. Kept at 2.9 for parity with e438721ae894 — see …/2026-07-26-wheelbase-impact/. Measured impact: ADE +0.0056 [+0.0007, +0.0113].` |
| 2 | `cosmos_drive.py:63` | `WHEELBASE = 2.9  # Hyperion platform class, shared with PhysicalAI-AV` | The justification is a **cross-reference to a false claim**. Cosmos-DD is synthetic; whether it even *has* a defined ego wheelbase is **UNKNOWN — not probed here** | same comment fix + an open item: probe whether RDS-HQ/Cosmos-DD publishes an ego wheelbase. Do **not** assume it inherits PhysicalAI's |
| 3 | `GEOMETRY_INTEGRITY_AUDIT.md:40, 77` | *"wheelbase 2.9 vs real **2.85** (1.5 %)"* | **INHERITED from a single chunk.** Real is five values; 2.85 covers **1.8 %** of our corpus; the corpus-weighted error is **+6.2 % to −9.8 %**, not 1.5 % | correct in place, citing this report |
| 4 | `RETRACTION_LOG.md` | — | the *"2.85 m for ~90 % of clips"* figure is a textbook **C2 (absence/statistic from a single probe)** — one chunk, and that chunk is 100 % US. It propagated into the feature probe's P0 recommendation and into this brief | append the entry with class **C2**, cost: *"would have justified a fix sized 1.75 % against a corpus whose modal error is 6.23 %, with the sign inverted for 47 % of clips"* |
| 5 | `l2d.py:53`, `nuscenes.py:331` | `KIA_NIRO_WHEELBASE_M = 2.72`, `L = 2.588` (Renault Zoe) | **These are correct and should not be touched.** They are published specs for single-vehicle corpora. ⚠️ The brief's claim that the 2.9 constant is *"replicated in `cosmos_drive.py` **and `l2d.py`**"* is wrong for `l2d.py` | no action; note the correction |
| 6 | `closedloop.py:99` | `WHEELBASE = 2.7` | consistent within the harness (path invariant ✅) but **feeds the model a 2.7-derived action while the model was trained on 2.9** — a +7.41 % train/serve skew, mixed with 2.9-derived actions in the same window | align the harness constant to the label-mint constant, or document the skew. **No retrain, no parity impact.** Open-loop cost measured: +0.0026 [−0.0006, +0.0062], not separated |

---

## 7. Limitations — read these with the recommendation

1. **Tier 2 is an input perturbation, not a retrain.** It bounds the input-side effect. The training-side
   effect is *bounded* by Tier 3's argument (a global gain is absorbable; the per-clip residual is 0.55 %
   of the steer channel's variance), **not measured**. A retrained arm could differ in either direction.
   The one experiment that would settle it — retrain one arm on corrected labels — is a ~91 A40-h
   pre-registered A/B, and **it is not justified at this effect size** unless the PI wants it for the
   record.
2. **The 40-episode eval deployment contains no 2.85 m clip**, so nothing here measures the one
   population 2.9 nearly fits.
3. **No per-population CI separates** (5–15 episodes each). The stratified point estimates are
   directional only.
4. **Tier 1's query grid is exact for 95 of 3 000 clips** and a measured rule for the rest; the rule's
   error is 0.24 % median / 4.59 % max against exact, and 0.45 % median against the built episodes.
5. **Cosmos-DD's true wheelbase was not probed.** Stated as UNKNOWN, not as "inherits 2.9".
6. **Δ is quoted as a fraction of `2 × p99|steer| = 0.8234` rad**, a stated denominator; against the
   *full* range (max 0.677 rad → 1.359) every percentage roughly halves. The rad values are primary.

---

## 8. Deliverable manifest

All paths relative to the repo root. **Nothing was `git add`ed.** No shipped file was modified.

| artifact | what | where |
|---|---|---|
| `WHEELBASE_IMPACT.md` | this report | `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-26-wheelbase-impact/` |
| `wheelbase_population.json` | per-split wheelbase population, platform/country/hour cross-tabs, full dimension rows | same dir |
| `tier1_label_delta.json` | the stratified delta distribution + the grid-rule validation block | same dir |
| `tier1_population_strata.json` | per-population country entropy, `I(wheelbase; country)`, speed regimes | same dir |
| `tier2_input_eval.json` | 7 arms, full-set suites, paired CIs, path displacement, per-population strata | same dir |
| `tier2b_along_cross.json` | along/cross levels + paired CIs on the sparse ego surface | same dir |
| `tier3_redundancy.json` | Q1/Q2 out-of-sample ridge probes with episode-cluster CIs | same dir |
| `tier3b_redundancy_reconciliation.json` | pooled vs per-episode vs oracle estimators of the `0.9865` claim | same dir |
| `wb_pull_dims.py` | HF pull of `vehicle_dimensions` for the 197 corpus chunks + the verified join | same dir |
| `wb_tier1_label_delta.py` | Tier 1 | same dir |
| `wb_tier2_input_eval.py` | Tier 2 (runs on `tanitad-eval`) | same dir · deployed copy `tanitad-eval:/root/wb_tier2_input_eval.py` |
| `wb_tier2b_alongcross.py` | Tier 2b | same dir · `tanitad-eval:/root/wb_tier2b_alongcross.py` |
| `wb_tier3_redundancy.py`, `wb_tier3b_reconcile_0p9865.py` | Tier 3 | same dir |
| pod-side raw | `wb_tier2_results.json`, `wb_tier2b_results.json`, `wb_tier2_windows.pt`, `wb_tier2.log`, `wb_tier2b.log` | `tanitad-eval:/root/` — the JSONs are copied into the repo dir above; the `.pt` window dump is **not** (it is regenerable and lives only there) |
| **gated-confidential, NOT in the repo** | `train_clip_order.tsv`, `val_clip_order.tsv`, `corpus_dims.json`, `tier1_per_clip.csv`, `tier1_samples.npz`, `val40_wheelbase.json`, `vd_cache/` — all UUID-bearing | session scratchpad `…/8fc25020-…/scratchpad/wb/` (+ `tanitad-eval:/root/val40_wheelbase.json`, which is keyed by `ep_%05d.pt`, **not** by UUID) |

**Gated-content re-scan (the prior probe's leak class).** Every file written into the repo directory was
re-read after generation and searched for UUID-shaped strings, `clip_id` keys and binary blobs:
**0 clip UUIDs, 0 raw content, 0 PNG blobs.** The only per-clip identifiers that reach the repo are
`ep_%05d.pt` filenames and order indices, both of which are already public in `parity_profile.csv`.

### Escalations (per the Agent Operating Standard — not left as a request in a doc)

1. **The PI decision this report exists to serve** — A / B / D. Recommendation **B**, §5.
2. **`closedloop.py:99` train/serve skew (§4.3.2)** — needs an owner. One constant, no retrain, no parity
   impact, and it touches every closed-loop number the program will produce from here.
3. **`GEOMETRY_INTEGRITY_AUDIT.md`'s "real 2.85 (1.5 %)"** is wrong and is cited elsewhere — needs the
   correction in §6 row 3 and a `RETRACTION_LOG` C2 entry (§6 row 4).
4. **The eval pod's `taniteval` is stale** — its `rollout.collect` predates the dense-path upgrade, so
   `lateral.block` cannot run there and `paired_cross_track` reports a horizon that does not match its
   input surface. Any agent quoting a lateral block off that pod today is quoting a mismatch.
