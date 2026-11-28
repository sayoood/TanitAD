# IDM v2 — results against the pre-registration

**Date:** 2026-07-26 · **Agent:** `idm-v2` · **Pod:** `tanitad-eval` (A40; pod1/pod2/pod3 untouched)
**Reads with:** `IDM_DIAGNOSIS.md` (the (a)–(d) verdict) and `PRE_REGISTRATION_IDMV2.md`
(written before any arm was trained).

**Substrate:** 104 held-out episodes → **68 train / 36 val**, **15,875 train / 4,195 val
windows**. Every arm scores the **identical** val windows. All intervals are
`taniteval.ci.(paired_)episode_cluster_bootstrap` over the **36 val episodes**,
n_boot = 2000. `overlapping_holdout_se` is not used anywhere.
**Evidence class: MEASURED (this run)** throughout.

---

## 0. The one number that carries the whole report

> **Deleting 9 windows out of 4,195 — the ones whose ground-truth yaw is
> physically impossible — moves the deployed head's pooled `yaw_rate` R² from
> 0.105 to 0.497.**

Nine windows. 0.21 % of the val set. They are all comma2k19 frames where the
vehicle is stationary and heading is `arctan2` of an undefined ENU velocity
(`IDM_DIAGNOSIS.md` §3.2). **That is what "yaw_rate R² 0.010 at scale" has been
measuring.** On PhysicalAI, where heading comes from the orientation quaternion,
the *same deployed head* reads **yaw R² 0.9035**.

---

## 1. Pre-registered endpoints — scored honestly

| # | endpoint | bar | result | verdict |
|---|---|---|---|---|
| 1 | **yaw_rate** | paired CI on Δ medAE vs A0 excludes 0 **and** pooled nMedAE < 1.0 | best arm V3sB: Δ MAE **−0.0047 [−0.0078, −0.0018] separated ✓**; nMedAE **1.342** ✗ | **FAIL** (2nd condition) |
| 2 | **speed** | paired CI on Δ MAE vs A0 excludes 0 **and** effect ≥ 0.5 m/s | V3wB: Δ MAE **−0.328 [−0.976, +0.340] not separated** ✗ | **FAIL vs A0** · **PASS vs the v1 recipe control B0** (−0.906 [−1.661, −0.298] **separated**, effect 0.91 m/s) |
| 3 | **long_accel** | derived accel R² ≥ 0.30 vs the kinematic target | best derived arm **−0.30**; post-hoc sliding-window derivative **−7.2 to −35.3** | **FAIL** |
| 4 | **steer** | derived ≤ regressed (would license dropping the channel) | regressed **0.520**, derived **0.424** — derived is *worse* | **FAIL — do not drop steer** |

**Two of the four bars were mis-set by me, and I am saying so rather than
re-defining them after the fact:**

- Endpoint 1's `nMedAE < 1.0` is **unachievable on this corpus by any model.**
  PhysicalAI is ~74 % straight, so MAD(yaw) = 0.0088 rad/s and "always predict the
  median" is a strong medAE baseline. A model with R² 0.90 on PhysicalAI still
  reads nMedAE 2.8. nMedAE remains the right *outlier-proof* statistic, but
  **"< 1.0" was the wrong threshold** — it encodes "beat the median on a mostly-
  straight corpus", not "be useful". *(Class C10 — the evaluator's own bar was not
  achievable; caught by running it, not by arguing it away.)*
- Endpoint 2 compares against **A0, which was trained on 160 clips (120 of them
  PhysicalAI) while every v2 arm here has 68 clips (26 PhysicalAI).** That is a
  2.4× data disadvantage and a very different domain mix, so "not separated vs A0"
  is a weaker statement than it looks. **B0 — the same recipe on the same data — is
  the controlled contrast, and v2 beats it with a separated CI.**

---

## 2. What was built

**`idm_head_v2` (arm V3w/V3wB) = four changes to `idm_head_v1`, each justified by a
measurement in `IDM_DIAGNOSIS.md`:**

| change | why (measured) |
|---|---|
| **Label winsorisation** at physical limits (yaw 1.5 rad/s, accel 12 m/s², speed 60 m/s, steer 1.0), standardiser refit on the cleaned labels | 0.30 % of comma frames carry |ω| up to 15.5 rad/s at v ≈ 0; they inflate the train std 0.429 → and the standardised Huber loss then divides the real yaw signal by that inflated scale |
| **Clip-context token** — mean & std of `z` over the whole clip, projected to one extra token. Legitimate: the IDM is an **offline, already non-causal** labeler | 56.6 % of speed MSE is a clip-level bias proportional to clip speed (gain 0.830). The head needs information about *which clip this is* |
| **17-frame window** (k=8) | long_accel probe rises 0.065 → 0.113 → 0.156 (1 → 9 → 17 frames) on PhysicalAI |
| **0.86 M head** (d_model 128), down from 2.90 M | the measured capacity curve: 0.86 M beats 2.90 M on 3 of 4 channels and 19.98 M is worse than 0.86 M on all four |

Deliberately **not** included, because each was measured and rejected (§4):
log-speed parametrisation, derived (differentiated) long_accel, derived steer,
MAD standardiser.

---

## 3. Before / after at scale — per channel, n = 4,195 windows / 36 episodes

Point estimates are on the **seed-averaged prediction**; `V3w` (seeds 0–1) and
`V3wB` (seeds 2–4) are the *same configuration run as two independent seed groups*
— they agree to 0.003 R², which is the replication.

### 3.1 speed

| arm | R² [95 % CI] | MAE m/s [95 % CI] | pai R² / MAE | cm R² / MAE |
|---|---|---|---|---|
| **A0** `idm_head_v1` (deployed) | 0.8651 [0.773, 0.918] | 2.994 [2.39, 3.70] | 0.9070 / 2.442 | 0.7590 / 3.217 |
| B0 (v1 recipe, this data) | 0.7815 [0.581, 0.892] | 3.573 [2.65, 4.71] | 0.5143 / 4.841 | 0.7973 / 3.063 |
| **V3w** (v2, seeds 0–1) | **0.9091 [0.849, 0.944]** | **2.654 [2.13, 3.27]** | 0.8771 / 2.640 | 0.8713 / 2.659 |
| **V3wB** (v2, seeds 2–4) | **0.9121 [0.859, 0.942]** | **2.667 [2.15, 3.24]** | 0.8659 / 2.954 | **0.8842 / 2.551** |

Paired, vs **B0**: Δ MAE **−0.906 m/s [−1.661, −0.298] separated ✓**.
Paired, vs **A0**: Δ MAE −0.328 [−0.976, +0.340] not separated.

**Where the gain is:** v2 is much better on **comma2k19** (R² 0.759 → 0.884, MAE
3.217 → 2.551) and somewhat worse on PhysicalAI (0.907 → 0.866) — exactly what a
68-clip train set with 42 comma / 26 PhysicalAI predicts against A0's 120/40 mix.
**The v2 change closes the cross-domain speed gap; it does not add PhysicalAI data.**

### 3.2 yaw_rate — report it per corpus, and say what was excluded

| arm | pooled R² | **admissible R²** (9/4195 impossible labels dropped) | medAE rad/s | nMedAE | **pai R²** | cm R² |
|---|---|---|---|---|---|---|
| **A0** | 0.1046 | **0.4967 [0.161, 0.872]** | 0.0212 | 1.806 | **0.9035** | 0.0719 |
| B0 | 0.0921 | 0.5004 | 0.0172 | 1.469 | 0.8508 | 0.1313 |
| B1 (+winsorise) | 0.0985 | 0.5014 | 0.0163 | 1.386 | 0.8429 | 0.1412 |
| **V3sB** (v2, k=4) | 0.1019 | **0.5101 [0.191, 0.866]** | 0.0158 | 1.346 | 0.8598 | 0.1422 |
| V3wB (v2, k=8) | 0.0977 | 0.4932 | 0.0166 | 1.413 | 0.8265 | 0.1407 |

Paired vs A0, V3sB: Δ MAE **−0.0047 [−0.0078, −0.0018] separated ✓** (a 22 % MAE
reduction). But the honest reading is the **pai** column: A0 is already at 0.90
there and **no arm improves on it**, because none of them has A0's PhysicalAI data.

**yaw's problem was never the model.** The three levers that matter are, in order:
(1) drop physically impossible labels — 9 windows move pooled R² 0.105 → 0.497;
(2) report per corpus — PhysicalAI 0.90 vs comma 0.07 are different facts;
(3) fix comma's heading derivation at the loader (`arctan2` of ENU velocity is
undefined at standstill) or exclude comma from any yaw claim.

> 🔴 **RE-ISSUED 2026-07-27 (C29) — lever (3) was done, and it beats lever (1).**
> Every `cm R²` cell in the table above, and every `pooled`/`admissible` cell, was scored with
> **`heading_repair` OFF**. Superseded values are kept above for audit. On the **identical
> 4,195 windows / 36 episodes, `v_min` 0.5**, with the **deployed head and nothing retrained**:
>
> | protocol | pooled | PhysicalAI | comma2k19 | n |
> |---|---:|---:|---:|---:|
> | legacy (the `A0` row above) | +0.1046 | +0.9035 | **+0.0114** | 4 195 |
> | 9 impossible windows **deleted** (the `admissible` column above) | +0.4967 | +0.9035 | **+0.0719** | 4 186 |
> | **`heading_repair` ON, `v_min` 0.5** | **+0.8108** | +0.9035 | **+0.3308** | **4 195** |
>
> **Repair beats deletion and discards nothing.** The 9 impossible rows were the visible tip: the
> repair touches **50 windows (1.19 %)**, on which GT speed maxes at **0.528 m/s** while the legacy
> label claimed up to **9.47 rad/s** and the head had predicted **0.023**.
> ⚠️ **Rows B0 / B1 / V3sB / V3wB are STALE-PENDING, not corrected** — no repaired re-score of those
> arms exists. (`B0`'s recipe was re-run in v3 as `R0LEG` and re-scored: comma **+0.5894**.)
> ⭐ **Honesty condition:** comma-only, MAE falls **42.5 %** but **medAE moves only −1.1 % and nMedAE
> gets 8.0 % WORSE**, Spearman ρ flat (+0.001). **The repair fixes the tail and the summary statistic,
> not typical accuracy** — which is also why the `medAE`/`nMedAE` columns above stay broadly readable.
>
> 🔴 **AMENDED 2026-07-27 (`anchor-settlement`, class C43): the `comma2k19` column of the table
> above is WITHDRAWN; the `PhysicalAI` column and the repair-beats-deletion ordering are NOT.**
> *(Nothing above is rewritten.)* BY CONTENT — sha256 of raw pose bytes **and** raw `frames_u8`
> sensor bytes — **2 of those 22 comma val episodes are bit-identical to 2 of the deployed head's
> own 40 comma TRAINING clips**; without them the same head reads comma yaw **−0.746**
> (CI [−1.574, −0.177]). The pooled **+0.8108** inherits it through its comma half. The ordering
> survives because the 2 leaked clips carry **0 of the 9** impossible legacy labels.
> `R0LEG`'s **+0.5894** is **not** contaminated (content-disjoint training split) but reads
> **+0.0639 [−0.374, +0.334] — not separated** on the 20 content-clean episodes, so it should not
> be read as a `B0` re-qualification. Record:
> `…/Benchmarks & Eval/Implementation/incoming/2026-07-27-anchor-settlement/ANCHOR_SETTLEMENT.md`.
> Source: `…/2026-07-27-idm-v3/results/compare_v3.json`. Inventory:
> `…/incoming/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md`.

### 3.3 long_accel — the channel should be removed

| arm | R² vs the CAN label | R² vs the **kinematic** target dv/dt |
|---|---|---|
| A0 | −0.240 | −0.599 |
| B0 | −0.214 | −0.533 |
| **best anywhere (S6, MAD standardiser)** | **+0.117** | −0.136 |
| V3wB (v2) | −0.217 | −0.525 |
| B4 / V2 (derived by differentiating a predicted speed *sequence*) | −1.10 / −0.86 | −1.52 / −1.07 |
| post-hoc: differentiate A0's sliding-window speed track (w = 9 / 17 / 25) | −25.97 / −11.06 / −5.48 | −35.29 / −14.79 / −7.18 |

Nothing works. And it cannot: `IDM_DIAGNOSIS.md` §3.3 measures that on PhysicalAI
the CAN `long_accel` label correlates **r = 0.434** with the vehicle's own dv/dt,
so a *perfect* kinematic estimator caps at **R² 0.188** against it; while on
comma, where the label *is* dv/dt, a linear probe on the frozen latent reads
**−0.095**. **Recommendation (pre-committed, independent of these arms): remove
`long_accel` from `SCALAR_NAMES`.** It consumes 25 % of the scalar loss to emit a
number no consumer may use.

### 3.4 steer — keep it, and retract "unusable"

| arm | regressed steer R² | **derived** steer R² (bicycle geometry from ŷaw/v̂) |
|---|---|---|
| A0 | **0.742** | 0.754 |
| B0 | 0.415 | 0.497 |
| V3wB (v2) | 0.520 | 0.424 |

The derived channel is not reliably better, so the pre-registered licence to drop
steer **fails** — steer stays a regressed output. Separately, the card's claim that
*"steer … [is] DROPPED as unusable (pilot NOTE 2026-07-24)"* is **retracted**: the
cited NOTE does not mention steer, no artifact measures it as unusable, and A0
reads **0.742** here. The defensible statement is *"PhysicalAI steer is
`atan(2.9·κ)` and comma steer uses `STEER_RATIO = 15.3` — they are not the same
quantity, so steer is per-corpus, not pooled."* **Root-cause class C4 (inherited
without re-verification).**

### 3.5 trajectory — with the lat/lon split (M1)

Seed-mean of the per-seed metrics; DE per horizon at 0.5 / 1 / 1.5 / 2 s.

| arm | ADE@2s | lon MAE @2 s | lat MAE @2 s | **lat p90 @2 s** | lon share of sq. err | DE per horizon |
|---|---:|---:|---:|---:|---:|---|
| **A0** | 3.828 | 5.999 | 0.829 | **2.224** | 0.983 | 1.50 / 3.02 / 4.58 / 6.21 |
| B0 | 4.659 | 7.346 | 0.897 | 2.164 | 0.990 | 1.83 / 3.67 / 5.57 / 7.57 |
| B1 | 4.630 | 7.304 | 0.914 | 2.186 | 0.989 | 1.82 / 3.65 / 5.53 / 7.53 |
| S3 (ctx only) | 3.881 | 6.060 | 0.912 | 2.322 | 0.981 | 1.53 / 3.05 / 4.64 / 6.31 |
| **V3w** | **3.658** | 6.033 | 0.908 | 2.235 | 0.978 | 1.36 / 2.75 / 4.26 / 6.26 |
| **V3wB** | **3.732** | 6.077 | 0.900 | 2.280 | 0.978 | 1.41 / 2.83 / 4.37 / 6.32 |

Paired ADE (on the seed-averaged prediction) vs **B0**: V3w **−1.014 [−2.064,
−0.221] separated ✓**; V3wB **−0.981 [−1.912, −0.259] separated ✓**. Paired vs A0:
−0.22 [−1.01, +0.60], not separated.

**Two things the decomposition says that the ADE alone hides:**
- The whole v2 gain is **longitudinal** (7.35 → 6.08 m at 2 s). **Lateral error is
  unchanged to slightly worse everywhere (0.83–0.92 m mean, p90 2.16–2.32 m)** —
  and A0 is still the best lateral arm. Nothing in v2 was aimed at lateral, and
  the instrument confirms it did not accidentally help.
- **lon share of squared error is 0.978–0.990 on every arm**, i.e. this val set sits
  at the very top of the 0.607–0.976 range measured across the 8 committed arms.
  Any lateral claim from *this* corpus would be underpowered, and I make none.

---

## 4. The positive control fired, and three candidate ideas were refuted

### 4.1 P1 — the contamination mechanism is CONFIRMED

`B0` retrained with **1 % of train `yaw_rate` labels replaced by ±8 rad/s`,
everything else identical. Paired vs B0:

| channel | Δ (P1 − B0) | separated? |
|---|---|---|
| **yaw_rate MAE** | **+0.0028 [+0.0010, +0.0047]** | **yes — degraded** |
| **yaw_rate medAE** | **+0.0035 [+0.0018, +0.0054]** | **yes — degraded** |
| speed MAE | +0.049 [−0.018, +0.117] | no |
| ADE@2s | +0.054 [−0.030, +0.141] | no |

**A 1 % label contamination degrades yaw by a CI-separated amount and leaves speed
and the trajectory untouched.** That is the mechanism the diagnosis proposed,
reproduced on demand and channel-specific. The pre-registered falsifier did **not**
fire.

### 4.2 Refuted: "predict a trajectory and differentiate for accel"

Measured **two independent ways**, both far worse than regressing the target:
in-model (arms B4/V2/V2w: R² −0.29 to −1.52 vs kinematic) and post-hoc on the
deployed head's sliding-window speed track (R² −7.2 to −35.3, `accel_derive.json`).

⚠️ **I predicted the opposite and was wrong — recorded rather than buried.** The
error budget in `accel_budget.json` estimated the *white* part of A0's per-frame
speed error at 0.360 m/s on PhysicalAI (vs 3.30 m/s smooth) and therefore predicted
a derived-accel **R² ceiling of +0.657 at w=9 / +0.898 at w=17**. Direct measurement
gives **−8.81 / −4.13**. **Root cause: the budget assumed the post-smoothing
residual is spectrally white. It is not — and the derivative amplifies exactly the
non-white part.** Class **C3 (mechanism instead of measurement)**; caught only
because the measurement was run. The budget JSON is kept in this directory *with*
this correction attached.

### 4.3 Refuted: log-speed parametrisation

Arm S2 (log-speed alone off B0): speed R² **0.7815 → 0.6749**, MAE 3.573 → 4.147.
Paired vs A0 Δ MAE **+1.153 [+0.100, +2.516] separated — significantly WORSE.**
A squared loss in log space optimises *relative* error while the metric is
*absolute*. (This also fired the pre-registered falsifier and forced a retraction
inside `IDM_DIAGNOSIS.md` §5.3 — see that note.)

### 4.4 Refuted: the MAD (median/1.4826·MAD) standardiser

Arm S6: it is the **best arm in the program for yaw (nMedAE 1.280) and the only one
with positive `long_accel` (+0.117)** — and it destroys speed: R² 0.7815 → 0.4433,
MAE 3.573 → 6.567, ADE 4.659 → 8.373, all CI-separated. It is not a label fix at
all: it silently **re-weights the loss across channels by up to 14×** (yaw's scale
goes 0.246 → 0.0175). Arm S7 (winsorise + MAD) is numerically **identical** to S6,
which confirms MAD is insensitive to the 0.3 % outliers — i.e. the two knobs are
genuinely orthogonal, and only one of them is a label fix.

> This is the single most useful negative result here: it is a **per-channel loss
> weighting dial**, and it shows the four channels are in direct competition for
> the same gradient. It should be used deliberately (as a weight), never as a
> "robustness" default.

### 4.5 Which single change actually did it

All measured off B0, single-change arms:

| change | speed R² | Δ MAE vs B0 | separated? | ADE |
|---|---|---|---|---|
| **clip context (S3)** | **0.8870** | **−0.641 [−1.506, +0.044]** | no (borderline) | 3.881 |
| winsorise (B1) | 0.7895 | −0.024 | no | 4.630 |
| 17-frame window (S5) | 0.7894 | −0.021 | no | 4.691 |
| log-speed (S2) | 0.6749 | +0.574 | no (worse) | 4.771 |
| derived targets (S4) | 0.7765 | +0.252 | no (worse) | 4.918 |
| MAD standardiser (S6) | 0.4433 | +2.994 | **yes — worse** | 8.373 |
| **all four v2 changes (V3w/V3wB)** | **0.909 / 0.912** | **−0.919 / −0.906** | **YES** | **3.658 / 3.732** |

**No single change is CI-separated; the combination is.** The clip-context token
carries most of it, and the label fix, the longer window and the smaller head each
add a sub-threshold increment that only becomes decision-grade together.

---

## 5. Answering the PI: more training, better architecture, better targets, or better substrate?

**Not more training — that is measured and closed.** Data flat past 34 clips, steps
flat past 25 epochs (and *harmful* past 25 for `long_accel`), and the shipped 2.90 M
head is beaten by a **0.86 M** one. Three axes, 8.5× / 20× / 23× ranges, all flat or
negative. A *linear* probe on the frozen latent matches the trained transformer on
every channel.

**Better targets — yes, and it is the cheapest win available.** Nine impossible
labels cost 0.39 of pooled yaw R². `long_accel`'s label is not the quantity the
video shows (r = 0.434 with dv/dt on PhysicalAI; a perfect estimator caps at 0.188)
and should be removed. `steer` is per-corpus, not pooled. **None of this needs a GPU.**

**Better architecture — one change, and it is small.** The clip-context token
(+0.106 speed R² off B0) is the only architectural change that pays, and it pays
because it supplies *information*, not capacity — it lets an offline labeler
condition on the clip it is labelling. Everything else in the architecture space
tested here (log-space outputs, derived targets, longer windows alone, more
parameters) is neutral or negative.

**Better substrate — yes, and that is where the next GPU-day belongs.** Two
measurements point at it and nothing else can fix them:
1. **`long_accel` is absent from the latent.** On comma2k19, where the label is
   clean, a linear probe reads **−0.095** and every head reads ≤ +0.12. Acceleration
   is not in this representation.
2. **Speed's residual is an information deficit, not a fitting deficit.** After the
   clip-level term is removed (oracle) A0 reaches R² 0.942 — the remaining 44 % of
   MSE is within-clip, and the clip-level part is a *shrinkage toward the training
   prior* (gain 0.830), which is the MMSE-optimal response to not knowing the metre.
   No loss and no output space can remove it. Only **metric grounding** can —
   ground-plane geometry with a real camera height, not the per-corpus constant
   (measured to be a **no-op**, §5.2 of the diagnosis), and not a per-corpus
   calibration table.

**Ranked next actions**

1. **(0 GPU) Data QC**: reject `|yaw_rate| > 1.5 rad/s` and any heading derived at
   v ≈ 0 at ingest; re-derive comma2k19 heading from a source that is defined at
   standstill. Fix the loader, not the head.
2. **(0 GPU) Contract change**: drop `long_accel` from `SCALAR_NAMES`; report
   `yaw_rate` and `steer` **per corpus**; publish R² only alongside nMedAE and ρ.
3. **(0 GPU) Card corrections**: retract "steer DROPPED as unusable"; re-issue the
   card's `val_heldout_traindomain` yaw number with the admissibility filter.
4. **(small) Ship `idm_head_v2`** = winsorised labels + clip-context token +
   17-frame window + 0.86 M head, retrained on A0's full corpus (not the 68 clips
   available here) so it inherits A0's PhysicalAI data as well as the v2 changes.
5. **(the real experiment) Metric grounding for speed** — a ground-plane /
   optical-flow scale module using a *measured* camera height, pre-registered
   against the 0.942 oracle ceiling. Note first that the repo carries **three
   mutually inconsistent `cam_h` values (1.5 / 1.43 / 1.22)**; reconciling them is
   a prerequisite, not a detail.
6. **Do not** spend a GPU-day on unfreezing/LoRA for `long_accel` before a cheap
   auxiliary-signal probe (flow or depth) shows acceleration is recoverable at all.
   *(Repo evidence, INHERITED: light fine-tuning of 4 encoder blocks moved accel
   from −0.006 to −0.023 — i.e. nothing.)*

---

## 6. Honest limits

- **68 train episodes** (26 PhysicalAI / 42 comma) vs A0's 160 clips. Every "v2 vs
  A0" statement is confounded by that; "v2 vs B0" is not, and is the one I lean on.
- **2–3 seeds per arm**; V3w's configuration was run as two independent seed groups
  (0–1 and 2–4) which agree to 0.003 R², but that is a replication, not a power
  analysis.
- **36 val episodes** is the resampling unit. It is enough to separate v2 from B0
  and P1 from B0; it is **not** enough to separate v2 from A0, and I do not claim it.
- The comma2k19 yaw mechanism is read from the shipped loader
  (`comma2k19.py:172` — `arctan2` of ENU velocity) — **INHERITED** at line level;
  the frame-level consequence (v ≈ 0, ±15.5 rad/s) is MEASURED here.
- Arm `V2*`'s regressed `steer` R² values of −26 to −256 are **not** a finding:
  those arms leave the steer head unsupervised by design. Read `steer_derived` for
  them.

## 7. Deliverable manifest

All in `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-idm-v2/`
(repo working tree — **NOT staged, NOT committed**, per the brief):

| file | what |
|---|---|
| `IDM_DIAGNOSIS.md` | the (a)–(d) verdict per channel, with probe/label/curve numbers |
| `PRE_REGISTRATION_IDMV2.md` | written before any arm ran |
| `IDM_V2_RESULTS.md` | this file |
| `labels.json`, `labels2.json` | label noise floors, comma yaw audit, long_accel provenance, variance decomposition |
| `probe.json` | the first per-channel linear probe on frozen v1 latents (centre / 9 / 17 frames, cross-domain) |
| `curve.json` | learning curves vs data / steps / capacity |
| `scale.json`, `speed_decomp.json` | per-corpus calibration no-op; the clip-level/within-clip speed error split |
| `accel_budget.json`, `accel_derive.json` | the (falsified) error budget and the direct derived-accel measurement |
| `v2_results.json`, `v2b_…`, `v2c_…`, `v2d_…` | all 22 arms, per seed, per channel, per domain |
| `compare.json` | every paired episode-cluster bootstrap |
| `analyze_console.log` | the console transcript of the comparison |
| `idm2_*.py` (11 files) | encode · lib · 3 diagnosis stages · scale · speed decomp · accel budget · accel derive · v2 ladder · analyze |

**Pod-side (`tanitad-eval:/root/idm2/`)**: `lat/` (104 encoded episodes, ~230 MB — the
only artifact not copied to the repo, since it is regenerable in 102 s by
`idm2_encode.py`), plus identical copies of everything above.
**Nothing is stranded on the pod.**

**Escalation (integration, not a README request):** items 1–3 of §5 are **loader and
contract changes outside this directory** — `stack/tanitad/data/comma2k19.py`,
`stack/scripts/idm_head.py::SCALAR_NAMES`, and
`…/2026-07-25-idm-youtube-validation/idm_head_v1_card.json`'s `usage_caveats`.
They are 0-GPU and they change published numbers. **They need an owner assigned, not
a note in this file.**
