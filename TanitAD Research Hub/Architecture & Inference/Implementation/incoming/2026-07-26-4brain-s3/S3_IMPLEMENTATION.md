# S3 — manoeuvre-initiation timing: IMPLEMENTATION

**Date:** 2026-07-26 (Europe/Berlin) · **Stream:** 4-Brain Dominance Program, problem **S3**
**Pre-registration:** `PRE_REGISTRATION_S3.md` (this directory) — **written and fixed before any number below**
**Compute:** dev box, **CPU only**. No GPU. No pod SSH — pod1 (v2corpus), pod3 (E1c) and the eval pod (wheelbase) untouched; pod2 untouched.
**Nothing `git add`-ed, committed or pushed. No training launched.**

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another of our docs, not re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. Headline — the five things that decide S3's fate

| # | Result | Class |
|:--:|---|---|
| **1** | ✅ **The target is NOT circular.** A scene-blind head reaches **QWK 0.3898** (lateral) / **0.2334** (longitudinal) against a ceiling of 1.0 — the spec §0.1 refusal threshold is 0.98. **R1 does not fire on either axis.** | `MEASURED` |
| **2** | ⚠️ **But S3 is a partial ECHO of the model's own conditioning.** Handing the blind head `route`/`route_graded` moves lateral QWK from **0.1128 → 0.3381**, paired Δ **+0.2254 [+0.0631, +0.3674]**, **CI-separated**. **69 % of everything a blind head knows about `ttm_lat` comes from a channel the model is handed at inference.** R2 is **armed**: the withheld variant **S3-W** must run beside the specified one. | `MEASURED` |
| **3** | ✅ **The clock artifact is CLEARED.** Adding the observable-horizon channel changes nothing: paired Δ **−0.0014 [−0.1270, +0.1101]** (lat) and **+0.0209 [−0.1593, +0.2455]** (lon), neither separated. **Rule M1 worked** — the target is not "how much clip is left". | `MEASURED` |
| **4** | ⛔ **S3 is UNUSABLE as a two-arm control on the 40-episode published eval set: 32 (lat) / 29 (lon) episode-clusters — below even the 40-cluster SINGLE-arm bar.** It **is** usable on the 600-episode `physicalai-val-0c5f7dac3b11`: **486 / 438** projected clusters, above the 200 bar. **Said now, not after a 30-pod-day run.** | `MEASURED` + `ESTIMATED` projection |
| **5** | ⚠️ **The corpus caps the horizon at ~12 s.** PhysicalAI clips are 18.8–20.5 s, so requiring a fully-observable decision horizon costs **58 %** of all windows before any other rule. **S3 covers ~1–12 s of the spec's nominal 5–25 s and structurally cannot test the upper half.** | `MEASURED` |

---

## 1. What was built

| Artifact | Path (repo working tree, **NOT staged**) | What |
|---|---|---|
| Pre-registration | `PRE_REGISTRATION_S3.md` | target · miner · option set · metric · **all three outcomes** · refusal conditions |
| Target + miner + metrics | `s3_labels.py` | `ttm_lateral` / `ttm_longitudinal`, `lat_in_progress` / `lon_in_progress`, `mine_episode`, QWK, per-band recall, `mae_skill_s`, bootstrap wrappers |
| The firewall | `s3_blind_baseline.py` | `blind_conditioning_baseline` (corpus-agnostic) + `refusal_verdict` |
| Driver | `run_s3_characterisation.py` | mines a cache, emits coverage / option set / power / firewall JSONs |
| Tests | `test_s3_labels.py` | **20 tests, all passing** — incl. the non-circularity assertion and the synthetic-echo REFUSAL regression |
| Results | `s3_{coverage,option_set,power,blind_baseline}_primary.json` | primary, `H_S3 = 12 s` |
| Results | `s3_{…}_sens_h8.json` | pre-registered sensitivity arm, `H_S3 = 8 s` |

Run: `python run_s3_characterisation.py --train-cache <dir> --test-cache <dir> --out-dir . --tag primary`

### 1.1 ⚠️ Corpus caveat that travels with every number below

**`MEASURED`.** The parity pose caches (`/workspace/v15/poses_{train,val}.pt`,
`…/_epcache/physicalai-train-e438721ae894`) are **pod-side**. The dev box holds two *different*
PhysicalAI-AV epcaches — `physicalai-train-14231cd29c74` (400 eps) and
`physicalai-val-bb543bdf7836` (100 eps) — built by the **same** `build_episode` pipeline from the same
R0 corpus. Overlap with the parity corpus by `episode_id`: **70/400 = 17.5 %** and **17/100 = 17.0 %**
(`MEASURED`, cross-checked against `parity_profile.csv`'s 2,376 rows).

> **Therefore: every number in §3–§6 is `MEASURED` on a NON-PARITY PhysicalAI-AV cache** and is a
> **distributional characterisation of the instrument**, not a parity-corpus fact. **No cross-arm
> comparison is made or implied.** The decision-grade re-run is a **`--cache-dir` swap** to
> `physicalai-train-e438721ae894` / `physicalai-val-0c5f7dac3b11`, pod-side, **zero code change**.
>
> **Transfer is bounded, not assumed.** The two caches match the parity corpus on the structural
> quantities S3 depends on: clip length `T ∈ [197, 205]` here vs `[188, 205]` parity (`MEASURED`,
> `parity_profile.json`), same 10 Hz contract, same builder. They are **not** matched on scenario mix
> — parity carries `junction` presence **0.3784** and `has_turn` **0.4259** (`MEASURED`), which the
> local caches were never profiled against. **Coverage and class balance may shift on parity; the
> firewall verdicts and the power arithmetic will not.**

The miner prints the resolved cache and `PARITY=YES/NO` on every run and **refuses**
`physicalai-val-f1b378f295ae` by name (78.5 % leaked into the parity train).

---

## 2. The target — definition and the non-circularity proof

### 2.1 Two clocks, never one

For a window whose last **observed** pose index is `L`:

* **`ttm_lat(L)`** — seconds to the start of the first **junction-scale curvature segment** in
  `poses[L+1 : L+121]`. Delegates to the **shipped** `refb_labels.route_from_future_v3` →
  `v4_labels.time_to_maneuver` with `horizon_steps = 120`.
* **`ttm_lon(L)`** — seconds to the start of the first **sustained longitudinal segment** in the same
  span: same-sign run of savgol-smoothed `|a| ≥ 0.5 m s⁻²`, `≥ 1.0 s`, total `|Δv| ≥ 1.5 m s⁻¹`.

**Why two.** The shipped `strat_scalars.ttm` is a **lateral-only** clock — a model can score perfectly
on it while being blind to *when to start braking*. That is the same defect class as the v1 5-way
softmax mixing lateral and longitudinal (`0/881 accelerate`). The two axes are minted and scored
**separately and never pooled**; `test_axes_are_independent_a_turn_is_not_a_brake` asserts that a pure
turn mints no longitudinal event and a pure brake mints no lateral one.

### 2.2 Fields the target touches — and the proof

| field | index range | model input? |
|---|---|---|
| `poses[:, 0:2]` x, y | `L+1 … L+120` — **strictly future** | ❌ |
| `poses[:, 2]` yaw | `L+1 … L+120` — **strictly future** | ❌ |
| `poses[:, 3]` v | `L+1 … L+120` — **strictly future** | ❌ |

Nothing else. No map, no agent track, no camera, no label, no `nav_cmd`.

> **Proof (index-range disjointness).** The observation ends at pose index `L`; the target reads only
> `poses[L+1:]`. The ranges are **disjoint**, so no fixed rule maps inputs onto the target. This is
> strictly stronger than the v1 route label, which *was* its input
> (`route_target(nav_cmd) = _NAV_TO_ROUTE[nav_cmd]`, `refb_labels.py:172-175`) ⇒ `route_skill = 0.0000`
> by construction.

**Asserted in code, not prose:** `test_target_never_reads_the_observed_window` overwrites `poses[:L+1]`
with Gaussian noise and requires every S3 target to be bit-identical; the complement test
`test_target_DOES_move_when_the_future_moves` stops that passing on a constant. **`MEASURED`: both pass.**

### 2.3 ⛔ The surface restriction that PC2 forces

S3 is scored on **`open-loop-choice` only**. It is **never** scored on `rollout_decode` under the
expert's true future actions — on that surface the future action sequence *is* the future path, so
`ttm` is recoverable from the inputs and S3 becomes circular. Registered in the pre-registration §1 so
it cannot be re-derived later.

---

## 3. The miner — rules, and what each one costs

`MEASURED`, `s3_coverage_primary.json`, `H_S3 = 12 s`, `MIN_TTM_S = 1.0 s`.

### 3.1 Coverage funnel

| stage | val (100 eps) lat | val lon | train (400 eps) lat | train lon |
|---|---:|---:|---:|---:|
| all dataset windows | 17,100 | 17,100 | 68,377 | 68,377 |
| **M1** full 12 s horizon observable | 7,200 (42.1 %) | 7,200 | 28,777 (42.1 %) | 28,777 |
| **M4** ego moving (≥1 m s⁻¹) | 5,079 (29.7 %) | 5,079 | 21,929 (32.1 %) | 21,929 |
| **M2** rejected — manoeuvre already begun | −1,202 | −1,925 | −5,722 | −8,825 |
| **M3** rejected — window still executing | −418 | −1,157 | −1,723 | −4,124 |
| ✅ **admissible decision points** | **3,459** | **1,997** | **14,484** | **8,980** |
| coverage of all windows | **20.23 %** | **11.68 %** | **21.18 %** | **13.13 %** |
| coverage of M1∩M4 windows | 68.10 % | 39.32 % | 66.05 % | 40.95 % |

**Against the reported in-corpus figures.** The brief carried `ttm 20.5 % / dist 62.9 %` as `INHERITED`.
**Verified `MEASURED`** by reading the artifact directly (`labels_train_v4_provenance.json`):
`strat_scalars.ttm = 0.2048` train / `0.2076` val, `dist_target = 0.6292` / `0.6339`. ✅ The brief's
numbers are correct.
**And they are not the same quantity as S3's.** The shipped 20.48 % is *"a curvature segment exists in
a 25 s lookahead and the future arc reaches it"* — right-censored on ~20 s clips, and it **includes
windows where the manoeuvre is already under way**. S3's lateral coverage lands at **20.23 %** for a
different reason (M1 discards 58 % of windows; M2/M3 discard another 32 % of what survives; but S3
counts `t_none` as a *label*, which the shipped mask does not). **The agreement is a coincidence of two
different constructions and must not be quoted as a validation of either.**

### 3.2 ⚠️ One instrument correction, made before any arm was scored — and one near-miss

**The correction (kept, documented in `s3_labels.py`).** The first M3 implementation tested
"has it begun?" with an **instantaneous** threshold on **raw** per-step acceleration, while the target
is a **sustained** segment of **savgol-smoothed** speed. That mismatch made M3_lon fire on ordinary
speed noise and admit only windows inside a steady-speed lull, so the next event was always imminent:
`MEASURED` on the 6-episode smoke, **3 of 5 longitudinal classes came out empty** (`t_5_10`,
`t_10_H`, `t_none` all 0). M3 now runs the **same segment detector as the target, backwards from `L`,
on the same smoothed signal**. No model existed, so nothing could be flattered.

**The near-miss, recorded because it is the more useful half.** On that 8-episode read the longitudinal
option set still looked degenerate (`t_none = 0.000`, median `ttm_lon` 2.2 s) and the obvious move was
to **raise the longitudinal manoeuvre threshold**. A pre-declared, outcome-independent sweep over
`(A_MAN, DV_MIN)` on **60 episodes** says the opposite (`MEASURED`, sweep in scratchpad, reproduced below):

| `A_MAN` | `DV_MIN` | admissible | `t_1_2` | `t_2_5` | `t_5_10` | `t_10_H` | `t_none` | majority |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.5** | **1.5** | 0.323 | 0.287 | 0.377 | 0.192 | **0.021** | **0.123** | **0.377** ✅ |
| 0.8 | 2.5 | 0.521 | 0.133 | 0.279 | 0.243 | 0.009 | 0.336 | 0.336 ✗ |
| 1.0 | 3.0 | 0.595 | 0.117 | 0.219 | 0.208 | 0.006 | 0.451 | 0.451 ✗ |
| 1.5 | 4.5 | 0.778 | 0.051 | 0.123 | 0.080 | 0.000 | 0.746 | 0.746 ✗ |
| 2.5 | 6.0 | 0.968 | 0.004 | 0.000 | 0.013 | 0.000 | 0.982 | 0.982 ✗ |

**Every higher threshold is worse** — `t_10_H` empties and `t_none` swallows the problem. **The
pre-registered threshold stands unchanged; the 8-episode signal was a small-sample artifact.** This is
the CLAUDE.md *"verify before alarming — take multiple samples first"* trap, caught at 8 episodes
instead of after a re-mint of the corpus.

### 3.3 Per-stratum — because a pooled S3 number cannot answer HP-2

`MEASURED`, episode-clusters with ≥1 admissible decision point, val 100 eps / train 400 eps:

| speed stratum | val lat | val lon | train lat | train lon | median `ttm_lat` | ≥40 bar (val) | ≥200 bar (train) |
|---|---:|---:|---:|---:|---:|:--:|:--:|
| city `< 8 m/s` | **65** | 51 | **250** | 189 | 3.7 s | ✅ | ✅ (lat) |
| mid `8–15 m/s` | **50** | 34 | **202** | 157 | 6.1 s | ✅ lat / ✗ lon | ✅ (lat) |
| ⛔ highway `≥ 15 m/s` | **2** | 2 | **7** | 4 | 4.8 s | ✗ | ✗ |

> ⛔ **The highway stratum does not exist for S3 on this corpus** — 2 val clusters, 7 train clusters.
> Any S3 result is a **city + mid-speed** result and must be labelled as one. Projected to the
> 600-episode val the highway stratum reaches ~12 clusters: still below the single-arm bar.

---

## 4. The option set — justified from the data's own distribution

### 4.1 The full distribution (reported *with* the bands, per the pre-registration)

`MEASURED`, val, event windows only (`t_none` excluded):

| axis | n | mean | deciles of `ttm` (s): 0/10/…/100 |
|---|---:|---:|---|
| **lateral** | 2,483 | **5.204 s** | 1.0 · 1.7 · 2.5 · 3.3 · 4.1 · **5.0** · 5.9 · 6.8 · 7.9 · 9.1 · 11.1 |
| **longitudinal** | 1,603 | **4.066 s** | 1.0 · 1.4 · 1.8 · 2.3 · 2.8 · **3.4** · 4.1 · 5.1 · 6.4 · 8.0 · 11.0 |

Train reproduces this closely (lat mean 5.203, median 5.0; lon mean 4.214, median 3.6), i.e. the
distribution is a corpus property, not a val artifact.

### 4.2 PRIMARY banding — the spec's own edges, adopted unchanged

`MEASURED`, val:

| class | interval | lateral | longitudinal |
|---|---|---:|---:|
| `t_1_2` | [1, 2) s | 9.37 % | 18.83 % |
| `t_2_5` | [2, 5) s | 26.37 % | **36.55 %** ← majority |
| `t_5_10` | [5, 10) s | **32.55 %** ← majority | 23.59 % |
| ⚠️ `t_10_H` | [10, 12) s | **3.50 %** | **1.30 %** |
| `t_none` | no event in 12 s | 28.22 % | 19.73 % |
| **majority-class rate** | | **0.3255** | **0.3655** |
| **QWK of the majority baseline** | | **0.0000** (exact, by construction) | **0.0000** |

The spec's `2 / 5 / 10 s` edges are adopted **unchanged** precisely so nobody can claim they were
tuned. The distribution above shows they are also reasonable: they land near the 20th, 50th and 90th
percentiles of the lateral distribution.

> ⚠️ **`t_10_H` is a thin band on both axes (3.5 % / 1.3 %).** It is 2 s wide against `t_5_10`'s 5 s and
> it sits at the edge of the observable horizon. **Per-band recall on `t_10_H` will be noisy and must
> not carry a verdict on its own.** This is the honest cost of closing the top band at `H_S3`; the
> alternative (leaving it open) is right-censoring, which is worse.

### 4.3 SECONDARY banding — equal-mass quartiles (pre-registered robustness)

`MEASURED`, val: **lateral edges 2.9 / 5.0 / 7.3 s** → balance 17.8/18.0/17.6/18.4/28.2 %, majority
**0.2822**. **Longitudinal edges 2.0 / 3.4 / 5.7 s** → 18.8/21.0/20.2/20.3/19.7 %, majority **0.2098**.

Equal-mass is the **least** flattering binning — it minimises the majority baseline — which is why it
is the robustness arm and not the primary.

### 4.4 The MAE baseline (co-primary)

`MEASURED`, val: the MAE-optimal constant is **5.0 s** lateral (`MAE = 2.3046 s`) and **3.4 s**
longitudinal (`MAE = 1.9910 s`). **`ttm_MAE_skill_s` is measured against these, never against 0.**

---

## 5. ⭐ The blind-conditioning baseline — the mandatory pre-flight

**Setup, `MEASURED`.** Train = 400-episode cache (14,484 lat / 8,980 lon rows) → test = 100-episode
cache (3,459 / 1,997 rows, **81 / 73 episode-clusters**). **Episode-disjoint by construction** (two
different caches). 2-layer MLP (64-64), class-weighted (so the firewall cannot pass merely by
collapsing to the majority class), CPU, seconds. Intervals: **`episode_cluster_bootstrap`, B = 2000**;
deltas: **`paired_episode_cluster_bootstrap`**. `overlapping_holdout_se` **not used anywhere**.

### 5.1 What `X_cond` actually is — and the finding the spec did not anticipate

**`MEASURED`** (`stack/tanitad/models/flagship_v4.py:198-203`; fed at eval by
`stack/scripts/eval_flagship_v4.py:333` via `train_flagship_v4._goal_inputs`):

```
forward(states, v0, imagined=None, vt_band=None, route=None,
        route_graded=None, vt_speed=None, steps=None, lambda_plan=1.0)
```

> ⚠️ **Four of those inference-time conditioning channels are themselves derived from the ego's
> FUTURE.** `route` / `route_graded` come from `route_from_future_v21(poses, L)` over a **25 s**
> lookahead — the *same function family* that produces S3's lateral target. `vt_band` / `vt_speed` come
> from `vtarget_v2(v, L, min_lookahead=50)` — the ego's own speed **≥ 5 s ahead**, which is a statement
> about the longitudinal target. **A model handed these has been handed a low-resolution copy of both
> S3 clocks.** This is the real circularity risk for S3, and it is *not* the one §0.1 anticipated.

### 5.2 Results — `ttm_band_QWK`, val

| arm | features | **lateral** QWK [95 % CI] | **longitudinal** QWK [95 % CI] |
|---|---|---|---|
| **B0** majority-class | — | **0.0000** (exact) | **0.0000** (exact) |
| **B1** sensor-only | `v0`, in-window v/Δv/κ | **+0.1128** [−0.0010, +0.2271] | **+0.0676** [−0.0959, +0.2204] |
| **B2** + route | B1 + `route`, `route_graded` | **+0.3381** [+0.1406, +0.5069] | **+0.2334** [+0.0286, +0.4139] |
| ⭐ **B3** FULL conditioning | B2 + `vt_band`, `vt_speed` | **+0.3898** [+0.2308, +0.5293] | +0.1200 [−0.0529, +0.2913] |
| **B4** + clock (`H_obs`) | B3 + observable horizon | +0.3884 [+0.2085, +0.5547] | +0.1410 [−0.0412, +0.3140] |
| | **operative blind floor** = max(B1,B2,B3) | **0.3898** (B3) | **0.2334** (B2) |

**Paired deltas** (`paired_episode_cluster_bootstrap`, B = 2000, same windows, same resampled episodes):

| delta | lateral | longitudinal |
|---|---|---|
| **B2 − B1** (route conditioning) | **+0.2254 [+0.0631, +0.3674]** ✅ **separated** | +0.1657 [−0.0451, +0.3518] not separated |
| **B3 − B1** (all future conditioning) | **+0.2770 [+0.1334, +0.4088]** ✅ **separated** | +0.0524 [−0.0866, +0.1911] not separated |
| ✅ **B4 − B3** (clock artifact) | **−0.0014 [−0.1270, +0.1101]** not separated | **+0.0209 [−0.1593, +0.2455]** not separated |

*(B3 < B2 on the longitudinal axis is an MLP-capacity artifact of adding two noisy features to a small
problem, not evidence that `vt_*` hurts. That is exactly why the operative floor is the **max** over
B1/B2/B3 and not B3 alone.)*

### 5.3 Verdicts against the pre-registered refusal conditions

| rule | test | verdict |
|---|---|---|
| **R1 circular** | blind ≥ 0.98 × ceiling(1.0) | ✅ **NOT REFUSED.** 0.3898 (lat) / 0.2334 (lon) — far below 0.98. **The target is admissible.** |
| ⚠️ **R2 echo** | blind ≫ majority **and** model−blind not separated | **ARMED on the lateral axis.** Blind − majority = **+0.3898**, and **+0.2254 [+0.0631, +0.3674]** of it is `route`/`route_graded` alone. The second half needs a model. **`S3-W` (conditioning withheld) must run beside the specified variant** — pre-registered, no architecture change (the head owns a learned null row per channel). |
| ✅ **R3 clock** | B4 − B3 separated | **CLEARED on both axes.** Neither CI excludes 0. **M1 worked**: the target is not "how much clip is left". Also cleared at `H_S3 = 8 s` (lat [−0.087, +0.062], lon [−0.057, +0.123]). |
| **R4 / R5 power** | ≥40 / ≥200 clusters | see §6 — **fails on the 40-ep val, passes on the 600-ep val.** |

### 5.4 ⭐ The skill bars a model must clear — measurable today, zero GPU

> **`skill = QWK(model) − blind`, never `QWK(model)`.**
>
> | variant | lateral bar | longitudinal bar |
> |---|---:|---:|
> | **S3 as specified** (route + vt given) | **0.3898** | **0.2334** |
> | **S3-W** (conditioning withheld; pixels + `v0` only) | **0.1128** | **0.0676** |
>
> A lateral QWK of 0.35 would look like a result and is **worse than a head with no camera at all.**

The firewall has its own regression test: `test_synthetic_echo_label_is_REFUSED` builds a label that
*is* a conditioning channel and asserts `REFUSED=True`; `test_pure_noise_label_is_NOT_refused` stops
the check degenerating into "refuse everything". **`MEASURED`: both pass.**

---

## 6. ⛔ Power — the verdict, stated now

Standing bars (spec §0.4): **≥40** episode-clusters per stratum single-arm · **≥200** two-arm.
`MEASURED` on the 100-episode val: episode yield **0.81** (lat) / **0.73** (lon).

| val deployment | episodes | lat clusters | lon clusters | ≥40? | ≥200? |
|---|---:|---:|---:|:--:|:--:|
| `tanitad-pod (pod1):/root/valdata` | 12 | 10 | 9 | ⛔ | ⛔ |
| ⛔ **`tanitad-eval:/root/valdata` — every published number** | **40** | **32** | **29** | ⛔ | ⛔ |
| ✅ **`physicalai-val-0c5f7dac3b11` on the training pods** | **600** | **486** | **438** | ✅ | ✅ |
| parity train (`physicalai-train-e438721ae894`) | 2,376 | 1,925 | 1,734 | ✅ | ✅ |

> ### ⭐ VERDICT
> **YES — S3 is usable as a two-arm control at n ≥ 200, but ONLY on the 600-episode
> `physicalai-val-0c5f7dac3b11` cache (486 lat / 438 lon clusters).**
>
> ⛔ **NO on the 40-episode published eval set: 32 / 29 clusters — below even the 40-cluster
> SINGLE-arm bar.** Running S3 there produces a number that cannot support a single-arm claim, let
> alone a paired Δ. **The 600-episode val is not the val the eval pod holds**, so this is an
> operational prerequisite, not a formality.
>
> **Per stratum the answer is narrower still.** Only *city* and *mid-speed* clear 200 on the 600-ep val
> (~390 and ~300 projected). **Highway reaches ~12 and never clears any bar** (§3.3). Since HP-2 is a
> *stratified* prediction, S3 can contribute to the capacity-vs-structure test on **city and mid-speed
> only**.

*(The 600/40/2,376-episode rows are `ESTIMATED` — episode counts are `MEASURED`
(`VAL_PARITY_REPORT.md`, `parity_manifest.json`, `labels_val_v4_provenance.json`); the yield rate is
`MEASURED` on the local 100-episode cache and projected linearly. Re-measure on the parity cache before
quoting; the arithmetic will not change the ⛔/✅ pattern — the 40-ep set would need a yield above 5.0
to clear 200, which is impossible.)*

---

## 7. The metric — implemented, tested, and immune to the `route_acc = 1.0` shape

| role | metric | status |
|---|---|---|
| **PRIMARY** | `ttm_band_QWK` per axis | ✅ `quadratic_weighted_kappa`; **majority-class predictor scores exactly 0.0** (`test_majority_class_predictor_scores_EXACTLY_zero_qwk`, `MEASURED` \|κ\| < 1e-9); ordinal (`test_qwk_is_ORDINAL_a_far_miss_costs_more_than_a_near_miss`) |
| **CO-PRIMARY** | `ttm_MAE_skill_s` vs the median constant | ✅ `mae_skill_s`; 0.0 for the median constant by construction |
| **REQUIRED** | per-band recall, all 5 classes, both axes | ✅ `per_band_recall`; a dead band shows as `0.0` — the metric that would have caught **0/881 accelerate** |
| **REQUIRED** | `early_late_bias_s` (mean signed error) | ✅ late = safety failure, early = comfort failure; an undecomposed MAE hides which |
| Secondary | band acc, off-by-≤1 acc, majority rate | ✅ emitted **only inside** `band_metrics`, never alone |
| Estimator | episode-cluster bootstrap B=2000, paired for two arms | ✅ via `taniteval/ci.py` (callable-reducer path for QWK) |

**Lateral/longitudinal decomposition.** S3's target is a scalar time, so `taniteval/lateral.py` does not
apply to it — **the two-axis split IS S3's lateral/longitudinal decomposition**, and it is enforced at
mint time. If an S3 arm also emits a trajectory, that trajectory is decomposed with
`taniteval/lateral.py` and reported as a secondary; cross-track remains the safety-relevant axis there.

---

## 8. Sensitivity — `H_S3 = 8 s` (pre-registered arm)

`MEASURED`, `s3_*_sens_h8.json`, val:

| | `H_S3 = 12 s` (primary) | `H_S3 = 8 s` |
|---|---:|---:|
| admissible windows (lat) | 3,459 (20.2 %) | 5,290 (30.9 %) |
| episode clusters (lat) | 81 | 95 |
| **majority rate (lat)** | **0.3255** | **0.5509** ← `t_none` takes over |
| `t_10_H` occupancy | 3.50 % | **0.00 %** (empty by construction — the band lies beyond the horizon) |
| blind floor (lat) | 0.3898 | 0.4469 |
| clock check B4−B3 (lat) | −0.0014 [−0.127, +0.110] ✅ | +0.0 [−0.087, +0.062] ✅ |

**Read:** a shorter horizon buys coverage and clusters but **degrades the option set** — the majority
baseline climbs to 0.55 and one class vanishes, so the 8 s arm is a **4-class** problem whose QWK is not
directly comparable to the primary's. **`H_S3 = 12 s` remains primary.** The clock check clears at both.

---

## 9. Honest limits (from the pre-registration, now with numbers)

1. ⚠️ **The corpus caps the horizon.** Clips are 18.8–20.5 s; **M1 alone discards 57.9 % of all
   windows** (`MEASURED`: 17,100 → 7,200). **S3 covers ~1–12 s of the spec's nominal 5–25 s and
   structurally cannot test the upper half.** Anything beyond ~15 s needs a longer-clip corpus.
2. **M1 selects clip-early windows** — at `H_S3 = 12 s` only `L ≤ T−121`, i.e. roughly the first 7.8 s
   of each ~20 s clip. The scene mix there is not guaranteed to match the clip mean.
3. ⛔ **No highway stratum** (§3.3): 2 val / 7 train clusters. Every S3 result is a city + mid-speed result.
4. **`ttm_lat` is kinematic** — it cannot distinguish *"the ego chose to turn"* from *"the road turned"*
   (spec §6). Admissible for S3 specifically because *when* is a kinematic fact even when *which branch*
   is not.
5. **`ttm_lon` has no lead referent** — `lead_state` is a `None` stub (0 % coverage), so a
   lead-caused deceleration is indistinguishable from a chosen one. `ttm_lon` is *"when the ego's speed
   changed"*, not *"when the ego decided"*.
6. **All §3–§6 numbers are on a non-parity cache** (§1.1). Re-run pod-side before any number enters the
   registry.

---

## 10. What S3 now says about the dominance proof

S3 was commissioned as a **deliberate near-control**, and the instrument is now built to the same
standard as the treatments. Three things it can do that it could not this morning:

1. **It can be run.** The target is non-circular (R1 clear), the clock artifact is dead (R3 clear), the
   metric is chance-corrected, and the option set is measured rather than asserted.
2. ⚠️ **It comes with a bar, not a zero.** A lateral QWK below **0.3898** is *worse than a head with no
   camera*. Any S3 number quoted without that bar is the `route_acc = 1.0` shape again.
3. ⛔ **It tells you where it cannot run** — not on the 40-episode published eval set, and not on
   highway. **Both are known before a GPU is booked, which is the whole point of building the control
   first.**

**The three pre-registered outcomes (EXPECTED near-null · NULL-uninformative · ⛔ CAPACITY verdict) are
fixed in `PRE_REGISTRATION_S3.md` §6 and are not editable after the first arm is scored.** The
interaction test `Δ_S3 − Δ_S1` is what separates them and is **not optional**: a bare per-problem Δ
table cannot distinguish capacity from structure, which is the entire reason S3 exists.

---

## 11. ⚠️ Escalations — these must not live only in this file

1. ⛔ **S3 (and any decision-point problem with a comparable yield) cannot be two-armed on the eval
   pod's 40-episode val.** This is not an S3 fact — it is an **arithmetic ceiling on the eval
   deployment**: 40 episodes can never yield 200 clusters. **Any HP-1…HP-8 prediction whose power need
   is "n ≥ 200 clusters" has the same problem.** Owner needed on the eval-pod val deployment
   (600-episode cache or a documented refusal).
2. ⚠️ **`route`, `route_graded`, `vt_band` and `vt_speed` are future-derived quantities fed at
   inference.** S3 measured the consequence for *its* target (**+0.2254 [+0.0631, +0.3674]** blind lift
   from `route` alone). **The same channels condition every flagship-v4 eval in the program.** Whether
   they inflate other reported capabilities is **not** established here and should be probed with the
   same firewall.
3. **Firewall duplication.** `…/2026-07-26-4brain-preconditions/` did not exist when this ran
   (`MEASURED`, dev box). `s3_blind_baseline.py` is written corpus-agnostic so consolidation is a move,
   not a rewrite. **Whichever lands second must delete its copy** — an orthogonality instrument sat
   unmerged for 10 days because the request lived in a file nobody re-read.

---

## 12. Deliverable manifest

| Artifact | Where it lives | Staged? |
|---|---|:--:|
| `PRE_REGISTRATION_S3.md` | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-4brain-s3/` | ❌ **not staged** (per brief) |
| `S3_IMPLEMENTATION.md` (this file) | same dir | ❌ not staged |
| `s3_labels.py` · `s3_blind_baseline.py` · `run_s3_characterisation.py` · `test_s3_labels.py` | same dir | ❌ not staged |
| `s3_coverage_primary.json` · `s3_option_set_primary.json` · `s3_power_primary.json` · `s3_blind_baseline_primary.json` | same dir | ❌ not staged |
| `s3_*_sens_h8.json` (sensitivity arm) | same dir | ❌ not staged |
| Longitudinal threshold sweep (§3.2) | `sweep_lon_threshold.py` + `s3_lon_threshold_sweep.json`, same dir | ❌ not staged |
| Pose caches read (**read-only**) | `C:\Users\Admin\tanitad-data\physicalai\_epcache\physicalai-{train-14231cd29c74,val-bb543bdf7836}` | n/a |
| Parity caches **NOT** touched (the decision-grade re-run target) | `<pod>:/workspace/.../physicalai-train-e438721ae894`, `physicalai-val-0c5f7dac3b11` | n/a |
| Tests | `python -m pytest test_s3_labels.py -q` → **20 passed** | ❌ not staged |
| Pods touched · GPU used · training launched · commits · pushes | **none · none · none · none · none** | ✅ |
