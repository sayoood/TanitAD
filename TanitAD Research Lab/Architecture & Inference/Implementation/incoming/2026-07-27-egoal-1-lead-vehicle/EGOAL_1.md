# E-GOAL-1 — does lead-vehicle context close the longitudinal gap?

**Stream:** `2026-07-27-egoal-1-lead-vehicle` · **Host:** dev box, CPU only · **Date:** 2026-07-27
⛔ **No pod contacted.** pod1 (training), pod2 (FOV sweep), pod3 and the eval pod were never touched.
**Pre-registration:** `PRE_REGISTRATION.md`, written **before any number below was computed.**
**Estimator:** episode-cluster bootstrap, `taniteval/taniteval/ci.py`, **B = 2000**, unit = **the clip**;
**paired** form for every contrast. `overlapping_holdout_se` is **never called**.
Every fitted number is **out-of-fold and clip-disjoint** (5 folds, seed 0).

---

## 0. HEADLINE — the lead-vehicle hypothesis fails, and the axis it was aimed at OPENS ANYWAY

> ### ⛔ **Lead-vehicle state is worth 2–3 %. It is not the longitudinal lever and never could have been: 41.65 % of windows have no vehicle ahead within 50 m at all.**
>
> ### ⭐⭐ **But run through the ACTUAL RULE, a goal head built from EGO DYNAMICS recovers `+23.6 %` of the fan's headroom, SEPARATED — where the parent's head, pushed through the IDENTICAL construction, is separated-WORSE at −33.1 %. The longitudinal goal is realisable. `obstacle.offline` is not why.**

### 0.1 The along-track head, out-of-fold, clip-disjoint (99,935 windows / 612 clips)

| arm | along-track RMS | 2 s-mean speed err | vs the pre-registered **0.813 m** |
|---|---:|---:|---|
| `CV` — constant velocity, no fit | 1.3827 m | 0.6913 m/s | 1.70× |
| `E0_v0` — **the parent head's ego content** (`v` only) | 1.5246 m | 0.7623 m/s | 1.88× |
| *(reference)* parent's best OOF goal head, canonical 881 `INHERITED` | *1.151 m* | *0.576 m/s* | *1.42×* |
| **`E1_ego` — ego kinematics + 1 s of history** | **0.9305 m** | **0.4652 m/s** | 1.14× |
| **`E1_L` — ⭐ + lead vehicle (gap / closing / TTC)** | **0.8983 m** | **0.4491 m/s** | 1.10× |
| `E1_L_X` / `E1_L_X_D` — best-effort lead + density | 0.9056 / 0.9106 m | 0.4528 / 0.4553 m/s | 1.11× / 1.12× |
| `L_only` — lead features **alone** | 18.1743 m | 9.0871 m/s | 22.4× |

### 0.2 ⭐ The requirement curve, recomputed for the family that actually applies

The pre-registered **0.813 m** was derived on the parent's **`ISO`** family. The measured heads match
neither pre-registered family: they are **near-unbiased** (shrinkage α ≈ 0.996, so not `SHRINK`) and
**strongly heavy-tailed** (RMS/MAE **1.867** against a Gaussian's 1.2533, so not `ISO`). Sweeping a
scale factor over **this stream's own OOF residual pool**, through the real REF-C rule:

| | family-matched (**EMPIRICAL**) | parent's `ISO`-conditional |
|---|---:|---:|
| **σ₀ — break-even** | **1.1434 m** | 0.813 m |
| **σ₅₀ — half prize** | **0.5907 m** | 0.439 m |

> ⇒ **The correctly-computed break-even is 1.41× more forgiving than the bar this experiment was
> pre-registered against**, and `E1_L`'s 0.8983 m **clears it by 1.27×**. Curve monotone ✅.
> *(This is retraction class `RMS-PLACED-ON-A-NOISE-CURVE` firing for the second time in two days —
> the parent measured it over-predicting damage 5.7×; here it under-states the admissible error
> by 1.41×.)*

### 0.3 The placement — the number that decides, `recovery` of the −0.2705 headroom

Measured by **running each error structure through the REF-C-xl rule**, not read off a curve.
Canonical 881 windows / 40 episodes; the parent's learned cross-track held fixed throughout.

| along-track error fed to the selector | RMS | realised `ade_0_2s` | **recovery** | paired vs as-trained | sep |
|---|---:|---:|---:|---:|---|
| `E0_v0` (the parent head's ego content) | 1.5246 | 0.6504 | **−66.1 %** | +0.1789 [+0.1111, +0.2435] | ✅ **worse** |
| `CV` | 1.3827 | 0.6275 | −57.7 % | +0.1561 [+0.0892, +0.2203] | ✅ **worse** |
| ⛔ **parent's own head residual, same construction** | 1.1510 | 0.5608 | **−33.1 %** | — | ✅ **worse** |
| **`E1_ego`** | 0.9305 | **0.4076** | **+23.6 %** | **−0.0638 [−0.1271, −0.0008]** | ✅ **better** |
| **`E1_L`** | 0.8983 | **0.4015** | **+25.9 %** | **−0.0700 [−0.1339, −0.0071]** | ✅ **better** |
| `E1_L_X` / `E1_L_X_D` | 0.9056 / 0.9106 | 0.4028 / 0.4018 | +25.4 % / +25.7 % | −0.0687 / −0.0696 | ✅ **better** |
| ⛔ `P_ORACLE` (future speed at t+2 s — **a bound, not a capability**) | 0.7857 | 0.3097 | +59.8 % | −0.1617 [−0.2234, −0.1037] | ✅ better |

⚠️ **The conservative resampler disagrees on separation, and that is reported, not buried.** With
residuals drawn within a matched ego-speed decile (`by_speed`, which preserves heteroscedasticity),
`E1_ego` gives **+20.3 %** and `E1_L` **+22.9 %** — same sign, same size, **not separated**
(`E1_L`: −0.0619 [−0.1245, +0.0010]). `E1_ego|iid`'s separation is also marginal (hi = **−0.0008**).
**The honest statement: positive and consistent across four arms and both resamplers; separated on
one of two.** At n = 40 episodes this is the ceiling of what the canonical val can resolve — and
**the 600-episode parity-preserving val build already exists** (§6.3).

### 0.4 ⛔ The control that decides whether ANY of the above is real

The injection replaces the along coordinate with `true + resampled residual`, so the injected error
is **independent of which window is hard**. A real head's error **correlates** with difficulty, and
decorrelating it could be optimistic. So: **the parent's own along residual, pushed through the
identical resampler.**

| | recovery |
|---|---:|
| parent's head, **actual and correlated** (`R_head_oof`, its measured value) | **−10.4 %** |
| parent's head, **resampled through this construction** | **−33.1 %** (iid) / −34.9 % (by_speed) |

> ⇒ **Decorrelation makes the outcome WORSE by 22.7 points, not better. The construction is
> CONSERVATIVE.** Every recovery above is if anything an **under**-estimate of what a real head with
> that error structure would achieve. *(This control could have voided the stream; it did the
> opposite, and it is reported because it was run either way.)*

### 0.5 ⭐ Why 0.90 m beats 1.15 m by 59 points of recovery — most of it is SHAPE, not RMS

`E1_L` (0.898 m) recovers **+25.9 %**; the parent's residual (1.151 m) recovers **−33.1 %** through
the identical construction — a **59-point** gap. It decomposes:

| | recovery | what changes |
|---|---:|---|
| parent's residual @ 1.151 m | **−33.1 %** | — |
| **this stream's residual, rescaled to ≈ the same RMS** (k = 1.25, 1.123 m) | **+2.1 %** | ⭐ **SHAPE alone ⇒ +35.2 points** |
| this stream's residual at its actual 0.898 m | **+25.9 %** | magnitude ⇒ a further +23.8 points |

⇒ **Roughly 3/5 of the advantage is the error's SHAPE, not its size.**

| along residual | RMS | MAE | RMS/MAE | median \|err\| | **% of windows under 0.5 m** |
|---|---:|---:|---:|---:|---:|
| parent's best head | 1.151 | 0.8213 | 1.4015 | 0.6114 m | **42.2 %** |
| **`E1_L`** | 0.8983 | 0.4811 | 1.8671 | **0.2959 m** | **68.8 %** |
| `E1_L` rescaled to the parent's RMS | 1.151 | 0.6165 | 1.8671 | 0.3792 m | 59.9 % |

> **The consumer of a goal does not feel the RMS — it feels how often the goal is nearly right.**
> `E1_L` is nearly-right on 68.8 % of windows against the parent head's 42.2 %, and pays for it with
> a thinner, worse tail. **A goal head that minimises L2 is optimising the wrong loss for this
> consumer** — the parent said so from the `SHRINK` family; this measures it from the tail.

### 0.6 The lead-vehicle contribution, three independent reads of the same contrast

| read | Δ along-track (m) | rel. | separated |
|---|---:|---:|---|
| **primary — full grid, RMS** (the bar's axis) | **+0.0322 [−0.0040, +0.0969]** | −3.46 % | ❌ (separates at ≈ **1,505 clips**) |
| secondary — full grid, **MAE** (higher power) | **+0.0113 [+0.0011, +0.0258]** | −2.29 % | ✅ |
| stride-matched grid (density removed), RMS | **+0.0187 [+0.0014, +0.0501]** | −1.90 % | ✅ |

**In recovery terms the whole lead block is worth +2.3 points** (`E1_ego` +23.6 % → `E1_L` +25.9 %).
Compare **ego kinematics**, on the identical windows:

| contrast | Δ along-track RMS | rel. | separated |
|---|---:|---:|---|
| `E0_v0` → `E1_ego` | **+0.5942 [+0.4458, +0.7493]** | **−38.97 %** | ✅ |
| **history ablation** — drop `dv_*` / `v_lag_*` only | **+0.1428 [+0.0686, +0.2516]** | −13.30 % | ✅ |
| lead-vehicle block, same axis | +0.0322 | −3.46 % | ❌ |

> ⭐ **One second of ego speed/acceleration history is worth 4.4× the entire lead-vehicle feature
> block, and unlike it, it separates on both axes.**

### 0.7 The pre-registered verdict — and its two clauses disagree

| clause | returned |
|---|---|
| **CONFIRM** = (a) RMS < 0.813 m **AND** (b) separated positive recovery OOF | ⚠️ **SPLIT** — (a) ❌ (0.8983 m); (b) ✅ (+25.9 %, separated) |
| **PARTIAL** = separated improvement, still past break-even | ✅ on the *lead* increment against the *pre-registered* bar |
| **REFUTE** = lead state does not move along-track RMS | ⚠️ the letter fires on the primary axis alone (full-grid RMS not separated) |

**I am not picking the flattering clause.** (a) was written as a **proxy** for (b) — the bar existed
to predict whether the recovery would be positive. **The measurement shows the proxy is wrong in the
conservative direction**, and §0.2 recomputes the bar on the family that applies: **1.1434 m**, which
`E1_L` clears. So:

> **VERDICT — two answers, and they are answers to two different questions.**
> **On the question the stream was named for — *does lead-vehicle state close the longitudinal gap?*
> — the answer is NO, and it is not close: 2–3 %, +2.3 points of recovery, and 41.65 % of windows
> have nothing ahead to measure. That is a REFUTE of the lead-vehicle hypothesis and it is not
> re-scoped.**
> **On the question the stream was launched to serve — *is the longitudinal goal realisable from our
> own corpus?* — the answer is YES: `+23.6 %` from ego dynamics alone, separated, with the
> lead-vehicle block contributing almost none of it.**

---

## 1. S0 — the fidelity gate. Nothing new was quoted before the old reproduced

`raw/eg_gate.json`, `_all_ok = true`. **Three controls, each able to fail.**

| control | required | measured |
|---|---|---|
| **F1 per-row identity** vs the repo reader's 2026-07-21 dump | exact | **max abs diff `0.0` over 21 columns, 104,652 common rows, 0 NaN mismatches** ✅ |
| **F3 failing control** — same comparison, deliberately mis-joined by +0.1 s | must FAIL F1 | max ‖Δv‖ **0.5679 m/s** ✅ (F1 has power) |
| **F2 coverage reconciliation** — predict the committed lead rate from mine | \|error\| < 0.005 | predicted **0.3863** vs measured **0.3863**, error **0.0000** ✅ |

F1 is a **per-row exact identity**, not a mean-vs-mean comparison, so it cannot be passed by a wrong
join — the same shape as the parent's Fidelity B, the control that actually caught a bug upstream.
My ingest **calls** `stack/scripts/lead_state_gate.py`'s reader rather than re-implementing it, and
F1 proves the call is faithful.

**S4 has its own gate** (`raw/eg_place.json:fidelity`, `passes = true`): the parent's whole pipeline
is reproduced from raw before a new number is quoted — as-trained **0.4714**, `R_goal2s` **0.2009**,
`R_head_oof` **0.4996**, `oracle_in_fan` **0.1640**, headroom **0.2705**, all to four decimals. The
script aborts on any mismatch > 5×10⁻⁴.

---

## 2. S1 — ⭐ THE ALIGNED `obstacle.offline` DUMP, and a live defect in the repo reader

`raw/eg_align.json` · `raw/eg_windows.parquet` — **612 clips, 104,652 windows, 26 chunks,
25 countries**, 10 Hz grid over t ∈ [1.0, 18.0] s, 2 s horizon.

### 2.1 The clock join — proven, not assumed

`obstacle.offline` boxes are `reference_frame = "rig"`, so a world-**static** object has a constant
world position **iff** the clock offset is right. Sweeping δ ∈ [−2, +2] s over 40 clips:

| | value |
|---|---:|
| best δ | **0.0 s** |
| clips whose own minimum is at δ = 0 | **95 %** |
| median p10 track dispersion at δ = 0 | **0.0637 m** |
| …at a deliberately wrong δ = +1.0 s | **0.3117 m** |
| **failing control** — +1 s must be worse | **4.89× worse** ✅ |

The statistic is recomputed fresh at every δ, so no track is pre-selected at any particular δ and
the test cannot be rigged toward zero. *(Independently consistent with the sibling
`…/2026-07-27-percandidate-labels/raw/t2a_clock_join.json`: 4.96× on a different 30 clips.)*

### 2.2 ⛔ THE SPAN — the hazard the brief named, and it is a live defect

| source | t₀ range | t₁ range |
|---|---:|---:|
| `egomotion` | −0.200 … −0.185 s | **20.19 … 140.19 s** |
| `obstacle.offline` | 0.000 … **19.234 s** | **2.699** … 20.000 s |

> ⛔ **10.59 % of clips have an `obstacle.offline` span that does NOT cover the feature grid, and
> 4.51 % of all windows (4,717 of 104,652) have no obstacle data at all.**
> `stack/scripts/lead_state_gate.py` **does not check this.** For those windows its reader finds no
> obstacle rows and records `lead_present = 0` — **which is indistinguishable from an empty road.**

**Measured, and its size measured.** The committed `lead_gate_result.json` reports
`lead_present_frac_windows = 0.3851`. The **corrected**, coverage-only rate is **0.4046** — the
committed figure is **deflated by 4.5 %** of fabricated zeros, and F2 reproduces that dilution
exactly (0.4046 × 0.9549 = **0.3863**). The defect is small *here* because coverage is 95.5 %; it is
**silent**, and on a worse-covered chunk set it would be arbitrarily large.

⭐ **The fix is three lines and it belongs in the repo reader** — see §7 escalation 1. A window with
no lead vehicle is a legitimate value; a window with no *data* is not, and the two must not collapse.

### 2.3 Coverage — what the corpus actually supplies

| quantity | value |
|---|---:|
| clips with `obstacle.offline` **and** `egomotion` in the R0 selection | **612** (2 further dropped: grid entirely outside the obstacle span) |
| windows with obstacle coverage | **99,935 / 104,652 = 95.49 %** |
| **windows with a lead vehicle** (vehicle class, gap ≤ 80 m, \|lat\| < 2 m) | **40.46 %** |
| clips with a lead vehicle at some point | **66.34 %** |
| per-clip lead fraction — p10 / p50 / p90 | 0.00 / 0.24 / 1.00 |
| lead gap when present — p10 / p50 / p90 | 9.03 / 26.79 / 59.92 m |
| **windows with ZERO vehicles ahead within 50 m** | **41.65 %** |
| 2 s along-track displacement — mean / SD | **26.34 m / 19.56 m** |
| *(canonical 881 reference — parent §2.1)* | *25.51 m / 18.73 m* |
| ego speed, mean | 13.17 m/s |

> ⭐ **The along-track difficulty of this window set matches the canonical 881 set to within 4 %**
> (SD 19.56 vs 18.73 m). That is what makes the bar comparison admissible at all, and it is
> **checked rather than assumed** (`PRE_REGISTRATION.md §8`).

### 2.4 ⚠️ "53.6 tracked agents per window" is true, and it is the wrong statistic

`raw/eg_agent_census.json` (125 clips, 21,375 windows, identical causal association):

| what is counted | per window |
|---|---:|
| **every** tracked agent, any direction, any range *(the quoted figure's family)* | **27.70** (median 17) |
| agents **ahead** of the ego | 13.92 |
| **vehicles** ahead of the ego | 10.42 |
| **vehicles ahead within 50 m in a ±8 m corridor** | **2.01** |
| **the lead vehicle** (±2 m corridor) | **0.40** |
| windows with **zero** vehicles ahead within 50 m | **41.65 %** |

**The richness that made the lead-vehicle hypothesis attractive is mostly parked cars, oncoming
traffic, pedestrians and agents behind the ego — none of which constrain how far the ego travels in
the next 2 s.** On 41.65 % of windows the road ahead is empty and there is nothing for a
lead-vehicle feature to say. That is not a bug; **it is the ceiling, and it was visible before the
fit.**

---

## 3. S2 — ⛔ THE LEAK AUDIT, by definition and not by name

Retraction class **`ORACLE-SHAPED-AS-EGO-STATE`**: the parent found `head_deg` — *future* net heading
change — sitting beside `v0` in every fan dump. The check that catches that class is **reading the
producing function**. Every column fed to a scored arm, with its source line:

| column | definition | source | verdict |
|---|---|---|---|
| `v`, `ax`, `ay`, `curv`, `abs_curv` | interpolated **at t** | `lead_state_gate.py:158-162` | ✅ present |
| `yawrate` | `(yaw(t) − yaw(t−0.2)) / 0.2` | `:144, :163` | ✅ **past** |
| `dv_0p5`, `dv_1p0` | `v(t) − v(t−0.5 / −1.0)` | `:145-146, :164-165` | ✅ **past** |
| `v_lag_0p5`, `v_lag_1p0` | `v(t−0.5)`, `v(t−1.0)` | `:145-146, :166-167` | ✅ **past** |
| `lead_present`, `gap_m`, `lead_lat_m`, `lead_is_big` | nearest obstacle sample **at or before t**, staleness ∈ [0, 0.5] s | `:199` `searchsorted(side="right")−1`; `:205` `stale ≥ 0` | ✅ **past** |
| `closing_ms` | **backward** difference over ≥ 0.15 s, older endpoint at or before `t − 0.5` | `:229-234` | ✅ **past** |
| `ttc_s`, `inv_ttc` | functions of `gap_m` / `closing_ms` | `:246-252` | ✅ **past** |
| `n_ahead_50m`, `n_vru_near` | same causal sample | `:207-212` | ✅ **past** |
| `headway_s`, `req_decel`, `lead_v_abs`, `gap2_m`, `inv_gap` | pure functions of the above; `gap2_m` repeats the identical association | `eg_common.lead_extra` | ✅ **past** |
| **`y_long`, `y_lat`** | `x(t+2) − x(t)` projected on `yaw(t)` | `:137-141` | ⛔ **THE TARGET** |
| **`y_long_h*`, `dv_h*`** | future horizons in `lead_gate_windows_h.parquet` | `:149-155` | ⛔ **ORACLE — never read** |
| **`v_fut_*`, `v_fut_int`** | future speed / its integral | this stream | ⛔ **ORACLE — controls only** |

Enforced at runtime: `eg_common.assert_no_oracle` aborts on any of these names and is called by
**every** feature builder (`eg_fit.build_arms.take`). The oracle arms bypass it **explicitly and
visibly**, in three lines that say so.

⚠️ **The lead-vehicle features were the exposed surface here** — a track "at t+2 s" is not
deploy-time — and the association is causal by construction: `searchsorted(side="right") − 1` returns
the last sample **at or before** t, and `stale ≥ 0` rejects anything later.

---

## 4. S3 — the out-of-fold re-fit

`raw/eg_fit_gbm.json` · `raw/eg_oof_pred_gbm.npz` (per-window OOF predictions, so any interval is
recomputable). 99,935 windows / 612 clips; `HistGradientBoostingRegressor` with **identical
hyper-parameters for every arm**, so extra columns can only help an arm through held-out
generalisation.

### 4.1 ⚠️ TWO AXES, and the reason is measured, not stylistic

The **bar's** axis is the RMS, so the RMS is primary. But the RMS is **tail-dominated**
(`eg_fit_gbm.json:tail_diagnostics`):

| | value |
|---|---:|
| share of total squared error in the worst **1 %** of clips | **49.06 %** |
| …worst 5 % / 10 % | 58.88 % / 65.77 % |
| per-window RMS / MAE (Gaussian = 1.2533) | **1.8895** |

**Half the squared error lives in six clips.** A clip-cluster bootstrap on that is wide —
`E1_ego`'s RMS interval is [0.6814, 1.2531] around 0.9305. The MAE on the *identical* per-window
errors is far better powered and is reported beside **every** contrast. **A claim is made only where
the two agree; where they disagree it is stated** (§0.6 is exactly such a place).

### 4.2 ⭐ What the tail actually is — `raw/eg_tail.json`

| arm | RMS (all) | RMS **excl. worst 6 clips** | MAE | median \|err\| | p90 | % windows > 2 m |
|---|---:|---:|---:|---:|---:|---:|
| `E1_ego` | 0.9305 | **0.6675** | 0.4925 | 0.2955 | 1.0933 | 2.32 % |
| `E1_L` | 0.8983 | **0.6579** | 0.4811 | 0.2959 | 1.0705 | 2.10 % |
| ⛔ `P_ORACLE_STRONG` | 0.8062 | **0.2648** | 0.1911 | 0.0769 | 0.2855 | 0.93 % |

**The six worst clips are not glitches — they are autobahn.** Mean speeds **36.5–45.0 m/s
(131–162 km/h)**, `y_long` up to **93 m**, near-zero curvature, `scenario = highway`. At 45 m/s a 2 s
displacement is ~90 m, so hitting 0.8 m of along-track error means predicting mean speed to **0.9 %
relative**. ⇒ **The along-track RMS bar is, in effect, a high-speed bar**, and one of the six worst
clips has a lead vehicle on 89 % of its windows — leads do not rescue it.

### 4.3 Both-directions validation

| control | required | measured | passes |
|---|---|---|---|
| **`N_SHUF`** — lead block permuted **across clips** | must NOT help | **−0.0439 m RMS [−0.1369, −0.0046]**, MAE −0.0140 [−0.0319, −0.0050] ⇒ shuffle is separated-**WORSE** | ✅ |
| **`P_ORACLE_STRONG`** — future speed profile **+ its integral** | must be separated-better | RMS +0.1243 ❌ sep; **MAE +0.3013 [+0.2646, +0.3276] = −61.2 %, ✅ sep** | ✅ on MAE, ❌ on RMS |
| `P_ORACLE` — `v(t+2 s)` | *(diagnostic)* | RMS +0.1447 ❌; **MAE +0.2357 = −47.9 % ✅** | — |
| `L_only` — lead features with no ego state | sanity | **18.17 m** — lead state alone says nothing about ego displacement | ✅ |

> ⚠️ **THE POSITIVE CONTROL FAILS ON THE RMS AXIS AND PASSES DECISIVELY ON THE MAE AXIS, AND THAT IS
> ITSELF THE RESULT: at n = 612 clips the RMS axis CANNOT separate even a near-perfect oracle.**
> ⇒ **No non-separation on the RMS axis may be read as evidence of absence in this stream** — which
> is why §0.6 quotes three reads and why §0.3 (running the rule) is the operative measurement.

⚠️ **The first form of the strong control was weak, and finding out why is a finding.** Handing the
tree ensemble the *raw* future speed profile moved the RMS only 0.9305 → 0.8107, which reads like
"even knowing the future barely helps". **It is a model-class artifact:** `y_long` is the *integral*
of speed, an additive function of five continuous inputs, which a 31-leaf tree approximates badly.
Handing the same information as **one monotone column** (the trapezoidal integral) takes the MAE from
−57.9 % to −61.2 %, and §4.2 shows the oracle is in fact nearly exact on typical windows
(median error **0.077 m**, RMS excl. the six autobahn clips **0.265 m**). **Both forms are reported,
because the gap between them is the evidence that the weak form's number was about the regressor and
not about the data.** *(Third stream in three days to find its own scoring code producing a stable,
plausible, wrong number. Assume yours has one.)*

### 4.4 ⛔ C19 — the conditional result, with its firing rate and its whole-set value

| | RMS axis | MAE axis (powered) |
|---|---:|---:|
| **firing rate** — windows with a lead vehicle | **40.46 %** (40,430 windows / 406 clips) | same |
| paired Δ **on those windows only** | +0.0018 [−0.0095, +0.0189] ❌ | **+0.0107 [+0.0027, +0.0182]** ✅ |
| **whole-set policy value** | **+0.0322** | **+0.0113** |

On the powered axis the effect is **essentially uniform** across lead-present and lead-absent windows
(+0.0107 on the 40.46 % where a lead exists; the whole-set 0.0113 implies ≈ +0.0117 on the rest).
On the RMS axis the subgroup effect is much smaller than the whole-set one, i.e. what tail-error the
lead block removes, it removes mostly on **no-lead** windows.
⇒ **`gap` / `closing` / `TTC` are not carrying a lead-following mechanism; the block behaves like a
weak "is the road ahead occupied" indicator.** Either way the size is 2–3 % and the deployable
whole-set value is what §0.6 quotes.

### 4.5 Is it features, or is it sample size? — `raw/eg_robust.json`

The obvious objection: this fit sees ~100,000 windows over 612 clips; the parent's head saw 881
windows over 40 episodes. Three controls, each able to overturn the headline.

**R1 — stride-matched** (every 9th grid point, ~19 windows/clip vs the canonical ~22, same 612 clips;
removes the autocorrelated density without touching `n_clips`):

| arm | `E0_v0` | `E1_nohist` | **`E1_ego`** | **`E1_L`** | `E1_L_X` |
|---|---:|---:|---:|---:|---:|
| along RMS | 1.5503 | 1.1322 | **0.9831** | **0.9644** | 0.9659 |

`E1_ego`→`E1_L` = **+0.0187 [+0.0014, +0.0501] ✅**; `E0_v0`→`E1_ego` = **+0.5671 [+0.4301, +0.7056] ✅**.
⇒ **the ordering survives de-correlation.**

**R2 — the parent's EXACT data regime** (40 clips × ~22 windows ≈ 827 windows, 5 clip-disjoint folds,
15 independent clip draws):

| arm | median along RMS | p10–p90 |
|---|---:|---|
| `E0_v0` | 2.4394 | 1.82 – 3.84 |
| **`E1_ego`** | **2.2759** | 1.34 – 3.85 |
| `E1_L` | 2.3258 | 1.36 – 3.86 |

| | value |
|---|---:|
| draws where `E1_ego` beats `E0_v0` | **86.7 %** |
| draws where `E1_L` beats `E1_ego` | **20.0 %** |
| **draws where `E1_ego` clears 0.813 m** | **0.0 %** |

> ⛔ **AT n = 40 EPISODES NOTHING GETS NEAR THE BAR — everything sits at ≈ 2.3 m, worse than the
> parent's 1.151 m head.** ⭐ **The along-track head is DATA-limited at that scale, not
> feature-limited, which reframes the parent's 1.151 m as a small-sample number rather than a
> ceiling.** ⚠️ And note row 2: **at the parent's n, adding lead features makes things WORSE in 80 %
> of draws** — a tiny true effect plus a fitting cost is a net loss at small n.

**R3 — history ablation** (drop `dv_*` / `v_lag_*` only, full set): 0.9305 → **1.0733 m**, paired
**+0.1428 [+0.0686, +0.2516]** RMS and **+0.1035 [+0.0838, +0.1273]** MAE, **both separated**.
⇒ **one second of speed/accel history = 4.4× the whole lead-vehicle block.**

---

## 5. S4 — placement by RUNNING THE RULE, and the family named

`raw/eg_place.json`. Method, fidelity, the decorrelation control, the family-matched curve and the
shape comparison are all in §0.2–§0.5. What remains to state explicitly:

**The family is `EMPIRICAL`, and neither pre-registered family fits.**

| arm | RMS/MAE (Gaussian 1.2533) | shrinkage α | bias | family |
|---|---:|---:|---:|---|
| `E1_ego` | 1.8895 | 0.9972 | −0.0014 m | **near-unbiased, heavy-tailed** |
| `E1_L` | 1.8671 | 0.9958 | −0.0099 m | **near-unbiased, heavy-tailed** |
| `CV` / `E0_v0` | 1.5124 / 1.5512 | 0.998 / 0.9948 | ≈ 0 | near-unbiased |

⚠️ **I expected `SHRINK` and it is not `SHRINK`.** The parent measured `SHRINK` (α < 1) to be the
strictest family and predicted an L2-trained regressor would land there. With ~80,000 training
windows the GBM barely shrinks (**α ≈ 0.996**); the dominant deviation from `ISO` is the **tail**,
not the bias. That is why the family-matched σ₀ came out **more** forgiving (1.1434 m) rather than
less, and it is why **the rule had to be run rather than a curve read** — the pre-registered
prediction about the family was wrong, and the instrument caught it.

**The full σ-sweep** (`eg_place.json:family_matched_curve.grid`, monotone ✅):

| scale k | 0.125 | 0.25 | 0.5 | 0.75 | **1.0** | 1.25 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| along RMS (m) | 0.112 | 0.225 | 0.449 | 0.674 | **0.898** | 1.123 | 1.347 | 1.797 | 2.695 | 4.491 |
| recovery | +81.1 % | +76.2 % | +61.5 % | +43.3 % | **+22.6 %** | +2.1 % | −20.8 % | −66.1 % | −158.1 % | −346.6 % |

⚠️ **It goes negative, and hard** — the parent's finding that *a confidently wrong goal is
destructive, not neutral* replicates here on a different error family: **−346.6 % at 4.5 m.**
A goal channel must still be gated on measured accuracy before it is wired in.

---

## 6. WHAT THIS LICENSES, AND WHAT IT DOES NOT

### 6.1 Settled

1. ⛔ **Lead-vehicle state is not the longitudinal lever.** Real (separated on 2 of 3 reads), worth
   **2–3 %** on the error axis and **+2.3 points** of recovery. The ceiling was set before the fit:
   **41.65 % of windows have no vehicle ahead within 50 m.**
2. ⛔ **`gap` / `closing` / `TTC` do not behave like a car-following mechanism** (§4.4). Whatever
   the block is worth, it is not concentrated where a lead vehicle exists.
3. ⭐⭐ **The longitudinal goal IS realisable from our own corpus — from EGO DYNAMICS.**
   `E1_ego` recovers **+23.6 %** of the fan's headroom through the real rule (separated), and
   `E1_L` **+25.9 %**, against the parent's head at **−33.1 %** under the identical construction.
4. ⭐ **The 0.813 m bar is the wrong number for this error family. The family-matched break-even is
   1.1434 m** (half prize 0.5907 m) — 1.41× / 1.35× more forgiving.
5. ⭐ **What the consumer feels is largely the SHAPE, not the RMS**: at near-matched RMS
   (1.123 vs 1.151 m) a residual that is under 0.5 m on 60 % of windows beats one that is under
   0.5 m on 42 % by **35.2 points of recovery** — about 3/5 of the total 59-point gap.
   **An L2 loss is the wrong objective for a goal head.**
6. ⭐ **The parent's 1.151 m is a small-sample number**, not a ceiling: at n = 40 episodes *nothing*
   here gets below ≈ 2.3 m. ⇒ **A goal head must be trained on the TRAIN corpus, never fitted on the
   40 val episodes.**
7. ⛔ **A defect in `stack/scripts/lead_state_gate.py` is measured and localised** (§2.2).
8. ⚠️ **At n = 612 clips the along-track RMS axis cannot separate even a near-perfect oracle**
   (§4.3). Report MAE beside it or do not report separation at all.

### 6.2 What I refuse to conclude

- **NOT** "agent tracks are useless." They are worth 2–3 % *for 2 s along-track displacement*. They
  are the natural input for **collision and TTC** metrics — which the sibling per-candidate stream
  measures ADE to be in *tension* with — and this instrument says nothing about those.
- **NOT** that `+25.9 %` is a deployable number. It is what a goal with **this measured error
  structure** buys when fed to the frozen REF-C-xl fan on 40 episodes. **No such head has been
  trained on our latent, on the canonical windows, or end-to-end.** §6.3 is the experiment that
  would make it deployable, and it can fail.
- **NOT** that the separation is settled. One resampler separates, the conservative one does not,
  and `E1_ego|iid`'s upper bound is **−0.0008**. At n = 40 this is the ceiling of what the canonical
  val can resolve.
- **NOT** anything about a lead vehicle a real perception stack would see. `obstacle.offline` is a
  **label**; a deployed estimate is noisier ⇒ every lead number here is an **upper bound**.
- **NOT** anything past **2 s**, and every number is a displacement/ADE number — blind to collision.
- **NOT** that 0.9305 m is a ceiling for ego features. It is what *this* feature set, *this*
  regressor and *this* corpus reach out-of-fold. **No world-model latent is in it.**

### 6.3 The next experiment — cheaper than this one, and it can fail

**E-GOAL-3 — the ego-dynamics goal head on the canonical val, trained on the TRAIN corpus.**
Everything above says the along-track head is data-limited below ~600 clips and that ego dynamics,
not agent tracks, are the lever. The one comparison this host cannot make is the decisive one: **the
same ego-kinematic head fitted on `physicalai-train-e438721ae894` and scored on the canonical 881**,
then placed through the REF-C rule with its *actual, correlated* per-window predictions rather than a
resampled residual. Cost: an **egomotion-only** pass — no camera decode, no GPU, no obstacle chunks —
plus this stream's code unchanged. **Failure mode:** the 0.93 m does not transfer off the dense grid,
in which case the +23.6 % evaporates and the parent's REFUTE stands unqualified.

**E-GOAL-2 remains the cheapest way to settle the separation** — the 600-episode parity-preserving
val build already exists (`2026-07-26-0757-program-report §2.3`) and converts every "not separated"
above into a sign.

---

## 7. ESCALATIONS — these must not sit in a file

1. 🔴 **`stack/scripts/lead_state_gate.py` needs an obstacle-span guard, and its committed result
   needs a footnote.** The reader builds features on a fixed t ∈ [1, 18] s grid without checking that
   `obstacle.offline` covers it. **Measured: 10.59 % of clips and 4.51 % of windows are not covered
   and silently become `lead_present = 0`.** The committed
   `…/2026-07-21-lead-state-gate/lead_gate_result.json` corpus block is affected
   (`lead_present_frac_windows` **0.3851 → 0.4046**). **The fix is `eg_common.span_of` plus
   `eg_ingest`'s per-window `obst_cov` mask — three lines — and it belongs in the repo reader, not in
   this folder.** *(An orthogonality instrument sat unmerged for 10 days on exactly this failure
   mode.)*
2. 🔴 **`V5_PLAN.md §8` and `Gates/flagship-v5-retrain.PREP.md` item 6 must be amended AGAIN, by
   their owner, and the correction is not the obvious one.** §8 currently says the +83.7 %
   longitudinal axis *"is UNBLOCKED and its signal is ALREADY IN PhysicalAI-AV — `obstacle.offline`"*.
   **Half right, and the wrong half is load-bearing:** the axis *is* open, but **`obstacle.offline`
   is not what opens it — ego dynamics are (+23.6 % vs +2.3 points).** Both documents name the wrong
   supplier, and a v5 decision taken on them would fund an agent-track ingest instead of a
   speed-history feature.
3. 🔴 **`V5_PLAN.md §8`'s conclusion that a realisable goal is "at or past break-even" needs its bar
   corrected.** The **0.813 m / 0.955 m** figures are `ISO`-family numbers, and the measured heads
   are **not** in that family. Family-matched: **σ₀ = 1.1434 m**. Quoting the `ISO` bar at a supplier
   would reject a supplier that is in fact net-positive — which is what nearly happened here.
4. 🟠 **The "97.44 % coverage, 53.6 agents/window" line has propagated as evidence of richness for
   the lead-vehicle hypothesis, and it is the wrong statistic** (§2.4). Same-family census on this
   grid: **27.70** agents/window total, **2.01** vehicles ahead within 50 m, **0.40** lead vehicles,
   **41.65 %** of windows with nothing ahead. The coverage figure is about *tracks existing*, not
   about *the road ahead being occupied*.
5. 🟠 **For `RETRACTION_LOG.md` — root-cause classes:**
   - **`COVERAGE-GAP-READ-AS-A-ZERO`** — a source whose SPAN does not cover the query grid returns
     "no rows", which a feature builder records as a real, meaningful zero, indistinguishable
     downstream from the genuine value. **The check is: measure the intersection of the spans and
     mask; never impute.** *(Measured here at 4.51 % of windows; arbitrarily large on a
     worse-covered chunk set.)*
   - **`CONTROL-WEAK-BY-MODEL-CLASS`** — a positive control that fails can indict the **regressor**
     rather than the data. A tree ensemble given an additive-in-5-inputs oracle moved the RMS 13 %;
     given the same information as one monotone column it is a different number. **A control must be
     expressed in a form the model class can consume, or its failure means nothing.**
   - **`BAR-INHERITED-FROM-THE-WRONG-FAMILY`** — the pre-registered 0.813 m was an `ISO`-family
     number and the measured estimator is near-unbiased and heavy-tailed. Reading it literally
     returns REFUTE; running the rule returns **+25.9 %, separated**. **A requirement expressed as a
     scalar RMS is only meaningful with its family attached** — and the family must be *measured on
     the estimator*, not assumed. *(Sibling of the parent's `RMS-PLACED-ON-A-NOISE-CURVE`; that class
     warned the curve over-predicts damage — this one is the same mechanism seen from the
     requirement side, where it under-states the admissible error by 1.41×.)*
   - **`SEPARATION-CLAIMED-ON-AN-UNPOWERED-AXIS`** — at n = 612 clips the along-track RMS could not
     separate a near-perfect oracle (§4.3). **Before reading a non-separation as absence, show the
     axis can separate something that must separate.**

---

## 8. THREATS TO VALIDITY I COULD NOT REMOVE

| threat | direction | status |
|---|---|---|
| ⚠️ **Not the canonical 881 windows.** The dev box does not hold `physicalai-val-0c5f7dac3b11`; the pods that do are training / sweeping. | unknown | **stated in advance** (`PRE_REGISTRATION.md §8`). Along-track difficulty **checked and matches to 4 %** (SD 19.56 vs 18.73 m). |
| ⚠️ **The placement injects a RESAMPLED residual, not a real head's per-window prediction.** | measured **conservative** | §0.4: the parent's own residual through the identical construction is **−33.1 %** against its actual **−10.4 %** — decorrelation costs 22.7 points. |
| Dense 10 Hz grid ⇒ ~171 autocorrelated windows/clip | inflates apparent n | clip-level folds **and** a clip-level bootstrap throughout; **R1 re-runs everything stride-matched** and the ordering holds. |
| RMS is tail-dominated; 49 % of squared error in 6 clips | widens everything | MAE beside every contrast; the positive control **fails on RMS**, which is reported as an axis-power statement (§4.3). |
| `obstacle.offline` is a **label**, not perception | favours the lead hypothesis | every lead number is an **upper bound**. |
| GBM hyper-parameters not tuned per arm | neutral by design | identical for every arm. |
| 2 s, displacement/ADE only | unknown, possibly large | §6.2. |
| 26 of 197 obstacle chunks are local | narrows the corpus | 612 clips / 25 countries / 26 chunks; the lead effect separates at ≈ 1,505 clips — **but even at face value it does not change the verdict.** |
| n = 40 episodes on the placement | widens every interval | one resampler separates, the other does not; **the 600-episode build exists** (§6.3). |

**Evidence classes.** §§1–5 are `MEASURED (ours)` with artifact paths. The bars 0.813 / 0.439 m and
the parent head reference 1.151 m / 0.576 m/s are `INHERITED` from `GOAL_INPUT.md §5` (quoted, not
re-derived); the parent's `−10.4 %` is `INHERITED` and is **re-verified here from raw**
(`eg_place.json:fidelity`, `R_head_oof` = 0.4996 ✅). §6.3 is `HYPOTHESIS` with a named test.
**Tier: CONFIRMED** for the lead-vehicle refutation (bit-level gate, both-directions validated,
three robustness controls); **PROVISIONAL** for the `+23.6 % / +25.9 %` recovery — it is separated on
one resampler of two at n = 40 and uses a resampled rather than a real per-window head.

---

## 9. DELIVERABLE MANIFEST

**17 of 18 artifacts are staged in the repo. Nothing was committed or pushed. ⚠️ EXACTLY ONE
artifact exists in only one place, and it is named as such below rather than left to an audit.**

| artifact | where | what |
|---|---|---|
| `EGOAL_1.md` | `repo:…/incoming/2026-07-27-egoal-1-lead-vehicle/` | this document |
| `PRE_REGISTRATION.md` | same | bars, arms, leak whitelist, failing-value proofs — **written before any fit** |
| `code/eg_common.py` | same | ⭐ **the reusable reader**: span audit, clock-join proof, derived lead features, oracle whitelist, folds, estimators |
| `code/eg_ingest.py` | same | S1 — the aligned per-window dump + coverage/alignment report |
| `code/eg_gate.py` | same | S0 — per-row identity vs the repo reader, failing control, coverage reconciliation |
| `code/eg_fit.py` | same | S2/S3 — arms, OOF fit, both-directions controls, C19, tail diagnostics |
| `code/eg_robust.py` | same | R1/R2/R3 — features vs sample size |
| `code/eg_place.py` | same | S4 — fidelity, decorrelation control, family-matched curve, placement |
| `raw/eg_align.json` | same | clock join, per-clip spans, coverage |
| ⚠️ `raw/eg_windows.parquet` | **dev box working tree ONLY — git-ignored** | 104,652 aligned windows × 612 clips (15.2 MB) — see the note below |
| `raw/eg_gate.json` | same | the S0 gate |
| `raw/eg_fit_gbm.json` | same | every arm, every paired contrast (RMS **and** MAE), controls, tail |
| `raw/eg_oof_pred_gbm.npz` | same | per-window OOF predictions — any interval recomputable |
| `raw/eg_robust.json` | same | R1/R2/R3 |
| `raw/eg_place.json` | same | fidelity, arms, decorrelation control, family-matched curve, error structure |
| `raw/eg_tail.json` | same | what the RMS tail is (the six autobahn clips) |
| `raw/eg_shape.json` | same | error shape at matched RMS, parent vs this stream |
| `raw/eg_agent_census.json` | same | the "53.6 agents/window" reconciliation |

### ⚠️ The one artifact in only one place — stated, not discovered later

`raw/eg_windows.parquet` **did not stage**: `.gitignore:21` bans `*.parquet` repo-wide. **I did not
`git add -f` it.** The ban is deliberate — a per-window dump derived from a gated-confidential corpus
is exactly what that rule protects — and overriding a privacy-shaped policy is not a subagent's call.

**Why this does not strand anything:**
- The **producer is staged** (`code/eg_ingest.py` + `code/eg_common.py`); the dump regenerates
  deterministically in ≈ 8 min of dev-box CPU:
  `OMP_NUM_THREADS=6 python eg_ingest.py` (reads only `labels/{obstacle.offline,egomotion}/*.zip`).
- The **numbers are staged**: `eg_oof_pred_gbm.npz` carries every per-window OOF prediction, so
  **every interval in this document is recomputable from staged artifacts alone**, without the
  parquet.
- The dump's own **fidelity reference** (`lead_gate/lead_gate_windows.parquet`, 2026-07-21) is
  likewise dev-box-only for the same reason, and `eg_gate.py` degrades gracefully when it is absent.

**Decision needed from the owner** (§7.1 is the natural place to take it): if a corpus-derived
per-window dump *should* be shareable, the right home is HF under `Sayood/` (gated), not a
`git add -f`. **Flagged rather than actioned.**

⭐ **WHERE THE READER SHOULD LIVE PERMANENTLY.** The `obstacle.offline` reader **already exists in
the repo** — `stack/scripts/lead_state_gate.py` (tracked, `52d089a`) — and this stream **imports it
rather than forking it**, with F1 proving the call is bit-faithful. What is new and belongs upstream
is the **span guard** (`eg_common.span_of` + `grid_inside_intersection` + `eg_ingest`'s per-window
`obst_cov` mask) and the **clock-join assertion** (`eg_common.assert_clock_join`). **Recommendation:
move those three functions into `stack/scripts/lead_state_gate.py` (or a new
`stack/tanitad/data/obstacles.py` if a second consumer appears) and have `build()` call them.** They
are written as free functions with no stream-specific state precisely so this is a move, not a
rewrite. **Escalated as §7.1 rather than left as a note in a README.**

**Inputs consumed** (read-only): `C:/Users/Admin/tanitad-data/physicalai/labels/{obstacle.offline,
egomotion}/*.zip` (26 chunks), `r0/phase0_selection.parquet`, `lead_gate/lead_gate_windows.parquet`
(the prior dump, used **only** as the F1 identity reference),
`taniteval/results/fan_refc-xl-30k.pt`, `…/2026-07-27-goal-input/raw/gi_head_preds.npz`,
`…/2026-07-27-goal-input/raw/gi_place.json`, `…/2026-07-21-lead-state-gate/lead_gate_result.json`.

⛔ **Parity untouched:** `_epcache` never written, no clip re-selected,
`physicalai-train-e438721ae894` / skip-hash `f09e44db` never read.
🔒 **No clip UUID or raw PhysicalAI content reaches any artifact** — clips appear as
`clip_<sha256[:8]>` throughout (`eg_common.clip_alias`).

**Suite green:** `cd stack && pytest -q` → **1253 passed, 7 skipped** in 87 s (2026-07-27, re-run at
the end of this stream). This stream added **no files** to `stack/` or `taniteval/` —
`git status --short stack/ taniteval/` is empty; all new code lives in the hub folder above and
**imports** the repo reader rather than modifying it.

**Total compute:** ~35 minutes of dev-box CPU. **No GPU. Zero pod load.**
