# E-GOAL-1 — PRE-REGISTRATION

**Written 2026-07-27 BEFORE any number in `EGOAL_1.md` was computed.** Host: dev box, CPU only.
Nothing here is edited after the first fit is run; corrections are appended with a timestamp.

---

## 0. The question, in one line

**Does lead-vehicle state (gap / closing speed / TTC, from `obstacle.offline`) reduce the
ALONG-TRACK RMS of a 2 s endpoint predictor below the pre-registered break-even bar?**

The parent stream (`…/2026-07-27-goal-input/GOAL_INPUT.md §5`) measured that the goal input's
value is ~entirely longitudinal (+83.7 % along-track vs +2.9 % cross-track) and published the
conditional spec below. This stream tests whether our own corpus supplies it.

## 1. Bars — taken verbatim from the parent, not re-derived here

| axis | quantity | bar |
|---|---|---|
| along-track RMS, given a **learned** cross-track | break-even (`σ₀`) | **0.813 m** |
| along-track RMS, given a **learned** cross-track | half prize (`σ₅₀`) | **0.439 m** |
| 2 s-mean speed error | break-even | **0.406 m/s** |
| 2 s-mean speed error | half prize | **0.219 m/s** |
| reference — best head to date (`H_ridge_all_raw`, canonical 881) | along RMS / speed | **1.151 m** / **0.576 m/s** |

`2 s-mean speed error = along-track error / 2.0 s`, exactly the parent's conversion.

## 2. Pre-registered decision rule

- **CONFIRM** — the lead-vehicle arm's OOF along-track RMS is **< 0.813 m** *and* its paired
  improvement over the matched no-lead arm is **separated** from zero (episode-cluster bootstrap).
  ⇒ the longitudinal goal is realisable from our own corpus; it enters v5.
- **PARTIAL** — RMS improves **separated** but stays **≥ 0.813 m**. ⇒ report the residual gap in
  metres and in m/s; that becomes the spec for whatever must close it.
- **REFUTE** — lead-vehicle state does not move along-track RMS (paired delta not separated, or
  separated-worse). ⇒ say so plainly; we stop paying for the longitudinal prize *from agent tracks*.
  **No re-scoping.**

### 2.1 ⚠️ What would make each rule return a FAILING value — and the proof it can

| rule | the input that would make it fire | can it fire? |
|---|---|---|
| CONFIRM | lead features carry real, separated information about how far the ego travels | **YES** — the positive control `P_ORACLE` (feeds `v(t+2 s)`, a future observable) must clear both bars by a wide margin. If it does not, the instrument cannot return CONFIRM and the whole run is void. |
| PARTIAL | a separated but small lead effect | **YES** — it is the arithmetic middle of the other two and needs no separate proof; the CONFIRM control proves separation is reachable and the REFUTE control proves non-separation is reachable. |
| REFUTE | lead features that carry nothing | **YES** — the negative control `N_SHUF` (lead block permuted across episodes) must come back non-separated / worse. If a *shuffled* lead block "helps", the pipeline leaks and every number is void. |

**Both controls are mandatory and are reported whatever the verdict.** A run in which either control
misbehaves is reported as VOID, not as a result.

## 3. Estimator, fixed in advance

Paired **episode-cluster bootstrap**, `taniteval/taniteval/ci.py`, **B = 2000**,
resampling unit = **the clip** (one clip = one episode = one independent 20 s recording).
`overlapping_holdout_se` is **never called**. Separation predicate: the CI excludes 0.
Anything fitted is **out-of-fold and episode-disjoint** (5 folds, clip-level, seed 0);
no clip is ever in both the fit and the score of the same number.
Not separated ⇒ **UNPOWERED NOT REFUTED**, quoted with the n at which it separates.

## 4. Arms, fixed in advance

| arm | features | why |
|---|---|---|
| `CV` | none (ŷ = 2·v₀) | the no-fit reference; the parent's `H1_cv` analogue |
| `E0` | `v` only | **the parent head's ego content** (`F_ego` is `v0` ⧺ 4 CV waypoints, which are pure functions of `v0`) |
| `E1` | 10 ego kinematics incl. 1 s of speed/accel history | what a deployed vehicle actually has from IMU/CAN |
| `E1+L` | `E1` ⧺ 7 lead-vehicle columns | **THE ARM UNDER TEST** |
| `E1+L+D` | `E1+L` ⧺ 2 density columns | traffic density as a weaker longitudinal cue |
| `E1+L+X` | `E1+L` ⧺ 5 derived lead columns (headway, required decel, lead abs speed, 2nd lead) | a fair best-effort for the lead hypothesis |
| `N_SHUF` | `E1` ⧺ lead block **permuted across episodes** | negative control (must not help) |
| `P_ORACLE` | `E1` ⧺ `v(t+2 s)` | positive control (must help enormously) — ⛔ **an oracle, never quoted as a capability** |

Regressor: `HistGradientBoostingRegressor` and ridge, **identical hyper-parameters for every arm**,
so extra columns can only help an arm through held-out generalisation.

## 5. ⛔ Leakage whitelist — audited BY DEFINITION, not by name (retraction class C23)

Admissible = a quantity computable at time *t* from data timestamped **≤ t**.

| field | definition (source line) | verdict |
|---|---|---|
| `v`, `ax`, `ay`, `curv`, `abs_curv` | interpolated at `t` from `egomotion` | ✅ present |
| `yawrate` | `(yaw(t) − yaw(t−0.2))/0.2` | ✅ past |
| `dv_0p5`, `dv_1p0`, `v_lag_0p5`, `v_lag_1p0` | `v(t) − v(t−0.5)`, `v(t−1.0)` … | ✅ past |
| `lead_present`, `gap_m`, `closing_ms`, `ttc_s`, `inv_ttc`, `lead_lat_m`, `lead_is_big` | nearest obstacle sample **at or before** `t`, staleness ≤ 0.5 s; closing from a **backward** difference | ✅ past — enforced by `searchsorted(side="right")−1` + `stale ≥ 0` |
| `n_ahead_50m`, `n_vru_near` | same causal sample | ✅ past |
| derived `headway_s`, `req_decel`, `lead_v_abs`, `gap2_m`, `inv_gap` | pure functions of the above | ✅ past |
| **`y_long`, `y_lat`** | `x(t+2) − x(t)` projected on `yaw(t)` | ⛔ **THE TARGET** |
| **`y_long_h*`, `dv_h*`** | future horizons in `lead_gate_windows_h.parquet` | ⛔ **ORACLE — inadmissible** |
| **`v(t+2 s)`** | future speed | ⛔ **ORACLE — admissible ONLY inside `P_ORACLE`, and quoted as a bound** |

Enforced at runtime by `eg_common.assert_no_oracle(names)`, which aborts on any name in the
`ORACLE` frozenset. ⚠️ The parent stream found `head_deg` (a *future* net heading change) sitting
beside `v0` in every fan dump; the check that catches that class is **reading the producing
function**, which is done above for every column.

## 6. Alignment — asserted, not assumed

`egomotion` and `obstacle.offline` are joined on a shared clock. The hazard is the **SPAN**:
`egomotion` runs to 20–140 s while `obstacle.offline` stops at ≈ 20 s. Before any feature is built:

1. **Clock offset** — sweep δ ∈ [−2, +2] s and minimise the world-frame dispersion of rig-frame
   tracks (a static object's world position is constant iff δ is right). δ = 0 must win, and a
   deliberately wrong δ = +1.0 s must be reported WORSE.
2. **Span** — report `ego_t0/t1` and `obst_t0/t1` per clip; the feature grid must lie inside the
   INTERSECTION. Any clip whose intersection does not contain the grid is **dropped and counted**,
   never imputed.
3. **Coverage** — report the fraction of grid points with any obstacle sample, and the fraction
   with a lead vehicle. ⭐ **A window with no lead vehicle is a legitimate value (`lead_present=0`),
   not a gap to impute.**

## 7. C19 — conditional results carry their firing rate

Any number computed on the lead-present subgroup is reported **with its firing rate and with its
whole-set policy value**. A stratum win is not a deployable win.

## 8. Corpus and parity

Read-only over `labels/{obstacle.offline,egomotion}` zips and `r0/phase0_selection.parquet`.
**No episode is re-selected; `_epcache` is never written; `physicalai-train-e438721ae894` /
skip-hash `f09e44db` are untouched.** Every number is a property of *labels*, not of a model arm.

⚠️ **Stated in advance, not discovered late:** the dev box does **not** hold the canonical
`physicalai-val-0c5f7dac3b11` 881-window set, and the pods that do are training / sweeping. The
along-track RMS measured here is therefore on a **different window set** drawn from the same corpus
and the same R0 selection. Its comparability to the 0.813 m bar rests on the two sets having the
same along-track difficulty, which is **checked and reported** (canonical: mean displacement
25.51 m, along SD 18.73 m). If the difficulty differs materially, the bar comparison is reported as
**UNVERIFIED** rather than quietly used.

## 8a. ⚠️ AMENDMENTS — appended after the first fit, timestamped, nothing deleted

*Recorded here rather than silently absorbed into §§0–8. Every original clause above stands as
written; these are additions, and the reader can see exactly what changed and when.*

**A1 (2026-07-27, after run 1).** §2.1's positive control `P_ORACLE` (= `v(t+2 s)`) **did not
separate on the RMS axis** (Δ +0.1447 [−0.0005, +0.3096]) while separating decisively on MAE
(−47.9 %). Two additions, neither of which touches an arm under test:
- `P_ORACLE_PROFILE` (the full future speed profile) and `P_ORACLE_STRONG` (the profile **plus its
  trapezoidal integral**) were added as stronger controls. **Reason, stated before re-running:**
  `y_long` is the *integral* of speed — an additive function of several continuous inputs — which a
  31-leaf tree ensemble approximates badly, so the weak control may be indicting the **regressor**
  rather than the data. Both forms are reported so the gap between them is visible.
- A **secondary MAE read** is reported beside every paired contrast. **The RMS remains primary**
  because it is the bar's axis; the MAE is reported because it is better powered on a tail-dominated
  statistic. **A claim is made only where the two agree, and disagreements are stated as such.**

⚠️ **The outcome of A1 must be read as a limit on this instrument:** `P_ORACLE_STRONG` still does not
separate on RMS (Δ +0.1243 [−0.1131, +0.3582]) while reaching **−61.2 % on MAE, separated**.
⇒ **At n = 612 clips the along-track RMS axis cannot separate even a near-perfect oracle, and no
non-separation on that axis in this stream may be read as evidence of absence.**

**A2 (2026-07-27, after run 1).** `eg_robust.py` (R1 stride-matched / R2 parent-regime / R3 history
ablation) was **not** pre-registered. It exists to answer an objection that arose from the data —
*is the gain features or sample size?* — and it is reported as a **robustness check, not a
hypothesis test**. It was written to be able to overturn the headline, and R2 partly does (at n = 40
the lead features go the wrong way in 80 % of draws).

**A3 (2026-07-27, at S4).** §1's bars are `ISO`-family numbers inherited from the parent. The
measured estimators turned out to be **near-unbiased** (α ≈ 0.996) and **heavy-tailed**
(RMS/MAE ≈ 1.87) — i.e. **neither** `ISO` **nor** the `SHRINK` family §5 expected. A
**family-matched requirement curve** was therefore computed by scaling this stream's own residual
pool through the real rule. **The original 0.813 m bar is still reported against every arm**; the
family-matched σ₀ = 1.1434 m is reported beside it, and §0.7 of `EGOAL_1.md` states plainly that the
two disagree and why.

**A4 (2026-07-27, at S4).** A **decorrelation control** was added: the parent's own along-track
residual pushed through the identical resampler, against its known correlated value (−10.4 %). It
was added because the injection could have been optimistic; it measures the construction to be
**conservative** (−33.1 % vs −10.4 %). It is reported whichever way it had come out.

## 9. Threats I expect and will report whatever they say

- Dense 10 Hz grid ⇒ ~171 near-duplicate windows per clip. Handled by clip-level folds and a
  clip-level bootstrap, but it inflates apparent n; every interval is clip-clustered.
- `obstacle.offline` is a **label**, not a perception output. A real vehicle's lead estimate is
  noisier ⇒ any positive result here is an **upper bound** on a deployed one.
- ADE/displacement only — blind to collision and TTC outcomes.
