# S3 — manoeuvre-initiation timing: PRE-REGISTRATION

**Date:** 2026-07-26 (Europe/Berlin) · **Stream:** 4-Brain Dominance Program, §2 problem **S3**
**Spec:** `…/2026-07-26-4brain-dominance-program/STRATEGIC_TACTICAL_PROBLEM_SPEC.md` §2 S3 · §0.1 firewall · §0.3 horizon bar · §0.4 power bar
**Status of this file:** written **before** any model is scored on S3. No arm exists. No GPU used. Nothing staged.

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) ·
`INHERITED` (another of our docs, **not** re-verified here) · `ESTIMATED` · `HYPOTHESIS` · `PLANNED`.

---

## 0. Why this document exists, and what it is allowed to fix

S3 is the program's **deliberate near-control**. The pre-registered expectation is that the hierarchy
does **not** win much on it, because *when to begin a manoeuvre* is largely a tactical/operative
competence. Its value in the proof is entirely negative-space:

> **If the hierarchy's advantage is UNIFORM across S1/S2/S3, it is CAPACITY, not STRUCTURE.**
> (`4BRAIN_DOMINANCE_PROGRAM.md` §3.2 HP-2 falsifier: *"uniform Δ across strata ⇒ the gain is capacity, not strategy"*.)

A control that is expected to be null is only informative if it is **built to the same standard as the
treatments**. This pre-registration therefore fixes, *before any arm is scored*:

1. the target and the proof that it is non-circular (§2);
2. the miner's admissibility rules (§3);
3. the option set and its band edges (§4);
4. the metric, its baselines, and the estimator (§5);
5. **what counts as the hierarchy winning, what counts as null, and what counts as the *expected*
   near-null** (§6) — all three committed here;
6. the refusal conditions that would remove S3 from the program entirely (§7).

**What this document may still change:** nothing in §4–§6 after the first arm is scored. §3's
sensitivity arms (H_S3 ∈ {8, 12, 15} s, `MIN_TTM_S` ∈ {0.5, 1.0, 2.0}) are pre-registered *as
sensitivity arms* — the **primary** is `H_S3 = 12 s`, `MIN_TTM_S = 1.0 s`, and a sensitivity arm may
never be promoted to primary after the fact.

> ### AMENDMENT LOG (append-only)
>
> **A1 · 2026-07-26, before any arm was scored — M3 made symmetric with the target.**
> The first implementation tested M3 ("has the manoeuvre begun?") with an **instantaneous** threshold on
> **raw** per-step acceleration, while the target is a **sustained** segment of **savgol-smoothed**
> speed. On a 6-episode smoke that mismatch left **3 of the 5 longitudinal classes empty**
> (`t_5_10` / `t_10_H` / `t_none` all 0). M3 now runs the **same segment detector as the target**,
> backwards from `L`, on the **same smoothed signal** (`s3_labels.lat_in_progress` /
> `lon_in_progress`). *Nothing in §4–§7 changed; no model existed, so no result could be flattered.*
>
> **A2 · 2026-07-26 — the longitudinal manoeuvre thresholds are NOT amended.** After A1 the 8-episode
> read still looked degenerate and the obvious move was to raise `A_MAN`/`DV_MIN`. A pre-declared,
> outcome-independent sweep on **60 episodes** shows every higher threshold is **worse** (`t_10_H`
> empties, `t_none` reaches 0.98 at `A_MAN = 2.5`). **`A_MAN = 0.5 m s⁻², DV_MIN = 1.5 m s⁻¹` stand
> unchanged.** Recorded because the near-miss is the useful half: this is the CLAUDE.md
> *"verify before alarming — take multiple samples first"* trap, caught at 8 episodes.

**Instrument design vs. result tuning — stated so it cannot be re-litigated.** The band edges (§4) are
chosen from the *label* distribution, with **no model number in existence**. There is nothing to
flatter. The full distribution is reported alongside the bands (`S3_IMPLEMENTATION.md` §4) so any
reader can re-band it.

---

## 1. The decision problem, restated precisely

> **Given the scene at time `t`, and given that the manoeuvre has not yet begun — WHEN does it begin?**

| Field | Commitment |
|---|---|
| **Surface** | **`open-loop-choice`** only. The model emits the band from the observation window. |
| ⛔ **Forbidden surface** | **`rollout_decode` under the expert's true future actions** (§0.2 PC2). On that surface the future action sequence *is* the future path, so `ttm` is recoverable from the inputs and S3 becomes circular by construction. **S3 is never scored there.** This is a concrete PC2 consequence, registered here so it cannot be re-derived later. |
| **Decision horizon `H_S3`** | **12 s** primary (`8 s`, `15 s` as sensitivity). Every mined window must be able to *observe* the full `H_S3`, so the target is **never right-censored**. |
| **Axes** | **two, minted and scored SEPARATELY** — lateral (`ttm_lat`) and longitudinal (`ttm_lon`). Never pooled, never a single mixed softmax. |
| **Output space** | one ordered 5-class variable **per axis** (§4). |
| **Resampling unit** | the **episode** (a PhysicalAI clip, ~20 s). |

### 1.1 ⚠️ The single-axis rule, and how S3 obeys it

`CLAUDE.md` / spec §1: the v1 5-way softmax (`lane_keep, turn_left, turn_right, accelerate, brake_stop`)
**mixes lateral and longitudinal classes**, one documented mechanism behind **0/881 accelerate
predictions** (`memory: longitudinal-blindness-root-cause`, `INHERITED`).

The shipped `strat_scalars.ttm` (`stack/scripts/v4_labels.py:147-164`) is keyed **only** to a
*junction-scale curvature segment* — i.e. it is a **lateral-only** clock. A model could score perfectly
on it while being completely blind to *when to start braking*. S3 therefore mints a **second,
symmetric longitudinal clock** and reports the two side by side. **A single `ttm` number is not an
admissible S3 result.**

This two-axis split *is* S3's lateral/longitudinal decomposition (the scalar target has no trajectory
to decompose with `taniteval/lateral.py`; if an S3 arm also emits a path, that path is decomposed with
`taniteval/lateral.py` as usual and reported as a secondary).

---

## 2. The target, and the proof that it is non-circular

### 2.1 Definition (`MEASURED` provenance for every field it touches)

For a window whose **last observed pose index is `L`** (`L = t + WINDOW − 1`, `WINDOW = 8`):

* **Lateral.** `ttm_lat(L)` = seconds until the ego reaches the **start of the first junction-scale
  curvature segment** in `poses[L+1 : L+1+H_S3/Δt]`. A *junction-scale curvature segment* is a
  contiguous same-sign run of arc-smoothed `|κ| ≥ 1/60 m⁻¹` sweeping `|Δψ| ≥ 15°`
  (`stack/scripts/refb_labels.py:935-959`, `:1045-1056`). Implemented by calling the **shipped**
  `v4_labels.time_to_maneuver` with `horizon_steps = H_S3/Δt`, so S3 inherits the audited label rather
  than a private fork.
* **Longitudinal.** `ttm_lon(L)` = seconds until the start of the first **sustained longitudinal
  segment** in the same future span: a contiguous same-sign run of Savitzky–Golay-smoothed
  `|a| ≥ 0.5 m s⁻²` lasting `≥ 1.0 s` with total `|Δv| ≥ 1.5 m s⁻¹`. (Thresholds chosen to sit inside
  the existing vocabulary's `DV_ACCEL_MS = +1.0` / `DV_BRAKE_MS = −1.0` over 2 s ⇒ ±0.5 m s⁻²,
  `refb_labels.py:58-59`; sensitivity arm at `A_MAN ∈ {0.4, 0.5, 0.7}`.)
* **No event within `H_S3`** ⇒ the ordinal top class `t_none` (§4). Because `H_S3` is fully observable
  by construction, `t_none` means *"no manoeuvre in the next 12 s"*, **not** *"the clip ended"*.

**Fields the target touches — the complete list:**

| field | index range read | is it a model input? |
|---|---|---|
| `poses[:, 0:2]` (x, y) | `L+1 … L+120` (**strictly future**) | ❌ no |
| `poses[:, 2]` (yaw) | `L+1 … L+120` (**strictly future**) | ❌ no |
| `poses[:, 3]` (v) | `L+1 … L+120` (**strictly future**) | ❌ no |

**Nothing else.** No map, no agent tracks, no camera, no label, no `nav_cmd`.

### 2.2 The non-circularity argument, and its exact form

The proof is an **index-range disjointness** argument, not a semantic one:

> The model's observation ends at pose index `L`. The target reads **only** `poses[L+1 : L+1+H]`.
> The two index ranges are **disjoint**. Therefore no fixed rule maps the model's inputs onto the
> target — the target is a fact about a segment of the trajectory the model has never seen.

This is a strictly stronger position than the v1 route label, which failed because
`route_target(nav_cmd) = _NAV_TO_ROUTE[nav_cmd]` read the input itself
(`stack/scripts/refb_labels.py:172-175`, `MEASURED`) ⇒ `route_skill = 0.0000` **by construction**.

**It is asserted in code, not in prose.** `test_s3_labels.py::test_target_never_reads_the_observed_window`
mutates `poses[:L+1]` arbitrarily and asserts every S3 target is bit-identical. A future refactor that
lets the observed window into the target fails that test.

### 2.3 ⚠️ Where disjointness is NOT enough — and what actually threatens S3

Index disjointness kills *copying*. It does not kill *echo*. The spec's §0.1 caution is exactly right,
and on this stack it has teeth:

**`MEASURED` (`stack/tanitad/models/flagship_v4.py:198-203`; fed at eval by
`stack/scripts/eval_flagship_v4.py:333` via `train_flagship_v4._goal_inputs`)** — the v4 head's
inference-time signature is

```
forward(states, v0, imagined=None, vt_band=None, route=None,
        route_graded=None, vt_speed=None, steps=None, lambda_plan=1.0)
```

Four of those conditioning channels are **themselves derived from the ego's future**:

| channel | how it is minted | what it leaks to S3 |
|---|---|---|
| `route` (v2.1 3-class + unknown) | `route_from_future_v21(poses, L)` over a **25 s** future lookahead | *whether* a lateral manoeuvre is coming, and its direction |
| `route_graded` (scalar) | `refb_labels.route_graded_target` — a **graded** version of the same future turn | a monotone proxy for the same event's magnitude |
| `vt_band` (24-way) / `vt_speed` | `vtarget_v2(v, L, min_lookahead=50)` — the ego's own speed **≥ 5 s ahead** | the future speed profile ⇒ *when the ego slows* = the longitudinal target |

> ⚠️ **This is the real circularity risk in S3, and it is not the one the spec anticipated.**
> `ttm_lat` and `route`/`route_graded` are derived from the **same** function family
> (`route_from_future_v21` ⊂ `route_from_future_v3`) over an overlapping future span.
> `ttm_lon` and `vt_speed` are both statements about the ego's future speed.
> **A model that is handed `route_graded` and `vt_speed` at inference has been handed a
> low-resolution copy of both S3 targets.**

**Consequence, committed in advance:** the `blind_conditioning_baseline` for S3 **must** include
`{v0, route, route_graded, vt_band, vt_speed}` — the *actual* inference-time conditioning set, not a
guess at it — and the reported capability is **`skill = score_model − score_blind`**, never
`score_model`. See §5.3 and §7.

---

## 3. The miner — admissibility rules, fixed in advance

A window at last-pose index `L` in an episode of `T` poses enters the S3 set **iff all** hold:

| # | Rule | Why (each has a named failure it prevents) |
|:--:|---|---|
| **M1** | `T − 1 − L ≥ H_S3/Δt` (**full decision horizon observable**) | otherwise `t_none` conflates *"no manoeuvre"* with *"clip ended"* — **informative censoring**, i.e. the target becomes partly a clock. PhysicalAI clips are `T ∈ [188, 205]` (`MEASURED`, `parity_profile.json` `T_out_min/max`), so this bites hard and is stated in §8. |
| **M2** | the manoeuvre has **not begun**: `ttm ≥ MIN_TTM_S = 1.0 s` | at `ttm ≈ 0.1 s` the segment starts at future index 0 — i.e. it is already underway and the answer is visible in the last observed frames. Brief's explicit exclusion. |
| **M3** | the **observed window itself** is not already executing the axis's manoeuvre: mean arc-smoothed `\|κ\|` over `poses[L−7 … L]` `< 1/60 m⁻¹` (lateral) / mean `\|a\|` `< 0.5 m s⁻²` (longitudinal) | belt-and-braces on M2, and the direct implementation of *"the answer would be visible"*. Reported separately so its marginal effect is countable. |
| **M4** | ego is **moving**: mean in-window speed `≥ 1.0 m s⁻¹` (`MOVING_V_MS`, `refb_labels.py:61`) | at standstill `κ = Δψ/Δs` blows up (`ds` floors at 0.1 m) and parking-lot yaw jitter reads as a junction. This is the **same** defect the `SEG_MIN_DYAW_RAD` gate was added for (`MEASURED`: `d_now` was 864/2201 = 39 % before that gate). |

**M0 (not a filter, a refusal):** the miner refuses to run against `physicalai-val-f1b378f295ae`
(**78.5 % leaked into the parity train**) and prints the resolved cache directory and its parity status
on every run. `sorted(glob("*val*"))[-1]` is never used — it **prefers** the leaked split
(`MEASURED`, `VAL_PARITY_REPORT.md` FINDING 1).

**Windows with no manoeuvre in `H_S3` are NOT excluded** — they become `t_none`. This is a deliberate
departure from the brief's *"exclude where no manoeuvre occurs"*: excluding them would condition the
mined set on the target's own value (selection-on-outcome), and it would delete the only class that
carries *"do nothing yet"*, which is half of a timing decision. The conditional-on-occurrence variant
is still reported as **S3-A′** (drop `t_none`) so the two are comparable.

---

## 4. The option set — five ordered classes, per axis

**Primary bands (fixed here, before any arm exists):**

| ix | name | interval | rationale |
|:--:|---|---|---|
| 0 | `t_1_2` | `[1, 2) s` | the spec's `≤2 s` band, floored by M2 |
| 1 | `t_2_5` | `[2, 5) s` | spec |
| 2 | `t_5_10` | `[5, 10) s` | spec |
| 3 | `t_10_12` | `[10, H_S3) s` | spec's `>10 s`, **closed** at the decision horizon so it is observable |
| 4 | `t_none` | no manoeuvre within `H_S3` | the "not yet" option; required or the problem has no null action |

Band edges are the spec's own (2, 5, 10 s) — **adopted unchanged so no one can claim they were tuned**
— with the top band closed at `H_S3` (§3 M1) and a `t_none` class added (§3).

**Secondary (robustness, pre-registered): equal-mass quartile bands** over the observed `ttm`
distribution among event windows, plus `t_none`. Equal-mass maximises class entropy and therefore
*minimises* the majority-class baseline — it is the **least** flattering binning available, which is
why it is the robustness arm and not the primary.

**Reported alongside, always:** the full `ttm` distribution (deciles + histogram) for both axes, so the
binning is auditable and re-derivable. A band table without the underlying distribution is not an
admissible S3 report.

---

## 5. The metric, the baselines, and the estimator

### 5.1 ⛔ Never bare accuracy

The strategic head reports `route_acc_nav = 1.0000` beside `route_skill = 0.0000`
(`MEASURED`, three artifacts, spec §0.1). **We will not be fooled by that shape twice.** Every S3
score is reported as a triple: `(chance-corrected statistic, raw statistic, baseline)`.

### 5.2 Primary and co-primary

| role | metric | why this one |
|---|---|---|
| **PRIMARY** | **`ttm_band_QWK`** — quadratic-weighted Cohen's κ over the 5 ordered classes, **per axis** | (a) **ordinal**: a 2-band miss costs 4× a 1-band miss, which is what a timing error means; (b) **chance-corrected**: a majority-class predictor scores **exactly 0.0 by construction** — the `route_acc = 1.0` shape is arithmetically impossible; (c) it is a single number that cannot hide a dead class. |
| **CO-PRIMARY** | **`ttm_MAE_skill_s`** `= MAE(median-constant) − MAE(model)`, over event windows only | the MAE-optimal constant is the median, so this is the honest "did you beat always-guess-the-middle" in seconds. Positive = better. |
| **REQUIRED DIAGNOSTIC** | **per-band recall** (all 5 classes, both axes) | this is the metric that would have caught **0/881 accelerate** the day it happened. A band with recall 0.000 is reported in bold; a head that never emits a band has not learned the option set regardless of its QWK. |
| **REQUIRED DIAGNOSTIC** | **`early_late_bias_s`** = mean signed error (pred − truth) | timing is directional: predicting *late* is a safety failure, predicting *early* is a comfort failure. An undecomposed MAE hides which one the arm makes. |
| Secondary | band accuracy, off-by-≤1 accuracy, majority-class rate | reported **only** next to QWK and the majority rate, never alone. |

### 5.3 Baselines — all three are mandatory

| id | baseline | what it tests |
|---|---|---|
| **B0** | **majority-class / median-constant** | the floor. QWK(B0) = 0.0 by construction; `MAE_skill`(B0) = 0.0 by construction. |
| **B3** | ⭐ **`blind_conditioning_baseline`** — 2-layer MLP on `{v0, in-window speed/Δv/κ, route, route_graded, vt_band, vt_speed}`, **no pixels, no history beyond the window's own kinematics** | §2.3. This is the firewall. `skill = score_model − score_B3`. |
| **B4** | B3 **+ the observable horizon `H_obs`** | ⚠️ **an S3-specific check the spec's firewall does not name.** If B4 ≫ B3, the target is partly *"how much clip is left"* — a dataset artifact, not a scene fact. On a 20 s corpus this is a live risk and M1 exists to kill it; B4 measures whether M1 worked. |

Two ablation rungs are reported between B0 and B3 so the leak is *attributable*:
**B1** = `{v0, in-window kinematics}` only · **B2** = B1 + `{route, route_graded}`.

### 5.4 Estimator — named on every interval

* **Episode-cluster bootstrap**, `taniteval/ci.py`, **B = 2000**, resampling unit = **val episode**.
* Two arms on the same windows ⇒ **`paired_episode_cluster_bootstrap`**, never a quadrature combination.
* ⛔ **`overlapping_holdout_se` is never used.** It is 1.28–2.06× too narrow (`MEASURED`, 10 arms) and
  moved a point estimate by up to ×2.97.
* QWK is not a per-window mean, so it is bootstrapped through the **callable-reducer** path
  (`ci.resolve_reducer`) on resampled window indices — the same estimator, no private interval.

---

## 6. ⭐ The three outcomes — all committed in advance

Let `Δ = QWK(A2-HIER) − QWK(A0-FLAT)`, paired episode-cluster bootstrap, B = 2000, per axis, with both
arms receiving the **identical working route input** (the `nav_cmd=None` void, spec §5.3, does not
recur here).

| outcome | definition | what it means |
|---|---|---|
| ⭐ **EXPECTED (near-null)** | the paired CI for Δ **contains 0** on **both** axes, **and** the same comparison on S1/S2 is CI-separated | **the pre-registered prediction.** The advantage is **concentrated at option-set problems** ⇒ it is **structure**. S3 has done its job as a control. |
| **NULL-UNINFORMATIVE** | CI contains 0 on S3 **and** on S1/S2 | the hierarchy shows nothing anywhere. S3 says nothing about structure-vs-capacity; the verdict is carried entirely by S1/S2 and by §5's falsifier. |
| ⛔ **CAPACITY VERDICT** | Δ is CI-separated on S3 **and** Δ_S3 is **statistically indistinguishable** from Δ_S1/Δ_S2 (paired interaction test, CI on `Δ_S3 − Δ_S1` contains 0) | **the hierarchy's advantage is UNIFORM across the battery ⇒ it is CAPACITY, not STRUCTURE.** This is a **secondary falsifier of the dominance claim** (HP-2). It is reported to the PI with the same prominence as a positive, and logged in `RETRACTION_LOG.md` with its root-cause class. |
| **HIERARCHY WINS ON S3 (unexpected, admissible)** | Δ CI-separated on S3 **and** Δ_S3 **< 0.5 × Δ_S1** (the interaction test separates) | the hierarchy helps timing *too*, but less. Compatible with dominance. **Requires an explicit "the near-control was not null" note** — a control that moves must be explained, not absorbed. |

**Binding:** none of these four rows may be edited after the first arm is scored. The
interaction test (`Δ_S3 − Δ_S1`) is what separates rows 1/3/4 and it is **not optional** — a bare
per-problem Δ table cannot distinguish capacity from structure, which is the entire point of S3.

**Direction of "winning" is fixed:** higher QWK is better; higher `MAE_skill` is better; `t_none`
recall counts equally with the event bands (a model that predicts "something is coming" everywhere is
not early, it is wrong).

---

## 7. ⛔ Refusal conditions — S3 leaves the program if any fires

| # | Condition | Verdict |
|:--:|---|---|
| **R1** | `QWK(B3) ≥ 0.98 × QWK_ceiling` (ceiling = 1.0) — the scene-blind head reaches ceiling | **the target is circular. S3 is REFUSED** and does not enter the program (spec §0.1 rule 5). |
| **R2** | `QWK(B3) − QWK(B0)` is large **and** `QWK(model) − QWK(B3)` has a paired CI containing 0 | the label survives R1 but is an **echo** of `route_graded`/`vt_speed` (§2.3). **S3 is only admissible with those channels WITHHELD**, and that variant becomes the primary. |
| **R3** | `QWK(B4) − QWK(B3)` is CI-separated and material | the target is partly a **clock artifact**. M1 failed; raise `H_S3` observability or refuse. |
| **R4** | fewer than **40** episode-clusters carry ≥1 mined decision point on the scored val | below the single-arm power bar ⇒ **no single-arm claim**. |
| **R5** | fewer than **200** episode-clusters on the scored val | **S3 may not be used as a two-arm control** on that val. Reported as a limitation, not worked around. |

R2's withheld variant is pre-registered *now* so that choosing it later is not a post-hoc rescue:
**S3-W** = identical in every respect except that `route`, `route_graded`, `vt_band`, `vt_speed` are
passed as their **DROPPED / null-row** values to **both** arms (the head already owns a learned
null row for each — `flagship_v15.condition`, `MEASURED`), so no architecture change is needed.

---

## 8. Pre-registered limitations — stated before, not after

1. ⚠️ **The corpus caps the horizon.** PhysicalAI clips are `T ∈ [188, 205]` poses ≈ **18.8–20.5 s**
   (`MEASURED`, `parity_profile.json`). With `WINDOW = 8` and M1, the largest fully-observable decision
   horizon is ~19 s at the *first* window and shrinks linearly. **S3 as built covers ~1–12 s of the
   spec's nominal 5–25 s range and structurally cannot test the upper half.** Anything beyond ~15 s
   needs a longer-clip corpus. This is registered as a **known incompleteness of the control**, not a
   defect discovered later.
2. **M1 selects clip-early windows.** At `H_S3 = 12 s` only `L ≤ T−1−120` qualifies ⇒ roughly the first
   **7.8 s** of each ~20 s clip. Scene mix at clip-start is not guaranteed to match clip-mean. Reported
   as a stratum, and the `H_S3 = 8 s` sensitivity arm exists to bound it.
3. **`ttm_lat`'s manoeuvre definition is kinematic**, so it inherits the spec §6 limit: *a description
   of what the ego did, never an instruction*. S3 is admissible under that limit specifically because
   **"when" is a kinematic fact even when "which branch" is not** (spec §2 S3 Ⓐ) — but it means S3
   cannot distinguish *"the ego chose to turn here"* from *"the road turned here"*.
4. **`ttm_lon` has no lead-vehicle referent.** `lead_state` is a `None` stub (`MEASURED`, 0 % coverage),
   so a deceleration caused by a lead car is indistinguishable from a chosen one. `ttm_lon` is "when
   the ego's speed changed", not "when the ego decided".

---

## 9. Compute and constraints honoured

* **CPU only, dev box.** No GPU requested. No training launched. pod1 (v2corpus) / pod3 (E1c) training —
  **not touched**. Eval pod (wheelbase) — **not touched**. pod2 — **not touched**.
* **Nothing `git add`-ed, committed or pushed.** Files written into the working tree only.
* Parity: the miner takes `--cache-dir` and prints the resolved cache + parity status; the
  decision-grade run is a **flag swap** to `physicalai-train-e438721ae894` /
  `physicalai-val-0c5f7dac3b11`, pod-side, zero code change.

---

## 10. Duplication to consolidate

The sibling brief places a reusable `blind_conditioning_baseline` in
`…/2026-07-26-4brain-preconditions/`. **That directory did not exist when this work ran**
(`MEASURED`, 2026-07-26, dev box). S3's firewall is therefore implemented **locally** in
`s3_blind_baseline.py`. It is written as a **corpus-agnostic function**
`blind_conditioning_baseline(X_cond, y, eid_train, eid_test, …)` with no S3-specific logic, so
consolidation is a move, not a rewrite. **⚠️ ESCALATION (do not leave this in a README):** whichever of
the two lands second must delete its copy — an orthogonality instrument sat unmerged for **10 days**
because the request lived in a file nobody re-read.
