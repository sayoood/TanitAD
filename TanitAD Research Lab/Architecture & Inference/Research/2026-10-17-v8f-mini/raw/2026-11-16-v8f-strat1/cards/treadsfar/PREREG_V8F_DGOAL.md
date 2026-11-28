# PREREG_V8F_DGOAL: SCOPING NOTE (not a frozen card): goals derived from detected strategic information

Status: written 2026-11-18 by the pre-registration designer for the MM. **It is not frozen and not a pre-registration**: it cannot honestly be one tonight, for the four reasons in section 0. Nothing is built or run. It is CKPT120's second committed next step on S120-FOLLOWS ("the derived-goals card") and a member of the PI horizon ruling (2026-11-08): *strategic 6-30 s incl. AV checkpoints, detected strategic information and derived goals from it*. PI decisions: `QUESTIONS_TFAR_DGOAL.md` Q5-Q6.
Evidence classes and tiers as in `PREREG_V8F_TFAR.md`.

## 0. Why this is a scoping note, not a card

1. **"Derived goals" has no operational definition in the repo.** Every STRAT1 / CKPT120 / NAVRB2 / weather-head card names it only as "a later card". Which goal variables S should derive is the PI's definition to give. Section 1 lists the candidates.
2. **The labels do not exist at any layer for the non-geometric goals.**
   * Detected information exists in the **augmented label release**: `strat_labels_table.npz`, sha256 `8055a321…`, with country, weather, surface, illumination and period of day per cache step (M).
   * There is no goal label in the **published corpus**, the **episode build**, or the **augmented release**: no 6-30 s speed goal, no time-gap goal and no stop goal.
3. **The base has no detected-information head.**
   * H2, the strategic base of record, is N1-based with `--ctx-heads off` (M, `ckpts/H2_s0/argv.json`).
   * A goal derived from detected information needs the ctx head. In the S1d form (`--ctx-detach on`), the S1 TAC-LAT cost is gone, and country, surface, illumination and period of day DETECT at s0/s1 (M, `read/s1d/out/RESULT_STRAT1_S1D.json`).
   * H2 + detached ctx heads has never been trained. That is a rung of its own before any goal rung.
4. **The measured information for the obvious candidate is near zero** (section 2). A card whose main lever the labels already predict null should not be frozen before the PI chooses between the candidates.

## 1. Candidate goal variables (the PI chooses; all in the 6-30 s band)

| id | goal | derived from | label layer it needs | status |
|---|---|---|---|---|
| G1 | geometric goal: S's route checkpoints incl. 120 m | S's inputs incl. route text | episode build (RC-A at 30-150 m, built) | **exists**: CKPT120 H2 S120-FOLLOWS; T reading it is `PREREG_V8F_TFAR.md` |
| G2 | strategic speed goal: mean speed over [t + 6 s, t + 30 s] | detected info (weather, surface, illumination, period of day, country) + limit + S's state | episode build (egomotion, CPU minutes, like RC-A120) | not built; information measured near zero (section 2) |
| G3 | time-gap / headway goal over 6-30 s | detected info (weather, surface, illumination) + S's state | program-wide layer (the `obstacle.offline` lead join, `lead_train4572.npz`, pod) | not built; information not measured (the lead table is not on the dev box, U) |
| G4 | stop goals (stop line, signal) in 6-30 s | detected strategic information (traffic light from the v7.x augmented release, the map stop-line card) | augmented release (779 / 4,572 v7.2 records carry a traffic-light goal, I from CLAUDE.md) + the pending stop-line card | not scoped here: depends on the stop-line card (PI 2026-11-16) and on the v7 TL label reaching training |

## 2. Design measurement for G2 (M, labels only, zero training: `DGOAL_INFO.json`, `measure_dgoal_info.py`)

* **Question.** Does detected information predict the driven speed in the 6-30 s band beyond the current speed? If it does not, a G2 goal derived from it has nothing to pass on.
* **Data.** Train split, 4,350 of 4,572 clips; the other 222 have no anchor with 30 s of log after t >= 0 (U, why not checked). 182,283 anchors every 2 s.
* **Variables.** v_goal = mean |v| over [t0 + 6, t0 + 30] s from egomotion; v_now = |v(t0)|. Each detected variable is the clip's modal valid label.
* **Method.** Least squares with a clip-grouped 2-fold split.

| features | R² | ΔR² vs v_now | RMSE m/s |
|---|---|---|---|
| v_now, v_now² | 0.8012 | | 3.397 |
| + all five detected variables | 0.8045 | **+0.0032** | 3.369 |
| + four (no country) | 0.8036 | +0.0024 | 3.376 |
| + illumination alone | 0.8038 | +0.0025 | |
| + weather alone / surface alone / period alone / country alone | 0.8012 / 0.8012 / 0.8012 / 0.8022 | 0.0000 / 0.0000 / −0.0001 / +0.0009 | |
| CONTROL: labels deranged across clips | 0.8001 | **−0.0012** (must be <= 0.002: passes) | |

* **Per-class means of v_goal − v_now:**
  * weather: clear +0.92 (3,720 clips), rain +0.82 (181), snow +0.86 (304), fog +0.52 (88);
  * illumination: day +0.95, lit night +1.15, dark +0.42.
* ⇒ **H:** on this corpus, a speed goal derived from detected information adds about 0.3 % of variance beyond the current speed. Drivers in rain or snow are already slower at t0, so v_now absorbs most of what weather would say. Illumination ("dark") is the only visible signal.
* This is a clip-level linear measurement. It does not rule out a non-linear or within-clip effect (TRAPS: a linear negative is not a negative about learnability). It does rank G2 below G1.

## 3. Admissibility plan (binding on any card built from this note)

* **Goal-input rule.** The ctx head is not the situation classifier: v8f-mini has none, and its inference read set is pinned (`assert_no_situation_classifier_input`, `v8f_mini.py:1543` at `aedf8716`). The rule's reasons (attribution, independent evaluability, echo) apply to any classifier whose output feeds a goal, though. So:
  * the goal head reads the ctx posterior **detached**;
  * the ctx head's own accuracy cells are read in every arm, so it stays independently evaluated;
  * an attribution arm derives the same goal from S's hidden state without the ctx posterior;
  * a donor-ctx eval variant (ctx posterior from the donor window) measures the goal's dependence on detected information.
  * State per arm what the goal is computed from. A shared adapter feeds the ctx head and S; that is the declared coupling STRAT1 measured with N1.
* **Vision-only at inference.** The ctx head reads `feats` and `frame_ok` only (STRAT1 T-VIS, I). The goal labels G2 / G3 use the ego's future (labels may use ego). No goal label is an inference input.
* **Optimism.** If the goal also reads the route text, every number carries "with the route supplied along the driven path".

## 4. The ladder a card would pre-register (each rung one lever; all PI go)

| rung | = | arms (s0, s1 + replicate) | cost (E, 16 min per arm, the TFAR measured rate) |
|---|---|---|---|
| D0 | H2 + `--ctx-heads on --ctx-detach on --w-ctx 0.5 --ctx-lr-mult 10` (S1d form on H2): does H2 keep S120-FOLLOWS and driving, and do the heads detect? | 2 + rep | about 0.8 pod-h |
| D1 | D0 + S's goal OUTPUT (G2 and/or G3 head on S's hidden + detached ctx posterior; loss and evaluator only, isolated from T and O as CKPT120 H1) | 2 + rep + a goal-label-permuted must-fail arm | about 1.1 pod-h |
| D2 | D1 + T reads the goal (the TFAR form: zero-init additive layer into T's token) | 2 + rep + donor-ctx must-fail + goal-from-hidden-only attribution | about 1.3 pod-h |

* Deciding cells, if G2: LONGITUDINAL tactical-band speed MAE, and the time-gap MAE where a lead exists. STRATEGIC: goal error against the 6-30 s label, versus a v_now-only baseline (the EXTRAP analogue).
* Controls as TFAR section 7, plus K-CTX: ctx accuracy inside its floor.
* Outcomes follow RULE ZERO; the replicate floor and the 4-seed rule are as TFAR.

## 5. What must be built first, and the cheapest next experiment

1. **The PI's definition** of the goal variables (Q5).
2. **G3 information measurement** (labels only, CPU): the same method as section 2 on the lead time-gap over 6-30 s. It needs `lead_train4572.npz` (pod `/root/strat1/inputs/data/v9/`) on the dev box, or it can run on the pod CPU. It is the cheapest discriminating step: if G3 is also near zero, the derived-goal line reduces to G1 (TFAR) and G4 (stop goals).
3. **The G2 / G3 label writer** (episode build layer): egomotion plus the lead join, row-aligned to the join like `build_rc120_labels.py`. About a minute of CPU (E, by analogy with RC-A120's 64 s, M).
4. **D0's build**: a `ckpt120_check`-style kind with ctx heads on and detached. `--ctx-heads` / `--ctx-detach` already exist in the `aedf8716` trainer (H2's argv names them, off).

## 6. What runs with no PI go

The section-2 measurement (done) and the G3 measurement (CPU). Nothing is built or trained.

## Artifacts (this folder)

| file | sha256 |
|---|---|
| `measure_dgoal_info.py` | `1a58361e…` |
| `DGOAL_INFO.json` (label table sha256 `8055a321…`) | `7ddde028…` |
