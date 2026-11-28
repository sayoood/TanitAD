# RESULT — WP-B stage 2: the R8-4 metric suite, and refcv7's baseline row

**Stamp.** MEASURED. Artifacts: `raw/r84_baseline_refcv7.json` (md5 `fe4ae3c9…`; produced by `code/r84_baseline_refcv7.py`, two
runs identical except `wall_s`) and `raw/d0_dose_response.json` (D0, section 4). Module: `taniteval/taniteval/tactical_conditioning.py`
(`code/fix/`, md5 `251d6e31…`; tests `taniteval/tests/test_tactical_conditioning.py`, 60/60, every metric with a literal
known-value check and a mutation that goes red). **Tier:** OPEN-LOOP, logged frames of the held-out eval139, refcv7-r101-s0
@ 50,400, one TRAINING seed; inputs are the route package's banked captures `eval_s0g` / `eval_s1` (md5 `1c48a53e…` /
`5f783c84…`). Intervals: episode-cluster bootstrap (B 2000) — the "another draw of EPISODES" question only.

## 1. The suite (what exists now, independent of the model's own tagger)

| R8-4 measure | function | refcv7 row |
|---|---|---|
| (i) tactical accuracy (macro-F1, per class with n, majority + train-marginal controls, partial labels) | `class_report` | below |
| (i) goal-token AP vs prevalence | `goal_ap` | ABSENT: the capture holds `p_goal` but no per-window GT goal bits |
| (i) constraint MAE vs a constant | `constraint_mae` | ABSENT: refcv7 has no constraint heads |
| (ii) controllability (forced TURN_L / TURN_R / LANE_KEEP / STOP-at-d) + the shuffled-condition comparison | `controllability`, `controllability_vs_shuffled` | ABSENT: refcv7 has no conditioning input that can be forced |
| (iii) consistency (candidates and pick vs their tag; pick vs tactical argmax) | `consistency` | below |
| (iv) route following (bars 0.95 / 0.70) | `route_following` | below |

Independence (the cross-check rule): the module imports numpy and `taniteval.ci` only — no `tanitad`, no
`refcv8_conditioning`, no route package (pinned by an AST test); its direction classes equal the route package's
`route_metrics` exactly on every fan candidate, pick and GT of both captures.

## 2. refcv7 baseline (eval seed 0; seed 1 reproduces within ~0.01, the tactical head is seed-independent)

**Controls (hard, all PASS):** GT classes 40 L / 67 R / 588 straight / 105 gentle / 312 unclassified, window for window equal to
the route package; pick on GT-turn windows direction-correct **90/107 = 0.841 [0.742, 0.920]**, heading-within-15° **55/107 =
0.514 [0.423, 0.603]** (route RESULT §1.1).

**(i) Tactical accuracy — no skill over the majority class, on every reading:**

| reading | windows | head | majority control | head − majority [paired CI] | macro-F1 |
|---|---|---|---|---|---|
| lat3 = 8-way argmax mapped (TURN_L→L, TURN_R→R, else K) vs the GT plan's 30° heading-excursion class | 834 with a valid 6-s GT | **0.856** [0.812, 0.896] | **0.8645** | −0.008 [−0.031, +0.013] | 0.492 |
| same, on the route-classified set | 800 | 0.8588 | 0.8588 | 0.000 [−0.019, +0.021] | 0.501 |
| D0 §4: side-collapsed posterior vs the GT terminal-heading direction class (τ 10.35°) | 800 | **0.7375** | **0.7388** | — | 0.586 |
| lon (v7 8-way, strict) vs the plan's v9-literal class | 834 | 0.418 [0.367, 0.470] | 0.456 (CRUISE) | −0.037 [−0.106, +0.028] | 0.351 |

**Why the two lat3 numbers differ (0.856 vs 0.7375) — same capture, same 1,112-window EVAL-DIAG grid, same head outputs; three
definitions differ:**
1. **Window set.** Stage 2's headline scores the **834** grid windows whose 6-s GT is valid (slot 60), which includes 34 windows
   whose GT path is under 5 m (the route rule leaves them "unclassified"; the lat3 rule calls them LANE_KEEP). D0 scores the **800**
   route-classified windows (turnL / turnR / straight / gentle).
2. **GT class.** Stage 2: TURN iff the GT plan's largest segment-heading excursion reaches **30°**, else LANE_KEEP (the R1 / v9
   turn bar) — so the 105 "gentle" windows are LANE_KEEP. D0: the route package's **direction class of the terminal heading at
   10.35°** — so the gentle windows count as L / R (GT L/K/R = 73 / 591 / 136 in D0 vs mostly K in stage 2), which is a harder,
   less K-dominated target.
3. **Readout.** Stage 2: the 8-way argmax, with NUDGE / LANE_CHANGE read as K. D0: the posterior SUMMED per side (NUDGE_L +
   LANE_CHANGE_L + TURN_L = L), then the argmax.

The majority control moves with the target (0.8645 vs 0.7388), and on every reading the head sits on it. The conclusion is the
same on both: **refcv7's tactical lateral head carries no skill beyond the class prior** on this grid.

**(iii) Consistency (tag := the tactical argmax; it is the same for every candidate of a window):** reading A (lat3): pick vs tag
0.886 on all classified, **0.439 [0.318, 0.564] on GT-turn windows**; reading B (side-collapsed, D0's): 0.751 / **0.449**; the
candidate share at the tag (0.416 / 0.352 on turns) sits at the permuted-tag null (0.361 [0.335, 0.378]). On turns the pick agrees
with GT (0.841) about twice as often as with the tactical layer.

**(iv) Route following:** both bars fail on GT-turn windows on both seeds (0.841 < 0.95; 0.514 < 0.70); straight windows 0.964 /
0.980.

## 3. What this sets for refcv8 (SPEC_WPB, registered `c952d4b4…`)

The R8-4 (i)/(iii) baselines above are the refcv7 rows of SPEC_WPB's TACTICAL and consistency measures; B-TAC's bar (lat3 ≥ 0.95 on
turn windows) starts from a head that does not beat the class prior. Goal AP, constraint MAE and controllability become computable
with the refcv8 seams (the instruments are built and tested).

## Deliverable manifest

| artifact | where |
|---|---|
| `RESULT_STAGE2.md` | this package |
| `code/fix/taniteval/taniteval/tactical_conditioning.py` (md5 251d6e31…) | → repo `taniteval/taniteval/` |
| `code/fix/taniteval/tests/test_tactical_conditioning.py` (md5 df78778e…) | → repo `taniteval/tests/` |
| `code/r84_baseline_refcv7.py` (md5 89b34750…), `raw/r84_baseline_refcv7.json` (md5 fe4ae3c9…) | this package |
