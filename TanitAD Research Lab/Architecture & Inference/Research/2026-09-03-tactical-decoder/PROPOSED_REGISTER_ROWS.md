# PROPOSED REGISTER ROWS — `GOALS_AND_CLAIMS.md`

⚠️ **This FlyWheel does not own `GOALS_AND_CLAIMS.md`.** These rows are proposed for the Master
Mind to paste. Every number is MEASURED, T0, and cites an artifact in
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-tactical-decoder/`.

---

## New DECIDED / MEASURED rows

### `D-REFAV1-SEED-GOAL-MISMATCH` — ⛔ **BLOCKING, and prior to both fixes R28 proposed**

ℹ️ **Attribution: escalated qualitatively by the steer-conversion agent in commit `4139203`** (*"a 2 s plan is judged against a 6 s goal, so the planner's own seed cannot reproduce its own goal"*). This row is the MEASUREMENT of that residual, and it shows the same commit's `cost_time_grid` repair does not close it.

**MEASURED at source, 0 GPU, exact arithmetic on the shipped code.** `plan()` rolls the imagined
goal from the FULL 30-step canonical control subsampled onto the tactical grid — operative indices
`[0,3,6,9,12,15,18,21,24,27]` (`refa_v1.py:1761`) — but seeds the search with that control
TRUNCATED to `cfg.plan_steps = 10` (`refa_v1.py:1932`) ⇒ `[0..9]` under the DEFAULT
`cost_time_grid="dense"` — **the behaviour every BANKED refav1 number was produced under; the
parameter did not exist before `4139203`** — or `[0,3,6,9,9,…]` under that commit's new
`"tactical"` regrid. **62 of 64 (lat, lon) v7.0 token pairs hand the goal and the seed DIFFERENT
tactical action sequences UNDER BOTH GRIDS** (48/64 differ in curvature alone, under both), because
the cause is the TRUNCATION, not the regrid — ⚠️ **so `4139203`'s time-grid repair does NOT close
this residual.** The only two pairs that agree, `(LANE_KEEP, CRUISE)` and `(ABORT_LC, CRUISE)`,
are the all-zero control — *the one manoeuvre the planner can faithfully chase is doing nothing*,
which is the behaviour observed on 140/140 windows. At v0 = 10 m/s under the DEFAULT grid:
`TURN_L`/`TURN_R` over-rotate their own goal by **±82.506°** (ψ 3.360 → 4.800 rad); `LANE_CHANGE_R`
loses its return arc's sign and reads **−27.072°**; `NUDGE_R`'s S-curve is stretched **2.5× in
time** so **8 of 10** tactical steps carry the wrong curvature (its NET heading still matches, which
is why heading excess alone is not a sufficient gate); `LANE_KEEP × BRAKE_TO` differs on **9 of
10** accel steps and ends at **6.66×** the goal's final deceleration (−0.5811 vs −0.0872 m/s²).
⇒ The cost is not measuring the right pair imprecisely — **it is measuring the wrong pair.**
Must land before any decoder or metric arm is spent. Evidence: `raw/seed_goal_mismatch.json`,
`RESULT.md` §8.

### `D-REFAV1-TAC-DECODER` — the lateral head IS the constant predictor, to the digit

**MEASURED, T0, 140 windows / 20 episodes, in-band n = 40.** The incumbent (fp32) step-1,000
checkpoint emits `LANE_KEEP` on **140/140** windows; its in-band accuracy **0.650** equals the
constant-only base rate **EXACTLY** and its macro-recall **0.200** equals **1/K = 1/5 EXACTLY**;
the clustered label-permutation null gives `p(macro ≥ observed) = 1.0000`. Per class: `LANE_KEEP`
1.000 (n 26), `NUDGE_L` 0.000 (6), `NUDGE_R` 0.000 (2), `TURN_L` 0.000 (2), `TURN_R` 0.000 (4).
The EMA+bf16 arm (ep2) emits `TURN_R` on 25/140, reaching accuracy 0.675 / macro 0.377
(clustered p = 0.0168) with `TURN_R` recall 1.000 — **but that is a nav echo**: under nav-shuffle
its macro falls to **0.177, BELOW the constant control**, and under nav-zero it returns to
`LANE_KEEP` 140/140. On the geometric turn stratum (`gt_turn_deg ≥ 5°`, n = 27/140, 10 episodes)
the incumbent asks for a turn on **0/27** (0.000 [0.000, 0.000]) and ep2 on **9/27**
(0.333 [0.000, 0.667]). ⚠️ Only 8 of those 27 are in-band and only 3 carry a `TURN_*` label — the
geometric stratum is not the v7.2 token. **Longitudinally the head is NOT collapsed** (accuracy
0.550 vs 0.450, macro 0.378 vs 0.200, clustered p = 0.0098) and its predictions are row-for-row
identical on both checkpoints. Evidence: `raw/decode_audit.json`, `RESULT.md` §6.

### `D-REFAV1-TAC-LABEL-SHARE` — ⚠️ **CORRECTS the R28 brief and inverts its lever ordering**

**MEASURED.** The R28 brief read the decoder's supervision as 0.1–0.8 % of the loss from
`loss_feat_tac`. That is the tactical **FIELD PREDICTOR's** MSE — a different term. The decoder's
own term is `w_tac_label · mean(CE_lat, CE_lon)` and on the 140 eval windows it is **0.11313
(incumbent) / 0.11695 (ep2)** against feature terms summing to **0.71918 / 0.45324** — i.e.
**13.59 % / 20.51 %** of the label+feature sum. `w_tac_label = 0.1` in both recorded configs and the
live ep2 run passed real `--labels`. ⇒ **"the head is unsupervised / the term is too weak" is
REFUTED, and raising `w_tac_label` is NOT the lever.** Two further reads settle the mechanism:
`CE_lat = 0.8665 nats` sits **BELOW** the in-band label entropy `H = 1.0944` (the head carries
0.228 nats beyond the marginal *while its argmax is constant*), and the head's own
`P(TURN_L)+P(TURN_R)` ranks the label at **AUC 0.873 [0.686, 1.000]** (ep2 0.922 [0.737, 1.000])
against a no-information 0.500. **The head ranks; it does not decide.** ⇒ the lever is the DECISION
RULE (class-balanced / focal CE), not the weight. Evidence: `raw/intent_probe_*.json`, `RESULT.md` §2.

### `D-REFAV1-TAC-NAV-ECHO` — every scrap of the tactical decision is oracle nav, not scene

**MEASURED, and it discharges the standing nav-shuffle obligation (`refa_v1.py:395`).** With
`nav_zero`, the incumbent's `CE_lat` rises 0.8665 → **1.3390** (ep2 0.9420 → 1.3684) — **above** the
marginal entropy 1.0944 — and its turn-ranking AUC collapses **0.873 → 0.520**. The longitudinal
head collapses to `CRUISE` 140/140 (accuracy 0.450 = the base rate, macro 0.200 = 1/K). A predictor
reading **nothing but `nav_cmd`** (leave-one-episode-out) scores **0.684** on the same 38 scorable
in-band rows — **above the model's 0.650 and above the constant control's 0.650.** ⚠️ `nav` is an
**ORACLE (ego-future)** derivation, training-input only. ⇒ The one signal the tactical head has
learned to read is the one that will not exist at deployment. ep2 alone retains a vision-borne
component (`nav_zero` label-TURN AUC **0.858 [0.667, 1.000]**, CI excluding 0.500).
Evidence: `raw/decode_audit.json`, `raw/intent_probe_*.json`, `RESULT.md` §6.

### `D-REFAV1-TAC-LABEL-SPARSITY` — the head sees ~2 labelled rows per batch-8 step

**MEASURED (this slice) / INFERRED (train corpus).** `s2_labels_v7.2_train.jsonl.gz` carries
**4,572 records**, one per clip, with `bands.tactical_s = [2.0, 6.0]` and **`t0_s ≡ 8.0` on
4,572/4,572** — so the label attaches only to windows whose NOW lies in `[6.0 s, 10.0 s]`
(`refav1_loader.py:443`). In the TRAINING loader shape (`str_ext_steps = 2`, reach 60) that is
**199 / 739 = 26.93 %** of windows (eval shape: 420 / 1,339 = 31.37 %, confirmed independently by
the loader's own banner). **8.13 %** of batch-8 steps carry NO labelled row and the term is
`continue`-skipped entirely. ⚠️ `F.cross_entropy` uses `reduction='mean'` over non-ignored rows, so
the term's magnitude does not shrink — its **sample size** does, to ~2.15 rows per step. **The
gradient is noisy, not weak; reading "27 % of rows" as "27 % less loss" picks the wrong lever.**
Label distribution: `LANE_KEEP` 2,958 (64.70 %), `NUDGE_R` 592, `NUDGE_L` 488, `TURN_L` 275,
`TURN_R` 259; **`LANE_CHANGE_L`, `LANE_CHANGE_R`, `ABORT_LC`, `YIELD_MERGE` = 0** — 5 of the head's
16 output units have zero support. Evidence: `raw/window_band_census.json`,
`raw/decode_audit.json` `label_census_train`, `RESULT.md` §§3–4.

### `D-REFAV1-CHORD-INSUFFICIENT` — ⚠️ **CORRECTS BACKLOG R29's framing**

**MEASURED, 0 GPU, on ep2's 25 already-curved-goal windows.** `chord = √(2(1−cos))` is strictly
monotone in `1−cos`, so it cannot change the goal term's own ordering; what it changes is that
term's **leverage inside the cost SUM**, by a measured **median 5,792.6× (conv A) / 4,096.0×
(conv B)**. That flips **1 of 25** windows under convention A and **0 of 25** under B. As shipped
the turn wins **0/25** under both. The deeper reason: on **17/25** (A) and **21/25** (B) windows the
canonical turn's goal term is **no better than constant velocity** — the direction is wrong, not
only the magnitude (see `D-REFAV1-SEED-GOAL-MISMATCH` for why). The goal term's κ-range is
**1.63e-10 (f64)** against the `0.05·κ²` penalty's **2.00e-03** — a factor **1.2 × 10⁷**, of which
the chord's √ recovers ~2 × 10³. ⚠️ And because the cost is a SUM, swapping the metric while holding
`0.05` fixed is **not a one-variable arm**: it is a metric change *and* an implicit 5,793×
reweighting. ⚠️ **SCOPE: the banked panel predates the steer-conversion repair** (`4139203`, same
day), whose tiny-model evidence puts the units-FIXED turn advantage at **+5.364e-07** (vs
**−2.384e-07** half-applied). Chord-transformed, that value would clear the 3.200e-04 charge by
**2.33×** — but on a TINY MODEL, not these checkpoints. The two readings disagree and only the
0-GPU E0 arm on the real checkpoints settles it. ⚠️ The κ² penalty is also not the only charge — the `0.02·jerk²` term from the
canonical control's own longitudinal step has mean **1.695e-04** and max **3.218e-03**, exceeding
the curvature charge on 2/25 windows. ⇒ **R29 is NECESSARY but far from SUFFICIENT and must be
declared jointly with the goal:penalty balance.** Evidence: `raw/turn_decomposition_{A,B}.json`,
`RESULT.md` §9.

### `D-REFAV1-TAC-LOSS-UNINSTRUMENTED` — the mandated term has no instrument

**MEASURED by CONTENT.** `refa_v1_train.py:699-734` writes `loss`, `loss_feat_op`, `loss_feat_tac`,
`loss_feat_str`, `grad_norm`, `adapter_std`, `participation`, `loss_sigreg`, `loss_varfloor`,
`tgt_std_{op,tac,str}`, `ema_decay`, `clip`, `skipped_steps`, `tac_target_s`, `str_target_s`,
`elapsed_s` — and **no `loss_lat_label`, `loss_lon_label`, `loss_route_label` or
`loss_feat_str_ext`**. Every row of all **nine** banked refav1 `train_log.jsonl` artifacts was
parsed: **0 rows** carry any of them. Consequence: the residual
`loss − (1.0·op + 0.5·tac + 0.25·str)` in any banked refav1 log conflates three terms and cannot be
attributed, which is why `D-REFAV1-TAC-LABEL-SHARE` had to be recomputed from checkpoints. ⇒ The one
term the PI made mandatory (2026-08-31, *"It must be trained with this data"*) is the one term with
no telemetry. **Four lines; a preflight condition for every arm in `PREREG_TACTICAL_DECODER.md`.**
Evidence: `RESULT.md` §7.

---

## New OPEN hypotheses (pre-registered, no arm launched)

| id | text (abridged — full text in `Project Steering/PREREG_TACTICAL_DECODER.md` §2) | status |
|---|---|---|
| **`H-REFAV1-TAC-DECODER-1`** | refav1's lateral tactical decision is a majority-class collapse of the DECISION RULE, not a missing gradient and not missing information in the logits; correcting the decision rule alone (class-balanced / focal CE at UNCHANGED `w_tac_label`, label set and trunk) will raise in-band lateral macro-recall above the constant-only 0.200 with a 95 % interval excluding zero, ≥ 2 minority classes at recall ≥ 0.25, and a gain that survives `nav_zero`. | **OPEN** — pre-registered 2026-09-03, no arm spent |
| **`H-REFAV1-COST-SEED-1`** | the planner cannot faithfully chase any manoeuvre except "do nothing" because the control it seeds is not the control its goal was rolled from (62/64 token pairs under BOTH `cost_time_grid` modes); making them identical will raise the count of windows where the canonical turn's goal term beats cv above 8/25 (A) / 4/25 (B), and is a precondition for any decoder improvement being observable at T1. | **OPEN** — pre-registered 2026-09-03, verifiable at 0 GPU |

**Registered prediction, so it is falsifiable:** `PREREG_TACTICAL_DECODER.md` §7 predicts S1 and S2
will hold and **S3 will FAIL** (the repaired cost will rank a correct turn in fewer than 13/25
windows). If that prediction is wrong it goes in `RETRACTION_LOG.md`.

---

## Consistency note the Master Mind should carry into the paper

`D-ACTDIV-ANCHORED-REFAV1` (anchored displacement response to the lateral channel at 2,298× / 11.8×
its permutation null) and `D-REFAV1-CHORD-INSUFFICIENT` (goal-term κ-range 1.63e-10) are
**CONSISTENT, not contradictory**: a response that is large in ‖Δz‖ but small **relative to ‖z‖**
(here ~5 × 10⁻⁴ of the norm) is exactly what produces a near-flat cosine over a flattened
64 × 1024 field. The predictor hears the lateral action; the *cosine* is the wrong instrument for
hearing it. Saying this in one place stops the two rows reading as a conflict.

---

## RETRACTION_LOG candidate

**R-2026-09-03-tacshare** — the R28 / `D-REFAV1-COST-SURFACE` brief characterised the tactical
decoder's supervision as "0.1–0.8 % of the loss", citing `loss_feat_tac`. That key is the tactical
FIELD predictor's MSE, not the decoder's cross-entropy. The decoder's term is **13.59 % / 20.51 %**
(MEASURED). The correction **inverts the lever ordering**: "raise the tactical loss weight" moves
from a leading candidate to a refuted one.
