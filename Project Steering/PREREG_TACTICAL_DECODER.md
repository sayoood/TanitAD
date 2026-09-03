# PRE-REGISTRATION — the refav1 TACTICAL DECODER, and the cost that cannot rank its output

**Date:** 2026-09-03 (Europe/Berlin) · **Author:** Architecture & Inference FlyWheel ·
**Status:** written **BEFORE any arm is launched**, **0 training spent**, **no model file edited**.
**Owner files if approved:** `stack/tanitad/refs/refa_v1.py` (the cost + the seed), the loss block,
`stack/scripts/refa_v1_train.py` (logging + one CLI flag). No other stream touches them.
**Diagnosis this rests on:**
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-tactical-decoder/RESULT.md`
(every number MEASURED, T0, on the two banked step-1,000 checkpoints, 140 windows / 20 episodes).
**Validation contract:** `.claude/skills/TanitAD_ValidateAIDesign` (as CORRECTED 2026-09-03).

> ⚠️ **THE RETIRED FLOOR.** No arm in this document is passed or failed on `participation ≥ 8.56`.
> That floor is retired and no live instrument reproduces it. Participation appears here **only**
> as a **regression tripwire against the arm's own baseline at matched corpus, matched episode
> count and matched `d`**; absent such a match the reading is **UNDECIDABLE — neither pass nor
> fail**. See §6c.

> ⚠️ **THIS DOCUMENT COMMITS BOTH OUTCOMES IN ADVANCE.** §7 names, with intervals, what counts as
> success and what refutes each hypothesis. §8 names the arm that could refute the WHOLE line for
> ~0 GPU before any training arm is bought, and §8.4 states what I predict it will show — so that
> prediction is falsifiable too.

---

## 0. The claim under test, and how the brief that produced it must be corrected

> **R28 / `D-REFAV1-COST-SURFACE`:** refav1 always drives straight because the tactical decoder
> emits `LANE_KEEP` everywhere, making the goal degenerate; where a turn IS requested the cost is
> 99.5 % its own penalty and the `1−cos` goal term is float32-saturated along κ.

**The first half is confirmed and made exact (§1). Three parts of the surrounding brief are
CORRECTED by measurement, and each correction changes which lever is load-bearing:**

| # | the brief said | MEASURED | consequence |
|---|---|---|---|
| C1 | the tactical term is 0.1–0.8 % of the loss (from `loss_feat_tac`) | that is the **field predictor's** MSE. The **decoder's** term is **13.59 % / 20.51 %** of the label+feature sum | **"unsupervised head" is REFUTED. Raising `w_tac_label` is not the lever.** |
| C2 | the fix candidates are the decoder and the cost | there is a **third, prior** defect — escalated qualitatively by the steer-conversion agent in `4139203`, **QUANTIFIED here**: the planner's SEED is not the GOAL's control on **62 / 64 token pairs** (48/64 in curvature alone), `TURN` over-rotating by **82.5°**, and **`cost_time_grid="tactical"` does NOT close it** | **a blocking lever, L0, ahead of both.** |
| C3 | a chord distance lets a decoded turn "at least be RANKED" | chord leverage **+5,792.6×** measured, and it flips **1 of 25** windows (0 of 25 under convention B) | **R29 is NECESSARY but far from SUFFICIENT; and it is not weight-neutral.** |

---

## 1. The defect chain — every link MEASURED, with its primary source

| # | fact | class · source |
|---|---|---|
| 1 | The decoder is **two independent 8-way softmaxes** on ONE 256-d `intent`: `LayerNorm(256) → Linear(256,8)` each. | MEASURED — `refa_v1.py:1030-1033`, `:1252-1253` |
| 2 | One (lat, lon) pair per MPC tick, **held constant over the whole 6.0 s goal**; nominal recompute cadence 1.0 s. | MEASURED — `refa_v1.py:1711-1761`; `config.py:124` |
| 3 | It **is** supervised: `w_tac_label = 0.1` in both banked configs; the live run passed real `--labels`; the trainer refuses `--cache` without them. | MEASURED — `ckpt_ep2/config.json`; `refa_v1_train.py:506-513`, `refa_v1.py:1578-1595` |
| 4 | Its share of the loss on the eval windows is **13.59 %** (incumbent) / **20.51 %** (ep2). | MEASURED — `raw/intent_probe_*.json` |
| 5 | **No training log in the programme has ever carried it.** `refa_v1_train.py:699-734` has no `loss_lat_label` / `loss_lon_label` key; **0 rows** across all **9** banked refav1 `train_log.jsonl`, verified by parsing every row. | MEASURED — grep, `RESULT.md` §7 |
| 6 | Label distribution (n = 4,572 train clips): `LANE_KEEP` **64.70 %**, `NUDGE_R` 12.95, `NUDGE_L` 10.67, `TURN_L` 6.01, `TURN_R` 5.66. **`LANE_CHANGE_L`, `LANE_CHANGE_R`, `ABORT_LC` = 0**; `YIELD_MERGE` = 0. | MEASURED — `raw/decode_audit.json` `label_census_train` |
| 7 | The label is attached only inside `t0_s ± 2.0 s`, and `t0_s ≡ 8.0` on 4,572/4,572 records ⇒ **26.93 %** of training-shape windows carry a lateral label; **8.13 %** of batch-8 steps carry none. | MEASURED (this slice) / INFERRED (train corpus) — `raw/window_band_census.json` |
| 8 | Incumbent lateral head: `LANE_KEEP` on **140/140**; in-band accuracy **0.650 = the base rate EXACTLY**; macro-recall **0.200 = 1/K EXACTLY**; clustered permutation `p(macro ≥ obs) = 1.0000`. | MEASURED — `raw/decode_audit.json` |
| 9 | Yet the head **RANKS**: `AUC(P(TURN) vs label TURN, in-band)` = **0.873 [0.686, 1.000]** (incumbent) / **0.922 [0.737, 1.000]** (ep2), no-information 0.500. `CE_lat = 0.8665 < H(p) = 1.0944 nats`. | MEASURED — `raw/intent_probe_*.json` |
| 10 | That ranking is **nav-borne** on the incumbent: `nav_zero` AUC **0.520**; a **nav-only** predictor scores **0.684 > the model's 0.650 and the constant control's 0.650**. `nav` is an ORACLE (ego-future) training input. | MEASURED — same |
| 11 | **Plumbing is intact**: `goal_lat == argmax(lat_head)` on **140/140**, both arms, both nav conditions; ep2's 25 `TURN_R` decisions produced 25 genuinely curved goals. | MEASURED — same |
| 12 | **THE SEED IS NOT THE GOAL** (residual escalated in `4139203`; measured here). Goal consumes operative indices `[0,3,…,27]` (`_imagine_tactical_goal`); the seed is truncated to `plan_steps = 10` (`refa_v1.py:1932`) ⇒ `[0..9]` under the DEFAULT `cost_time_grid="dense"` (what every banked number used) or `[0,3,6,9,9,…]` under the `"tactical"` repair. **62/64 token pairs differ under BOTH grids** (48/64 in curvature alone); the only agreeing pairs are `(LANE_KEEP, CRUISE)` and `(ABORT_LC, CRUISE)` — the all-zero control. ⚠️ The `4139203` regrid does NOT close it: the cause is the TRUNCATION, not the regrid. | MEASURED — `raw/seed_goal_mismatch.json` |
| 13 | On ep2's 25 curved-goal windows the canonical turn beats constant velocity on the **goal term** in **8/25** (conv A) / **4/25** (conv B) and **wins 0/25**; under the chord, **1/25** / **0/25**. | MEASURED — `raw/turn_decomposition_{A,B}.json` |
| 14 | The goal term's κ-range is **1.63e-10 (f64)** against the `0.05·κ²` penalty's **2.00e-03** over the same box — a factor **1.2 × 10⁷**. The jerk charge is not negligible: mean **1.695e-04**, max **3.218e-03**, exceeding the κ charge on 2/25. | MEASURED — banked R10 summary + `raw/turn_decomposition_A.json` |

---

## 2. The hypotheses

### `H-REFAV1-TAC-DECODER-1` (proposed, NEW)

> **refav1's lateral tactical decision is a majority-class collapse of the DECISION RULE, not a
> missing gradient and not missing information in the logits.** The head already ranks turns above
> chance against the v7.2 label (in-band AUC 0.873 / 0.922, CI excluding 0.500) while emitting
> `LANE_KEEP` on 140/140 windows and reading its accuracy and macro-recall EXACTLY off the
> constant-only control. **Therefore: correcting the decision rule alone — a class-balanced or
> focal cross-entropy on the lateral head, at UNCHANGED `w_tac_label`, UNCHANGED label set and
> UNCHANGED trunk — will raise in-band lateral macro-recall above the constant-only control by a
> margin whose episode-cluster 95 % interval excludes zero, and will not be reproduced by the
> shuffled-label control.**
>
> **Corollary registered with it, and separately falsifiable:** doing so will **NOT** by itself
> produce turns in the plan, because `H-REFAV1-COST-SEED-1` binds first.

### `H-REFAV1-COST-SEED-1` (proposed, NEW — beyond the brief, and why)

> **The planner cannot faithfully chase any manoeuvre except "do nothing", because the control it
> seeds is not the control its goal was rolled from** — 62/64 token pairs under BOTH `cost_time_grid`
> modes, 48/64 in curvature alone; `TURN` over-rotates its own goal by 82.5°, `LANE_CHANGE_R` loses
> its return arc's sign, and `NUDGE_R`'s S-curve is stretched 2.5× in time so 8 of 10 tactical
> steps carry the wrong curvature. **Therefore: making the
> seed and the goal consume the same control sequence will, on the ep2 banked checkpoint's 25
> already-curved-goal windows and with the decoder frozen, raise the number of windows on which
> the canonical turn's goal term beats constant velocity from the measured 8/25 (conv A) / 4/25
> (conv B) — and it is a precondition for any decoder improvement being observable at T1 at all.**

*I am registering a second hypothesis the brief did not ask for. The reason is §1 fact 12: it was
found during this diagnosis, it is prior to both fixes the brief names, and a decoder arm run
without it would read as a null through no fault of the decoder — the exact failure of "the
target-scale instrument was blind to the collapse it was built to detect" (`e1352da`).*

---

## 3. The pre-registration block

```yaml
# ---------------- ARM SET 1: the COST/SEED line (0 GPU) ----------------
hypothesis:   H-REFAV1-COST-SEED-1
one_variable: seed_goal_identity        # the ONLY difference between C0 and C1
held_constant: [checkpoint, windows, episodes, nav, labels, plan_cfg.seed,
                plan_cfg.n_samples, plan_cfg.n_iters, plan_cfg.n_elites,
                w_goal, w_jerk, w_kappa, w_vend, model_action_units,
                cost_time_grid, dtype_of_the_cost]
success: "n_windows where the canonical turn's goal term beats cv rises from
          8/25 (conv A) with a paired episode-cluster 95 % interval on the
          delta that excludes 0; AND the per-window heading excess
          (seed minus goal) reads 0.000 rad on 64/64 token pairs"
failure:  "the interval includes 0, OR the count does not rise, on BOTH
           conventions. Then the seed/goal mismatch is not what is costing the
           turn, and H-REFAV1-COST-SEED-1 is REFUTED."
controls: [identity_control, constant_only, deliberate_regression,
           convention_AB, fp32_vs_fp64]
splits:   {fit: "none — no parameter is fit", val: "none",
           test: "the 140 banked windows / 20 episodes, scored once"}

# ---------------- ARM SET 2: the DECODER line -------------------------
hypothesis:   H-REFAV1-TAC-DECODER-1
one_variable: lateral_loss_reweighting   # {none | class_balanced | focal}
held_constant: [w_tac_label, w_str_label, w_feat_op, w_feat_tac, w_feat_str,
                label_set, label_band, vocabulary_version, trunk_weights,
                intent_bank, seed, corpus, steps, batch, window,
                nav_condition]
success: "in-band lateral MACRO-RECALL on the TEST split exceeds the
          constant-only control (0.200 = 1/K) by a margin whose episode-cluster
          95 % interval excludes 0, AND at least 2 of {NUDGE_L, NUDGE_R,
          TURN_L, TURN_R} reach recall >= 0.25, AND the shuffled-label control
          on the same arm stays inside [0.150, 0.250]"
failure:  "macro-recall interval includes the constant-only 0.200, OR only the
           single class the nav command already identifies improves (i.e. the
           nav_zero arm shows no gain). Then the collapse is NOT a decision-rule
           problem and H-REFAV1-TAC-DECODER-1 is REFUTED in favour of the
           representation branch."
controls: [constant_only, shuffled_label, nav_zero, nav_shuffled,
           raw_input_floor, deliberate_regression]
splits:   {fit:  "training clips MINUS a carved val; eval-clip exclusion ON",
           val:  "400 episode-disjoint clips carved from FIT ONLY — the ONLY
                  split on which the class-weight scheme / focal gamma is
                  selected",
           test: "the 147-clip v7.2 eval set and the 140-window banked grid;
                  SCORED ONCE, never tuned on"}
```

⛔ **Preflight refuses the run if** the arms differ in more than the one variable (diff the launch
commands, not the intent — the row-bank λ lesson); a control is missing; the hypothesis ID is not
in `GOALS_AND_CLAIMS.md`; any hyper-parameter would be selected on the scored split; or
`loss_lat_label` / `loss_lon_label` are still absent from the trainer's log row (§5 L1).

---

## 4. Arms

| id | arm | one variable | compute |
|---|---|---|---|
| **C0** | banked cost, as shipped | — (the baseline) | 0 GPU (re-read) |
| **C1** | seed = the goal's own control | `seed_goal_identity` | 0 GPU (re-score) |
| **C2** | C1 + `1−cos → chord` | metric | 0 GPU |
| **C3** | C1 + chord + `w_kappa` scaled so goal:penalty leverage matches C0's *intent* | metric + weight, **declared jointly** | 0 GPU |
| **C-REG** | **deliberate regression**: C1 but with `plan_steps` forced back to 10 and `tac_idx` saturating | re-creates the defect | 0 GPU |
| **D0** | head refit, plain CE — **must reproduce the collapse** | — (the baseline) | CPU seconds |
| **D1** | head refit, class-balanced CE (weights = inverse in-band frequency, computed on FIT only) | `lateral_loss_reweighting` | CPU seconds |
| **D2** | head refit, focal CE (γ selected on VAL only) | `lateral_loss_reweighting` | CPU seconds |
| **D-REG** | **deliberate regression**: D1 with the class weights set to the label frequency (i.e. anti-balanced) | re-creates the collapse | CPU seconds |
| **D3** | *(only if D1 or D2 passes)* full training arm, class-balanced CE, everything else identical to the live ep2 launch | `lateral_loss_reweighting` | see §9 |

**⭐ THE TINY RIG FOR THIS EXPERIMENT IS AN `intent` BANK, and it is exact for the question asked.**
Both heads are linear read-outs of a 256-d vector (§1 fact 1). Freeze the trunk, dump `intent` once
per labelled window, and a decoder arm becomes a 256→8 multinomial fit that trains in CPU seconds.
This **isolates the DECISION RULE from the REPRESENTATION**, which is precisely the discrimination
`H-REFAV1-TAC-DECODER-1` needs. ⚠️ Its limit, stated here rather than discovered later: the CE
gradient in a real run also flows into `intent_proj`, the tactical transformer, the strategic
policy and the adapter, so **D1/D2 measure what the decision rule alone can buy and nothing more**.
A full arm (D3) is bought only if the head-only arm passes.
⛔ `refa_v1_train.py --smoke` is **NOT** a usable tiny rig here: it runs on synthetic `SmokeData`,
which carries no v7.2 labels.

---

## 5. The levers, ranked by cost and evidence — and the ordering

| # | lever | cost | evidence | order |
|---|---|---|---|---|
| **L0** | seed = goal's control (the truncation to `plan_steps`, NOT the regrid) | ~3 lines, 0 GPU to verify | 62/64 mismatch under BOTH `cost_time_grid` modes; 82.5° over-rotation; only "do nothing" is faithful; `4139203`'s regrid does not close it | **1st — blocking** |
| **L1** | log `loss_{lat,lon}_label`, `loss_feat_str_ext`, `loss_route_label` | 4 lines, 0 GPU | 0 rows in 9 banked logs | **1st — free, and a preflight condition** |
| **L2** | class-balanced / focal lateral CE | 1 flag; CPU-seconds on the intent bank | ranks (AUC 0.873) but does not decide; acc = base rate EXACTLY | **3rd** |
| **L3** | `1−cos → chord` (R29) **coupled to** the goal:penalty balance | ~2 lines + 1 weight, 0 GPU | +5,792.6× leverage, flips 1/25; κ-range 1.63e-10 vs 2.00e-03 | **2nd, jointly with L4** |
| **L4** | `0.05·κ²` and `0.02·jerk²` | one number each | κ charge 3.2e-04 on every turn window; jerk mean 1.70e-04, **max 3.218e-03** | **2nd, never alone** |
| **L5** | the label distribution (±2 s band = 26.93 % of windows; one token per clip; 5 of 16 classes dead; `t0_s ≡ 8.0`) | Data-FlyWheel re-emit, days | 64.7 % is not a pathological imbalance — the *time sparsity* and the *dead classes* are | **open with Data FlyWheel NOW; longest lead** |
| **L6** | raise `w_tac_label` | one number | **REFUTED** — already 13.6–20.5 % of the loss | **do not spend an arm** |
| **L7** | remove/de-weight oracle `nav` into the tactical head | one flag | nav-only beats the model (0.684 > 0.650); every head signal is nav-borne | **measure, and pre-register that T0 gets WORSE first** |

### ⭐ Which must land first, and why — the question the brief asks explicitly

**L0 + L1 before any training arm; then L3+L4; then L2.**

* **L1 first** because a loss term with no instrument cannot be tuned, and the preflight refuses an
  arm whose objective is invisible.
* **L0 next** because it is the only lever that is simultaneously (a) blocking, (b) verifiable at
  0 GPU on artifacts already on disk, and (c) *upstream of the endpoint*. Until the seed IS the
  goal's control, a decoder improvement is unobservable at T1: ep2 already emits 25 turns and wins
  **0** of them, so a better decoder would still read as a null.
* **L3+L4 jointly**, never L3 alone: `chord = √(2(1−cos))` is monotone in `1−cos`, so it cannot
  change the goal term's own ranking; what it changes is its **leverage inside the SUM**, by a
  measured **5,792.6×**. Swapping the metric while holding `0.05` fixed is therefore **not a
  one-variable arm** — it is a metric change *and* an implicit 5,793× reweighting.
* **L2 last**, because L0/L3/L4 change what a decoder improvement is worth, and because it is the
  only lever on this list that ultimately needs GPU-days.

**A decoder fix and a cost fix are INDEPENDENT and BOTH are necessary.** A turn that is requested
but unrankable is exactly as useless as one never requested — **measured: 0/25.** Neither arm may
be reported as "the fix"; the pair is the fix, and the register rows must say so.

---

## 6. Gates, IN ORDER — an arm earns the next only by clearing the previous

### 6a. G-DECODE′ (the decoder gate; this replaces G-RANK as the *first* gate for this experiment)

An arm passes only if **all** hold on the TEST split:

1. in-band lateral macro-recall > the **constant-only control**, episode-cluster 95 % interval on
   the paired delta excluding 0;
2. the **constant-only control reads its known values EXACTLY** — accuracy = the base rate,
   macro-recall = 1/K — asserted in code (`decode_audit.py::constant_only` raises otherwise);
3. the **shuffled-label control** (clustered, episode blocks permuted whole) does **not** clear (1);
4. the **`nav_zero`** arm also improves. ⛔ Without this an arm can pass by decoding the oracle nav
   command better — the C6 / nav-echo family, and the reason `RESULT.md` §6 reports every quantity
   under three nav conditions;
5. **n and d are printed** beside every number, and `n ≪ d` is named as underpowered BY
   CONSTRUCTION rather than reported as a negative.

### 6b. G-COST′ (the cost gate)

1. the seed's tactical action SEQUENCE equals the goal's **element for element on 64/64 token pairs**, in BOTH `cost_time_grid` modes. ⚠️ **Heading excess is NOT a sufficient criterion**: `NUDGE_R` already reads 0.000° excess under the default grid while 8 of its 10 steps carry the wrong curvature. It is an identity, so it must be exact, not "small";
2. the `C-REG` deliberate-regression arm **FAILS** gate 1. If it does not, a pass on C1 means
   nothing;
3. the count of windows where the turn's goal term beats cv rises, paired interval excluding 0, on
   **both** conventions A and B;
4. reported in **fp64** as well as fp32, with the ULP of every difference printed
   (`turn_decomposition.py` already prints `resolvable_in_fp32` per window).

### 6c. G-RANK — how participation is handled here

This experiment changes a **decision rule and a cost**, not the representation, so participation is
**not a gate**. It is recorded **only** as a regression tripwire against the arm's own baseline at
**matched corpus, matched episode count and matched `d`** (D3 vs the live ep2 run). ⛔ **The bare
`8.56` floor is retired and is not used to pass or fail anything here.** If a matched reference is
unavailable, the reading is **UNDECIDABLE — neither pass nor fail**. Val-side is quoted for any
representation statement; the O4-weighted train-stream pooled value is not a val-side measure.

### 6d. G-DRIVE

T1, four metric families (longitudinal / lateral / tactical / strategic beside ADE), paired
episode-cluster bootstrap over the val episodes. **No arm reaches this gate in this document** —
G-DRIVE is where a merged L0+L2 candidate is judged, and it needs the Benchmarks FlyWheel.

---

## 7. Success and failure, committed now, with intervals

| # | claim | SUCCESS | FAILURE |
|---|---|---|---|
| S1 | seed/goal identity (C1) | the two action SEQUENCES are equal element-for-element on **64/64** token pairs in **both** `cost_time_grid` modes; `C-REG` reproduces the 62/64 mismatch | any pair unequal, or `C-REG` also reads 0/64 (then the instrument, not the code, is what changed) |
| S2 | C1 raises the turn's goal-term wins | count rises above **8/25** (A) and **4/25** (B), paired episode-cluster 95 % interval on the delta excluding 0 | interval includes 0 on both conventions ⇒ **`H-REFAV1-COST-SEED-1` REFUTED** |
| S3 | C3 makes a correct turn rankable | ≥ **13/25** windows where the turn's TOTAL cost beats cv (i.e. a majority), fp64, both conventions | < 13/25 ⇒ the cost line is insufficient even repaired; escalate to the goal-SPACE branch (§10) |
| S4 | decoder decision rule (D1 or D2) | in-band lateral macro-recall − constant-only **> 0**, 95 % interval excluding 0; **≥ 2** minority classes at recall ≥ 0.25; shuffled-label control inside **[0.150, 0.250]**; `nav_zero` also improves | interval includes 0.200, or only the nav-identified class improves ⇒ **`H-REFAV1-TAC-DECODER-1` REFUTED** in favour of the representation branch |
| S5 | `D-REG` deliberate regression | `D-REG` **FAILS** S4 | `D-REG` passes ⇒ the gate cannot see the defect it exists to catch; **no D1/D2 pass may be reported** |
| S6 | the pair is the fix | with L0 + L2 both landed, plan-level turn rate on the geometric turn stratum rises above the ep2 baseline of **0.333 [0.000, 0.667]**, interval excluding it | it does not ⇒ a third defect is load-bearing; §10 |

**My registered prediction, so it is falsifiable:** S1 and S2 will hold; **S3 will FAIL** — I expect
the repaired cost to rank a correct turn in fewer than 13/25 windows, because the goal term's
f64 κ-range (1.63e-10) is 1.2 × 10⁷ below the penalty's, and 2,008× of that gap is all the chord's
√ can recover. If I am wrong I will say so in the RETRACTION_LOG.

⚠️ **AND THE PREDICTION IS ALREADY UNDER PRESSURE FROM A SOURCE I FOUND AFTER WRITING IT — recorded
here rather than quietly folded in.** The steer-conversion package landed the same day
(`4139203`) and its tiny-model evidence
(`…/2026-09-03-steer-conversion-completion/raw/evidence.json`, `P1_tiny_model_costs`) measures the
canonical turn's goal advantage at **+5.364e-07** once the units crossing is fixed on BOTH sides
(it is **−2.384e-07** half-applied, which is what the banked panel ran). Applying the chord to
those two values gives an advantage of **7.465e-04** against a `0.05·κ²` charge of **3.200e-04** —
**the turn would WIN by 2.33×**. That is a TINY MODEL and not window-for-window comparable to the
25 banked windows, so it does not overturn the prediction; it makes S3 genuinely uncertain rather
than a formality. **The prediction above STANDS AS WRITTEN** — revising it after seeing new evidence
would defeat the purpose of registering it — and E0 (§8) is now the experiment that decides, on the
real checkpoints, between a banked reading that flips 1/25 and a tiny-model reading that flips.

---

## 8. ⭐ §8 THE CHEAPEST DISCRIMINATING EXPERIMENT — run this before any arm is bought

### 8.1 It has already been run, in part, and it is in the package

`turn_decomposition.py` and `seed_goal_mismatch.py` (both **0 GPU**, both reading only banked JSON
and the shipped source) delivered facts 12–14 today. They cost minutes and they are what reordered
the levers.

### 8.2 The arm that could refute the WHOLE line — **E0**, ~0 GPU

**Population:** ep2's **25 windows whose imagined goal already carries curvature** — a natural
experiment with the decoder held fixed.
**Procedure:** re-score those 25 windows' named seeds, in **float64**, under the 2 × 2 × 2 cross of
`{seed as shipped, seed = goal's control} × {1−cos, chord} × {w_kappa as shipped, w_kappa scaled}`,
plus the `C-REG` regression arm. The 4060 needs one tactical rollout per candidate per window
(~10³ rollouts total, minutes), and `cost_surface_probe.py` already carries the harness.
**What it answers, before a training arm is spent:**

* **Q1 — does the tactical 6 s goal field prefer a turn AT ALL, once the seed is the goal's own
  control?** If the goal-term advantage stays ≤ 0 on a majority of the 25 **even at seed/goal
  identity and in f64**, then the goal SPACE — not the decoder, not the metric, not the weights —
  is what cannot represent a manoeuvre, and **the whole decoder-then-cost line is REFUTED.**
* **Q2 — if it does prefer the turn, by how much,** and what `w_kappa` / `w_jerk` would let it win?
  That number sizes L4 exactly instead of by sweep.

### 8.3 Why this is the right refutation arm

It is the only experiment on the table that (a) costs no training, (b) uses a population that
already exists (25 curved goals, banked), (c) can return a result that kills the programme's current
plan rather than merely tuning it, and (d) is upstream of every lever in §5. **⛔ No training arm is
authorised until E0 has run.**

### 8.4 The known confound E0 must carry

The goal is rolled by `self.tactical` and the candidate is rolled by `self.tactical` — the goal is
defined by the model's own rollout. A goal space that is curvature-insensitive therefore produces a
curvature-insensitive goal *and* a curvature-insensitive candidate, and their cosine can be flat for
a reason that is not a bug. **E0 must therefore also report the RAW-INPUT FLOOR for the cost**: the
same 25 windows scored with the goal replaced by the *ground-truth* 6 s field (`cl_oraclegoal`, an
arm the banked dump already carries). If the oracle-goal cost also fails to prefer the turn, the
defect is the cost; if it succeeds, the defect is the imagined goal.

---

## 9. If a training arm IS needed — sizing, and the fleet constraint

**D3 (full class-balanced arm), sized from MEASURED numbers:**

| quantity | value | source |
|---|---|---|
| Thor step time, bs 8, fp32, TF32 off | **20.66 s/step** MEASURED | `2026-09-02-refav1-step-profile/RESULT.md` §6a |
| one epoch | **21,109 steps** | `ckpt_ep2/config.json` `args.steps` |
| epoch wall-clock, fp32 | **5.05 days** | same |
| epoch wall-clock, bf16 + TF32 (the live config) | **≈ 1.9 days** ESTIMATED (2.6×) | same §6a — ESTIMATED, not measured, and it says so |
| a **decision-only** read | **step 1,000** suffices — both banked checkpoints are step-1,000 and the collapse is already fully expressed there | this package |

⇒ **A D3 arm to a decision-readable checkpoint is ~2.2 h of Thor at bf16** (1,000 steps × ~8 s),
not days. Budget one arm + one deliberate-regression arm = **≈ 4.5 h of Thor**.

⛔ **THE DEV BOX CANNOT RUN THIS ARM.** MEASURED: a full refav1 fwd+bwd step **fits no batch on the
4060's 8 GB** (`step-profile` §2 — `save_on_cpu` with pinned memory OOMs at ~7 GB; the pageable
offload path runs at 9.89 s/step but holds 18.7–22.7 GB of host working set). The 4060 is a
**forward-only** box for refav1. It can build the `intent` bank (D0–D2) and run E0; it cannot train.

⛔ **THOR IS BUSY.** refav1's speed epoch runs until ≈ **2026-09-04 02:00Z**; refcv3 ends ≈ 23:00Z
today. **No arm in this document may touch Thor before 2026-09-04 02:00Z**, and D3 is a PI /
Master-Mind scheduling decision, not this FlyWheel's.

**The `intent` bank (D0–D2) sizing:** ~300 training episodes × ~10 in-band windows = **~3,000
labelled rows**. Forward-only on the 4060 at ≤ 0.6 s/window ⇒ **≈ 30 min**, plus an fp8 cache pull
of **≈ 19.9 GB** (66.2 MB/episode MEASURED on the local slice: 66,193,268 B). Detached via
`powershell.exe Start-Process` because it exceeds 55 min end to end.

---

## 10. What would refute the whole line, and what happens next if it does

If **E0 Q1** returns "the goal term does not prefer the turn even at seed/goal identity, in f64,
with the oracle goal" then:

* `H-REFAV1-TAC-DECODER-1` and `H-REFAV1-COST-SEED-1` both become **irrelevant rather than wrong** —
  the tactical **goal SPACE** is the defect, and the next pre-registration is about what the 6 s
  tactical query field encodes, not about how it is compared. Candidates already visible from this
  diagnosis: the cosine over a flattened 64 × 1024 field is dominated by everything two rollouts
  share (a curvature change moves the field by a **relative displacement of ~5 × 10⁻⁴**), so a
  distance on the ACTION-INDUCED DELTA, or a per-token / whitened distance, is the lever — not the
  precision of the comparison.
* That branch must be reconciled with `D-ACTDIV-ANCHORED-REFAV1` (anchored displacement response to
  the lateral channel at 2,298× / 11.8× its permutation null): a response that is large in ‖Δz‖ but
  tiny relative to ‖z‖ is exactly what produces a flat cosine. **The two findings are consistent;
  saying so is part of the register row.**

---

## 11. Threats to validity this pre-registration refuses in advance

1. **Nav echo.** Every decoder number is reported under `nav_true`, `nav_shuffled` and `nav_zero`.
   An arm that improves only under `nav_true` does not pass (§6a gate 4). `nav` is an ORACLE
   (ego-future) input and a nav-only predictor already beats the model — this is a live risk, not a
   theoretical one.
2. **The clustered null.** The v7.2 label is emitted once per clip, so an episode's in-band rows
   carry the SAME label. A free row-wise permutation is **anti-conservative** here (measured: it
   moves `p` from 0.0098 to 0.0001). **The clustered null is the one quoted.**
3. **Underpowered panels are named, not reported as negatives.** In-band n = 40 against
   d_intent = 256 is `n ≪ d` BY CONSTRUCTION; the linear-probe panel in `RESULT.md` §6 is stamped
   *reported, not decisive*, and its function class (`PCA-8 → multinomial logistic`) is stated.
4. **The geometric stratum is not the label.** `gt_turn_deg ≥ 5°` selects 27/140 windows, of which
   only 8 are in-band and only 3 carry a `TURN_*` token. No claim mixes the two denominators.
5. **fp32 differences below their own ULP are not reported as effects.** Every cost difference
   carries `resolvable_in_fp32` and the ULP it was taken against.
6. **One variable, diffed at the command line.** The chord/weight coupling (§5 L3) is the specific
   trap here: an arm that swaps the metric and holds `0.05` is silently a 5,793× reweighting.
   The launch commands are diffed, not the intent.
7. **Verify by CONTENT.** An arm "succeeded" only if its artifact exists and is non-trivial —
   assert on bytes and on the control reading its known value, never on an exit code.

---

## 12. Close-the-loop obligations (owner: Master Mind — this FlyWheel does not own the register)

* `GOALS_AND_CLAIMS.md`: add `H-REFAV1-TAC-DECODER-1` and `H-REFAV1-COST-SEED-1` as **OPEN**, plus
  the five diagnosis rows proposed in
  `.../Research/2026-09-03-tactical-decoder/PROPOSED_REGISTER_ROWS.md`.
* `RETRACTION_LOG.md`: the R28 brief's "0.1–0.8 % of the loss" reading of the tactical term
  (correction C1 above) — it names the wrong term, and the correction inverts the lever ordering.
* `stack/scripts/refa_v1_train.py`: L1 (four log keys) is a preflight condition for every arm here.
