# PREREG_V8F_OACC: train O to match the human's acceleration, not only its positions (written 2026-10-25, before any OACC data)

**Status: pre-registered.** One training arm on box 2, behind a launch-gate PASS, under `PI-V8F-2026-10-25b` and `PI-V8F-2026-10-25c` (*"Go ahead with your proposed ideas … our priority is to proof very quickly dominance"*). It is `PREREG_V8F_HSPEC`'s committed next step (outcome SMOOTHER, NOT DOMINANT).

## Why (MEASURED; `raw/2026-10-25-hspec/READ.json`, eval-141, paired episode-cluster bootstrap)

HSPEC stage 2 (H plus per-coefficient shrinkage with a second-difference objective) against P10-yaw-4512:
* separated-better on ADE2 (−0.005 / −0.005 m) and ADE6 (−0.511 / −0.490 m);
* not separated-worse on any other family metric, consistency and distance keeping included;
* separated-better on accel MAE at **neither** seed: s0 +0.026 [+0.011, +0.044] (worse), s1 +0.006 [−0.011, +0.028];
* separated-better on curvature MAE at neither seed: +0.0012 and +0.0031 m⁻¹, not separated.

Post hoc, no bar: even weights fitted on the eval itself do not reach P10's accel at s0 (+0.017, separated-worse). ⇒ **O's residual carries no acceleration skill that post-processing can extract.** O is trained on positions only (`w_traj`); nothing in its loss asks for the right acceleration. ⇒ The lever is the loss.

## The arm X (ONE lever: an acceleration-matching term on O's plan)

$$\mathcal{L}_X = \mathcal{L}_B + w_{acc}\cdot \frac{1}{|M|}\sum_{t\in M}\big\|\Delta^2\tau_{30}(t) - \Delta^2 y_{30}(t)\big\|^2$$

* $\Delta^2$ is the second difference on the 0.2 s plan grid; the plan includes the origin knot, as in `upsample_prior`.
* $M$ masks the steps where the GT is valid. Units are m per (0.2 s)², the same quantity as HSPEC stage 2.
* The term acts on O's own plan, on every training window; teacher forcing is unchanged.

**$w_{acc}$ is fixed by a rule, computed once before launch:**
$$w_{acc} = 0.5\cdot\frac{\mathcal{L}_{traj}(B_{s0})}{\mathcal{L}_{acc}(B_{s0})}$$
Both terms are evaluated with B's s0 checkpoint on the training windows of HBLEND's fit set. The value is written into the argv and quoted in the result.

**Everything else is B's argv verbatim:** seeds 0, then 1, with the same steps, data and schedule.

**New flag:** `--w-acc` (default 0.0 = B), with a G-DVB entry, a G-LIVE term (the accel term must appear finite and non-zero, and O's head must receive a non-zero gradient from it), and a deliberate regression: with the term's line deleted, the gate must go RED.

## The reading (T1-class open loop, eval-141, paired episode-cluster bootstrap, B = 2,000; F = 0.1207 m; the eval's own `evaluate()`, `families()`, distance keeping)

| read | definition | bar |
|---|---|---|
| **PRIMARY-ACC** | accel MAE X − B | **< 0 and separated, at the seed run** |
| **PRIMARY-KEEP** | ADE6 X − B | **upper bound ≤ +F/2 = +0.060 m** |
| **DOMINANCE** (the programme read) | HX − P10, where HX is X with HBLEND's per-step weights re-fitted on X's own training windows (the deployed form) | HSPEC's PRIMARY-DOM row verbatim: separated-better on ADE2, ADE6, accel MAE and curvature MAE; not separated-worse on any other family metric |
| beside | the four families for X and HX; CONSISTENCY(a); HS on X (descriptive) | descriptive |

| outcome | condition | next (committed) |
|---|---|---|
| **DOMINATES ITS PRIOR** | DOMINANCE at both seeds (s1 runs on s0's PRIMARY-ACC and PRIMARY-KEEP pass) | HX is the deployed v8F output and the register states dominance over the **kinematic floor** on every family; ⛔ never as state of the art. Next: the same row against FLAT |
| **SMOOTHER, NOT DOMINANT** | PRIMARY-ACC and PRIMARY-KEEP pass, DOMINANCE fails | X becomes the base for later arms. The acceleration gap is quantified per horizon; next, the longitudinal decision (`PREREG_V8F_KHYP`) |
| **COSTS POSITION** | PRIMARY-ACC passes, PRIMARY-KEEP fails | the trade-off is reported; the weight rule is not re-tuned post hoc |
| **NO EFFECT** | PRIMARY-ACC fails | the acceleration error is not reachable through O's loss at this capacity; next, an O-capacity read |

⛔ No weight rule, mask, bar or seed order changes after any OACC number is seen.

## Compute and order

Box 2, one job at a time, about 4.4 h per seed (ESTIMATED from B). The build waits for `PREREG_V8F_FLAT`'s instruments to land, because both edit `v8f_mini.py` and the trainer. Queue position: after FLAT s0 / s1, before DSEFF, unless the PI allocates other compute.

## Amendment 1 (2026-10-25, MM): written BEFORE any OACC data (the instruments are built, not landed; nothing has trained)

Building the instruments needed eight readings that the text leaves open. They are fixed here. **No bar, rule, mask or seed order changes.**

**The weight rule**
* **R1.** $\mathcal{L}_{traj}$ is the trainer's own trajectory term, `w_traj·(ADE/3 + 0.5·FDE/6)`, as it enters the `total` loss of `v8f_mini_train.py`. It is evaluated with B's s0 checkpoint on the training windows of HBLEND's fit set.
* **R2.** $\mathcal{L}_{acc}$ is this PREREG's term at unit weight: the second difference on the 0.2 s grid, origin knot included, averaged over GT-valid steps.
* **R3.** $w_{acc}$ is rounded to 6 significant figures and written into X's argv. The box-2 stage recomputes it on the checkpoint, and the result quotes that value.
  * Offline dry check on HBLEND's banked B_s0 arrays (9,582 windows): $\mathcal{L}_{traj}$ 1.24598, $\mathcal{L}_{acc}$ 0.0145114, so $w_{acc}$ = **42.9311** (`raw/2026-10-25-oacc/wacc_offline_dry_check.json`).
  * An ADE-only $\mathcal{L}_{traj}$ would give 74.6976. It is shown beside and not used.

**The reading**
* **I1.** PRIMARY-KEEP's bar is F/2, with F the exact |ADE6(B_s0) − ADE6(B_s1)| = 0.12074 m. The bar is therefore **+0.06037 m**; the table's "+0.060" is its rounding.
* **I2. Outcome partition**, applied in this order:
  1. PRIMARY-ACC fails at any seed that has run ⇒ **NO EFFECT**.
  2. Else PRIMARY-KEEP fails at any seed that has run ⇒ **COSTS POSITION**.
  3. Else only seed 0 has run ⇒ **PENDING SEED 1**. This is not an outcome: seed 1 runs, as the table's DOMINATES row requires.
  4. Else DOMINANCE at both seeds ⇒ **DOMINATES ITS PRIOR**.
  5. Else **SMOOTHER, NOT DOMINANT**.
* **I3.** A pass at seed 0 alone is PENDING SEED 1, never a dominance claim (this is I2's third step).
* **I4.** "Separated-better", "not separated-worse" and the distance-keeping directions follow HSPEC's reader conventions, so DOMINANCE is HSPEC's PRIMARY-DOM row verbatim.
* **I5.** The STRATEGIC family is **read** for X and HX. S is retrained in X, so it is not identical to B by construction. It is not paused for v8F (`PI-V8F-2026-10-17`; R72 / R73).

The instruments land only after their mutation proof is clean. The first proof had 44 surviving mutants, several of them on the reading above (the ACC and KEEP metrics, HX's re-weighting, DOMINANCE's pair). Killing tests are being written, and the proof is then re-run.

## Amendment 2 (`A2-2026-10-26`), POST-DATA and CONTROL ONLY: the one-lever control compared two trainer record formats

**When, and what had been seen.** Written 2026-10-26 ~10:40.
* X_s0 (`M2b_s0_oacc`) had trained, been evaluated and been re-fitted on box 2.
* The read of seed 0 was then REFUSED by the B-vs-X built-lever control in `check_x_run`. The refusal message, in full:
  `built levers ['residual_zero_init', 'w'] differ from B_s0's`, for both `summary.json` and `selfcheck.json`.
* The reader refuses before it computes any row, and it wrote no read file. No outcome row, no family row and no eval number of X has been read
  by anyone. This amendment is written without them.

**The defect: a control comparing record formats, not levers.** MEASURED on the real records:
1. **`residual_zero_init` is a readout, not a lever.** It is True exactly while O's last residual layer `head_o.2` is all zero.
   * B (banked M2b, trained 2026-10-17 by an older trainer) writes it AFTER training in `summary.json` (False) and BEFORE training in
     `selfcheck.json` (True).
   * The current trainer writes the before-training value in both.
   * X's head trained: in its checkpoint, `head_o.2.weight` has an L1 norm of 164.4, against B's 137.4.
   * `test_v8f_grid.py` and `test_v8f_hist.py` already drop exactly this difference ("written BEFORE training (True) and AFTER it (False)").
2. **`w`: the current trainer records `w_tac` and `w_cons` inside it; B's older record lists only `w_pen`, `w_pen_self` and `w_wm`.**
   * B's own checkpoint cfg reads w_tac 1.0 and w_cons 0.5 at both seeds, and so does X's.
   * B's argv names neither.

**Fix, in `code/v8f_oacc_read.py`.** The control stays strict:
* `residual_zero_init` joins `lever_skip`.
* `w` is compared key by key. A weight that B's record lacks must equal B's BUILT config (`W_B_BUILT = {w_tac: 1.0, w_cons: 0.5}`). Any other
  difference is refused, by name.
* Proof that the fix does what it should:
  * the reader as it ran reproduces the refusal on the real records;
  * the fixed reader returns no defect on them;
  * a test refuses each real weight change (w_tac 2.0, w_cons 0.25, w_wm 0.2, a missing or added weight);
  * a new mutant turns the weight check off, and it must go RED.

**Root-cause class.** A cross-version control: an older banked arm compared field by field with a newer trainer's record. The generic rule
already lives in two tests and was not carried into this reader.

Nothing else moves: the rule's w_acc, the rows, the bars, the outcome table, the seed-1 condition and every committed next step.
