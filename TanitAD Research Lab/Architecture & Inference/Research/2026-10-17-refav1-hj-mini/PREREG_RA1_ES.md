# PREREG_RA1_ES — does a shorter training budget fix RA-1's memorisation, and with it the A1x instrument?

**Written 2026-10-18, AFTER the compliant RA-1 verdict and BEFORE any sweep run.** This is a new, exploratory follow-up
to `Project Steering/PREREG_HJ_STAGE0.md` (H-HJ-K-1). It does not amend that file, and it moves none of its bars.

## What the compliant RA-1 verdict said (MEASURED, T1-class open loop, eval-141, 2,399 windows / 141 episodes)

Source: box 2 `D:\tanitad\runs\ra1_yaw\verdict.json` (pulled to `box2_ship/logs/ra1_verdict_yaw.json`). Estimator: paired
episode-cluster bootstrap, B = 2000 (answers EPISODE variance only).

* **A1x gate: INSTRUMENT BROKEN.** The time-shuffled-target control BEATS A1: A1x − A1 = −0.1268 [−0.2281, −0.0276].
  PREREG_HJ_STAGE0 binds: *"A1x not losing to A1 ⇒ the instrument is broken; nothing else is read until it is fixed."*
  So K, SENS and COST from that run are **not read**. The numbers below are reported, not used.
* K (not read): R0 − P10-yaw-540 = +0.53 / +0.51 at seeds 0 / 1, both separated; A1 − P10 = +0.95 / +0.80.
* **Diagnosis: memorisation.** R0's training-batch ADE@6s falls from 2.093 at step 500 to 0.226 at step 6,000, while its
  eval-141 ADE@6s is 3.09 and the frozen prior P10 scores 2.5627. On the withheld train holdout (60 clips) R0 − P10 =
  +0.3971 [+0.1806, +0.6519]. A readout that is a residual on P10 and ends up worse than P10 has learned clip-specific
  structure that does not transfer.

## The hypothesis (H-RA1-ES-1)

Memorisation by the trajectory readout dominates every arm's K. If so, it also explains the broken instrument: when
every arm's error is mostly readout overfit, A1 vs A1x measures overfit, not world-model content. A shorter budget
should then (a) bring R0 to or below P10 and (b) restore A1x losing to A1.

Why a budget sweep and not intermediate checkpoints: the trainer keeps only `latest.pt` (overwritten every 500 steps),
and its cosine schedule depends on `--steps`. So each budget is a full run, with the schedule compressed to it.
Training constants other than `--steps` are unchanged. ⚠️ That includes `--warmup 200`, so at S = 250 the warmup is
80 % of the run and the cosine phase is 50 steps. This is disclosed, not corrected: changing the warmup per budget would
be a second lever moved at once.

## Design

* **ES-1 (selection):** R0 at seeds 0 and 1, `--steps` S ∈ {250, 500, 1000, 2000}. The banked S = 6000 runs join the set.
* **S\*:** the S with the lowest seed-mean holdout delta (R0 − P10, ADE@6s, the 60-clip Amendment-1 holdout). Ties
  (|Δ| < 0.01 m) go to the shorter S. **eval-141 is never used to select S\*.**
* **ES-1 stop rule:** if no S has a seed-mean holdout delta below 0, ES-2 is NOT run. At budgets this short the residual
  readout sits near P10 by construction (its output layer starts at std 1e-3), so an A1x-vs-A1 comparison there could not
  discriminate anything.
* **ES-2 (instrument, then K):** at S\*, A1 at cell b0.08g50 (the holdout-selected cell) at seeds 0 and 1, plus A1x
  b0.08g50 at seed 0. The budget is taken from R0's S\* for all three; this is disclosed, not tuned per arm.
* **Runner:** `ra1_box2_queue.cmd es "<S list>"` and `es2 <S*> 0.08 50`, box 2, SAFE MODE, runs in
  `D:\tanitad\runs\ra1_yaw_es\st<S>\`. The command lines are the queue's own `:run` block, unchanged.
* **Verdict:** `code/ra1_es_verdict.py` reads the per-run `eval.json` files and applies the rules below.

## Readings, committed before any data

Read in this order. A later reading is not made if an earlier one stops.

1. **ES-1 NULL.** No S has seed-mean holdout delta < 0. ⇒ No training budget lets the frozen pooled readout beat P10 at 540
   clips. Memorisation is real, but stopping early does not turn it into skill. **Next lever:** the data scale that
   PREREG_HJ_STAGE0 already plans ("600 TRAIN clips first, then all 4,572"). Its blocker is named with its encode cost.
2. **ES-2 INSTRUMENT.** At S\*: A1x − A1 on eval-141 ADE@6s > 0, separated.
   * If not ⇒ the instrument is still broken, and K is not read. Memorisation alone does not explain A1x. **Next lever:**
     find out what A1x's shuffled target leaves intact, starting with whether the world-model gradient changes the latent
     the readout reads at all.
   * If yes ⇒ the instrument is restored, and memorisation was what broke it.
3. **K at S\*** (only if 2 passes). This is PREREG_HJ_STAGE0's K bar, unchanged: `best(R0, A1) − P10` ADE@6s < 0, separated
   at both seeds, |Δ| > 2× the training-seed floor |A1_s0 − A1_s1|. A2 is not in this sweep, so K is read on {R0, A1}
   and says so.
   * **PASS** ⇒ early stopping is the lever. Next: A2 and SENS (with A0) at S\*, under PREREG_HJ_STAGE0's other bars.
   * **FAIL** ⇒ the pre-registered K-FAIL branch applies as written. SENS is unread here, so the LoRA branch is not
     taken on this evidence. The data-scale lever from reading 1 is next.

## Variance named

* **Episodes:** the paired bootstrap above.
* **Training seed:** seeds 0 and 1 for R0 and A1; the floor |s0 − s1| is reported per budget.
* **Inference seed:** none. The readout is deterministic (no sampling planner), so there is no inference floor.

## What this sweep does not do

* It does not re-read the 6,000-step K, SENS or COST. The binding A1x rule voids them until the instrument passes.
* It does not touch the vision-only, goal-input or yaw-rate rulings. Action channel 1 stays the measured yaw rate.
* It runs nothing on Thor or the A40.

## Addendum A — ES-3 attribution (written 2026-10-18 04:40, AFTER ES-2, BEFORE any ES-3 data)

**What ES-2 measured** (`es_final.json`; eval-141, paired episode-cluster bootstrap B = 2000):

* At S\* = 250 the A1x gate still FAILS: A1x − A1 = −0.0011 [−0.0046, +0.0026], not separated. Reading 2 binds, so
  **K is not read.**
* Reported, not read: A1 − P10 = −0.0517 [−0.0792, −0.0234] and −0.0480 [−0.0764, −0.0201], separated at both seeds.
  The seed floor is 0.0038 m. R0 − P10 at the same budget is +0.0144 / +0.0020, not separated.

**What A1x leaves intact** (source, `ra1_hjepa.py` `forward_batch`):

* A1x permutes the latent sequence `z1` in time before `stack.loss`. Both the prediction loss and the IDM loss therefore
  see shuffled frame order while the actions stay in order, so the IDM cannot fit: `idm_term/L1` is 14.42 at step 6,000
  against A1's 0.5862.
* Two things survive the shuffle:
  * **SIGReg**, which is a marginal over frames and so invariant to their order;
  * the fact that a world-model gradient reaches the adapter at all.
* R0 has neither. ⇒ **HYPOTHESIS H-RA1-ES-2:** the gain over P10 at 250 steps comes from the world-model losses acting
  as a **latent regulariser on the adapter**, not from temporal prediction and not from the IDM.

**ES-3 design.** A0 (SIGReg β 0.08 + prediction, γ = 0, no IDM) at seeds 0 and 1, `--steps 250`, everything else
unchanged. Runner: `ra1_box2_queue.cmd es3 250 0.08`, which also runs the queue's verdict over R0, A0, A1 and A1x at 250.
The comparisons are paired over the same 2,399 eval-141 windows.

**Readings, committed now:**

1. **A0 − P10 < 0, separated at both seeds, AND A1 − A0 not separated at either seed** ⇒ neither the IDM nor temporal
   order is needed for the gain. ⇒ **H-RA1-ES-2 supported:** a world-model loss is acting as a regulariser.
   **Next:** isolate SIGReg itself (R0 plus SIGReg on `z1`, a new one-lever arm that needs its own launch-gate entry).
2. **A0 − P10 not separated, AND A1 − A0 < 0 separated at both seeds** ⇒ the IDM term carries the gain. A1x would then
   keep it through its gradient into the adapter despite being unable to fit. **Next:** an IDM-only arm (no prediction
   loss).
3. **Anything else** (mixed seeds, or A0 between the two) ⇒ reported as UNRESOLVED, with both deltas and the seed floor.
   **Next:** the replicate (seed 2) of the discordant arm before any attribution.

⚠️ None of these readings reopens K. The A1x gate binds until an arm built to test temporal content loses to its own
shuffled control. The ridge probe (`PREREG_RA1_RIDGE.md`) runs in the same job, after ES-3, on CPU.

## Addendum A — result (2026-10-18 04:45; `raw/ra1_es/es3_attribution.json`, `code/es3_attribution.py`)

**READING 2.** A0 − P10 = −0.0249 [−0.0557, +0.0060] and −0.0247 [−0.0573, +0.0079], not separated. A1 − A0 =
−0.0268 [−0.0508, −0.0044] and −0.0233 [−0.0480, −0.0004], separated at both seeds. Seed floors: A0 0.0003, A1 0.0038.
A1x − A0 = −0.0279 [−0.0504, −0.0063], separated. ⇒ **The IDM term carries the gain, and its gradient helps even when the
IDM cannot fit (A1x).** The pre-registered next arm (IDM-only, no prediction loss) needs new code: the stack has no
prediction weight (`hjepa.py` `loss`), and an unbuilt predictor would trip the parameter audit. It is scoped as its own job.

## Addendum B — A1's own budget (written 2026-10-18 04:47, BEFORE any ES-4 data)

ES-2 ran A1 at R0's S\*, not at A1's own. **ES-4:** `es2 500 0.08 50` and `es2 1000 0.08 50`: A1 b0.08g50 at seeds 0 and 1
plus A1x at seed 0, per budget. R0 at those budgets is already banked from ES-1.
* **Selection:** the budget with the lowest seed-mean A1 HOLDOUT ADE@6s among {250, 500, 1000}. Ties within 0.01 m go
  to the shorter budget.
* **Reported at that budget, NOT read as K (the A1x gate binds):** A1 − P10 on eval-141 at both seeds, A1x − A1, and the
  seed floor.
* **Purpose:** the configuration that carries forward. It goes into the instrument redesign (a control that removes the
  TRUE action information, which A1x's shuffle does not) and into data scale.
* **Committed:**
  * if A1's holdout keeps improving to 1,000, a 2,000-step arm is next;
  * if it is best at 250, the IDM gain is also a short-training effect, and that is reported as such.
