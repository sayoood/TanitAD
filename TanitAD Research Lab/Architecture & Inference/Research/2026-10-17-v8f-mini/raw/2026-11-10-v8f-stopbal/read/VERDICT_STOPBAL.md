# VERDICT: PREREG_V8F_STOPBAL = BAL-RECALL-SHORT for both levers, branch (a): the second dose (Master Mind, 2026-11-11 ~18:20)

Seven runs banked on box 2 at code of record `715ef8fc` (2026-11-11 17:30). Read 17:35-18:20: seven T1 dumps (rc 0), STOPDEC per arm, `stopbal_read.py` (`RESULT_STOPBAL.json`, 36.9 s, **0 control problems**). Tier **T1-kin** (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene, deterministic planner), eval-141. Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0. It answers "another draw of episodes"; "another training run" is the BAL-NULL_rep floor. STRATEGIC: **PAUSED** (PI item 25). All numbers MEASURED from `RESULT_STOPBAL.json` unless stated.

Two read-side fixes were made after the banks and before any read number, and neither changes a rule (`READ_NOTE_COMMIT.md`): (1) the commit pin follows AMENDMENT 2 (`715ef8fc`; the read's exported paths are byte-identical to `5ecdd7e6`); (2) the eval-only `forward_topk` / `topk_codes` of `8e9499e0` are attached at read time (`topk_shim.py`, sha-pinned). The STOPDEC control "TOP1 vs dump plan" reads 0.0 m on every arm, so the attached method reproduces the dumped plans exactly.

## Mechanism gate M (hit >= 0.30 AND false-positive rate <= 0.05, at both seeds)
| lever | seed | hit (NULL) | hit arm [CI] | fpr (NULL) | fpr arm [CI] | AUROC NULL / arm | M |
|---|---|---|---|---|---|---|---|
| BAL-G (goal W 4.5616) | s0 | 0.060 | **0.222** [0.097, 0.373] | 0.0054 | 0.028 [0.016, 0.042] | 0.819 / 0.816 | hit fails |
| BAL-G | s1 | 0.120 | **0.291** [0.139, 0.456] | 0.0085 | **0.051** [0.028, 0.076] | 0.850 / 0.848 | hit and fpr fail |
| BAL-L (lon W 4.6991) | s0 | 0.060 | **0.291** [0.162, 0.432] | 0.0112 | 0.046 [0.030, 0.063] | 0.816 / 0.819 | hit fails |
| BAL-L | s1 | 0.145 | **0.393** [0.250, 0.537] | 0.0134 | **0.072** [0.045, 0.099] | 0.851 / 0.857 | fpr fails |
Floors from BAL-NULL_rep against BAL-NULL_s0: hit 0.068 (goal) / 0.137 (lon), fpr 0.010 / 0.011, AUROC 0.038 / 0.036.

## P and NR (tactical ADE, knots 11-30; delta = arm minus BAL-NULL of the same seed, m)
| lever | seed | P: stop-point windows (117, 25 episodes) | NR: all windows (2,399, 141 episodes) |
|---|---|---|---|
| BAL-G | s0 | +0.020 [-0.258, 0.291], floor 0.703 | +0.013 [-0.022, 0.048], floor 0.183 |
| BAL-G | s1 | +0.106 [-0.107, 0.363] | +0.035 [-0.005, 0.074] |
| BAL-L | s0 | +0.218 [-0.054, 0.477] | +0.022 [-0.019, 0.061] |
| BAL-L | s1 | +0.198 [-0.079, 0.457] | +0.028 [-0.015, 0.076] |
P is not a gain at either seed for either lever; NR is not a cost anywhere.

## Four families (48 cells per lever and seed: 2 window sets x 2 bands x 11 metrics + 2 TACTICAL accuracies)
* **Gains: none.** **Costs (separated, worse side, above the floor), each at ONE seed only:** BAL-L s0, stop-point operative acceleration MAE +0.0175 m/s² [0.0048, 0.0324], floor 0.0133 (s1: -0.0010, not separated); BAL-L s1, stop-point tactical FDE +0.966 m [0.035, 2.034], floor 0.401 (s0: +0.793, not separated). No cost on the all-window set for either lever.
* LONGITUDINAL, LATERAL and TACTICAL are read per cell in `RESULT_STOPBAL.json`; STRATEGIC: PAUSED (PI item 25).

## Outcome, as committed
* **BAL-RECALL-SHORT for BAL-G and for BAL-L** (hit < 0.30 at either seed). The predicates were evaluated literally; no other predicate holds, and BAL-REGRESSES does not hold (NR is not a cost at either seed).
* **Branch (a) at both levers:** recall rose beyond the floor at both seeds (hit gains of 0.16 / 0.17 for BAL-G and 0.23 / 0.25 for BAL-L, against floors of 0.068 and 0.137), and AUROC stayed within its floor at both seeds. **Committed next step: the second dose, full balance, W 20.81 (goal) / 22.08 (lon), seeds 0 and 1.**
* The card's two-lever ADOPT rule does not apply (`both_levers` null).

## What the numbers say beyond the outcome (MEASURED, no bar)
* **The levers moved the operating point, not the discrimination.** AUROC is unchanged at all four lever-seed reads (largest move 0.006 against a floor of about 0.037). Hit and false-positive rate rose together, along the same ROC. The card itself noted that on DAG-OACC's ROC, recall 0.32 sits at a 5 % false-positive rate, so M can pass only in a narrow band. Two reads landed at that band's edge: BAL-L s0 at 0.291 / 0.046 and BAL-G s1 at 0.291 / 0.051.
* **The plan did not follow the better decision.** At BAL-L, where the hit rate on stop-point windows roughly tripled (0.060 to 0.291 at s0, 0.145 to 0.393 at s1), stop-point tactical ADE moved +0.22 / +0.20 m, worse in point estimate and not separated. The two single-seed costs are on the same stop-point windows.

## What follows (Rule Zero)
1. **Committed: the second dose** (four runs on box 2: BAL-G2 s0 / s1 at W 20.81, BAL-L2 s0 / s1 at W 22.08; BAL-NULL and its replicate are reused). **Pre-registered prediction, written now, before any second-dose run (HYPOTHESIS):** because the ROC did not move, full balance moves the operating point further along it, and the second dose reads **BAL-FP-COSTS** for both levers (hit >= 0.30 at both seeds, fpr > 0.05 at at least one). If it reads that, the dose path has no M-passing point, and the next lever is discrimination (the perception card: stop-region resolution, LoRA arm L's read, a stop-line auxiliary).
2. **Cheaper, zero training, before the dose spends box-2 time: does O plan a stop when T's code says stop?** The dose path can only deliver P if it does. `PREREAD_V8F_STOPORC.md` (written beside this verdict, frozen before any of its numbers) forces T's stop decision on the existing dumps and reads the ceiling.
3. Box-2 slot: the queue is OBD seed 3 (running) then SLOWWM (PI-approved). The second dose follows SLOWWM unless the PI reorders it.
