# LAUNCH-INTENT: the Master Mind's pre-promotion redesign of the bar and the arms (2026-10-28 ~12:00)

**Status.** A design decision on a DRAFT pre-registration. No LAUNCH-INTENT run, rollout or row exists. The draft (`PREREG_V8F_LAUNCH_INTENT.md`)
is edited to this design before promotion, and the edit is disclosed in its status block.

**Seen before deciding:**
* `RESULT_V8F_DAG_ORC` (HARMS; the REP pair sets every crawl bar);
* the REP s0 diagnosis (a slow-T training draw: its T's five speed cells read 0.73–1.17 m/s slow);
* `RESULT` of SPEEDSUB-T1 (NULL; speed cells do not bind on healthy DAG-OACC draws).

None of these is a LAUNCH-INTENT number.

## 1. The defect: the registered bar cannot be cleared by any fix

The draft's section 6 reuses DAG-ORC's floor: `bar_r = 2 × max(F_r, RF0_r, RF_r)`, with RF = max over seeds of |REP − DAG-OACC|, from
`RF_REP.json` sha256 `64c9444d…`. REP s0 sets RF on every crawl row. Against the whole gap that DAG-OACC leaves to P10, MEASURED in
`RESULT_V8F_DAG_ORC.md` sections 2.1 and 4 (b):

| crawl row | bar | CTRL − P10, s0 / s1 | bar ÷ gap |
|---|---|---|---|
| yaw-rate MAE | 31.7106 | 3.4726 / 8.6844 | 9.1 / 3.7 |
| curvature MAE | 0.2648 | 0.0854 / 0.1352 | 3.1 / 2.0 |
| L2 curvature MAE (GT ≥ 5.2) | 0.1740 | 0.0443 / 0.0529 | 3.9 / 3.3 |
| M1 crawl share | 0.0776 | 0.0568 / 0.0604 | 1.4 / 1.3 |

R-MOVE needs an improvement of at least the bar on every crawl row. A lever that removed the crawl completely, bringing the model to P10
parity on all four rows, would still fail R-MOVE at both seeds. **The decisive condition is unreachable by construction**, whatever the data
say. That is a design defect, not a strict bar. It is fixed here, before any LAUNCH-INTENT data exists.

## 2. The cause

RF answers a real question: how far apart two training runs of one recipe land (`H-ESTIM-SEED-1`). But it answers it with n = 2 draws per
row, and the maximum of two draws is set by the worse one: here a slow-T draw. Using it as a *level* bar on a cross-run contrast
(`A − CTRL`) then demands that a lever's effect exceed the largest run-to-run gap ever seen. That conflates two questions:

* *would another training run say this?* That is answered by **replication across runs**;
* *is the effect large enough to matter?* That is answered by a **size bar**, which must be reachable.

## 3. The redesign (one change of the bar, one added arm per seed; the outcome table's structure is kept)

1. **Arms.** At each seed s ∈ {0, 1}, two same-seed one-lever pairs:
   * the main pair: **A_s − CTRL_s**. A_s = DAG-OACC_s's argv + `--launch-intent-w 1.58926`, `--seed s`. CTRL_s = DAG-OACC_s, banked.
   * the replicate pair: **A_rep,s − REP_s**. A_rep,s = REP_s's argv (DAG-OACC_s's argv at `--seed 10 + s`) + `--launch-intent-w 1.58926`.
     REP_s is banked from DAG-ORC r2 (`M2b_s{s}_dagoaccorcrep`, under the draft's reuse licence).
   * Each pair differs in the lever alone (same init, seed, bank and argv, apart from `--out` and the lever).
   * New box-2 runs: A_0, A_1, A_rep,0, A_rep,1. ESTIMATED ≈ 2.8 h at ≈ 40 min each (DAG-ORC's measured run times).
2. **The bar.** `bar_r = 2 × max(F_r, RF0_r)`:
   * F_r = 0.1207 m on ADE6 and 0 elsewhere;
   * RF0_r is the frozen same-seed cross-build floor `P0_RF0.json`, sha256 `1608d70b…`.
   * The crawl bars become yaw-rate 1.1070, curvature 0.0526, L2 0.0136, M1 0.0096. Each is reachable: it is below the CTRL − P10 gap.
   * **RF is not a bar term.** The training-run question moves into the per-seed predicates (item 3). RF is still computed from the
     REP pair and reported beside every row.
3. **The per-seed predicates.** Each condition the draft states "at seed s" holds only when it holds on **both pairs of that seed**:
   * R-MOVE at s: every crawl row clears in A_s − CTRL_s **and** in A_rep,s − REP_s.
   * R-CRAWL at s: A_s − P10 and A_rep,s − P10 are each not separated-worse on every crawl row.
   * R-ADE / R-TLAT / R-TLON at s: not separated-worse in either pair.
   * R-T0 at s: the T0 ADE6 point is ≤ +F/2 in both pairs.
   * R-NEW at s: no row fires in either pair (fire is as in the draft, with each pair's own control in place of DAG-OACC, and the new bar).
   * R-KEEP at s: the T1-kin ADE6 point is ≤ +F/2 in both pairs.
   * R-NOBASE at s: holds in both pairs.

   The outcome table, its precedence and its enumeration over the per-seed booleans are unchanged: 524,288 cells, each matching exactly one
   row. A lever effect is credited at a seed only when two independent training draws show it. Across both seeds that is four draws, the
   same replication standard as `PREREG_V8F_SPEEDSUB_T1`.
4. **The slow-draw flag (beside; decides nothing).** For every trained arm, the reader reports its T0 speed-cell shift against its pair's
   control. Use the five cells and the method of `rep_s0_diag/rep_s0_speedcells.py`. An arm whose five cells are all separated by more
   than 0.5 m/s is labelled a slow-T draw in the RESULT.
5. **The dose fallback B** (share 1/4, W_B 0.726175) follows the same pair design: 4 runs, its own PI go, only after A's HARMS or NOT
   ADOPTED.

## 4. What this costs and what it buys

* **Cost:** 2 more box-2 runs than the draft's two (≈ 1.4 h); the T1-kin rollouts double on the dev-box CPU (≈ 140–305 s each, MEASURED in
  the SPEEDSUB-T1 and DAG-OACC logs).
* **What it buys:** a decisive condition that a real fix can meet, and a training-run standard that one slow draw can neither manufacture
  nor block.
* ⚠️ Stricter in one way: every guard must hold in both pairs.
* ⚠️ Looser in one way: the size bar drops from 2 × RF to 2 × RF0. That is the point of item 2, and it is why replication in both pairs
  is required.

## 5. Disclosure (goes into the PREREG status block)

The bar and arms were changed by the Master Mind before promotion, after DAG-ORC's floors and the REP s0 diagnosis were known, and before
any LAUNCH-INTENT number existed. The change follows from arithmetic on banked numbers (section 1). It is not a reaction to a LAUNCH-INTENT
outcome, which does not exist.
