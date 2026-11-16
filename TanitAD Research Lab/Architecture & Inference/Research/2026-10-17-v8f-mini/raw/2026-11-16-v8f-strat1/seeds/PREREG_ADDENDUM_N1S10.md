# PREREG addendum to PREREG_V8F_STRAT1: N1 at seeds 10 and 11 (written 2026-11-16 ~22:45Z, before either arm is trained)

## Why

The SEEDS read (`RESULT_STRAT1_SEEDS.json`, MEASURED, T1-kin, 141 episodes / 2,399 windows per seed, paired episode-cluster bootstrap B 2,000 seed 0) reads **COSTS-HEADS** as written.
* The cell is TAC-LAT. S1 − C0 costs at seeds 0, 10 and 11: −0.0225 [−0.0438, −0.0025], −0.0150 [−0.0304, −0.0013], −0.0250 [−0.0441, −0.0054].
* At seed 1 it is −0.0150 [−0.0358, +0.0063], not separated.
* The floor is F 0.0067.
* The card's attribution rule (r3) calls a COST4 cell NAV only when N1 − C0 costs at both seeds 0 and 1. N1 − C0 on TAC-LAT is −0.0158 [−0.0329, +0.0012] at seed 0 (not separated) and −0.0233 [−0.0450, −0.0037] at seed 1. So the rule reads HEADS.
* The banked read's own S1 − N1 pair (the heads with the nav text held equal) reads −0.0067 [−0.0188, +0.0042] at seed 0 and +0.0083 [−0.0037, +0.0221] at seed 1. That is no heads effect on TAC-LAT at either seed (`RESULT_STRAT1.json`).
* The HEADS attribution therefore rests on one unseparated N1 seed. N1 was trained only at seeds 0 and 1.

## What runs (zero new code; kind N1 and seeds s10 / s11 exist at a4357e79)

* **N1_s10** (init M2b_s0) and **N1_s11** (init M2b_s1) on the LoRA pod, in the same tree as C0 / S1 at seeds 10 / 11 (new4 = a4357e79, G-SUITE result `gsuite_result6.json`), with the same chain, mint, gate and post checks.
* They run beside the committed S1d branch. **The committed next step S1d (`--ctx-detach on`) still runs**; this addendum changes no outcome of the card.

## Read and rules (fixed now)

* **Same reader and constants** as the SEEDS read: the 24 driving cells, the floor F = |C0_rep − C0_s0| per cell, and cost / gain as the card.
* **NAV4(cell)**: N1 − C0 costs at ≥ 3 of seeds {0, 1, 10, 11}, with ≥ 1 seed in each init.
* **HEADS4(cell)**: S1 − N1 costs at ≥ 3 of the 4 seeds, with ≥ 1 in each init.
* **Attribution of each COST4 cell** (today: TAC-LAT):
  * NAV4 and not HEADS4: the cost is the nav text's. The card's COSTS-NAV next step applies in addition: the N2 card, whose named suspect is the 38.1 % FOLLOW-type text on v9-turn windows.
  * HEADS4 and not NAV4: COSTS-HEADS stands, and S1d decides.
  * Both: both next steps.
  * Neither: the cost is not attributable at 4 seeds. Reported as written, and S1d is read as the card says.
* **Reported beside, no bar:** C0_s10 − C0_s0 on TAC-LAT is −0.0475 [−0.0692, −0.0275] (MEASURED, the seeds read). That is a second replicate pair, about 7x the card's floor F. The single-pair floor understates this cell's training-run variance (`H-ESTIM-SEED-1`), and the next STRAT1-family card should set F from at least two replicate pairs.
