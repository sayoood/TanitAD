# RESULT STRAT1 SEEDS (frozen card PREREG_V8F_STRAT1, the SEEDS-DISAGREE branch): C0 and S1 at seeds 10 and 11

**Outcome, as committed (rule r0: a cell costs at ≥ 3 of 4 seeds with one per init): COSTS-HEADS on one cell, TAC-LAT.**
* No other cell costs at ≥ 3 of 4 seeds. No cell is SPLIT-2of4.
* The committed next step is arm S1d = S1 with `--ctx-detach on` (built, default off), with the same seeds and reads.
* S1d needs a checker kind and a training go: S1d is not a kind of `strat1_check.py` at a4357e79, so it needs a small build.

Tier of the 24 driving cells: T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene), 141 episodes / 2,399 windows per seed.

Estimator: paired episode-cluster bootstrap B 2,000 seed 0, which answers "another draw of EPISODES". "Another TRAINING run" is the four seeds plus the card's floor F = |C0_rep − C0_s0| (TAC-LAT 0.0067). "Another INFERENCE run" does not apply (deterministic planner, one environment, one thread count).

Evidence: MEASURED (ours, zero training in the read).

The rules r0-r6 were written before any seed-10 / seed-11 number (`strat1_seeds_read.py` docstring). Controls:
* K1 self-pair reads exactly 0.0.
* K-RE: the recomputed seed-0 / seed-1 rows equal `RESULT_STRAT1.json` exactly.

## 1. TAC-LAT (tactical lateral decision accuracy), S1 − C0

| seed (init) | Δ | 95 % CI | cost |
|---|---|---|---|
| 0 (M2b_s0) | −0.0225 | [−0.0438, −0.0025] | yes |
| 1 (M2b_s1) | −0.0150 | [−0.0358, +0.0063] | no (not separated) |
| 10 (M2b_s0) | −0.0150 | [−0.0304, −0.0013] | yes |
| 11 (M2b_s1) | −0.0250 | [−0.0441, −0.0054] | yes |

The point estimate is negative at all four seeds, about 2 points of decision accuracy.

The one-seed costs of the first read do not recur at seeds 10 and 11: dk_hw and dk_tg at 0-2 s, and accel at 2-6 s. The accel cell is a gain at seed 11.

## 2. Attribution: why the HEADS label is fragile (beside, reported with the label)

* **Rule r3, as committed.** NAV needs N1 − C0 to cost at both seeds 0 and 1. On TAC-LAT it reads −0.0158 [−0.0329, +0.0012] (s0, not separated) and −0.0233 [−0.0450, −0.0037] (s1, cost). So the rule reads HEADS.
* **The read's own S1 − N1 pair** (the heads, with the nav text held equal) shows no heads effect on TAC-LAT at either seed: −0.0067 [−0.0188, +0.0042] at s0 and +0.0083 [−0.0037, +0.0221] at s1.
* **The point estimates point at the nav text, not the heads.** S1 − C0 and N1 − C0 are of the same size.
* **The discriminator is pre-registered** (`PREREG_ADDENDUM_N1S10.md`, written before either arm trains): N1 at seeds 10 and 11, with the NAV4 and HEADS4 attribution rules. It needs a training go (2 arms, about 30 pod-min, ESTIMATED).
* **A second replicate pair:**
  * C0_s10 − C0_s0 on TAC-LAT reads −0.0475 [−0.0692, −0.0275], about 7x the card's single-pair floor.
  * C0_s11 − C0_s1 reads +0.0075 [−0.0088, +0.0255].
  * The single-pair floor understates this cell's training-run variance (`H-ESTIM-SEED-1`). The seed-matched S1 − C0 difference is steadier than C0 alone, so the pairing removes much of it.
  * The next STRAT1-family card sets F from at least two replicate pairs.

## 3. STRATEGIC at seeds 10 and 11 (T0; the s0 / s1 read is in RESULT_STRAT1)

* **Route following: S-PARTIAL**, as at seeds 0 and 1. real − donor is separated at both seeds; real − best constant is not above 0 at ≥ 1 seed. The mechanism is in register row D-STRAT1-SPARTIAL-MECH: the nav announcement horizon.
* **Route setting (RC-A, S1 − C0):** no cell costs at both seeds.
* **Detected information (shipped `ctx_head`):**
  * Country (also beyond the rig), weather, surface, illumination and period_of_day all DETECT at seeds 10 and 11.
  * Accuracies at s10 / s11: country 0.4739 / 0.5169, weather 0.9133 / 0.9208, surface 0.9312 / 0.9141, illumination 0.9678 / 0.9580, period_of_day 0.4927 / 0.4865.
  * The shipped head's weather read is therefore seed-dependent: it detects at seeds 1, 10 and 11 and not at seed 0.
  * The weather head card (H-V8F-WEATHERHEAD-1) is unchanged by this. Its bar is at seeds 0 and 1, read on the pooled head.
* **Derived goals:** not in this arm. **Intent 6-30 s:** not computable. **Visibility:** not built (as RESULT_STRAT1).

## 4. Next (Rule Zero)

* **S1d** (committed). It needs the checker kind (a small build), the G-SUITE, and a PI training go: 4 arms at seeds 0, 1, 10 and 11, about 1 pod-h ESTIMATED.
* **N1 at seeds 10 and 11** (pre-registered addendum). It needs a PI training go.
* **The 120 m card (H-V8F-CKPT120-1).** Its rebase rule (a) reads "COSTS-HEADS at ≥ 3 of 4 seeds, so the base becomes N1".
  * That trigger has fired as written.
  * The attribution rows above say N1 may carry the same TAC-LAT cost. The base choice goes to the PI with these rows (CKPT120 Q4).
