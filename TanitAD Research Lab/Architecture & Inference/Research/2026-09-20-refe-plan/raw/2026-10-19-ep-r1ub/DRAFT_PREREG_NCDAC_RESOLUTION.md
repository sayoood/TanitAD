# DRAFT PREREG (NOT REGISTERED, NOT RUN): the NC/DAC-row resolution read on the S0 bank (H-REFE-RL1-NCRES), the committed next lever of EP-R1ub outcome N

**Written 2026-10-19 by the EP-R1ub agent as PREREG_EPR1UB row N requires.** Nothing here has been run. It needs the Master Mind's registration (and its sha256 written before any number) before anyone executes it. Part A trains nothing; Part B trains and needs a launch-gate token.

## 0. What it rests on (all MEASURED, ours)

| fact | value | artifact |
|---|---|---|
| EP-R1ub NONE-FEASIBLE: in all 18 (config, seed) cells the OOF unsafe-pick-rate delta vs P2 is wholly above 0 (point +0.0135 to +0.0182, lower bound >= +0.0095) and dPDMS x100 wholly below 0 (point -0.586 to -1.135, upper bound <= -0.200) | paired log-cluster bootstrap over 117 fit logs, 10,000, seed 20260927; P2 is IN-SAMPLE on these sets | `raw/f10/epr1ub_result.json` sha256 `10562ac7defe299b5bd051648a024b81c0334b503276a51f5e2beb120f49cafa` |
| the u anchor did its job on the EP row's LEVEL: mean sigmoid(EP) on unsafe proposals 0.43 / 0.26 / 0.14 at u = 0.03 / 0.1 / 0.3 (lr 1e-3; P2: 0.436; the u = 0 row: 0.843), yet the unsafe-pick rate stayed at 0.044-0.049 against P2's 0.0304 | C7 reads | same file, `grid[*].per_seed[*].leak_c7` |
| a NULL control (P2's EP row plus isotropic noise, no learning, same sets and estimator) that changes 69-73 % of the picks moves the unsafe-pick rate by only +0.0006 to +0.0034; the fitted rows move it by +0.0135 to +0.0182. POST-HOC, descriptive, two noise seeds, isotropic noise only | `raw/f10_null/null_control.json` | |
| `D-REFE-RL1-HEADWISE`: on navtest an EP-led pick change loses NC although every safety head is the base scorer's | register | `GOALS_AND_CLAIMS.md` |

**Reading.** A learned preference for progress moves the pick onto proposals that are unsafe, and neither a lower EP level on unsafe proposals (the u anchor) nor P2's NC / DAC heads stops it. Three mechanisms remain, and they have different cures:
* **RES (resolution):** P2's NC and DAC heads rate the new violators at least as safe as the pick they replace. The heads cannot see them. Cure: better NC / DAC rows, or features.
* **OVER (overpower):** the heads do rate the new violators lower, and the EP / TTC / C weighting of the rule outweighs them. Cure: a constraint at the pick (a floor on `pNC * pDAC`), the registered `R0e-pick` family. No head retrain helps.
* **FEAT (features):** the 256-d pre-head features do not carry the distinction for any linear row. Cure: the trunk.

## 1. Hypothesis and the read (Part A: zero training)

**H-NCDAC-RES.** On the S0 bank at f = 10 % (117 logs, 15,808 sets), among the sets where a progress-preferring EP row replaces P2's safe pick by an unsafe one, P2's own `log(pNC * pDAC)` does not rank the new pick below the old one.

* **Reference rows (registered, hashed, applied not fitted):** the f10 EP-R1 rows `ep_row_seed{0,1}.npy` (sha256 `f775753f...`, `636f6270...`), applied with the landed `epr1.apply_row` and the landed `picks_np`. They were fitted on all fit-set sets, so the picks are in-sample for the EP row too; disclosed. An OOF replication (a gated fit) is the follow-up if Part A is inconclusive.
* **Sets:** V = {P2's pick safe, the new pick unsafe}; S = {P2's pick unsafe, the new pick safe}; M = {picks differ, both safe}; per seed.
* **Statistics (per seed, paired log-cluster bootstrap 10,000, seed 20260927, over the 117 logs):**
  1. `m = log(pNC*pDAC)[new] - log(pNC*pDAC)[old]` from the stored P2 logits (columns 0 and 1). Report the fraction of V with `m >= 0`, the median `m`, and the same two numbers on M.
  2. The within-top-8 AUC of `log(pNC*pDAC)` for unsafe versus safe contenders, on V and on all sets.
  3. The share of V sets whose new pick is the contender with the highest `pNC*pDAC` of the top-8.
* **Committed outcomes (each with its next arm):**
  * **R-RES:** fraction(V, m >= 0) >= 0.50 AND AUC on V <= 0.60 on both seeds. Next arm: **Part B**, an NC / DAC head-only retrain (section 2). 
  * **R-OVER:** fraction(V, m >= 0) < 0.50 AND AUC on V >= 0.70 on both seeds. Next arm: a pick-level constraint (a floor on `pNC*pDAC` at the pick, the `R0e-pick` family); Part B is NOT run (a head retrain cannot cure an overpowered weighting).
  * **R-MIXED:** anything else, or the seeds disagree. Next arm: the OOF replication of Part A (gated) before any lever is chosen.
  * **R-NONE:** V has fewer than 100 sets on a seed. The read is void for that seed; report V's size.
* **Controls:** (C0) EP row = P2's: V and S are empty and the picks are P2's, bit for bit. (C1) shuffled picks: replacing the new pick by a uniformly random contender gives AUC 0.5 within sampling error. (C2) the statistics reproduce when computed by an independent re-statement. (C3) a mutation arm per statistic (sign of `m`, the unsafe mask, the top-8 membership) goes RED.

## 2. Part B (only if R-RES): an NC / DAC head-only retrain, the analogue of EP-R1

NC row (column 0) and DAC row (column 1) of `score_head`, head-only, same bank, folds and seeds as EP-R1: target the banked NC / DAC label, a pairwise term over the frozen top-8 that ranks safe above unsafe contenders, an anchor on all 64 proposals. Selection on OOF contender-level AUC at the pick-level constraint of EP-R1ub (the unsafe-pick delta upper bound <= 0 with the retrained rows AND the f10 EP row). A launch-gate token with a G-DVB entry for every lever and a RED mutation arm per lever, as for EP-R1ub. Dev gate and navtest only under the Master Mind's rulings D1 / R9.

## 3. Cost and scope

Part A: CPU, seconds (the stored logits and labels; no training). Part B: CPU, minutes per config. No GPU, no navtest, no new bank. The navtest form of the read (`H-REFE-RL1-NCRES`: the deployed NC head at V1's 24 NC-worse picks) stays with the PI, because it opens rows beyond RL-1's look.

## 4. Limits

P2 is in-sample on the bank sets: its heads are better calibrated there than on unseen scenes, so Part A can understate how badly the heads resolve (a conservative direction for R-RES). One head-fit seed pair; the intervals answer "another draw of logs" only.
