# PREREG_V8F_PROOF_ADD29: RULEATTR. Which part of the vision lead read holds the head-fed distance rule back: presence, gap or closing speed? Mixed-source rollouts, zero training.

Frozen before any number of this card is computed. Addendum to `PREREG_V8F_PROOF.md` and to card ADD26 with its note N1. Tier T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class of every result: MEASURED once run. LONGITUDINAL is the family measured (distance keeping and the four-family cost read); LATERAL and TACTICAL are reported by the cost read; STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it (MEASURED, card ADD26 with note N1, 642 lead windows of 56 episodes, controls passed)

* Label-fed rule (privileged, an upper bound): the model-induced unsafe rate U falls in six of six seeds for HIER (-0.0786 [-0.1015, -0.0557]) and for G (-0.0798 [-0.1024, -0.0596]); contact rate -0.0447 and -0.0444; no cell of the four families worse under the rule.
* Head-fed rule (vision only), U against no rule: LS -0.0409 / -0.0391, LSH -0.0071 / -0.0641, LSD -0.0196 / -0.0142. Label-fed on the same checkpoints: LS -0.0792, LSH -0.0863, LSD -0.0872. Head-fed minus label-fed: LS +0.0391 [+0.0197, +0.0623], LSH +0.0507 [+0.0227, +0.0848], LSD +0.0703 [+0.0426, +0.0996]. The head-fed rule on LS has one cell worse in LONGITUDINAL (along-track error +0.417 m).
* The head (card ADD24): gap R2 0.54 to 0.74, presence AUROC 0.90 to 0.92, closing speed flat. In the rollouts the head source equals the banked head read within 0.007 m and 0.001 m/s on all ticks of three arms.

## 2. The experiment

From the label-fed rule, one part of the lead source at a time is taken from the head; the other two stay from the labels. Constants, projection (cap-only), replanning and everything else as card ADD26 with note N1.

| variant | presence | gap | closing speed |
|---|---|---|---|
| XP | head (probability at least 0.5 and a full history) | label | label |
| XG | label | head | label |
| XC | label | label | head |
| XZ | label | label | 0 (the lead is taken to move at the logged ego speed) |

Rule for a missing value: on a tick that the presence part calls present, a gap or closing speed whose chosen source has no finite value is taken from the other source (in XP: a lead the head sees and the labels do not have gets the head gap and the head closing speed). The count is written into the log.

Arms: LS_s0, LS_s1, LSD_s0, LSD_s1 (the weight-1 arms and the detached arms), four variants each: 16 rollouts, named `<BASE>XP_s<seed>`, `XG`, `XC`, `XZ`.

## 3. Read

`headway/headway_read.py` (cells of card ADD25: U primary, C, F, gap bias), paired on windows, two-level bootstrap (seeds, then episodes), B 2,000, rng 0: each variant minus the label-fed arm of the same checkpoint (`LSXP:LSRL` and so on), and each variant minus the no-rule arm. `spatial/q3/q3_read.py` four families, each variant against the no-rule arm, spread of G on six seeds (reported).

Reported, naming nothing: each difference in U as a share of the head-fed minus label-fed difference of the same checkpoint; the counts of present ticks, of ticks where the rule changed the plan, and of missing-value substitutions per variant.

## 4. Controls (all must pass before any outcome is named)

* m1 the mixed wrapper with all three parts from the labels equals the label-fed dump of the same checkpoint (LS_s0; arrays of all 141 files, 0 differing).
* m2 the mixed wrapper with all three parts from the head equals the head-fed dump of the same checkpoint (LS_s0; 0 differing).
* m3 no plan ahead of the issued plan in any rule log (lift count 0).
* m4 the pinned eval tool has its recorded sha256 in every run (the wrapper refuses otherwise).
* m5 the controls of the two readers (h1 to h6; q3 controls) on the new dumps.

## 5. Outcomes (both committed), per arm type (LS, LSD), both seeds needed

* For X in presence (XP), gap (XG), closing speed (XC): `PRESENCE-LIMITS-RULE`, `GAP-LIMITS-RULE`, `CLOSING-LIMITS-RULE` if U of the variant minus U of the label-fed arm is above 0 in both seeds with a two-level interval that excludes 0; else `PRESENCE-NOT-LIMITING`, `GAP-NOT-LIMITING`, `CLOSING-NOT-LIMITING`.
* `CLOSING-ZERO-COSTS` if U of XZ minus U of the label-fed arm is above 0 in both seeds with an interval that excludes 0; else `CLOSING-ZERO-EQUAL`.

## 6. Committed next steps

* `CLOSING-NOT-LIMITING` and `CLOSING-ZERO-EQUAL` in both arm types: the closing speed is not a lever for this rule at this replanning interval. The closing-speed branch (cards ADD27 and ADD28 and the re-encoding question for the PI) is parked for distance keeping; it stays open only for imagining the gap ahead.
* `CLOSING-LIMITS-RULE` or `CLOSING-ZERO-COSTS`: the closing speed is a lever with a measured size; the probe arms of card ADD27 are read for it, and the re-encoding step is put to the PI with that size.
* `PRESENCE-LIMITS-RULE`: the presence threshold and a persistence rule over ticks are read on a tuning split of the evaluation episodes fixed before any number (zero training), own card.
* `GAP-LIMITS-RULE`: the gap read is the lever: a bias calibration on the train clips (zero training), then the gap of the probe arms of card ADD27, own card.
* In every case the training card that card ADD26 commits (the rule in the loop: the model learns from its own capped plans) follows, built on the parts that limit.

## 7. Cost and limits

* 18 rollouts of about 3.3 minutes on the pod GPU, one at a time beside the training chain and the token probe; no training.
* Two seeds per arm type; the two rollouts of a pair share the checkpoint and inference is deterministic, so a difference is the source part alone; the interval answers the question about episodes.
* The parts are swapped one at a time from the label-fed rule; interactions between parts are not read. The lead in the rollout does not react.
