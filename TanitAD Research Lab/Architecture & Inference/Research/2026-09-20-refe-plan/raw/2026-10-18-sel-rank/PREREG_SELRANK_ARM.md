# PREREG (REGISTERED 2026-10-18 by the Master Mind; nothing run): SR-TTC, a head-only retrain of P2's TTC output on the scorer's own contenders

**Registration (Master Mind, 2026-10-18, before any SR-TTC number exists).** The SEL-RANK agent's draft is registered unchanged, except for this note, the title and the section-6 heading. Every `[DRAFT]` value below is now FROZEN; the sha256 of this file is in `PREREG_SELRANK_ARM.sha256`. At registration, P2's navtrain features do not exist, and P2's own held-out table (the dev GATE) is being built, its first gate being disjointness from P2's training logs. No SR-TTC label, feature or gain has been read. **The training run itself still needs the PI's go-ahead (GPU-h) and a launch-gate PASS token (C1).** Changes from here on are appended as `Amendment n`.

**Registration change R1 (made before any data; it overrides the hyper-parameter clause in section 2).** The draft chose hyper-parameters on the held-out GATE set and then read the dev gate on that same set, which is the trap "a probe that tunes on the data it scores will manufacture a result" (`Project Steering/TRAPS.md`). Therefore:
- The grid stays as committed: lr {1e-4, 3e-4, 1e-3}, anchor {0.03, 0.1, 0.3}, K = 8.
- One configuration is chosen by **5-fold grouped-by-log cross-validation on the FIT set** (P2's training labels). The criterion is the mean out-of-fold contender concordance of the TTC head.
- That configuration and its sha256 are written down **before** the gate set is read.
- The dev gate is then read ONCE, on the held-out set, for that configuration and both seeds.
- The 8 unchosen configurations are never scored on the gate or on navtest.

**Status.** Drafted 2026-10-18 by the SEL-RANK agent from `RESULT.md` in this folder. **Not registered, not run, needs a GPU the PI or the Master Mind has not allocated.** Every number marked `[DRAFT]` is a proposal to be frozen (sha256 of this file written to `PREREG_SELRANK_ARM.md.sha256`) before the first training label is read. Later changes are appended as `Amendment n`; the body is never edited.

## 1. What this draft rests on (all MEASURED in `RESULT.md`, T1 NAVSIM v1.1)

| fact | value | where |
|---|---|---|
| P2's non-zero selection pool, full navtest (12,146 tokens / 136 logs) | **5.875 [5.535, 6.213]** x100 | `raw/attribution.json` pool |
| the oracle proposal wins on | EP in 97.3 % of pool tokens (77.1 % of the gap), TTC in 4.9 % (22.3 %) | section 2 |
| perfect-TTC-head bound on the pool | **+1.833 [1.569, 2.112]** = 31.2 % [27.3, 35.1] | section 5 |
| perfect-EP-head bound on the pool | +1.573 [1.349, 1.796] = 26.8 %; **+0.801 [0.500, 1.103] = 13.6 % without its violation signal** | section 5 |
| TTC - EP on the pool, paired | +0.260 [-0.066, +0.594] (not separated); vs the violation-free EP +1.031 [+0.672, +1.395] | section 5 |
| scorer's own top-8: pairwise concordance | EP 0.887 [0.876, 0.899]; **TTC 0.722 [0.686, 0.759]** | section 3 |
| CPU recalibration of the six logits (5 or 14 parameters), dev, grouped-by-log CV | **-1.6 % to +1.5 %** of the dev pool | section 8 |
| dev table, nine sibling scorers | TTC 24-26 % > EP 21-23 % of the pool in 9 of 9 | section 7 |
| training-seed floor on the dev pick | range 0.33-0.44 PDMS points over nine SFT-4 scorers | section 7 |
| in-flight levers (INHERITED, register rows, not re-verified here) | `D-REFE-FREEPOINT-1`: RL-S1 R0B reaches PDMS +0.55 [+0.24, +0.84] over the deployed scorer but pays NC -0.25, TTC -0.40; `D-REFE-SAFEOBJ-R0D`: EP +3.41, TTC -0.77 (breach, killed) | `GOALS_AND_CLAIMS.md` |

**Reading that motivates a TTC lever.** Selection-aware training (EPO) already buys EP and pays TTC; a perfect TTC head frees the scorer to take the higher-progress proposal safely (Shapley: a perfect TTC head's realised gain is 1.30 through TTC and 0.65 through EP). The TTC output is the weakest ranker among the contenders and is the largest single-head counterfactual. This is a design hypothesis, not a result.

## 2. Hypothesis and the single lever

**H-SELRANK-TTC.** Retraining only the TTC output of P2's scorer with a ranking loss restricted to the scorer's own top-8 contenders raises P2's navtest PDMS by at least the bar in section 6, with no sub-score and no longitudinal/lateral family component adverse.

**One lever (single-factor):** the TTC output row of `score_head` (and nothing else). **Frozen, and proved frozen:** the trunk, the visual context, the proposal fan, the EP/NC/DAC/C/DDC rows, the selection rule `NC*DAC*(5EP+5TTC+2C)/12` (`refe/planner.py:558-561`).

**Consequence worth stating: the fan is frozen, so the 64 proposals per token are bit-identical to `table_p2_full.npz`.** The navtest confirm then needs NO harness scoring: the PDMS of the new pick is the lookup `pdms[token, new_pick]` in the existing table, after a gate that the re-dumped proposals equal the table's (max abs pose diff 0.0 m on all 12,146 tokens, as in the table's own same-pick proof).

**Loss.** For every set, take the top-8 proposals under the FROZEN P2 selection score (membership computed once, never refreshed during training). Pairwise logistic loss on the TTC logit over (safe, violating) contender pairs, `[DRAFT]` plus a BCE anchor on all 64 proposals at weight 0.1 so the head stays calibrated. Hyper-parameters (learning rate, anchor weight, epochs, K) are chosen on the held-out gate set only, from a grid committed here: lr {1e-4, 3e-4, 1e-3}, anchor {0.03, 0.1, 0.3}, K = 8 fixed.

## 3. Data: fit, gate, confirm

| role | set | size | status |
|---|---|---|---|
| **FIT** | P2's own navtrain on-policy PDM labels (the 174,912 labels `RESULT_P2.md` section 4 says P2 trained on; 174,912 = 64 x 2,733, so about 2,733 sets, UNVERIFIED) with P2's pre-head scorer features cached from a P2 forward | about 2,733 sets | needs the forward (GPU) |
| **GATE (dev)** | navtrain held-out: 3,137 sets / 24 logs, labels `s1_read_cache/labels/x/heldout_pdm/*.jsonl`, **0 logs shared with navtest (checked)** | 3,137 sets | labels exist; P2's features on them need the forward |
| **CONFIRM** | `table_p2_full.npz` pdms (navtest), **read once**, after the head and every hyper-parameter are frozen and their sha256 written down | 12,146 tokens / 136 logs | exists |

**Where this departs from the brief's wording, and why.** The brief asks for the fit on navtrain held-out sets. A recalibration of the six logits with 5 or 14 parameters fitted on the held-out table has a measured out-of-fold reach of about zero (`RESULT.md` section 8), and a feature-level head cannot be fit on 3,137 sets of 24 logs without leaking its own selection. So the fit set is P2's training labels, and the held-out set is the gate. If the PI wants a held-out-only fit, the arm degenerates to the refuted recalibration and should not be run.

**First gate, before any training:** prove the held-out logs are disjoint from P2's training logs (UNVERIFIED here; u400 excluded them by construction, P2's exclusion was not checked). If they overlap, the GATE set is void and the dev gate is replaced by a log-disjoint carve of the fit set.

## 4. Cost (ESTIMATED; arithmetic)

Rate: **1.44 s per set** for a P2 forward = 907 s for the last 629-token navtest chunk (`STATUS.json` `last_dump_seconds`, `chunks.json`), which includes model and map load.

| step | sets | GPU hours |
|---|---|---|
| S0 P2 forward on the FIT sets, cache pre-head features and logits | 2,733 | 2,733 x 1.44 / 3600 = **1.09** |
| S1 same on the held-out GATE sets | 3,137 | **1.25** |
| S2 fit the TTC row (cached features; minutes, seed 0 and seed 1) | - | negligible |
| S3 navtest forward with the fitted row (CONFIRM; pdms by lookup, 0 harness hours) | 12,146 | **4.86** |
| total | 18,016 | **7.2** |

The dev-box GPU is contended (`P2F-1` waited more than 19 h for the lock), so this is a pod or a scheduled slot decision. S3 runs only if the dev gate passes, so a failed gate costs 2.3 GPU-hours.

## 5. Controls (each is a gate; a miss voids the arm)

- **C0 identity.** The TTC row left unchanged reproduces `pick` on 12,146 / 12,146 tokens and the gain reads exactly 0.0.
- **C1 reachability** (the "built, tested, and unreachable from its caller" class). Trace the flag to the loss: argparse -> namespace -> the `score_head` TTC row -> the term in the loss -> `.backward()`, `file:line` at each hop. A non-zero gradient on the TTC row, and **exactly 0.0** on every frozen module (a control that must read 0.0, so a broken probe cannot read as a zero). Mutation test: delete the line that adds the pairwise term and require the check to go RED. `stack/scripts/launch_gate.py` must issue a PASS token bound to the commit and argv before the first step (G-DVB entry for the new flag).
- **C2 negative control.** The same fit with TTC labels shuffled across proposals within each set must give a dev gain <= 0 (the shuffled-realised-EP control in this package reads -3.756, so the probe machinery can read negative).
- **C3 replicate** (H-ESTIM-SEED-1). Two training seeds, same flags. A separated navtest interval from one seed is necessary, not sufficient; the claim needs both seeds to clear the bar.
- **C4 fan-frozen proof.** Re-dumped proposals equal the table's, max abs pose diff 0.0 m on all tokens.
- **C5 estimator.** Paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (`zero_tail.Boot`), 136 logs. It answers "another draw of logs" only; the replicate answers training noise; inference is deterministic.

## 6. Reach (ESTIMATED; arithmetic) and the bars (frozen at registration)

**Ceiling (MEASURED, not a reach):** the TTC pool bound is 1.833 [1.569, 2.112] on navtest and 1.21-1.31 on the dev table (24-26 % x 5.07).

**Reach = ceiling x f**, where `f` is the fraction of that ceiling a retrained head captures. `f` has no measured anchor on this rig (this arm would measure it first), so three values are shown:

| f | navtest pool gain | dev pool gain | share of the 5.875 pool |
|---|---|---|---|
| 0.10 | 0.10 x 1.833 = **0.18** | 0.10 x 1.21-1.31 = 0.12-0.13 | 3.1 % |
| 0.25 | 0.25 x 1.833 = **0.46** | 0.30-0.33 | 7.8 % |
| 0.50 | 0.50 x 1.833 = **0.92** | 0.61-0.66 | 15.6 % |

Why these values: the in-flight EPO point moved PDMS by +0.55 (INHERITED), which is inside this range; the recalibration proxy captured about 0 (MEASURED), so `f` near 0 is a live outcome and is committed as one.

**Minimum detectable effect (ESTIMATED).** The oracle-TTC selection changes 6,408 picks (4,334 of them with a PDMS difference) and has a paired half-width of 0.43. An arm that changes about 1,000 picks (the lam = 0.1 dose model changes 1,033) scales to 0.43 x sqrt(1,000 / 6,408) = **0.17**, assuming the same per-pick variance and no inflation from log clustering. So a bar below about +0.2 cannot be separated on navtest, and the f = 0.10 row (0.18) sits at the instrument's resolution.

**Bars.**
- *Dev gate* (held-out, before navtest is read): paired out-of-sample total gain **>= +0.30** on BOTH seeds (about the dev training-seed range 0.33-0.44, so a smaller dev effect is indistinguishable from retraining noise), and C0-C2, C4 pass. 24 logs cannot support a lower-bound rule.
- *Navtest* (CONFIRM, read once): total paired gain point **>= +0.30** AND lower bound > 0 AND no sub-score (NC, DAC, EP, TTC, C, DDC) with an interval wholly below 0 AND no longitudinal or lateral family component separated adverse (families6, unpaired, conservative). TACTICAL reported; STRATEGIC **PAUSED** (PI 2026-09-19, item 25), never a pass. Longitudinal distance-keeping UNAVAILABLE until lead-agent tracks are supplied to `families6.py`.

## 7. Both outcomes, committed in advance, each with its next arm (Rule Zero)

| outcome | rule | what it means | next arm |
|---|---|---|---|
| **ADOPT-SIGNAL** | both seeds clear the dev gate and the navtest bar | the TTC head was the binding constraint on the pool; one seed pair, direction not yet a lever claim | (i) feed the new TTC row to the EPO/SAFEOBJ arms as their TTC constraint head, which is exactly the TTC cost they pay; (ii) the EP-non-violating listwise head (pool bound 0.80, 13.6 %); (iii) a third seed before any registry claim |
| **NOT PROVEN** | dev gate passes, navtest point > 0 but lower bound <= 0, or one seed misses | effect smaller than the instrument resolves (about 0.2) | stack it with (i) so the effect is read at a larger size, and add a third seed |
| **DEV-FAIL** | dev gain < +0.30 on either seed | the head, as trained, does not rank contenders better out of sample | **feature-vs-capacity probe**, CPU, minutes: a 2-layer TTC probe on the cached pre-head features with 5-fold-by-log CV. If its contender concordance stays <= 0.72, the features do not carry the TTC signal and the lever moves to the trunk (a TTC gradient into the trunk, the P6 pattern), not the head |
| **REFUTED** | navtest upper bound < 0 | selection got worse | log the cause class (head over-fit vs contender-set mismatch), then the feature-vs-capacity probe above |

No outcome ends at the verdict: each names the cheapest next experiment.

## 8. What this does not claim

- Nothing about driving quality beyond T1 NAVSIM v1.1 PDMS (self-action open loop, not closed loop).
- Nothing about the zero tail (P6 and SAFEOBJ own it); the zero-pick tokens' share of any oracle-head bound (1.21 for TTC) is excluded from the reach on purpose.
- The pool is bought with progress (best-of-64 drives at 1.177 x the human's progress); an EP-leaning gain will move the longitudinal family away from the human and must be reported as such.

## 9. Alternatives considered and set aside (so nobody re-derives them)

| arm | why not first |
|---|---|
| CPU recalibration / re-weighting of the logits | MEASURED out-of-fold reach about 0 (-1.6 % to +1.5 % of the dev pool); consistent with `D-REFE-SELRULE-FIT` and `-CONFIRM` |
| EP-head retrain | non-violating bound 0.80 vs TTC 1.72-1.83; the realised-EP target carries the violation signal, so a naive EP loss would chase the zero tail; and EP-leaning selection moves the longitudinal family away from the human |
| NC / DAC heads | pool bounds 0.15 / 0.68; they belong to P6 and SAFEOBJ |
| selection-aware retrain of the whole scorer (EPO) | in flight; this draft is the cheaper, single-factor complement |
