# PREREG_V8F_PROOF_ADD13: DATAEFF, how much data the hierarchy needs and whether choosing the clips helps (2026-11-13)

Frozen before any data of this card is read. Runs on the pod A40 after the HIERREP chain (addendum 11) and after stage 1 of SPATIAL (addendum 12), each arm under its own selfcheck gate token.
Theorems served: T1 / T2 (the value of the hierarchy sits on the selection-critical windows), T8 (the value of a training sample is its share of the decision loss).

## 1. Questions

Q1 (data need). Is v8F-mini limited by the number of clips? This decides the scaling step: more clips, or a better cue and state.
Q2 (hierarchy and data). Does the hierarchy lose less than the flat control when clips are removed? A structural advantage should be largest where data is scarce.
Q3 (choice of clips). At equal clip count, does choosing the clips by their share of selection-critical windows beat a random choice?

## 2. Arms

All arms: the HIER argv of HIERREP (`--arm M2b --grid pool4x10 --kin-mode yaw --batch 64`, same cache, same labels), from scratch, the SAME 19,164 steps set explicitly, so that only the clip set moves.
The clip set is cut with `--exclude 540:B` in the fixed hash order of the train manifest (a random subset by construction; the sets are nested). Full = 4,512 clips kept (HIERREP).

| arm | clips kept | control flag | seeds |
|---|---|---|---|
| R-1/2 | 2,256 | hierarchy | 0, 1 |
| R-1/4 | 1,128 | hierarchy | 0, 1 |
| R-1/8 | 564 | hierarchy | 0, 1 |
| F-1/4 | 1,128 (the same clips as R-1/4) | flat (`--o-interface cut`) | 0, 1 |
| S-1/4 | 1,128 chosen by rule 3 | hierarchy | 0, 1 |

10 new arms. Baselines without new training: the six HIER and six FLAT seeds of HIERREP at full data.

## 3. The choice rule of S-1/4 (fixed now; train labels only, no eval data, no model)

Clip score = the share of the clip's train windows that are selection-critical under the rule the eval uses (the logged joint mode is not the bulk mode).
S-1/4 keeps 846 clips with the highest score (ties by hash order) plus the first 282 of the remaining clips in hash order (a uniform floor of one quarter, so that the bulk is still seen).
It is passed as a prebuilt join restricted to those clips (`--join-train`), with no trainer change; the selfcheck must report 1,128 clips.
Design gate g1 (label statistics only, read before any arm): the mean score of the S-1/4 set must be at least 1.3 times the mean score of the R-1/4 set. Below that the clip-level choice has no contrast; S-1/4 is then not trained and the lever becomes window-level weighting (own card).

## 4. Reads

T1-kin strict (`--limit causal`), eval-141, the read tool of HIERREP; four families, STRATEGIC reported PAUSED (PI item 25). Primary cell: ADE over 6 s on the selection-critical windows; secondary: the same on bulk windows and every family cell.
Unit = seed. Noise floor = the six-seed spread of HIERREP for the same cell (15 pairs). A difference is an effect only if both seeds have the same sign, the two-level interval (seeds, then episodes) excludes 0, and it lies outside the six-seed spread.
No exponent is fitted to four points; the curve is reported as matched ratios to the full-data arm.

## 5. Outcome names

* Q1: `DATA-SATURATED` if R-1/2 is not worse than full on the primary cell under the rule of section 4; `DATA-HUNGRY` if it is worse; the first fraction that is worse is named (`KNEE-1/2`, `KNEE-1/4`, `KNEE-1/8`, or `NO-KNEE`).
* Q2: with d(f) = HIER minus FLAT on the primary cell at fraction f: `HIER-GAP-GROWS` if d(1/4) is below d(1) under the rule of section 4; `HIER-GAP-SHRINKS` if above; else `HIER-GAP-EQUAL`.
* Q3: `SELECT-BEATS-RANDOM` if S-1/4 is better than R-1/4 on the primary cell and no bulk or family cell is worse; `SELECT-TRADES` if better on the primary cell and worse on a bulk or family cell (named); else `SELECT-EQUAL`.

## 6. Controls

* d1: the selfcheck of each arm reports the declared clip count; the clip lists of R-1/8, R-1/4, R-1/2 are nested and F-1/4 equals R-1/4 (list hashes compared).
* d2: no eval clip is in any train list (the 141 eval clips against every list; count must be 0).
* d3: every arm logs 19,164 steps; a run with another count is not read.
* d4: the full-data reference of the read reproduces the HIERREP value of HIER seed 0 exactly (same dump, same tool).
* d5: the S-1/4 join restricted to the R-1/4 clip list reproduces the R-1/4 selfcheck window count (the restriction tool is a pure filter).

## 7. Both outcomes, committed

* `DATA-HUNGRY`: the scaling step is data. Next: the larger train corpus with the same recipe, sized from the measured ratios; `SELECT-BEATS-RANDOM` then sets how the added clips are chosen.
* `DATA-SATURATED`: more clips of this kind are not the lever. The scaling step is the cue and the state (addendum 12 and the stop-cue input), and the model width; data collection is not asked for.
* `HIER-GAP-GROWS`: a data-efficiency advantage of the hierarchy, stated with its seeds and its floor. `HIER-GAP-EQUAL` or `-SHRINKS`: reported as written; the dominance claim then rests on HIERREP alone.
* `SELECT-EQUAL`: the next choice rule is model-based (clips where the kinematic prior fails), own card.

## 8. Limits and budget

Two seeds per new arm. One fraction (1/4) carries Q2 and Q3; if R-1/4 is not worse than full, Q2 and Q3 move to 1/8 with the same rules (F-1/8 and S-1/8, 4 arms in place of 4).
Fixed steps mean more passes over fewer clips; a fixed-pass design is a different question and is not run. Budget: 10 arms, ESTIMATED 10 to 15 A40-h, replaced by the measured HIERREP time per arm.
All numbers are T1-kin (kinematic rollout on the model's own actions) on logged frames.
