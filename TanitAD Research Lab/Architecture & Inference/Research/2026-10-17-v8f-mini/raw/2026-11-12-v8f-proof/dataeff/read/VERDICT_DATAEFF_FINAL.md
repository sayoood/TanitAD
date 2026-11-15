# VERDICT DATAEFF, final read with six baseline seeds (card PREREG_V8F_PROOF_ADD13)

Written 2026-11-15. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours; `out_final/VERDICT_Q3.md`, `out_final/RESULT_Q3.json`, `out_final_eighth_choice/VERDICT_Q3.md`). Reader: `spatial/q3/q3_read.py read --same-sign`, spread arm HIER, spread seeds 0 to 5, B 2,000, rng 0. Controls of the reader: all five PASS in both reads (same windows, window counts 2,399 / 141 / 1,305 / 1,094, self is zero, swap is minus, planted regression read back as 0.899 m/s and named COST-LONGITUDINAL).

Rule of the card (section 4): a difference is an effect only if both seeds have the same sign, the two-level interval (seeds, then episodes) excludes 0, and it lies outside the six-seed spread of the baseline for the same cell. Primary cell: ADE over 6 s on the selection-critical windows (spread max 0.4093 m).

## Q1: how much data does the recipe need

| arm minus full (HIER), primary cell | seed 0 / seed 1 | D (m) | two-level interval | rule |
|---|---|---|---|---|
| half, R-1/2 | -0.1375 / -0.0789 | -0.1082 | [-0.2279, +0.0159] | OPEN |
| quarter, R-1/4 | -0.0423 / +0.2639 | +0.1108 | [-0.1505, +0.3822] | OPEN |
| eighth, R-1/8 | +0.3014 / +0.5609 | +0.4311 | [+0.1627, +0.6845] | WORSE |

Outcome: **`DATA-SATURATED`**, **`KNEE-1/8`**. The half is not worse than the full corpus; the first fraction that is worse is the eighth. The margin over the spread at the eighth is small (0.4311 against a spread max of 0.4093), and it is stated with the name. On all windows the eighth reads +0.2800 m [+0.1036, +0.4477], inside the spread (0.2845). The tactical-band ADE of the eighth is WORSE (+0.4204 m [+0.1617, +0.6664], spread 0.4169).

## Q2: does the gap between hierarchy and flat control grow with less data

d(f) = HIER minus FLAT on the primary cell at fraction f, seeds 0 and 1.

| fraction | seed 0 / seed 1 | D (m) | two-level interval |
|---|---|---|---|
| full, d(1) | +0.1085 / -0.1863 | -0.0389 | [-0.2574, +0.1951] |
| quarter, d(1/4) | -0.2457 / -0.0722 | -0.1590 | [-0.3452, +0.0551] |
| d(1/4) minus d(1) | -0.3542 / +0.1141 | -0.1201 | not needed, see below |

Outcome: **`HIER-GAP-EQUAL`**. The two seeds of d(1/4) minus d(1) differ in sign, so the first condition of the rule fails and no interval can change the name. Beside the name: at the quarter both seeds of the hierarchy are ahead on the primary cell (two of two negative), cross-track error -0.1746 m [-0.2647, -0.0787] and heading error -1.57 deg [-2.29, -0.83], all inside the spread; the flat control is ahead on the longitudinal mode error rate (+0.0919 [+0.0392, +0.1456], WORSE) and on speed error (+0.1968 m/s, inside the spread). With six seeds at the full corpus (HIERREP, secondary protocol) d(1) reads -0.1582 m [-0.2656, -0.0338]. As the card commits: the dominance claim rests on HIERREP alone.

## Q3: does choosing the clips by selection-criticality beat a random draw

| S-1/4 minus R-1/4 | seed 0 / seed 1 | D | two-level interval | rule |
|---|---|---|---|---|
| ADE 6 s, selection-critical (primary) | +0.0225 / +0.0469 | +0.0347 m | [-0.1750, +0.2608] | OPEN |
| ADE 6 s, common-case | +0.8026 / +0.7078 | +0.7552 m | [+0.4619, +1.0573] | WORSE |
| ADE 6 s, all windows | +0.3782 / +0.3483 | +0.3633 m | [+0.1705, +0.5617] | WORSE |
| speed error | +0.3478 / +0.3186 | +0.3332 m/s | [+0.1999, +0.4691] | WORSE |
| tactical-band ADE | +0.5644 / +0.5218 | +0.5431 m | [+0.2561, +0.8412] | WORSE |

Outcome: **`SELECT-EQUAL`**. The chosen quarter is not better on the primary cell, and it costs on the common-case windows, on speed and on the tactical band. Secondary, one seed, names nothing: at the eighth the chosen subset reads -0.1966 m [-0.4837, +0.0892] on the primary cell (OPEN) and +0.6405 m [+0.2163, +1.0280] on the common-case windows (WORSE). The second seed of the chosen eighth (S1_8_s1) was still training when this read ran.

## Four families of the named cells (all windows, two seeds)

* LONGITUDINAL: half and quarter OPEN on speed and along-track error against full; eighth OPEN; chosen quarter WORSE on speed against the random quarter. Headway and time gap: not measured (no lead block in the dumps).
* LATERAL: heading and yaw-rate error of every reduced arm below full, all inside the spread; cross-track OPEN.
* TACTICAL: eighth WORSE on the tactical-band ADE; mode error rates of the eighth below full, inside the spread. Stop-goal decision: not measured.
* STRATEGIC: PAUSED (PI item 25).

## What follows (committed in the card, section 7)

* `DATA-SATURATED`: more clips of this kind are not the lever at this model size; data collection is not asked for. The scaling step is the cue and the state, and the model width.
* `HIER-GAP-EQUAL`: reported as written.
* `SELECT-EQUAL`: the next choice rule is model-based (clips where the kinematic prior fails), own card.

## Limits

Two training runs per reduced arm; the interval over two seeds is wide by construction. One corpus (4,512 clips at full), one evaluation set of 141 episodes. A subset arm trains for the same number of steps as the full arm, so a smaller subset is seen more often per clip; the read answers how many distinct clips the recipe needs, at a fixed step budget. The planner is deterministic, so inference variance is not sampled.

## Addendum 2026-11-15T00:56Z: the one-eighth choice at both seeds (section 8 of the card moves Q2 and Q3 to one eighth with the same rules)

Source: `out_final_eighth_choice_s01/VERDICT_Q3.md` (reader controls all PASS; S1_8_s1 finished on the pod at 00:27Z, pulled with equal sha256). T1-kin, two seeds, six baseline seeds for the spread.

* Q2 at one eighth: d(1/8) = R-1/8 minus F-1/8 on the primary cell -0.1717 m [-0.3466, +0.0027] (seeds -0.2084 / -0.1349); d(1/8) minus d(1) per seed -0.3169 / +0.0514: the seeds differ in sign, **`HIER-GAP-EQUAL`** at one eighth as at one quarter. Beside the name: the hierarchy is ahead of its flat control on the primary cell in both seeds at one quarter and at one eighth (four of four), each time with an interval that includes 0.
* Q3 at one eighth: S-1/8 minus R-1/8 on the primary cell -0.1722 m [-0.3719, +0.0245] (seeds -0.1966 / -0.1479), OPEN; common-case windows +0.6681 m [+0.4101, +0.9193], WORSE; all windows +0.2110 m [+0.0334, +0.3795], inside the spread: **`SELECT-EQUAL`** at one eighth as at one quarter, with the same cost on common-case driving.
* S-1/8 against the full corpus: all windows +0.4910 m [+0.2804, +0.6879] WORSE; tactical-band ADE +0.7369 m [+0.4282, +1.0278] WORSE.

The names of the first section stand. STRATEGIC: PAUSED (PI item 25).
