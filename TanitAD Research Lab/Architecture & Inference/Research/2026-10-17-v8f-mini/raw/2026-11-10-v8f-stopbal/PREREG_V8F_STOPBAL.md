# PREREG_V8F_STOPBAL: on DAG-OACC's recipe, does a positive weight on the rare STOP class move T's stop decision and the stop-point driving error? (FROZEN 2026-11-10 by the Master Mind, before any STOPBAL run)

Status: FROZEN (sha256 in `PREREG_V8F_STOPBAL.sha256`). Drafted by a Sonnet agent (`PREREG_V8F_STOPBAL_DRAFT.md`, kept beside), changed at freeze by the Master Mind (Log). **Code of record: commit `2bca1c52abbc2346d7380d33ae78b324cbda9df7`** on branch `agent/v8f-stopbal-20261110` (parent: the landing tip `5514a684`; it adds `stopbal.diff`, sha256 96dcc667...). **Box-2 slot: PI go 2026-11-10, verbatim "Yes, STOPBAL next" (after OBD, once this card is frozen).** Every number carries its evidence class; every eval number its tier.

## Why
* **STOPDEC (MEASURED, T1-kin first tick, `theory/stopdec/VERDICT_STOPDEC.md`, H-V8F-STOPDEC-1):** on DAG-OACC, T's goal head says STOP_POINT in 0.103 / 0.128 of GT stop-point windows (false positives 0.006 / 0.009), and 0.94 of the speed-cell error mass of those windows sits in the misses (DECISION-MISSES). Where T's decision sees a slowdown, its speed cells follow.
* **STOPCUE (MEASURED, H-V8F-STOPCUE-1):** AUROC of p(STOP_POINT) 0.836 / 0.857 (SEEDS-DISAGREE: PARTIAL / CUE-PRESENT; neither CUE-ABSENT); recall at an eval-chosen 5 % false-positive rate 0.321. Both seed triggers put rebalancing first.
* **Training prevalence (MEASURED, `prevalence_stopbal.json`, 306,566 training windows of the 4,512 training clips; training labels only):** STOP_POINT positive on 13,116 of 286,035 observed (0.0459); lon STOP 13,116 of 302,740 valid rows (0.0433). Square-root rule W = sqrt(n_neg / n_pos): **goal 4.5616, lon 4.6991** (full balance 20.81 / 22.08 is the second dose).
* **The lever (code of record above):** `--goal-pos-weight-stop W` (positive weight of the STOP_POINT entry of T's goal BCE) and `--lon-class-weight-stop W` (weight of STOP rows in T's partial lon CE). Default 1.0 = the tip's loss bit for bit (MEASURED, `tests/pytest_stopbal.log`, against a pristine export of 5514a684); both read back from the BUILT model into `selfcheck.json` `levers`; mutation of either weighting line turns 2 tests red (`tests/pytest_mutation_{goal,lon}.log`).

## Arms (box 2, init M2b s{k}; DAG-OACC s{k}'s argv verbatim, `--out` moved, only the listed flags added; 3,000 steps each)
| arm | flags added | seeds | role |
|---|---|---|---|
| **BAL-NULL** | none (both levers at their 1.0 default) | 0, 1 | same-code reference for every delta |
| **BAL-NULL_rep** | `--seed 2` on BAL-NULL s0's argv | 2 | run-to-run floor (a same-seed replicate is bit-identical on this rig, so the seed changes) |
| **BAL-G** | `--goal-pos-weight-stop 4.5616` | 0, 1 | the goal lever |
| **BAL-L** | `--lon-class-weight-stop 4.6991` | 0, 1 | the lon lever, separately |
Seven runs. ESTIMATED about 4.4 h of training on box 2 (MEASURED 2,276 s per 3,000-step DAG-OACC run) plus ship, gates and pulls; the chain refuses a projected total above 12 h, and that case goes to the PI. Run order: BAL-NULL_s0, BAL-NULL_rep, BAL-G_s0, BAL-L_s0, BAL-NULL_s1, BAL-G_s1, BAL-L_s1 (seed 0's full comparison first). Beside, no bar: DAG-OACC s{k} banked (parity of BAL-NULL against it, tensor basis).

## Read (T1-kin, self-action open loop, m = 10, eval-141, per seed, every arm dumped in one environment)
Instruments unchanged: `theory/stopdec/stopdec_read.py`, `theory/stopcue/stopcue_read.py`, and the OTBAND four-family reader (`theory/otband/read/`). Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0 ("another draw of episodes" only; "another training run" is the floor; the planner is deterministic). Four families per band, each with its CI: ADE / FDE; LONGITUDINAL speed, acceleration, along-track, distance keeping; LATERAL heading, curvature, yaw-rate, cross-track; TACTICAL T's decisions; **STRATEGIC: PAUSED** (PI item 25), reported as paused. Bands: operative knots 1-10, tactical 11-30.

## Committed rules (before any run)
* **Delta** of a cell = arm minus BAL-NULL of the same seed. **Floor** of a cell = |BAL-NULL_rep - BAL-NULL_s0| on that cell, read at seed 0 and applied to both seeds.
* **Stop-point windows** = the STOPDEC set (GT STOP_POINT goal, no traffic-light goal; 117 at the banked dump; re-asserted per arm). **All windows** = the eval windows of the four-family reader.
* **Gain**: delta separated on the better side AND |delta| above the floor. **Cost**: separated on the worse side AND |delta| above the floor.
* An outcome needs both seeds to agree, else SEEDS-DISAGREE.

## PRIMARY and gates (per lever arm, both seeds)
* **P (driving, PRIMARY):** tactical ADE (knots 11-30) on the stop-point windows is a **gain** at both seeds.
* **NR (non-regression):** tactical ADE on all windows is **not a cost** at either seed; the four families are reported with every cost named.
* **M (mechanism gate):** T's STOP_POINT decision at 0.5 (BAL-G) or lon argmax STOP (BAL-L), hit rate on the stop-point windows **>= 0.30**, with the false-positive rate on the other windows **<= 0.05**, at both seeds. (On DAG-OACC's own ROC, recall 0.32 sits at a 5 % false-positive rate, MEASURED by STOPCUE, so a correct shift of the operating point can pass M; it does not need a better cue.)

## Committed outcomes (per lever arm)
* **BAL-ADOPT:** P, NR and M at both seeds.
* **BAL-GAIN-OTHER-PATH:** P and NR, M fails. Not adopted until a third training seed per init (`--seed 10` / `11`) repeats P.
* **BAL-DECISION-ONLY:** M and NR, P fails (T now says stop, the plan does not follow).
* **BAL-FP-COSTS:** hit rate >= 0.30 at both seeds, false-positive rate > 0.05 at either.
* **BAL-RECALL-SHORT:** hit rate < 0.30 at either seed.
* **BAL-REGRESSES:** NR fails at either seed, whatever else reads (cells named with delta, CI and floor).
* **SEEDS-DISAGREE:** per seed; repeated with a third training seed for the arm concerned before any outcome is quoted.

## What each outcome triggers (Rule Zero)
* **BAL-ADOPT:** the flag joins the recipe of record. Both adopt: one stacked arm (both flags, seeds 0, 1) is the next card, read against the better single arm. One adopts: it is the recipe lever; the other is closed with its measured row.
* **BAL-GAIN-OTHER-PATH:** the third-seed repeat, plus a read of which T cells moved (the CONSORC decomposition on the arm).
* **BAL-DECISION-ONLY:** a read of O's use of T's stop signal on existing dumps (no training), then the stacked arm.
* **BAL-FP-COSTS:** W = sqrt of the arm's W (goal 2.14 / lon 2.17), seeds 0, 1.
* **BAL-RECALL-SHORT:** (a) recall up but short and AUROC within the floor: the second dose (full balance, 20.81 / 22.08). (b) recall unchanged: check the lever's read-back and gradient for that run first, then the perception lever (LoRA arm L's read adds STOPCUE; a stop-line auxiliary) goes to the PI as a named blocker.
* **BAL-REGRESSES:** the arm is not adopted; the next arm halves W (square root) and re-reads.

## Controls (the read is INVALID if one fails)
1. **Lever read-back per run:** `selfcheck.json` `levers.goal_pos_weight_stop` / `lon_class_weight_stop` equal the argv (BAL-NULL 1.0 / 1.0; BAL-G 4.5616 / 1.0; BAL-L 1.0 / 4.6991); G-DVB passes. An arm whose selfcheck reads 1.0 for its own lever did not test it.
2. **Gradient reachability:** G-LIVE reads finite goal and lon terms and a non-zero gradient on T's goal and lon heads for every run.
3. **Instrument controls** as in STOPDEC / STOPCUE (TOP1 = dump 0.0, GT future = dump 0.0, `v0` = dump 0.0, planted AUROC 1.0 / 0.5 / 0.0, probabilities in [0, 1], windows equal to the banked dump).
4. **Floor present:** BAL-NULL_s0 and BAL-NULL_rep banked at the code of record; without them no gain or cost is read.
5. **Suite:** the chain's G-SUITE on the shipped tree, plus the launch_gate-reading tests and `test_v8f_dagger_train` on an export of the code of record before ship (failures that also fail on the pristine parent 5514a684 are named, not waived silently).

## Log
* 2026-11-10 DRAFT by a Sonnet agent (`PREREG_V8F_STOPBAL_DRAFT.md`).
* 2026-11-10 FROZEN by the Master Mind before any STOPBAL run. Changes from the draft, all before data: (1) the decision rests on driving: PRIMARY = stop-point tactical ADE gain + NR; the draft's P1 / P2 (recall >= 0.40 at <= 3 % false positives, a bar the Master Mind's brief set) lay ABOVE DAG-OACC's measured ROC, so a correct operating-point shift could never pass it; it becomes the mechanism gate M (recall >= 0.30, false positives <= 0.05), a point on that ROC. (2) Same-code null arms BAL-NULL s0 / s1 and BAL-NULL_rep replace OBD-NULL as reference and floor (OBD-NULL runs commit cdbff971, not this code of record). (3) Code of record named: 2bca1c52.
