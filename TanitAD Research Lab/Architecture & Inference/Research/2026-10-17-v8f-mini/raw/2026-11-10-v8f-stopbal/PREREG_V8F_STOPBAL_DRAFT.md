# PREREG_V8F_STOPBAL (DRAFT, NOT FROZEN): on DAG-OACC's recipe, does a positive weight on the rare STOP class move T's stop decision and the stop-point speed error? (drafted for the Master Mind to freeze)

Status: DRAFT 2026-11-10. Nothing here is frozen, launched or landed. The Master Mind freezes this file (sha256 into `PREREG_V8F_STOPBAL.sha256`) and names the code-of-record commit; until then every `<...>` is open. Style and rules follow `theory/obd/PREREG_V8F_OBD.md` (floors, outcomes, controls). Every number carries its evidence class; every eval number its tier.

## Why
* **STOPDEC (MEASURED, T1-kin first tick, `theory/stopdec/VERDICT_STOPDEC.md`):** on DAG-OACC, T's goal head says STOP_POINT in only **0.103 / 0.128** of GT stop-point windows (s0 / s1), false-positive rate 0.006 / 0.009, and **0.94** of the speed-cell error mass sits in the missed windows. Class: DECISION-MISSES.
* **STOPCUE (MEASURED, `theory/stopcue/VERDICT_STOPCUE.md`):** AUROC of p(STOP_POINT) **0.836 / 0.857** (SEEDS-DISAGREE: PARTIAL / CUE-PRESENT); recall at an eval-chosen 5 % false-positive rate only 0.321. Both seeds are far above CUE-ABSENT: the cue is in T's features, the 0.5 operating point is too conservative for a ~5 % class. Its triggers overlap on "rebalancing first".
* **Training prevalence (MEASURED, `prevalence_stopbal.json`, 306,566 training windows of the 4,512 training clips, `--exclude 540:600`, training labels only):** STOP_POINT observed on 286,035 windows, positive on **13,116 (0.0459)**; lon STOP **13,116 of 302,740** valid rows (**0.0433**; 13,116 rows have STOP as their only allowed class). Square-root rule W = sqrt(n_neg / n_pos): **goal 4.5616, lon 4.6991**. The full-balance ratio n_neg / n_pos is 20.81 / 22.08 (kept as the second dose).
* **The lever exists, built and tested off the shared checkout, unlanded** (`stopbal.diff`, `tests/`): `--goal-pos-weight-stop W` (positive weight of the STOP_POINT entry of T's goal BCE) and `--lon-class-weight-stop W` (weight of STOP rows in T's partial lon CE). Both default 1.0 = the tip's loss bit for bit; both are read back from the BUILT model into `selfcheck.json` `levers`.

## Arms (box 2, init M2b s{k}; DAG-OACC s{k}'s argv verbatim, only the listed flags added)
| arm | added to DAG-OACC s{k}'s argv | seeds | answers |
|---|---|---|---|
| **BAL-G** | `--goal-pos-weight-stop 4.56` | 0, 1 | does a positive weight on the goal BCE's STOP_POINT entry move T's decision and the speed error |
| **BAL-L** | `--lon-class-weight-stop 4.70` | 0, 1 | the same for T's lon STOP class, as a separate lever |
| **floor** | **OBD-NULL_s0 and OBD-NULL_rep** (`theory/obd/argvs_obd.json`: DAG-OACC s0's argv with only `--out`, and the same at `--seed 2`) | 0, 2 | the run-to-run floor of the base recipe at the code of record |
* **What exists (UNVERIFIED at freeze time):** per `theory/obd/HANDOFF.md` (2026-11-10) OBD-NULL_s0 / OBD-NULL_rep are DEFINED and queued, **not yet trained**. The Master Mind confirms they have banked at the STOPBAL code of record before freezing. If they have not, the card adds **BAL-NULL_rep** (DAG-OACC s0's argv, `--seed 2`); it is the same argv as OBD-NULL_rep up to `--out`, so one training serves both cards. A same-seed replicate is bit-identical (memory `same-seed-replicate-is-bit-identical`), so the floor pair must change the seed.
* Reference for every delta: DAG-OACC s{k} (banked `v8f-dagoacc/M2b_s{k}_dagoacc`), valid at the STOPBAL code of record because the levers at 1.0 reproduce the tip bit for bit (`tests/pytest_stopbal.log`, pristine comparison) and OBD's NULL parity row covers the tip against DAG-OACC. If NULL parity fails, every delta is read against OBD-NULL of the same seed (the OBD fallback, unchanged).
* Five new runs (BAL-G, BAL-L at 2 seeds, plus BAL-NULL_rep only if needed), 3,000 steps each (MEASURED: DAG-OACC s0 trained in 2,276 s on box 2). ESTIMATED about 3.2 h of training plus ship, gates and pulls for four arms; the chain refuses a projected total above 12 h and that case goes to the PI. Box-2 slot is the PI's call.
* **Second dose, only if triggered below:** the full-balance ratio (goal 20.81 / lon 22.08), same seeds. Not part of this card's launch.

## Read (T1-kin: self-action open loop, m = 10; eval-141, per seed)
Instruments, unchanged: `theory/stopdec/stopdec_read.py` (T's STOP_POINT decision at 0.5 on GT stop-point windows) and `theory/stopcue/stopcue_read.py` (AUROC, recall at fixed false-positive rates), run on every arm's dump in one environment, windows and model-free arrays asserted equal to the banked DAG-OACC dump; plus the OTBAND/OBD four-family reader (`theory/otband/read/`). Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0; it answers "another draw of EPISODES" only. "Another TRAINING run" is the floor below; the planner is deterministic.
Four families per band (BINDING), each with its CI: PRIMARY ADE / FDE; LONGITUDINAL speed, acceleration, along-track MAE, distance keeping; LATERAL heading, curvature, yaw-rate, cross-track; TACTICAL TAC-LAT / TAC-LON accuracy and T's own decisions. **STRATEGIC: PAUSED** (PI item 25, `PI_DECISION_QUEUE.md`): reported as paused with this ruling, not as a pass, not omitted. Bands: operative knots 1-10, tactical knots 11-30. Beside (no bar): STOPCUE's AUROC and recall at 5 % false positives per arm (separates a shifted operating point from a better cue), lon STOP argmax counts, the speed-cell MSE in hit and miss windows.

## Committed rules (before any run)
* **Floor** of a cell = |OBD-NULL_rep - OBD-NULL_s0| on that cell, read at seed 0 and applied to both seeds (OBD rule r4); for the recall and false-positive rows, the floor is the same pair's |delta| on that row.
* **Stop-point windows** = the STOPDEC set (GT STOP_POINT windows, 117 at the banked dump; the count is re-asserted per arm). **All windows** = the 2,399 eval windows the reader uses for tactical ADE.
* **A gain** on a cell: delta (arm minus DAG-OACC, same seed) separated on the better side AND |delta| above the floor. **A cost**: separated on the worse side AND |delta| above the floor.
* An arm's label is read per seed; an outcome needs both seeds to agree, else SEEDS-DISAGREE.

## PRIMARY (committed, per arm, both seeds)
* **P1 recall:** T's STOP_POINT decision at 0.5, hit rate on stop-point windows, **>= 0.40 at both seeds**.
* **P2 false positives:** the false-positive rate on the other windows **<= 0.03 at both seeds**.
* **P3 speed error:** tactical ADE (knots 11-30) on the stop-point windows is a **gain** (separated better than DAG-OACC and beyond the floor) at both seeds.
* **Non-regression NR:** tactical ADE on all windows is not a **cost** at either seed.
* ⚠️ **Calibration note (UNVERIFIED, for the Master Mind):** STOPCUE's recall at a 5 % false-positive rate is 0.321 (threshold 0.142 / 0.214). Recall >= 0.40 at <= 3 % false positives is above that point of the existing ROC, so P1 and P2 together can only pass if training changes the cue and not only the threshold. That is the bar the brief set; it is committed as written, and the beside rows make a threshold-only shift visible.

## Committed outcomes (per arm)
* **BAL-ADOPT:** P1, P2, P3 and NR hold at both seeds.
* **BAL-DECISION-ONLY:** P1, P2 and NR hold, P3 does not (the decision moved, the speed error did not).
* **BAL-FP-COSTS:** P1 holds at both seeds, P2 fails at either.
* **BAL-RECALL-SHORT:** P1 fails at either seed.
* **BAL-REGRESSES:** NR fails at either seed (cells named with delta, CI and floor), whatever the primaries read.
* **SEEDS-DISAGREE:** reported per seed; repeated with a third training seed per init (`--seed 10` / `11`) for the arm concerned before any outcome is quoted.

## What each outcome triggers (committed now, Rule Zero)
* **BAL-ADOPT:** the arm's flag joins the recipe of record. If both BAL-G and BAL-L adopt, one stacked arm (both flags, seeds 0, 1) is the next card, read against the better single arm. If one adopts, it is the recipe lever, and the other lever is closed with its measured row.
* **BAL-DECISION-ONLY:** T now says STOP but O's speed does not follow. Cheapest next experiment: read O's use of T's STOP signal (a T-code ablation on the stop-point windows, a read of existing dumps, no training), then the stacked G+L arm. Not a perception card.
* **BAL-FP-COSTS:** the dose is too high. Next arm: W = sqrt of the arm's W (goal 2.14 / lon 2.17), seeds 0, 1.
* **BAL-RECALL-SHORT:** two branches by the beside rows. (a) AUROC unchanged (|delta| within the floor) and recall up but short: the second dose (the full-balance ratio) is the next arm. (b) AUROC unchanged and recall unchanged: weighting is not reaching the decision; first check the gradient and the checkpoint's levers read-back for that run, then the perception lever (LoRA arm L's read adds this AUROC; a stop-line auxiliary) goes to the PI as the named blocker with what unblocks it.
* **BAL-REGRESSES:** the costing cells are named; the arm is not adopted; the next arm halves W (square root) and re-reads.
* **A refutation is a waypoint:** an arm that fails its bar leaves its next arm named above; the card is not finished at a verdict.

## Controls (the read is INVALID if one fails)
1. **Lever read-back per run:** `selfcheck.json` `levers.goal_pos_weight_stop` / `lon_class_weight_stop` equal the argv value (BAL-G: 4.56 / 1.0; BAL-L: 1.0 / 4.70), and `launch_gate.py`'s G-DVB passes for each. An arm whose selfcheck reads 1.0 for its own lever did not test it.
2. **Gradient reachability:** the launch gate's G-LIVE reads a finite goal and lon term and a non-zero gradient on `h_goal` / `h_lon` (the unit tests prove the weighted gradient reaches both heads and that the other head reads exactly 0.0).
3. **Instrument controls** as in STOPDEC / STOPCUE: TOP1 = dump 0.0, GT future = dump 0.0, `v0` = dump 0.0, planted AUROC 1.0 / 0.5 / 0.0, probabilities in [0, 1], windows equal to the banked dump.
4. **Floor present:** OBD-NULL_s0 and OBD-NULL_rep (or BAL-NULL_rep) banked at the same code of record; without them no gain or cost is read.
5. **Default-off parity:** the code of record reproduces DAG-OACC's path at both flags 1.0 (OBD NULL parity row; the unit test against the pristine tip).

## Landing notes for the stack session (not part of the card)
The diff touches `stack/tanitad/models/v8f_mini.py` and `stack/scripts/v8f_mini_train.py`; per memory, a land of those between an OVF record bind and its chain restart voids the record, a launch-gate pin or sha edit moves profile fingerprints (run all launch_gate-reading tests), and the Thor suite precheck needs `test_text_encoding_is_explicit`. New test: `stack/tests/test_v8f_stopbal.py`. The two config fields are new keys in `built_levers`, so any golden file of the levers key set is affected (see the existing-test log).
