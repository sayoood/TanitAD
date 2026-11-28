# PREREG_V8F_STOPBAL, AMENDMENT 3: the second dose, as the card committed it (Master Mind, 2026-11-11 ~19:40, before any second-dose run)

Status: written after the first-dose read (`read/VERDICT_STOPBAL.md`: BAL-RECALL-SHORT, branch (a) at both levers) and its three zero-training follow-ups (`oracle/VERDICT_STOPORC.md`, `VERDICT_STOPHARD.md`, `VERDICT_STOPTAU.md`), and before any second-dose run or number. The frozen card and AMENDMENTS 1 and 2 are not edited. This file fixes the arms of the card's committed branch (a) step; rules, read, PRIMARY, gates, outcomes and controls are the card's.

## Arms (box 2; init M2b s{k}; DAG-OACC s{k}'s argv verbatim, `--out` moved, only the listed flag added; 3,000 steps; code of record `715ef8fc`)
| arm | flag added | seeds |
|---|---|---|
| **BAL-G2** | `--goal-pos-weight-stop 20.81` | 0, 1 |
| **BAL-L2** | `--lon-class-weight-stop 22.08` | 0, 1 |
* 20.81 / 22.08 are the card's full-balance values (line 8: n_neg / n_pos on the training prevalence, 272,919 / 13,116 and 289,624 / 13,116).
* **Reused, not retrained:** BAL-NULL s0 / s1 (the reference of every delta) and BAL-NULL_rep (the floor). Same code of record, same box, same recipe, banked under `C:/Users/Admin/t2v8f_pulls/stopbal_<tag>/` with their T1 dumps and STOPDEC outputs. The read re-asserts their checkpoint sha256.
* Order: BAL-G2_s0, BAL-L2_s0, BAL-G2_s1, BAL-L2_s1. ESTIMATED 4 x about 40 min plus ship, gates and pulls (MEASURED about 40 min per first-dose run).
* Unhardened: the STOPHARD / STOPTAU transform is closed as a lever (`VERDICT_STOPTAU.md`) and is not applied.

## Read
The first dose's read package (`read/`), unchanged in rules: dumps, STOPDEC through `stopdec_arm.py` + `topk_shim.py`, then `stopbal_read.py`, with BAL-G2 / BAL-L2 as the lever arms and the banked NULL arms as reference and floor. The only read-side additions: the two new tags and their expected lever values in the refusal checks (control 1), and a `--levers` selection. The second dose is read against BAL-NULL exactly as the first was, and the first dose's rows stay as measured.

## Prediction (HYPOTHESIS, written before any second-dose number; from `read/VERDICT_STOPBAL.md`)
The ROC did not move at the first dose, so full balance moves the operating point further along it. Predicted: **BAL-FP-COSTS for both levers** (hit >= 0.30 at both seeds, false-positive rate > 0.05 at either). If it reads that, the dose path has no M-passing point, and discrimination (the perception card) is the next training lever.

## Box-2 slot
After the SLOWWM chain (PI-approved, queued behind OBD seed 3), unless the PI reorders it. The chain is built and dry-run tested before it is started; it waits for SLOWWM's end marker.
