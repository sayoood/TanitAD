# HANDOFF -- stream YSEL (REFe u400 CRUISE slow-down: selection probe), 2026-11-07

**State: DONE.** Nothing is running. Nothing was staged, committed or pushed. The folder sits outside the repo, and the Master Mind lands it.

## Result
- **P1:** the slow-down is a SELECTION defect.
  - On 85.7 % of slow CRUISE picks, a proposal that holds speed has true PDMS at least as high.
  - The scorer's safety heads fall with speed while their truths are flat (TTC −0.48 vs −0.05).
  - The lane head is the largest margin term (31/55).
- **P2:** the frozen rule gBG(0.20, 0.05) is **NOT PROVEN**.
  - Slow-down share −38.2 pts [−56.9, −22.1].
  - CRUISE v4 bias −1.36 → −0.40 m/s.
  - ΔPDMS mix +0.47 [−0.62, +1.51], which misses the −0.5 non-inferiority bar by 0.12 (a power miss, no harm shown).
- **Sim token `41d7b533797c5209`:** the rule changes the pick 37 → 7 and v4/v0 0.43 → 0.63, a partial fix only.
- **P3 speed-recalibration screen:** REFUTED at screen.

## Proposed register rows (the MM writes them into `GOALS_AND_CLAIMS.md` from the tip)
| id | proposed status |
|---|---|
| `H-SEL-1` | gBG(0.2, 0.05) re-pick: **NOT PROVEN** (T0, 62 held-out logs). |
| `D-REFE-SPEEDPESS-1` | MEASURED, T0: the u400 scorer reads proposal speed as danger. Within-token Spearman of pred vs true against v4/v0 on CRUISE: TTC −0.479/−0.049, NC −0.443/−0.039, DAC −0.290/−0.039, EP +0.086/+0.408. |
| `H-SEL-RECAL-1` | Global speed-covariate recalibration: **REFUTED at screen** (TRAIN 5-fold CV, slow-down share unchanged). |
| `H-SEL-LANE-1` (new HYPOTHESIS) | Lane temper L(0.25) read +1.03 [+0.67, +1.53] PDMS on held-out, but that read was post hoc and descriptive. It needs its own prereg plus a fresh set. It does not fix the slow-down. |

## Exact next step (Rule Zero)
1. **Lever 1 (blocked on compute and the PI):** dump u400's 64 proposals and logits on ≥ 2× fresh navtest logs outside the C3 1,058 (GPU).
   - Score only the banked pick and the gBG pick per changed token, with `eval/score_navtest_refe.py` (CPU).
   - Rerun `p2_score.py`'s statistic UNCHANGED on that set. The rule stays frozen as `rule_vec`.
2. **Lever 3 (blocked on GPU, the PI and a training change):** retrain the scorer on speed-balanced TTC/NC targets.
3. **Sim:** the sim may run gBG only as a labelled experimental arm, never as the default. The spec is in `P2_RESULT.md`.

## How to re-run (CPU only)
`C:/Users/Admin/venvs/tanitad/Scripts/python.exe` with `CUDA_VISIBLE_DEVICES=-1`, cwd = this folder:
1. `g0_gate.py` (exit 0 = PASS, and every RED arm fails)
2. `make_split.py`
3. `p1_diag.py --scope train|all|heldout` (`p1_print.py` prints the result)
4. `p2_explore_train*.py`
5. `p2_score.py`. It refuses to run unless `PREREG_SEL.md` matches its sha256.
6. `p3_recal_screen.py`

Every script finishes in under a minute.
