# The hierarchy audit, and the T → O intervention's proofs (2026-10-20, before any run)

**The PI's question (verbatim):**

> *"Did you implement the tactical and operative hierarchacy as included in h-jepan end2end planner and in alignment to my 4b conceot using the tactical goals and constraints in our training corpus"*

All numbers below are MEASURED. Source is quoted at tip `142e169d`; logs come from the banked runs.

## The audit

| question | answer | evidence |
|---|---|---|
| Is there a trained S → T → O hierarchy? | **Yes, in v8F-mini** | `stack/tanitad/models/v8f_mini.py` docstring and `_forward`: S (1 Hz) → route checkpoints at 30 / 50 / 80 m; T (2 Hz) → goals, lat / lon decisions, constraints; O (10 Hz) → the 6 s plan as a residual on P10. Sample-and-hold; scheduled sampling p_gt 1.0 → 0.2. |
| Does T use the corpus's tactical goals and constraints? | **Yes** | v9 labels: **22 goals** (incl. TRAFFIC_LIGHT_REACT_RED / _YELLOW / _GREEN), **5 lateral, 7 longitudinal** decisions, **26 constraints** (`v8f_mini_data.py` GOAL22 / LAT9 / LON9 / constraint names). Train coverage 4,572 / 4,572 clips; eval 139 / 141 (`LABEL_JOIN_REPORT.md`). |
| Do the labels reach training? | **Yes** | `raw/full/runs/M2b_s0/train_log.jsonl`, last record (step 19,164): goal 0.1203, lat 0.1532, lon 0.7709, cons 0.1402, all non-zero. |
| Are the constraints enforced? | **Yes, twice** | Differentiable SPEED / STOP / TURN / LEAD hinge penalties (non-zero in the same record), and a hard inference projection onto the R1 speed cap and the Kamm circle. Emitted plans violate active GT constraints at 0.155–0.210, against P10's 0.311 (`RESULT_FULL.md`). |
| How good is T? | Lateral good, longitudinal weak | Tactical-head macro-F1: lateral 0.82–0.84 (majority 0.317); **longitudinal 0.47–0.53** (majority 0.071) (`RESULT_FULL.md`). |
| Is the H-JEPA the hierarchy, as `PLAN_V8F_REFAV1_HJEPA` §4 specifies? | **No** | `WMAux` is an AUXILIARY on the shared adapter. Its z¹ / z² / z³ are not S / T / O, and **no v9 label supervises z²** (plan §4.4 R3 says all should). In M2b_s0's last record the WM term is 162.36 × w_wm 0.1 = **16.24 of the total loss 18.60**. That is a LOSS-VALUE share, not a gradient share. |
| Does the H-JEPA line (RA-1) have T / O? | **No** | `ra1_hjepa.py`, `ra1_hjepa_train.py`, `ra1_pooled.py`: **0** matches for tactical / g_tac / constraint / nav / vocab_v7 / speed_cap / traffic. Its levels are time scales (0.2 / 0.6 / 1.8 s) feeding one trajectory readout. |
| Does O use T? | **Never measured** | S's use of nav is measured (M1-navshuf − M1 = +4.25 [+3.02, +5.57] m at 50 m). T → O has no intervention. ⇒ `PREREG_V8F_TAC` (`Project Steering/`). |
| Did the H-JEPA plan with tactical goals (β)? | It planned in the UNSUPERVISED z² | FAIL-GOAL. Even the GT goal latent gives 3.36 / 3.24 m vs M2's 2.18 / 2.05 (`…/2026-10-18-v8f-beta/RESULT_FULL.md`). |

## The proofs (all before any run)

* `mutate_tac.py` / `mutate_tac.out`: **15 / 15 mutations RED** against `stack/tests/test_v8f_tac.py`, across five files:
  * `v8f_mini.py`: 3;
  * `v8f_mini_eval.py`: 3;
  * `v8f_tac_read.py`: 6;
  * the queue: 2;
  * `v8f_guard.sh`: 1.
* Each file is restored byte-identical.
* The identity control is bit-exact: `forward_diag` with T's own code equals `forward_infer`.
* CPU suites (`CUDA_VISIBLE_DEVICES=-1`, tip-`142e169d` export plus these changes):
  * `test_v8f_tac.py`, 6 passed;
  * the existing v8F files (`test_v8f_mini`, `_beta`, `_mini_full`, `_mini_full_report`, `test_launch_gate_v8f`, `test_v8f_smooth_read`): green, 141 passed together with the first 4 tac tests.
  * One export-only failure, a missing RA-1 raw file in the scratch tree, passed once that file was exported.
