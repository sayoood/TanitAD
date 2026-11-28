# HANDOFF_N1: lead reader extended with note PREREG_V8F_PROOF_ADD24_N1

Tier T0, never driving performance; STRATEGIC: PAUSED (PI item 25). Nothing in `leadsup_read.py`, `run_leadsup_read.sh`, `test_leadsup_read.py`, `in/`, `out/`, `out_prelim_s0/` was touched.

## Files (all in F:/TanitAD-scratch/v8proof/spatial/leadsup/read/)
- `leadsup_read_n1.py` (copy of the reader plus the additions), `run_leadsup_read_n1.sh` (default OUT `out_n1`), `test_leadsup_read_n1.py` (9 tests, all pass, about 18 s).
- Preliminary run on the three seed-0 arms (names nothing): `out_prelim_s0_n1/RESULT_LEADSUP_T0_N1.json`, `out_prelim_s0_n1/VERDICT_LEADSUP_T0_N1.md`.
- Scratch: `tmp/n1/` (pytest basetemps pt, pt2, pt3, probe.py; well under 100 MB). Nothing deleted.

## What was added
1. R3n: `n1_inputs` calls the objslot rule path (`head_pipeline` / `fit_r3`) with the closing column replaced by 0 BEFORE the fit (NaN kept NaN so both fits see the same rows): coefficient a is exactly 0.0 and (b, d) equal an explicit two-column least squares on the same folds. The test also shows it differs from R3 fitted with the closing column and zeroed afterwards.
2. DELTA = 1 - SSE(R3c)/SSE(R3n) through `skill_row` (R3n as reference), under `M3.delta_n1` (eval and train blocks); printed as "M3 DELTA (note N1; names the slot)".
3. Controls: `k3_slot_n1|ARM` (closing output permuted among scored windows, seed 0, four more printed; bar +0.02), `k3_closing|ARM` (closing R2 of the card k3), `k3_card_form_slot|ARM` (card k3 slot part, R3 skill over the bias-corrected copy, same number as the old `k3_shuffled_states`; printed PASS/FAIL, does not gate).
4. Gating (`gate_controls`): closing names and gradient: k1, k2, k3_closing, k4, a-checks; slot names: those plus k3_slot_n1. A failure of any arm blocks the class for all arm types (same all-arms rule as the original reader; stated here as a reader decision). Names: `SUP-SLOT-IMAGINES-N1` (DELTA above 0 and interval above 0 at both seeds) else `SUP-SLOT-FLAT-N1`; `SUP-SLOT-BEATS-ROLLOUT` only in addition to IMAGINES-N1, as in the card. Verdict sections after the controls: "Card form (as written)", "Names under note N1". R3 over bias-corrected and raw copy print and name nothing.
5. Outputs `RESULT_LEADSUP_T0_N1.json`, `VERDICT_LEADSUP_T0_N1.md`; exit 0 if the slot gate passes, 2 otherwise, 3 for input records.

## Commands
- Tests: `cd /f/TanitAD-scratch/v8proof/spatial/leadsup/read && mkdir -p tmp/n1 && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 /c/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_leadsup_read_n1.py --basetemp F:/TanitAD-scratch/v8proof/spatial/leadsup/read/tmp/n1/pt`
- All six arms (about 80 s per three arms at B 2000): `cd /f/TanitAD-scratch/v8proof/spatial/leadsup/read && STATES_DIR=/c/Users/Admin/v8proof_c/s1states bash run_leadsup_read_n1.sh /f/TanitAD-scratch/v8proof/spatial/leadsup/read/in /f/TanitAD-scratch/v8proof/spatial/leadsup/read/out_n1`
- Preliminary run (done): same with `ARMS=LS_s0,LSH_s0,LSD_s0` and OUT `out_prelim_s0_n1`.

## Open for the lead
- k5, k6, k7 are the lead's, as before. The cost read is untouched.
- The preliminary run names nothing (one seed per arm type).
