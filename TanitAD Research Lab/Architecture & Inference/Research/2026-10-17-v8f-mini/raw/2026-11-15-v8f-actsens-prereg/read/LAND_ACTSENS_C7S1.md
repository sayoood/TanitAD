# LAND_ACTSENS_C7S1 (Master Mind, 2026-11-15): control C7 passes on chassis s1

* **Run:** `pkg/c7diag/c7c_run.sh`, the patched `read_actsens.sh` (torch threads 1 for windows and plans, as in phase 1) with R35_s1 itself as the arm, chassis 1, the banked R35_s1 train latents. It ran 06:25 to 07:57 on the dev-box CPU, rc 0.
* **C7 PASS on chassis s1 (MEASURED,** `pkg/c7diag/out/C7c_R35_s1/c7_result.json`**):**
  * all four plan sets (HWM and FLAT, oracle seeds 0 and 1) are bit-equal to the banked phase-1 plans;
  * `contrast_l1full --check` compares 75 rows with 0 mismatches;
  * arm minus R35 is 0 of 39 non-zero for both HWM and FLAT;
  * IA-3 matches the banked read on all 486 numbers, max diff 0.0;
  * decode_check step3_nn is equal.
* **What this licenses:** with C7 passing on both chassis (s0 in `LAND_ACTSENS_READ.md`, s1 here) and C8 passing, the read package reproduces R35's phase-1 read exactly on each chassis, so an ACTSENS arm's difference from R35 is the arm's effect plus training noise, not instrument drift. The plan files (`win/`, 50 MB) are not banked; they rebuild from the banked code and latents.
* **Tier:** T0 (oracle goal), an instrument control, never driving performance. STRATEGIC PAUSED (PI item 25).
