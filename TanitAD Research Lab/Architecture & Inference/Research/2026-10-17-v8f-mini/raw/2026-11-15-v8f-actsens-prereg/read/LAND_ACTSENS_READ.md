# LAND_ACTSENS_READ (Master Mind, 2026-11-15): the ACTSENS read package and control C7 on chassis s0

* **Package** (`pkg/`, from `F:/TanitAD-scratch/actsens/read/`): `read_actsens.sh <tag> <chassis>` with helpers, `verdict_actsens.py` (9 synthetic tests) and the controls. Not banked here: `tree/` (an export of code at 6b2bca1f), `hf_phase1/` (R35's phase-1 plans, PRIVATE HF rev aa2d8014), plan/window npz files, and `publish_context_labels.py` (a copy of a repo file).
* **C8 PASS (MEASURED):** with R35_s0's level-1 action embedding zeroed, all level-1 IA-3 gains read exactly 0.0; the intact control reads gain_lon 0.384 and gain_lat -0.038 at L1 2.0 s (`pkg/out/C8/`).
* **C7 first run FAILED exactness on the HWM path** (`pkg/out/C7_R35_s0/`). FLAT plans were bit-equal; HWM plans were not (a1_raw max diff 0.39). **Root cause (MEASURED, `HANDOFF_ACTSENS_C7DIAG.md`):** torch's thread count. The read set MKL_NUM_THREADS=2 where phase 1 planned at 1, and torch takes MKL over OMP. The first divergence is in CPU `nn.Linear` (act_embed); the HWM planner amplifies it. That rounding-level perturbation alone produced 2 of 39 separated cells (LATERAL head_err 2-6 s and full 6 s, -0.0022 [-0.0034, -0.0009]).
* **Fix:** the windows and plan stages run at torch threads 1, behind a fail-closed probe. MM decision: one numeric environment (phase 1's), and the references stay the banked plans.
* **C7b PASS on chassis s0 (MEASURED,** `pkg/c7diag/out/C7b_R35_s0/c7_result.json`**):**
  * all four plan sets bit-equal to the banked phase-1 plans;
  * `contrast_l1full --check` reproduces the phase-1 rows exactly;
  * arm minus R35 is 0 of 39 non-zero for both HWM and FLAT;
  * IA-3 matches the banked read on all 486 numbers, max diff 0.0;
  * decode_check step3_nn is equal.
* **C7 on chassis s1:** running (C7c); it is banked when done. No ACTSENS arm has been read yet.
