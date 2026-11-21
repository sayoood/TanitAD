# HANDOFF P7MECH (2026-11-16)
State: item 1 DONE (DAC-zero P6 249 / P7 260, diff +0.09 pp [-0.19, +0.39], churn 119 fixed / 130 broken). Items 2-3 (head IoU, head-vs-scorer AUC) NOT RUN: need the planner forward (4-cam ViT-L, nuPlan) = GPU-lock window, over the 3 CPU-h bar.
Artifacts: hold `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/` : P7_MECH_READ.md, dac_zero_p7_vs_p6.json. Scratch `F:/tanitad_scratch/p7mech/` : dac_zero.py, ckpt_keys.py (own scratch, no deletions made).
Next step (needs a GPU window, PI/MM go): adapt `PK/eval/aux_navtest_forward.py` (PK = TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan) to load trunk from `F:/TanitAD_p7read/run/model_final.pt` and the head from its `aux_drv` entry (grid 2 m, 40x32, x0 -8); run the chunked command in `PK/raw/2026-10-13-aux-navtest-forward/HANDOFF.md` over the 1,058 C3 tokens; then `eval/aux_s1_readout.py` for IoU [0,40) m and P3 AUC. Interpreter: C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe.
Decision rule: IoU >= 0.80 => lever is the scorer feed; < 0.80 => 1 m grid / --w-field.
Caveat: SPEC_P7.md not found on E:.
