# HANDOFF SCORER1-IMPL (2026-11-08) - stopped at the context budget
Pinned tip: b626d941688271dfe843fb560eaa907d05d2a57a (agent/arch-inf-20260803). Venv with pytest+torch: C:/Users/Admin/venvs/tanitad (driverl-eval has no pytest).
DONE: deliverable 1; the code half of 2 and 3 (see MANIFEST.md). test_pair_loss.py -> ZZTEST_PASS (T5 needs REFE_BASE_SCORER=<base scorer_finetune.py>).
NOT DONE, in order:
 A. Profile `refe_scorer1` in launch_gate.py (model on refe_sft4r: runner scorer, pinned argv per arm C0/C0r/S1, --max-updates 400 allowed, --ckpt u400, forbid --pair-calibrate, S1 names --pair-w/--pair-heads/--pair-dv, C0/C0r carry none, suite entry test_pair_loss.py, navtest log/token refusal on the training data flags, RED-tested). Then run the mutation arms: REFE_PAIR_MUTATE_NOBALANCE=1 and REFE_PAIR_MUTATE_DROP_LOSS=1 must each make G16 / judge_g16 RED. judge_g16 and the trainer's G16 exist; the env arms were NOT run end to end (needs a CUDA preflight, shared dev-box lock).
 B. test_launch_gate.py --static passes its rls1 section and stops at a missing eval/rls1b_arms.json (my export held refe/* only). Re-export eval/ and rerun; then run every launch_gate-reading test (pin-edit hazard; I changed no pin or sha).
 C. Card 8.1 share-per-head on real labels: G16 computes it from the bank at preflight (pod or dev box). I did not probe for a label sample on the dev box.
 D. Stage 0 tool: not started. Reuse sel_common.py / u400_reference.py (E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107, refe_behaviour_20261107). Held-out set locations not probed (UNVERIFIED).
 E. Pod chain (wait for P7 end marker, refuse without a PASS token; per arm: preflight -> S1: --pair-calibrate, read pair_weight.json into --pair-w -> token -> 400 updates): not started.
