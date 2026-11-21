# HANDOFF: aux drivable-area head wired into `scorer_finetune.py` (S1 host), default OFF, nothing trained

Written 2026-10-13 ~09:40 local. All paths relative to `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/` unless absolute.

## State
- Wiring, gate entries (`gate_checks.py` G12 + 8 registry rows + DVB rows, `launch_gate.py` profile `refe_aux_s1`), tests and the prereg proposal are DONE and STAGED (nothing committed, nothing pushed).
- Fast suites, all GREEN on the final source (quoted in the hand-back): `test_launch_gate.py --static` 552, `--static --no-banked` 538, `test_gate_gpu_probes.py` 12, `test_dump_scores.py` 16,
  `test_field_loss.py` 42, `test_rls1_chain.py` 355, `test_p2r_chain.py` 139, `test_aux_s1_wiring.py` 20, `test_rl_epo.py` 42; `test_aux_drivable_head.py` 68 (module and test bytes unchanged since that run: md5 in the hand-back).
- **The FULL `test_launch_gate.py` PASSED: `ZZLG_FULL_PASS 583`, rc 0, 268 s** (log: `out_final3/full_launch_gate.log` in the session scratchpad; marker `FULL_DONE`).

## Exact next step
1. (done) full gate green.
2. Land with the coordinated lander (single committer). `gate_checks.py` and `launch_gate.py` also carry ANOTHER agent's P2 changes, staged together with mine; review the combined diff before landing.
3. Approve or amend `raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md`, then land it BEFORE the first S1 step (the profile's `AUXS1_*` constants mirror it; an amendment is a one-line change there).

## Escalations (each blocks something named)
- **No navtest PDMS / DAC-zero read of H2 exists**: the planner/harness do not apply the feed (needs a `model.py` + `planner.py` change). S1's H2 verdict is held-out only until then.
- **P1/P2/P3 perception reads are UNBUILT** (the trainer logs overall + per-log IoU / acc / SDF-MAE only): K2 cannot be declared passed.
- `refe/rls1_pod.sh` line 108 file list lacks `aux_drivable_head.py` and `field_loss.py`, but `scorer_finetune.py` now imports the first unconditionally: add both at the next RL-S1 re-ship (the running RL-S1 copy `a120418` is unaffected).
- G12 (a) determinism on the real ViT-L CUDA kernels is UNVERIFIED (identity measured on a stub trunk only); whole-run fit on the 8 GB card with the trunk resident is UNVERIFIED.
- Dev-box data paths for `$U400`, on-policy banks, held-out sets, teacher-lane and PDM labels are UNVERIFIED.
