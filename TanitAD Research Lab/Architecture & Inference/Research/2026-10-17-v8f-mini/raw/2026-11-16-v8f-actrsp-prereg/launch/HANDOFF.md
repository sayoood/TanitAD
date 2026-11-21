# ACTRSP launch preparation: HANDOFF (2026-11-16, Arch+Inference agent)

## State
* **B-CAL TRIGGERED** (card section 6): z_t-only linear R2 exceeds 0.25 x the pair R2 on BOTH channels, on BOTH chassis (MEASURED, TRAIN latents only, 31,575 transitions / 2,397 windows, 5-fold cross-fit by window, ridge lambda chosen on inner folds, window-cluster bootstrap CI):

| chassis | channel | z_t only | pair [z_t, z_t+1] | ratio | rule (> 0.25) |
|---|---|---|---|---|---|
| R35_s0 | acc | 0.159 [0.134, 0.182] | 0.181 [0.156, 0.206] | 0.876 | TRIGGER |
| R35_s0 | yaw rate | 0.194 [0.157, 0.218] | 0.209 [0.174, 0.235] | 0.930 | TRIGGER |
| R35_s1 | acc | 0.214 | 0.242 | 0.886 | TRIGGER |
| R35_s1 | yaw rate | 0.207 | 0.220 | 0.939 | TRIGGER |

  **Caveat found while checking (MEASURED, same files, same probe, folds grouped by EPISODE, 141 clips, instead of by window; windows of one clip share >= 77 of 91 frames, so a window split leaks):** `bcal/bcaleid_s{0,1}.json`. R2 collapses: R35_s0 acc z_t 0.036 [0.013, 0.055] vs pair 0.030 [0.005, 0.052]; yaw 0.089 [0.069, 0.110] vs 0.089 [0.066, 0.110]; R35_s1 acc 0.035 vs 0.033; yaw 0.065 vs 0.064. The rule still TRIGGERS on all four cells (ratio 1.0-1.2), but both numerators and denominators are near zero: a LINEAR probe on `[z_t, z_t+1]` has no skill beyond `z_t` alone (linear on `z_t+1 - z_t` alone: acc 0.025, yaw 0.001, window split). Diagnostic MLP probes on R35_s0 (window split, leaky, NOT the card's rule): z_t alone 0.867 / 0.918, pair 0.905 / 0.943 (`bcal/bcalmlp_s0.log`). Reading (HYPOTHESIS): the level-1 IDM head's R2 0.49 / 0.64 (`PROBE_ACTSENS_RESULT.json`, same 2,397 TRAIN windows) is a nonlinear, in-sample skill, and the linear B-CAL readout cannot separate state skill from pair skill; the MM's amendment should say which readout (episode-grouped, nonlinear, or the IDM head itself with held-out clips) the slope rule rests on.
  Files: `bcal/bcal_s0.json`, `bcal/bcal_s1.json`, `bcal/b_cal.py` (also `bcal/b_cal_mlp.py`, a diagnostic outside the card's rule; result in the report if it finished).
* Per the card the run STOPPED here: the MM freezes an amendment (slope rule `s_c`) BEFORE any run. Nothing was shipped to box 2, no pilot, no G-SUITE, no token (all of them bind the commit; an amendment that changes the trainer changes `a13f6f16`, so they must be redone on the amended commit).
* The box-2 package is built and dry-run (stub transport, fake box, stand-in trainer): `pkg/` (see `pkg/README_ACTRSP.md`). The real-commit tree export was tested: `ship_a13f6f16/` (42 members, needles pass, blob ids re-computed).

## Exact next step (after the MM freezes the amendment)
1. Land the amended trainer on ONE commit; set `ACTRSP_COMMIT` (40 hex).
2. If `--wm-rsp-w` argv text or any flag changes: regenerate argvs (`make_argvs_actrsp.py`), re-pin `ARGVS_SHA256` in `pkg/actrsp_check.py`, re-run `pkg/tests/test_actrsp_check.py` and `pkg/tests/test_actrsp_dryrun.sh`.
3. G-SUITE (full `stack/tests`, new commit vs `81cee6c1`, same environment, pod CPU or box-2 CPU; NOT the dev box): export both trees with `git -c core.autocrlf=false archive`, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=4` (the pod shows 128 cores; BLAS without a thread cap crawls).
4. Box 2: GPU idle was read 2026-11-16 (537 MiB, no python.exe); D: free 5.06 GB; no BOX2_BUSY marker. Launch the chain (ship, pilot) as in `pkg/README_ACTRSP.md`; the pilot stage runs the real `--selfcheck-only` gate of RSP_s0 and RSP_s2 and issues tokens only via `actrsp_check.py gate`.
5. Read the pilot (s/step from `PILOT.verified.json`, max_memory_allocated, `rsp/L1` trajectory) and decide the chain.

## Deletions log (own temporary data, literal paths)
* pod `/root/actrsp/` 40,063,059 B (two 20 MB latents copies, scripts, logs; sources F:/TanitAD-scratch/hwm_plan/latents_train_R35_s{0,1}.pt and `bcal/` here are intact): deleted.
* dev box `C:/Users/Admin/AppData/Local/Temp/actrsp_dry_run_dbg` and `actrsp_dry_run_dbg2` (kept dry-run trees): deleted, size not measured before the delete; the final dry run cleans its own folder.
