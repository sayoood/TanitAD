# HANDOFF -- the full 1,058-token C3 trunk pass (not run: projected 53 min to 5 h against a 15 min bar)

State (2026-10-13 ~11:30 Berlin): `eval/aux_navtest_forward.py` + `eval/test_aux_navtest_forward.py` are done and green (`ZZTEST_PASS 21`). The trunk path is verified on 32 C3 tokens:
scorer logits bit-identical to `raw/2026-10-11-c3-selection-headroom/table.npz` (32/32, 14,336 values, max |d| 0.0). Nothing of the 1,058 is cached yet except the 32-token smoke.

## Exact next step (ONE agent, no S1 head needed)
Fill the fp32 ctx cache for all 1,058 tokens in bounded chunks (each run holds the shared GPU lock about 10-20 min, then exits; **do not wait inside a run**: start it, END, relaunch from the result):

```
PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
L="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-13-aux-navtest-forward/launch.py"
C=D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full_ctx          # 1,058 x 7.9 MB = 7.75 GB; D: had 45 GB free at 99 % use
$PY "$L" D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full --ctx-cache $C --max-trunk-tokens 100 --heads H2-s0=D:/Projects/TanitAD-artifacts/aux_navtest_forward/fixture_fresh_head_seed0/aux_drv.pt
```
Re-run the SAME command until it prints `ZZAUXFWD_OK 1058` (each earlier run prints `ZZAUXFWD_PARTIAL chunk N tokens (identity PASS), M still uncached`, exit 0; roughly 11 runs). A chunk is resumable and crash-safe (logits are written before the ctx; an entry without both counts as a miss).
Exit codes: 0 ok/partial | 2 identity FAILED (stop and diagnose: re-run those tokens, c3_run groups ONE infer per token) | 4 refused | 5 GPU busy (nothing loaded; just relaunch).

How to check: `ls D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full_ctx | grep -c ^ctx_` (target 1058); `c3_full/progress.json` (done / remaining); `c3_full/identity_partial.json`. The final run writes `c3_full/identity.json` over ALL 1,058 (bar: bit-identical) and `head_pred_H2-s0.npz` for the FIXTURE head (never trained; plumbing).

## Then, when the S1 heads exist (arms H1-sK / H2-sK, files `aux_drv.pt` from `scorer_finetune.py --aux-drv`)
Head-only pass, no trunk, no nuPlan (still takes the GPU lock; add `--device cpu --no-gpu-lock` for a lock-free but not bit-comparable pass):
`$PY "$L" <out> --ctx-cache $C --heads H2-s0=<aux_drv.pt> H1-s0=<aux_drv.pt>` -> `<out>/head_pred_<ARM>.npz`, then the readout:
`$PY eval/aux_s1_readout.py perception --pred H2-s0=<out>/head_pred_H2-s0.npz H1-s0=... --labels-set navtest --dac-csv raw/p2_r2_instrument/baseline/u400_navtest_full.csv --c3-table raw/2026-10-11-c3-selection-headroom/table.npz --c3-tokens raw/2026-10-11-c3-selection-headroom/tokens.json --out p.json`
(the readout alone took 4 min of CPU on 32 tokens, almost all in label-store verification; budget for it.)

## Decisions this needs from the owners (not made here)
1. **Which ctx do the S1 heads get read on?** Planner fp32 (default, the only path the identity check vouches for) or the trainer's bf16 autocast (`--head-ctx trainer`, no cache, a second trunk pass). Measured on 32 tokens the two ctx differ by 1.05-1.46 % rel L2 (`smoke.json`).
2. **A never-trained-head control for P3** (see RESULT.md section 5): the fixture head scored pooled AUC 0.82 on 32 tokens of one log. Whether PREREG P3 should carry that control is the PREREG owner's call.

Files: code `eval/aux_navtest_forward.py`; tests `eval/test_aux_navtest_forward.py`; this dir; dev-box artifacts `D:/Projects/TanitAD-artifacts/aux_navtest_forward/` (single copy: fixture head, smoke outputs, 32-token ctx cache).
