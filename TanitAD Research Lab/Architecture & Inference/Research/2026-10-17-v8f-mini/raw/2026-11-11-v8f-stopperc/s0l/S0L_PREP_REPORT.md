# S0-L prep report (2026-11-13): B-prime s0 T1-kin dumps and STOPDEC, without L

**Headline: no decision needed.** Both L-free work items are done and the README step-5 gate passes. Nothing about L was computed. `s0l_read.py read` was not run; no pinned file was edited.

## What ran (all CPU, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`)
| step | result | evidence |
|---|---|---|
| `run_dumps_s0l.py` written (tags `BPRIME_s0`, `L_s0` only) | refuses any other tag, a missing/unreadable ckpt, cache or input, a sha mismatch; `L_s0` refused until env `S0L_L_CKPT`, `S0L_L_CACHE`, `S0L_L_SHA256` are set and the cache exists | `s0l/run_dumps_s0l.py` |
| tests `test_run_dumps_s0l.py` | 6/6 PASS. Deliberate regression (keep the baseline `--eval-cache`) made the argv test RED; file restored, sha256 `2779fe97...a273` identical before and after | MEASURED, run in this session |
| B-prime checkpoint sha256 | `b9aaf33e458c4da7a4e38becae905fbdd20b35b7dccf8858c973791e7e61c1a7`; matches the INHERITED `b9aaf33e` prefix (LOOP_STATE:544). No mismatch | MEASURED, `lorapod_eval/bprime/M2b_s0/ckpt_last.pt` (86,970,645 B) |
| T1-kin dumps `BPRIME_s0` | rc 0, 141 episode dumps, ticks file, done file; wall 401.7 s | `F:/TanitAD-scratch/s0l_read/t1/` (`done_BPRIME_s0_m10_causal_heads.txt`) |
| STOPDEC `BPRIME_s0` | rc 0, ~7 min | `s0l/out/stopdec/stopdec_BPRIME_s0.npz`, `STOPDEC_BPRIME_s0.json` |

`lorapod_eval` lives at `E:/Projects/TanitAD-artifacts/lorapod_eval` (not on D: or F:). The B-prime cache root is that folder (`eval/clips`, meta pipeline `ra1a-pooled-v1`). `lorapod_eval/native_tracks.pt` is byte-identical (`cmp`) to the baseline's `sfresh_inputs/inputs/native_tracks.pt`, so `--tracks` stays the baseline's.

**Argv:** equal to the DAG-OACC baseline except `--ckpt`, `--tag`, `--out-dir`, `--eval-cache`, and `--dump-ticks`. The last one is derived from `--out-dir` (`OUT + "/ticks"`) in the pinned driver, so it moves with it. The test asserts exactly that set.

**Tree choice: STOPBAL tree `F:/TanitAD-scratch/stopbal_read/tree` (715ef8fc).** Reasons: (a) it is `stopdec_arm.py`'s native tree and the one that made the parity file; (b) the two trees' `v8f_mini.py` differ (md5 `f123ba68...` vs `1a6b8e37...`), and the B-prime ckpt loads with `strict` (0 missing, 0 unexpected) on BOTH (probe, MEASURED), so the load does not decide it; (c) L must later be dumped in the same tree. Not tested: whether the dumps would differ between the two trees.

## Gate (README step 5), MEASURED against `stopbal/read/...` `stopdec/out/stopdec_PARITY_DAGOACC_s0.npz`
* Model-free arrays `clip_index, ws, eid, has, stop_hit, v_end_gt, v2_gt`: all 7 `array_equal` = True. PASS.
* Counts: windows 2,399; `has` 2,365; episodes 139; STOPCUE positives 131; STOP_POINT group 117 (STOPDEC.json `group_counts_has`: also REG_STOP 137, REG_DECEL 487). PASS (2,399 / 2,365 / 117 / 131).
* Own-plan identity controls (README step 6): `own_all_vs_ident_max_abs_m` 0.0, `top1_vs_dump_ol_max_abs_m` 0.0, `g_vs_dump`, `v0_vs_dump`, `ident_vs_top1` all 0.0; goal_prob in [0,1]. PASS. (`manifest_ok` is null: no BAL bank manifest exists for this arm; the sha is checked by the driver instead.)
* Note, UNVERIFIED cause: `cons_gt` and `dec_goal_stop` differ from the parity file. Neither is in `s0l_read.py` `MODEL_FREE`; `dec_goal_stop` is model-derived, `cons_gt` was not traced. Other label-like arrays (`gt_lon_cls`, `strat_*`, `tl_hit`, `v0`) are equal.

## CONTROL number only (not an outcome)
B-prime s0 STOPCUE AUROC (p(STOP_POINT) = `goal_prob[:,6]` vs `stop_hit`, `has` windows, n 2,365, 131 positives, via `stopperc_s1_read.auroc_fast`): **0.8561**, tier T1-kin first tick, eval-141. For reference, the same statistic on the parity arm's file reads 0.8357 (control, not a gate), and the B-prime mini-eval (different tier) read 0.8748 (README_S0L). S0-L compares L against this, and L does not exist yet. No CI computed here.

## What remains when L_s0 is banked (README steps 1-3, 6-9)
1. Confirm bank (HF ledger, sha256); `check_l_ckpt.py --expect-sha256` rc 0.
2. Step 3: build the L-merged eval cache (GPU, re-encode eval-141 with pipeline `ra1a-pooled-v1`, bf16, `local256`; LoRA=0 reproduction check; venue decision still open, README section 5).
3. `S0L_L_CKPT=... S0L_L_CACHE=... S0L_L_SHA256=... python -I run_dumps_s0l.py L_s0` with the same tree and env as above (`job_s0l_bprime.cmd` is the template; set `STOPBAL_DUMPS`/`S0L_DUMPS` to the same `t1` folder), then `stopdec_arm.py --tags L_s0 --outdir s0l/out/stopdec --split-json`.
4. Check L's identity controls read 0.0, model-free arrays equal these; then `s0l_read.py read --L stopdec_L_s0.npz --B stopdec_BPRIME_s0.npz`, controls, register.
Dumps on F: use ~24 MB; F: had ~8 GB free at start, and free RAM fell to ~2.6 GB during STOPDEC (other sessions share the box).

## Deliverable manifest (nothing committed or pushed)
| artifact | where | copies |
|---|---|---|
| `run_dumps_s0l.py`, `test_run_dumps_s0l.py`, `job_s0l_bprime.cmd`, `S0L_PREP_REPORT.md` | `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0l/` | ONE place (dagland is outside git) |
| `stopdec_BPRIME_s0.npz`, `STOPDEC_BPRIME_s0.json`, `STOPDEC.json` | `.../s0l/out/stopdec/` | ONE place (re-creatable in ~12 min) |
| `run_s0l_bprime.log`, `S0L_BPRIME.done`, `DUMPS_S0L_<stamp>.json` | `.../s0l/` | ONE place |
| T1-kin dumps (141 episodes, ticks, argv, log, done file, 24 MB) | `F:/TanitAD-scratch/s0l_read/t1/` | ONE place; kept as the STOPDEC input and for L's later same-environment check |

## Deletions (rule 4)
* `F:/TanitAD-scratch/s0l_orig_test.py`, 5,138 B (my own backup copy of the test, not needed after the byte-identity check). Nothing else deleted. A probe script remains in the session scratchpad (`.../scratchpad/probe.py`, tiny).

## MM review (2026-11-13 ~23:00)
* Re-checked: `test_run_dumps_s0l.py` + `test_s0l_read.py` 19 passed; B-prime `ckpt_last.pt` sha256 `b9aaf33e458c4da7...c1a7` (MEASURED, matches).
* Gate re-computed key by key against `theory/stopdec/out/stopdec_PARITY_DAGOACC_s0.npz`: all 7 MODEL_FREE arrays are equal (`has` 2,365, `stop_hit` 131), and so are `gt_lon_cls`, `strat_*`, `tl_hit` and `v0`. The arrays that differ are all model outputs (`dec_*`, `goal_prob`, `lon_prob`, `lon_argmax`, `err_mps`), as they must between two models.
* Resolves the report's UNVERIFIED `cons_gt` note: `cons_gt` is EQUAL under a NaN-aware comparison. It holds 36,340 NaNs out of 62,374 entries, and NaN != NaN made a plain comparison read "differs". `dec_goal_stop` is the model's stop decision, so it differs by construction.
* B-prime s0 STOPCUE AUROC 0.8561 (T1-kin) is a control value only. S0-L compares L with it once L_s0 is banked (Amendment 5).
