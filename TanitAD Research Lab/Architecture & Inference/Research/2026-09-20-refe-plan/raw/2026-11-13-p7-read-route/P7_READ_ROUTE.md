# P7 READ ROUTE: runnable from a tip export in one detached chain (2026-11-12)

`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Source tip `924901b7` (mirror `C:/Users/Admin/tanitad-push`, `ls-remote` of `agent/arch-inf-20260803`). Audit read first: `PK/raw/2026-11-12-read-readiness-p7-p5c/READINESS_P7_P5C.md` (Q1, Q2, Q3, B2-B4). CPU only: no pod, no ssh, no GPU, no launch, no navtest score. Nothing staged, committed or pushed. Evidence class of every test line: MEASURED (logs below, scratch export, deleted at the end).

## What changed (hold: `files/<PK>/...` full files + `p7_read_route.diff`)

| # | item | file | change |
|---|---|---|---|
| 1 | PKG from `__file__` (gap 3) | `eval/stage2_sft.py:39` | `PKG = $TANITAD_REFE_PKG or <this file>/../..`. DATA paths untouched (`E:/Projects/TanitAD/data/refe_navtest`, nuplan splits, navsim root) |
| 1 | same | `eval/stage3_full.py` | `S2_RAW` was a second hard-coded E: PKG (where stage 2 writes the arm's families / `result.json`); now `PKG/raw/2026-10-05-sft-stage2` (the same place a tip-export stage 2 writes) |
| 1 | same | `eval/harness/score_e.py` | PKG derived from `__file__` (2 levels up) with the same env override. It is one of the 21 pinned harness files: `HARNESS_PINS.json` score_e `sha256_lf` 0b5d7bd2... -> 264d1135..., `bytes` 1557, `p7_read_route_2026_11_12` provenance record added (file CRLF kept) |
| 1 | pins moved | `eval/stage3_full.py` `STAGE2_SFT_MD5` | 3764778748d6... -> `3147254ba8128c5dde1b33dbd6cb2b81` (the pin and the file move as a pair). `harness_pins.py` NOT touched |
| 2 | B2 `p6` comparator | `eval/stage3_full.py` `BASELINES["p6"]` | csv md5 `2b8f7701ab35ac90380a272a8e48e39b` (CRLF, E: disk) / `9b1a6c38f95319849fa526b5f6ae0f35` (LF, git); seam `4733c3f843aba0cc9313d19b2c6426b3`; families `b9dcbf62dd2ddf93bda26c89970c0797`; PDMS 91.1965. MEASURED by `md5sum` of the files that made the P6 read. `--base p6` / `--also p6` are choices |
| 2 | P6 banked in-repo (it was a single copy on F:) | `raw/p2_r2_instrument/baseline/p6_navtest_full{.csv,_seam.npz,_seam.report.json,_families.json}` | copies of the E: csv (LF form), seam, report and `F:/TanitAD_p6read_work/stage3_p6_full/p6_full_families.json`, md5-equal to the pins (5 MB) |
| 2 | LATENT HOLE FIXED (found by the tests) | `BASELINES["u400"]`, `["deployed"]` | their pins held only the CRLF form of the csv and families file, so a tip export (LF) could not resolve them and only the E: working copy satisfied them. LF md5s added as tuple members (`4ade3a8b...`, `0ea39583...`; `dbcda5b6...`, `2e26a6c3...`; each LF text converted to CRLF hashes to the old pin). Also `selfcheck` line `failed_as_required` compared `!=` with a now-tuple pin (vacuously True); now `not in _md5_set(...)` |
| 3 | B3 `pick20` | `eval/refe_navtest_seam.py` | `--dump-proposals` npz gains `pick20` [N,20,3] float32 = `planner.executed(traj,k)`, the same array that feeds `poses = to_navsim(ex20)`; new `pick20_equals_seam(pick20, seam_poses)` (bit-for-bit `to_navsim(pick20)` == seam row) asserted at dump time. All 10 existing keys unchanged, same order; the seam npz itself unchanged; nothing runs with the dump off |
| 4 | B4 chain | `eval/p7_read_chain.py` (new) | see below |
| tests | | `eval/test_p7_read_route.py` (19), `eval/test_p7_read_chain.py` (16), both new | known-value controls and mutation arms |

**launch_gate:** MEASURED by grep over `refe/launch_gate.py` and `refe/p7_ship_list.json` at the tip: neither names `stage2_sft`, `stage3_full`, `refe_navtest_seam`, `score_e` or `HARNESS_PINS`; the ship list pins `eval/harness_pins.py` only, which is unchanged. So no launch_gate profile fingerprint moves (not re-run: another agent is editing `refe/launch_gate.py` and `refe/vdac_pod.sh`).

**Control limit of B3, stated:** `to_navsim` reads only 0.2 s grid indices 1,2,4,6,7,9,11,12,14,16,17,19. Indices 0,3,5,8,10,13,15,18 of `pick20` are not covered by the bit-for-bit control; they are covered by the wiring test (`pick20` and `poses` come from the same variable `ex20`).

## The chain `eval/p7_read_chain.py`

Generalised from `F:/TanitAD_p6read_work/chain_p6.py` (read, not modified). CLI `--ckpt --name [--summary | --expect-ckpt-sha256] [--base p6] [--also p2] [--ego-yaw-table|--no-ego-yaw-table] [--work] [--only|--skip] [--dry-run] [--detach]`, so a P5-C arm is `--name <arm> --ckpt <arm ckpt>`.

| step | what | accepted only if |
|---|---|---|
| preflight (always) | ckpt sha256 vs `summary.json` `model_final_sha256` (and `done: true`), yaw table sha prefix `f5ebe24f`, a yaw checkpoint without a table refused, tokens md5 `3a8b3653` + n 12146, `HP.require` (21 pinned files), `pick_baseline(base/also).pin_ok`, `$TANITAD_NAVSIM_ROOT` pinned (a different value is refused) | all |
| seam1 | `stage2_sft.py --dump-proposals --ego-yaw-table` (attempt 1, Deviation 1) | rc 0, or rc 1 with report rows 12146, misses 0, ckpt = this ckpt, rule navsim_v1, A7 on, frame-control max == 6.18063877792441 m (the declared C1 value), `pick20` present and equal to the seam; `result.json` records the yaw sha |
| score | the same command again (seam exists: harness + families only) | rc 0, `harness_status` PASS |
| stage3 | `stage3_full.py --base p6 --also p2 --expect-rule navsim_v1 --expect-ckpt-sha256` | rc 0 (gates ok) |
| adherence | `command_adherence.py read --dump ... --base p6 --also p2` | rc 0 |
| rep | `<name>_rep` seam only (no dump, rc 1 expected), compare to the first seam | tokens equal, poses bit-identical |
| families | seam -> `plans8` for the arm and for the base -> `transp_score.py` 2 shards, ONE pass for both arms, harness MetricCache -> `families_run.py --per-window` per arm -> `fam_ci.py` paired (base = A, arm = B, so the diff is arm - base) | each rc 0 |

Markers `<work>/markers/<step>.DONE` (JSON), `.FAILED` (traceback; renamed `.FAILED.attempt_N` on retry), `ident.json` (a resume under another ckpt / yaw / comparator is refused), `chain.DONE`; `<work>/CHAIN.log`; per-step logs in `<work>/logs/`. An existing seam without a `seam1` marker is refused (stage 2 would reuse a seam of unknown yaw). Children get `TANITAD_NAVSIM_ROOT`, `TANITAD_REFE_PKG` (the export's own PKG), `PYTHONPATH` (devkit + refe), `HF_HUB_DISABLE_XET=1`. The GPU lock is taken inside `stage2_sft.py`, never in the chain. Exit codes: 0 done, 1 a step failed, 2 refusal, 3 dry-run that would refuse.

## Launch command for P7's read (after the MM has landed the hold)

```
# 0. tip export on F: (read-only git; ls-remote first, and the tip must exist in the mirror)
TIP=$(git -C C:/Users/Admin/tanitad-push ls-remote origin refs/heads/agent/arch-inf-20260803 | cut -f1)
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $TIP "$PK/eval" "$PK/refe" "$PK/raw/2026-10-05-sft-stage2" "$PK/raw/p2_r2_instrument/baseline" "$PK/raw/p2_readout/p2_full" "$PK/raw/2026-11-12-instruments" taniteval stack/tanitad | tar -x -C F:/TanitAD_p7read_tip
# 1. fetch Sayood/tanitad-refe-p7-reserve-head run/model_final.pt + run/summary.json (runbook step 1; path UNVERIFIED until P7 pushes)
# 2. dry run (prints every command and every preflight check; exit 0 only if the preflight would pass)
C:/Users/Admin/venvs/tanitad/Scripts/python.exe "F:/TanitAD_p7read_tip/$PK/eval/p7_read_chain.py" --ckpt <F:/.../model_final.pt> --summary <F:/.../summary.json> --name p7_full --dry-run
# 3. the read, detached (same line without --dry-run, plus --detach)
C:/Users/Admin/venvs/tanitad/Scripts/python.exe "F:/TanitAD_p7read_tip/$PK/eval/p7_read_chain.py" --ckpt <F:/.../model_final.pt> --summary <F:/.../summary.json> --name p7_full --detach
```
Defaults: `--base p6 --also p2`, yaw table `E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtest.json.gz`, `TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim`, work `E:/Projects/TanitAD-artifacts/p7_full_read`. Never create `E:/Archive/devbox-C/navsim` (the root is pinned, so it cannot be picked by accident). Do not reuse a seam of unknown yaw. A reader seeing `INCOMPLETE (seam failed)` in `seam1` should not worry: it is the expected Deviation-1 failure and the chain checks it.

Verified on real data without any GPU: the real preflight PASSES on P6's own checkpoint (`--name p6_check --expect-ckpt-sha256 968a469d...`, dry-run, all 7 checks ok, 21 harness files, `baseline_p6` and `baseline_p2` pins hold); `stage3_full.py --arm-from p6 --base u400 --also p2 --no-families` (2 min) gives gates_ok True and P6 arm mean 91.19647788427798 (= level_ci.json to the bit), P6 - u400 = +2.9495 [+2.1872, +3.7669] (n 12,146, 136 logs; paired log-cluster bootstrap; answers "another draw of logs" only); `seam_to_plans8` on the real P6 seam writes [12146,8,3].

## Tests: pristine tip export vs patched export

| test (CPU, `CUDA_VISIBLE_DEVICES=-1`, tanitad venv unless stated) | pristine tip export | patched export | verdict |
|---|---|---|---|
| `eval/harness_pins.py --verify` | ok, 21 files, caller NT_FIX1 | ok, 21 files, caller NT_FIX1 (score_e pin moved, recorded) | SAME |
| `eval/test_p6_read_gaps.py` | 35/35 | 35/35 | SAME |
| `eval/selftest_stage3_full.py` | 11/11 | 11/11 | SAME |
| `eval/test_command_adherence.py` | 24/24 | 24/24 | SAME (it FAILED 2/24 mid-way in the patched export until the LF pins of u400 / deployed were added: the latent hole above, FIXED) |
| `eval/test_p7_read_yaw.py` | 5/5 | 5/5 | SAME |
| `eval/test_teacher_seam_archive_root.py` | 6 tests OK | 6 tests OK | SAME |
| `refe/test_ego_yaw.py` (reads harness_pins) | ZZEGOYAW_PASS 187 | ZZEGOYAW_PASS 187 | SAME |
| `eval/stage3_full.py --selfcheck` | checks_ok False: only `families6_reproduces_banked_u400` (families6 "no output" in the scratch export, ~9 s); the 4 regression arms fail as required | identical: same single failing check, same 4 regressions | SAME (pre-existing, not caused here; the 4 known-regression arms still fail as required) |
| `refe/test_p6_planner_command.py` (driverl-eval venv) | rc 1: `D:/Archive/.../navtest_inputs.json.gz` absent | same rc 1, same cause | SAME (environment) |
| `eval/test_p7_read_route.py` (new) | absent | 19/19 | FIXED (new coverage: PKG x4 incl. 3 RED, env override, no-hardcode grep, md5 pin, p6 choice/pins/level/bank-only/all-comparators-in-repo/RED/distinct, pick20 control/RED/AST wiring/RED/default-off) |
| `eval/test_p7_read_chain.py` (new) | absent | 16/16 | FIXED (new coverage: dry-run plan, 12 preflight refusals, resume, failure+retry, seam acceptance rules, unknown-seam refusal, identity refusal, determinism rep, only/skip, plans8, detach, 5 mutation arms) |

**REGRESSION: 0.** The pristine export gave 35/35, 11/11, 24/24, 5/5, 6 OK, 187 on the same machine the same day; the patched export equals it on every one. Logs: `logs/pristine_tip/`, `logs/patched/`. (The two new test files read the real `E:/Projects/TanitAD/data/refe_navtest` and `F:/TanitAD_p6read_work` files for the p6 pins; set `REFE_NAVTEST_DIR` to move the data dir.)

## What stays open (reported only)

1. **End to end is UNVERIFIED**: no P7 checkpoint, no GPU, so `seam1` through `fam_ci` have run only against a fake runner (35 chain/route tests) plus the real preflight and the real `stage3` self-comparison above. The `transp_score` / `families_run` commands are the ones `INSTRUMENTS.md` documents (navsim-crun venv, `;`-joined PYTHONPATH); first real use is the first real run. ESTIMATED wall time 5.7 h on 2 processes (INSTRUMENTS.md).
2. **`pick20` is dumped but the Q3 route feeds `plans8`** (P6's base has no `pick20`; the arm and the base must share a form). The transparent frame is therefore the dt 0.5 form (about 1.5 cm coarser, reported not guarded). The registered P5-C form needs `pick20` for BOTH arms: the arm gets it from the new dump, the P6 base does not.
3. **`families6.py` (stage 2's families) still reads the E: `taniteval` adapter** (`ev6e/families6.py:34-35`, a pinned harness file). Audit: default-identical to the tip; left alone. `families_run.py` uses `--tanitad-repo` = the export root, so its adapter is the export's.
4. **Head endpoints** (DAC-zero share vs P6 from the csv is available; head IoU [0,40) m, head-vs-scorer DAC AUC NOT readable): `aux_navtest_forward.py` loads a standalone `aux_drv` file and P7's head is a state dict inside `model_final.pt`; the aux label store refuses navtest, so navtest drivable GT does not exist. Reported, not gating; ESTIMATED 150 lines; the navtest GT source is an MM question.
5. Stage 3's family comparison is unpaired and conservative; the paired family CIs and distance keeping come from the `fam_ci` step. A separated CI is necessary, not sufficient (`H-ESTIM-SEED-1`).
6. `refe/test_p6_planner_command.py` fails on the PRISTINE tip too (`D:/Archive/...navtest_inputs.json.gz` absent here); it is an environment fail, unchanged by this hold.
7. The E: tree is still stale and is not used: the chain runs the export's own code.

## Manifest (all in the hold `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_read_route/`; single copy until the MM lands them)

`P7_READ_ROUTE.md`, `p7_read_route.diff`, `files/<PK>/eval/{stage2_sft,stage3_full,refe_navtest_seam,p7_read_chain,test_p7_read_route,test_p7_read_chain}.py`, `files/<PK>/eval/harness/{score_e.py,HARNESS_PINS.json}`, `files/<PK>/raw/p2_r2_instrument/baseline/p6_navtest_full*` (4 files), `logs/` (test logs of both exports). Land together: the stage2 md5 pin, `score_e.py` and its pin move as pairs.

## Deletion log (CLAUDE.md rule 4)

Deleted by literal path `F:/tanitad_scratch/p7rr` (pristine tip export, patched export, test logs, s3out, work_dry): 819 MB (MEASURED `du -sh` before the delete). Re-creatable (git export of `924901b7` + the hold); the deliverables and test logs were `cmp`-verified in the hold first. Not deleted: small `p7chain_mod_*` temp dirs of the chain tests under `C:/Users/Admin/AppData/Local/Temp` (tens of KB each).
