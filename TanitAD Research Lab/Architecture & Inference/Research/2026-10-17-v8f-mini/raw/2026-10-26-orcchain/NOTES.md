# O-RECOVER box-2 chain: build + dry-run only (CPU). Nothing launched, shipped, staged, committed or pushed. Base = mirror 98713259. Updated for PREREG Amendment 3.

## Amendment 3 applied (Master Mind, pre-data): trainer guard UNCHANGED; ORC-0 names ONLY `--ego-perturb-p 0` from the ego-perturb family
ORC keeps the five law flags (vscale 0.5 1.0, vmax 8.0, adrop 1.5, yawrate-std 2.0, seed = the M2b seed). ORC argv == ORC-0 argv + those five flags, with p 0.3 vs 0 and --out differing.
At p = 0 the law is inert and vmax's default 8.0 is ORC's, so the eligible denominator is identical. `ego_problems_p0.diff` was DELETED (not wanted).
- queue: `:orcarm` sets `ORCL` (the five flags) for ORC only; `ORCF = ... --ego-perturb-p %ORCP%%ORCL%`. ORC-0 lines END with `--ego-perturb-p 0`. ORC token hashes unchanged; ORC-0's are new (out/chain_dry_run.out).
- chain gate assertions: ORC-0's BUILT law values must be the trainer's DEFAULTS (yawrate-std 0.0, seed 0, ...); ORC-0's queue lines may NAME none of the five (refused by name, any value); ORC must name all five.
- MEASURED on the real trainer (synthetic data, unpatched): the ORC-0 gates PASS at both seeds (probe `off`). Scope, honestly: the trainer's guard compares VALUES with the defaults, so it refuses a law flag on
  ORC-0 only when its value is not the default (yawrate-std 2.0, seed 1, vscale 0.6, vmax 6, adrop 1.0 are refused); `vscale 0.5 1.0`, `vmax 8.0`, `adrop 1.5`, `seed 0` named on ORC-0 PASS it, so the CHAIN refuses the naming.
- ⚠ OPEN, not mine: the reader's `orc_read.g_pair` (raw/2026-10-26-orcread/orc_read.py L518) still requires ORC and ORC-0 argv identical except p/out, so it will REFUSE the Amendment 3 pair
  (differs in the five law flags). Its owner must drop the five flags from the compared set and require ORC-0 to name none. `g_argv`, `g_token`, `g_pert` are unaffected. I did not touch the reader.

## What is here (`files/` = full content at repo paths; `out/*.diff` = vs the 98713259 blobs, CR stripped)
- queue `v8f_mini_box2_queue.cmd` (CRLF): `gateorc|trainorc|evalorc ARM SEED` + helper `:orcarm`; ARM in {orc, orc0}, SEED in {0,1}, else REFUSED rc 2. 79 older labels byte-identical, 41 other stages' dry-run output identical.
- chain `box2_ops/chain_v8f_orc.sh` (LF, `bash -n` OK): `--dry-run` (out/chain_dry_run.out) prints 12 exact commands, the 2 asserted ckpt sha256, 4 token argv_sha256 (gate==train argv: SAME), token schema; ships nothing.
- `stack/tests/test_v8f_orc_chain.py` (new) + 5 older queue-pin tests WIDENED for the 4 new labels (kind, mixo, flat, lora_ar2, oacc; pins only, see `out/widened_*.diff`). No xfail left: the old blocker test is now `test_orc0_gates_pass_*`.
- `tools/`: apply_queue_edit.py (queue, incl. Amendment 3), apply_amendment3_tests.py, widen_pins.py, gen_pins.py, mutate_orcchain.py, orc0_relax.py (the pre-Amendment-3 blocker repro), run scripts.
- Reader note: for `orc_read.py` on a real summary.json pass `--key-built-p ego_perturb.ego_perturb_p`.

## Results (CPU, CUDA_VISIBLE_DEVICES=-1, basetemp under E:; pytest cap respected)
- new test file `out/orc_tests_full3.out`: 45 passed, 0 xfailed, rc 0 (the 4 ORC-0 gates pass on the unpatched trainer; 10 non-default law flags refused by it; 4 default-valued ones pass it and are refused by the chain).
- 14-file queue suite `out/queue_tests_final3.out`: 3 failed, 83 passed, 2 skipped. Pristine 98713259 (`out/queue_tests_base.out`): the SAME 3 failures (kindx_s1 trainmixo-lines, mixo real-lines x2), 79 passed, 6 skipped. None new.
- mutation proof `out/mutation.out`: CONTROL GREEN (45 passed), SUMMARY 37 of 37 mutants RED (queue + chain, each ONE edit of a copy, suite runs -x; the law-flag mutants Q11-Q14, C20-C23 included).
  Run by `tools/run_a3_final.sh` (waits for the pytest cap).
## Relaunch (when the PI arms it; the trainer needs no change now)
0. Land first (the chain refreshes box 2 from the REMOTE tip and checks it carries the lever and the 3 queue stages): queue + 5 widened tests + new test (suggested `raw/2026-10-26-orcchain/`, chain beside chain_v8f_orc.sh). Update the reader's g_pair first (above) or its pair gate refuses the data.
1. `cp chain_v8f_orc.sh /c/Users/Admin/box2_ship/` (LF; Git Bash)
2. `AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_oacc.out nohup bash /c/Users/Admin/box2_ship/chain_v8f_orc.sh > /c/Users/Admin/box2_ship/logs/chain_v8f_orc.out 2>&1 &`
WAITS ON `^ZZV8FMINI-DONE` in AFTER_OUT (OACC's log; stops on `^ZZV8FMINI-STOP` there). AFTER_OUT has NO default: unset = refused, nothing touched
(MIXO's log already holds an end marker; an old log would satisfy the wait at once). Check first: `bash chain_v8f_orc.sh --dry-run`.
Runs: ORC s0, ORC-0 s0, ORC s1, ORC-0 s1, always all four, after ALL FOUR gates; markers ZZV8FMINI-DONE/-STOP in chain_v8f_orc.out. Pulls to E:/Projects/TanitAD-artifacts/v8f-orecover/<run>/ (sha256 printed).
NOT in the chain: T1-kin rollouts and the reader (dev box). The token is written by the chain, never by hand; it binds the TRAINING argv.

## Rebased and re-verified by the Master Mind before landing (2026-10-26 ~12:10, tip 04d56671 = OACC Amendment 2)
- The queue was unchanged on the tip since 98713259, so the agent's edit lands as built. The five pin widenings were re-derived on the tip's bytes by `widen_pins.py` (`mk_hold_orc.py`); kind / flat / lora_ar2 equal the agent's files byte for byte, oacc / mixo carry the tip's later edits plus the same widening.
- Verification tree = a full archive of the tip with this hold overlaid (`E:/Projects/TanitAD-artifacts/oc/tree`, short path on purpose), run by `run_oc.sh`:
  `out/rebased_04d56671_orc_chain_tests.out` 45 passed; `out/rebased_04d56671_orc_reader_tests.out` 53 passed (reader at Amendment 3); `out/rebased_04d56671_queue_tests.out` the 14-file queue suite **88 passed, 0 failed**.
- The 3 "pre-existing" queue failures above were the ENVIRONMENT: a missing raw file in the partial export (kindx) and cmd.exe's 8,191-char line limit on the long `orcchain_build/tree0` path (mixo x2). On a full, short-path tree they pass.
- `g_pair` (the open item above) is closed by the reader's Amendment 3 (`raw/2026-10-26-orcread/apply_a3.py`).
- The 37-mutant proof ran on the agent's tree (98713259 base, pre-A3 reader); it was not re-run on the rebased tree.
