# HANDOFF - PREREG_V8F_KINDX seed-1 confirmation (X_s1): BUILT, TESTED, NOT LAUNCHED

Built and tested against tip `d3d314c7fd13e1c3a941720d9417d188cc2effc9` (FLAT's `o_interface` lever included). Nothing was run on box 2,
nothing was written into `C:/Users/Admin/box2_ship/` (KSCORE's busy marker there is untouched), no git add/commit/push.

## What it does (one line per stage; `chain_v8f_kindx_s1.sh` header has the full text)
wait for KSCORE's chain -> refresh box 2 from the REMOTE tip, sha-check key files -> B's two banked checkpoint sha256 -> pre-flight (argv diff,
B s1 evals staged, `kindx_read.py --seed 1 --check-inputs`) -> `gatemixo` + `assert-gate` + token -> `trainmixo 1` (MIXO's argv, only `--seed 1`)
-> sha256 + `assert-run` -> `evalmixo 1` (7 evals) + the 10 KINDX evals in one process -> pull -> `kindx_controls.py --seed 1 1|2` ->
`kindx_read.py --seed 1` -> `READ_s1.json` (CARRIER / NOT A CARRIER; the PREREG's CANDIDATE row applied verbatim) -> `ZZV8FMINI-DONE|-STOP`.
Outputs: `C:/Users/Admin/box2_ship/logs/v8f_kindx_s1/` (READ_s1.json, controls_s1.json, ...), log `logs/chain_v8f_kindx_s1.out`.

## BEFORE LAUNCH (the coordinator's steps; I could not do them)
1. **LAND the 5 files (see FILES.txt) on `agent/arch-inf-20260803` and push.** The chain refreshes box 2 from the REMOTE tip and stops with
   `the tip has no stack/scripts/v8f_mini_eval_kindx.py` (or the modified reader) otherwise. `git ls-remote` must equal the landed sha.
2. Tip export (fresh dir, no autocrlf): `mkdir C:/Users/Admin/box2_ship/v8fkindx_tip_<TS>` then
   `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive --format=tar <tip> stack taniteval "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini" | tar -x -C <dir>`
3. Write `C:/Users/Admin/box2_ship/run_chain_v8f_kindx_s1.cmd` (CRLF; this exact text, `<TS>` filled). `--dry-run` through this very form was tested:
```
@echo off
"C:\Program Files\Git\bin\bash.exe" -lc "cd /c/Users/Admin/box2_ship/v8fkindx_tip_<TS> && bash './TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/box2_ops/chain_v8f_kindx_s1.sh' < /dev/null" < NUL > C:\Users\Admin\box2_ship\logs\chain_v8f_kindx_s1.out 2>&1
```
   Default predecessor = `logs/chain_v8f_kscore.out` + `logs/v8fkscore_guard.out` (override with `AFTER_OUT=... AFTER_GUARD=...` before `bash`).
   Launch detached (WMI, `< NUL`) per the memory note; **only on the PI's go for box 2** (it is his machine; KSCORE currently holds it).
4. Optional last look, no side effects: add `--dry-run` after the .sh path (reads the remote tip's queue unless `V8F_QUEUE=<file>` is set).

## Dry run (printed, d3d314c7 queue; identical through the .cmd wrapper)
`== predecessor, checkpoint sha256 (B s0 601cb435..., B s1 6104fb15..., X_s0 c6a62876... must differ), gatemixo (--code-mix kind once, no --seed, no --o-interface),
trainmixo 0, trainmixo 1 (--seed 1 ... M2b_s1_mixo), evalmixo 1 (7 lines), argv diff: exactly the seed (and the output directory that carries it), kindx evals (job line + 10 argvs), read`;
`== stages`: `wait(KSCORE) refresh(tip) ckpt-sha(B) pre-flight(argv diff, B s1 evals, check-inputs) gatemixo(+assert-gate,+token) trainmixo(1) ckpt-sha(X) assert-run evalmixo(1) evals(kindx) ckpt-sha(X, again) pull controls read DONE`.

## Estimated wall-clock after KSCORE ends: ~4.5-5 h
refresh + checks ~5 min, gate ~5 min, train ~4.0-4.5 h (MIXO s0: 4.0 h; guard cap 10 h, wait cap 11 h), evalmixo ~4 min, KINDX evals ~6 min, pulls + controls + read ~5 min.

## Tests: `stack/tests/test_v8f_kindx_s1.py`, 188 passed, 0 skipped, ~56 s, CPU
`CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<export>/stack;<export>/taniteval <venv python> -m pytest stack/tests/test_v8f_kindx_s1.py --basetemp C:/Users/Admin/pt_kindx_s1/<new dir>`
(use WINDOWS-style paths in PYTHONPATH; the venv's editable `tanitad` points at G:). 11 mutation tests, all detect their mutation (RED when the mutation is applied).
The tests found one real hole in my first argv check (a `--out` identical to X_s0's directory passed the diff, i.e. a seed-1 run could overwrite X_s0): fixed in the driver.
The stop table is checked against KSCORE's chain SOURCE (every `stop "..."` text, 24 + 3) and the running KSCORE chain file equals the tree's.

## Open risks
- `gatemixo` re-uses `gate_M2b-mixo` on box 2 and overwrites the s0 gate dir there (the s0 gate is banked in the repo; `assert-gate` compares against the banked copy).
- The predicate's PREFLIGHT class (KSCORE stopped on the dev box before box 2 ran anything -> PROCEED) goes one step beyond "run / verdict / gate"; every box-level or unknown text HOLDs.
- NOT A CARRIER has no committed next step in the PREREG (only the CANDIDATE row is verbatim); the reader flags its NOT-A-CARRIER text as its own reading, nearest committed text = KIND's MIXED row.
- `assert-gate`'s same-model check compares X's BUILT model with MIXO's banked s0 gate apart from `o_interface=full`; if the tip changes another lever it stops at the gate (box idle, cheap).
- The driver is named `v8f_mini_eval_kindx.py` so the guard that is already running (`v8f_mini_eval` in its kill regex) stops it on a thermal event.
- s1 result is T0 tier (teacher-forced oracle), one extra seed: separated-CI is necessary, not sufficient (H-ESTIM-SEED-1); the bootstrap here is over episodes.
