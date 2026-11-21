# HANDOFF: PREREG_V8F_KINDX (2026-10-25 ~14:00). STOPPED BY THE RESOURCE GUARD BEFORE EVAL 1. NO KINDX NUMBER EXISTS.

OUT = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/kindx`. Nothing staged, committed or pushed. box2_ship only read.

## Why stopped
Commit headroom (Win32_OperatingSystem.FreeVirtualMemory) was 9.5 GB at start, then 0.8-2.5 GB for 2+ min (rechecked 8x over 2 min, and 3 more times): other jobs' python processes
(5.1 + 3x2.3 GB private) hold it. Rule: need C: >= 1.5 GB free (ok, 3.4-4.6 GB) AND commit headroom >= 4 GB. `guard.ps1` prints OK/FAIL; `run_one.sh` runs it before every eval (exit 3 on FAIL).

## Done (all read-only on inputs, zero eval run)
* PREREG + KIND prereg/result/reader/queue read. X sha256 asserted = `c6a62876...126250d` (matches).
* `code/` = `git archive eeb2043e` export (stack, taniteval minus results, v8f-mini code + eval grid/kin/join, strata npz, K=10 and K=30 lead blocks). Inputs for the eval: `sfresh/` (cache, v9 release, lead, speed, P10, tracks; md5-verified vs box 2 by SFRESH).
* `b_kind/` = B s0's 15 banked KIND npz + `kind_read.json` (from tip), for the beside shift read.
* Scripts (syntax-checked, NOT run): `kindx_eval.py` (one eval via the tip's own `v8f_mini_eval.py run`, CPU, 6 threads, sha re-asserted, tanitad import path asserted under `code/`),
  `run_one.sh <name>` (guard + eval), `kindx_controls.py 1|2`, `kindx_read.py [--selftest-b|--no-beside]`.

## Declared interpretations (made before any data)
1. Own code = `--tac pred --project --families` (= the banked standard eval E0's flags). Oracle arms = KIND's lines: `--tac oracle --tac-groups <kind>|constraints|all --tac-alpha 1`.
2. IDENTITY ORACLE: `v8f_mini_eval` refuses alpha 0 and has no empty group spec, so the driver patches ONE thing: `M.code_group_index('none') -> []` (empty selection => `where(sel, code, code0)` = code0).
   Run as name `none`. If `sel[[]] = True` raises in torch, replace the patch by an empty LongTensor index; do not alter anything else.
3. PREREG gap: its beside table names necessity_k(X) but its 10-eval list has no leave-one-out arm. Do the 10 first; run the 6 `loo-<k>` evals (`run_one.sh loo-speed` ...) only AFTER READ.json holds the outcome,
   and label them an addendum. `kindx_read.py` picks them up if all six exist, else prints NOT COMPUTED.

## Exact next steps (when `guard.ps1` says OK; ~30 s per eval measured on SFRESH's identical CPU eval)
```
cd $OUT ; bash run_one.sh own ; bash run_one.sh none ; bash run_one.sh all ; bash run_one.sh union
python -I kindx_controls.py 1        # exit 2 => STOP, report, read no gain
for k in speed angle tte timegap geom leadgap; do bash run_one.sh $k; done     # tags ...-kindx-only-<k>
python -I kindx_controls.py 2
python -I kindx_read.py --selftest-b # must PASS (reproduces KIND's s0 gains) before the real read
python -I kindx_read.py              # writes READ.json (deciding block first, then beside)
```
Then write `KINDX_RESULT.md` (outcome first, controls with values, gain_k(X) with CI, the committed next step quoted from PREREG, union / A_X / shift vs B, families; T0; one seed: no lever claim).
Use `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`. Do not edit bar 0.2415, kinds, cells, estimator or routing after any number is seen. If a script has a bug, fix it BEFORE running on X data.
