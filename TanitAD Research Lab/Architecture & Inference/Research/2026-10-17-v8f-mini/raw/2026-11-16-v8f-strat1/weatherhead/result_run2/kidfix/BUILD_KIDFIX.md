# BUILD_KIDFIX: K-ID read 1.1e-5 on H_s0 / H_s1, cause and fix (CPU only, no bar changed)

## Cause (MEASURED)
The two paths differ in ONE thing: the window batch size. Both ran under the same `child_env` (OMP/MKL threads 2, CPU, fp32), same checkpoint, same 2,399 windows, same fp8 feature tensors.
- Evaluator: `v8f_mini_eval.py:304` loops `for b0 in range(0, N, a.batch)`, `--batch` default 64 (`:672`); the T0 argv carries no `--batch`.
- Fit-side K-ID: `wh_fit.py cmd_kid` called `W.predict(...)` whose default is `bs=BS=256` (`wh_lib.py:48,415`).
A different batch changes the CPU matmul/attention reduction order: fp32 round-off, not a wiring difference.
Evidence (`kidfix/diag.py`, threads 2, same model, compared with the banked eval npz):
| arm | batch | max abs diff | windows > 1e-5 |
|---|---|---|---|
| H_s0 | 256 | 1.144e-05 (reproduces the read) | 2 of 2,399 |
| H_s0 | 64 | 0.0 (bit-identical) | 0 |
| H_s1 | 256 | 1.335e-05 (reproduces the read) | 4 of 2,399 |
| H_s1 | 64 | 0.0 (bit-identical) | 0 |
Median per-window max diff at batch 256 is 1.9e-06; the worst columns sit at |logit| 15.7 to 16.3 (logit absmax 19.5), i.e. ~1 ulp-scale noise at large magnitude. The G-EVAL gate subset (a smoke of few windows, one batch) could not show it.
Same pooled input: the identical `StreamSource` feature tensors feed `ctx_pool_fit_logits` vs `forward_infer`'s `_ctx_pool` (`v8f_mini.py:1140`); with batch matched the outputs are bit-equal, so no upstream or wiring difference exists.

## Fix (kid side only; pinned evaluator and atol 1e-5 untouched)
`build/wh_fit.py` (diff in `kidfix/wh_fit_kidfix.diff`): new `kid --batch` (default 64 = the evaluator's batch) passed as `bs=a.batch` to `W.predict`. `W.predict` / `BS=256` (used by the fit loop) unchanged.
Mutation: swapping logit columns 0 and 1 of the new kid output gives K-ID max_abs_diff 18.105 (H_s0) and 18.975 (H_s1), `pass: False`. Fixed outputs read `0.0`, `pass: True`.

## Re-run
Only `wh_fit.py kid` for H_s0 and H_s1, then `wh_chain.py read --runs .../run2`. `run2/instrument.json`: all checks pass incl. `K-ID: pass`. `run2/read/WH_READ.json` outcomes: country POOL-GAIN, weather WH-DETECTS + WH-INPUT-FEATURES, surface POOL-REPLACES, illumination POOL-REPLACES, period_of_day POOL-REPLACES, driving NO-COST-BY-IDENTITY (no INSTRUMENT-FAIL). No fit, T0 or T1 re-run.

## Integration note for the MM
`wh_fit.py` is in `BUILD_FILES` of `wh_check.py:41`; its sha changed, so any gate token minted on the old sha is stale for a new launch (the run2 arms are finished). Land the patched `build/wh_fit.py`. Consider a G-EVAL addition that compares at the evaluator's batch over more than one batch (instrument-design note, not done here).

## Manifest
| artifact | where |
|---|---|
| patched `build/wh_fit.py` | F:/TanitAD-scratch/strat1/whead/build/wh_fit.py (staged nowhere; not in git) |
| diff | F:/TanitAD-scratch/strat1/whead/kidfix/wh_fit_kidfix.diff |
| old script / old outputs / old read | kidfix/wh_fit.py.before_kidfix, kidfix/old_kid_bs256/, kidfix/WH_READ.before_kidfix.json |
| diag + mutation scripts | kidfix/diag.py, kidfix/mut.py |
| new kid npz, re-read | run2/kid/kid_H_s{0,1}.npz, run2/read/WH_READ.json, run2/instrument.json |
## Deletions
tmp/*.npy (4 files, ~0.7 MB each) and tmp/mut_H_s*.npz, by literal path. Kept: the `.before_kidfix` files and `old_kid_bs256/` (evidence, under 2 MB); the MM may delete once landed.
