# HANDOFF - PREREG_V8F_KSCORE instruments (built against tip 4a59a711; nothing run on real data, nothing committed)

## Built (all NEW files; manifest in FILES.txt)
- `stack/scripts/v8f_kscore.py`: `hyps` (7 hypotheses per window = khyp's code-override with T's longitudinal slice replaced by one-hot k, plus B's own plan, p_lon, scene-at-NOW mean per frame, ego token; streamed clip by clip into resumable fp16 shards, per-shard sha256, meta JSON last), `fit` (scorer 205,825 params; CE to k*=argmin ADE6; AdamW 1e-3, 3 epochs, batch 256; SEL-CONST chosen on TRAIN windows; --perm-labels = control 2), `eval` (plans of scorer / permuted / forced-to-T-argmax / const), `read` (PRIMARY-B, PRIMARY-SEL, SECONDARY, the PREREG table verbatim; rc 2 on a failed stop-control or any unmapped state).
- Scorer launch gate inside `fit`, before step 1: G-DVB (a FLAG_CHECKS entry per flag, argv sha, optimiser holds exactly the scorer's params), G-LIVE (CE finite, scorer grad non-zero, scene-zeroed control EXACTLY 0.0, B's parameters EXACTLY 0.0 with a leaky positive control, B state unchanged), G-INPUTS (shards == what B makes now, fp16-aware), admissibility check. PASS token bound to commit + argv sha + code sha + input manifests; `fit --require-token` refuses (rc 3, ZZDVB-REFUSEDZZ) without a matching token.
- `stack/scripts/chain_v8f_kscore.sh`; `stack/tests/test_v8f_kscore.py`; `WIRING.patch`.

## Proven (CPU, synthetic / random-init B): 28 passed (OUT/pytest_log.txt)
Controls 1-2 and the SEL-CONST-on-train rule; gate refuses a missing/tampered token; 256-state table incl. Amendment 1 (every boolean state maps to exactly one outcome: 192 NOT PREDICTABLE / 48 BIAS / 12 COSTS SAFETY / 3 COSTS DYNAMICS / 1 PAYS; UNAVAILABLE readings or a wrong seed set still rc 2); khyp parity (functions ast-equal, hypotheses bitwise); real-size hyps peak commit 1.08 GB (< 2 GB, batch 16); guard reader parses our logs; chain dry-run == job command lines.
Source-level mutations, each turned the suite RED and was reverted (file sha restored): production featuriser builds a graph (B unfrozen): G-LIVE fails, 9 errors + 1 failure; scene input zeroed: G-LIVE `dead_params: scene_proj.weight`, 9 errors; two outcomes swapped in `decide`: 2 failures; Amendment 1: the two COSTS branches swapped in the real source of `decide` (safety no longer takes precedence): RED, only the both-fail states move. Test-level mutations (argmin, identity permutation, SEL-CONST on eval) also go RED.

## NOT proven / open risks
1. Never run on the real 4,512-clip corpus or real B; hyps timing and disk are ESTIMATES.
2. Train-window plans are IN-SAMPLE for B (B trained on those windows) while eval-141 plans are held out: the scorer learns k* from plans that are better fits than the ones it is scored on. Known risk of the PREREG's design; reported, not corrected.
3. Control 2 (permuted-k* scorer must not separate from SEL-CONST) can fail legitimately when the best constant is not the majority class; it is FLAGGED in READ.json, not a stop. Controls 1, 3 (B reproduces 2.1753 / 2.0546) and the harness identity are stops (rc 2).
4. HS stack is NOT-INSTRUMENTED (descriptive only). RESOLVED by PREREG Amendment 1 (tip ff58be9d): speed/accel-only SECONDARY failure = COSTS DYNAMICS; COSTS SAFETY (CONSISTENCY 'any' or LEAD-VIOL separated-worse at either seed) takes precedence. Next-step text is quoted verbatim from the PREREG (tested). The tree copy of the PREREG in OUT/tree is the ff58be9d version.
5. Chain default predecessor is now FLAT's log `chain_v8f_flat.out` (box-2 order MIXO -> FLAT -> KSCORE -> OACC; markers ZZV8FMINI-DONE/-STOP, taken from chain_v8f_flat.sh). NOTE: this chain stops on ANY predecessor ZZV8FMINI-STOP, whereas FLAT's own chain proceeds past a verdict-level MIXO stop; copy that classification (FLAT chain lines ~165-185) if KSCORE should proceed after a verdict-level FLAT stop. Override: `AFTER_OUT / AFTER_DONE / AFTER_STOP`. Gate runs on box-2 CPU; hyps on GPU: tolerances 1e-3 m (tau) / 5e-3 (features) not exercised on GPU.
6. The chain REFUSES until WIRING.patch is on the tip AND copied over `box2_ship/v8f_guard.sh` (otherwise the temperature guard prints STOP and kills nothing).

## Exact commands (dev box, from a checkout of the tip that contains the three files)
```
bash stack/scripts/chain_v8f_kscore.sh --dry-run          # prints stages, shas, exact box-2 command lines; ships nothing
git apply --check WIRING.patch && git apply WIRING.patch; cp "<patched v8f_guard.sh>" /c/Users/Admin/box2_ship/v8f_guard.sh
# after pushing the three files + patch to origin/agent/arch-inf-20260803 and ONLY on the PI's go for box 2:
nohup bash stack/scripts/chain_v8f_kscore.sh > /c/Users/Admin/box2_ship/logs/chain_v8f_kscore.out 2>&1 &
# overrides: AFTER_OUT=<log> AFTER_DONE=<marker> AFTER_STOP=<marker>; markers ZZV8FMINI-DONE / ZZV8FMINI-STOP in chain_v8f_kscore.out
```
Result: `<logs>/v8f_kscore/{READ.json,KSCORE_RESULT.md,kscore_numbers.json}`; hyps/fit/eval live under `D:\tanitad\runs\v8f_kscore`.

## ESTIMATES (ESTIMATED, not measured here)
- Wall-clock, train-window hyps: khyp MEASURED 34 ms/window on the dev-box CPU (9,582 windows, 326 s); ours runs 9 passes per window vs khyp's 8 => ~38 ms; 306,566 windows => ~3.2 h per seed on CPU; box 2 safe-mode GPU 1-3 h per seed (PREREG), reading the cache from the USB HDD may dominate. Two seeds: ~3-7 h; eval-141 (2,399 windows) ~1.5 min. Fits: 3,594 steps each, minutes. Chain waits up to 11 h per hyps stage.
- Disk: MEASURED 5,004 B/window at real dims (d=256, hist 4, fp16 scene/ego; test shard 2,768 B at toy d=32) => 1.53 GB per seed train, 3.07 GB for two seeds, + 12 MB eval; fit checkpoints negligible. fit loads the arrays into RAM: scene+ego fp16 ~0.8 GB per seed (fit-stage RAM not measured).
