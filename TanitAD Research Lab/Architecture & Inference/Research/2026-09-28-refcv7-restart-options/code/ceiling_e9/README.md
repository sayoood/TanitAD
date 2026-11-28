# Item 26.1: the speed ceiling reaches the EMITTED plan (added 2026-09-28 by the Master Mind; SPEC_REFCV7 26.1)

**Defect (MEASURED).** On the step-1,500 checkpoint the NavSim bridge produced bit-identical plans with the
ceiling filter ON and OFF on 204/204 warmup scenes, and 2 emitted plans exceeded the ceiling on CUDA.
- `refc.py`'s decoder masks only its local `rank`.
- `RefCV3Model`'s E9 re-ranks from the unmasked `score` with `reach_keep` only.

**Fix (inference-only; training arithmetic unchanged):**
- the decoder exports `ceil_keep`, added to `DECODER_PASSTHROUGH`;
- E9 ranks through `e9_rank` = reach AND ceiling;
- a row with no candidate under both keeps its reach-only ranking and is counted in `e9_ceil_dead_frac`.

**CLOSURE-TOUCHING** (`refc.py` and `refc_v3.py` are in both binding closures), so it stays **package-only** until the R4 restart.

## Runbook step (add to RESTART_OPTIONS.md sec. 6, after `apply_bundle.py --option R4`)
```
python code/ceiling_e9/apply_ceiling_e9.py <clean tree root>
cp code/ceiling_e9/stack/tests/test_speed_ceiling_reaches_emitted_plan.py <tree>/stack/tests/
```
- The patcher refuses unless every replacement matches exactly once, and it keeps each file's EOL.
- It is verified on BOTH the tip blobs (launch `refc_v3.py` a5e71567) AND on the bundle's frozen `refc_v3.py` (72b62a0c): both compile.

## Evidence
- `raw/ceiling_e9/test_ceiling_after.log`: 52/52. That is the new test (literal `e9_rank`, decoder export at eval / none in training, the REAL `RefCV3Model` E9 path, a RED arm with the pre-fix E9), plus `test_speed_ceiling_inference_only.py` and `test_declared_vs_built.py`.
- `raw/ceiling_e9/neigh_*.log`: neighbour suites, patched tree vs unpatched tip, test by test.
