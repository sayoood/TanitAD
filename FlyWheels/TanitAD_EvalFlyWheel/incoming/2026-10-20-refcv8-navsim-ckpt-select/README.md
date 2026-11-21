# refcv8 checkpoint selection on NavSim navtrain (PI D-R8-CKPT (b), D-NS8-1 Thor) -- groundwork + tested pipeline

Rule: GOALS_AND_CLAIMS `D-R8-CKPT-RULE` + `AMENDMENT 1` (both registered before any navtrain score).
* `GROUNDWORK.md` / `.json`: asset inventory (dev box, Thor, A40 pod), download options (S1 approved by the PI 2026-10-21),
  Thor CUDA smoke on 5 NAVTEST tokens per checkpoint (VALIDATION ONLY, never quoted), timings, blockers.
* `build/`: the 0-GPU pipeline -- `select_ckpt_r8.py` (the rule, exactly), `fix_navtrain_sample.py`,
  `export_navtrain_inputs.py`, `build_navtrain_frames.py`, `navtrain_cache_score.py`, `make_score_jobs.py`; 194 tests pass,
  63 deliberate regressions go red (`build/raw/`); `RUNBOOK_NAVTRAIN_SELECT.md` from downloaded shards to DECISION.json.
* `code/`: Thor smoke tools, `derive_rt_seed.py` (seed-1 R8_RT; acceptance-tested bit-exact vs seed 0), and the
  launch script `r8ns_navtrain_select_LAUNCH.sh` (not yet executed).
* `raw_thor_smoke/`: the Thor smoke artifacts (navtest tokens only). `thor_bundle/BUNDLE_SHA256.txt` pins the code shipped
  to Thor (`/home/nvidia/r8ns/code/`); the bundle itself is repo code and is not duplicated here.
No score exists on any navtrain token at the time of banking.
* Five zero-byte cache row files (`*_rows.jsonl` under `build/raw/e2e_evidence/.../cache/`; the cache step emits no rows by design) are omitted: git landing refuses empty files.
