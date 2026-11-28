# HANDOFF (STRAT1 implementer), written at ~150k context

Tree: `F:/TanitAD-scratch/strat1/build/tree` (cdbff971 `stack` + `taniteval` code), baseline copy `tree_base`, originals of the 5 edited files in `orig/`.
Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`; env: PYTHONPATH=<tree>/stack (cygpath -w), CUDA_VISIBLE_DEVICES=-1, TMP/TEMP under build/tmp.

## DONE (files in tree/stack)
- tanitad/models/v8f_mini.py (ctx head, loss, groups, levers, guards), tanitad/data/v8f_mini_data.py (ctx_label_arrays, nav_text_arrays),
  scripts/v8f_mini_train.py (9 flags, bad_ctx, opt groups, ctx_stage, ctx_head_probe, nav_source_probe, loop, blob, summary),
  scripts/v8f_mini_eval.py, scripts/v8f_t1_eval.py (nav-text refusal/overwrite, --nav-withheld, --nav-variant, --dump-nav, ctx_logits),
  scripts/strat1_check.py (argv render, gate, post, geval, hyg), tests/test_v8f_strat1.py (31+ tests, all passed), tests/v8f_mini_tip_snapshot_strat1.txt.
- Platform sidecar written: labels/strat_platform_by_clip.npz (sha b18705db...), train 1925/2647, eval 67/74, 0 unmapped.
- Pod facts: cache_pod train 4572 clips + eval 141 + native_tracks.pt (16.6 MB) present, ClipStore format (clips/<id>.pt, manifest.jsonl with sha12, md5, n_frames, dtype float8_e4m3fn);
  pod /root/tad is NOT a git repo (no .git); runs/bprime/join_train.npz + p10/p10_prior.pt exist on the pod; no join_eval141 on the pod.

## RUNNING
- suites: run/suite_base.txt (tree_base) and run/suite_new.txt (tree), started by run_suite.sh; both end with a DONE line. Compare pass/fail counts.

## TODO (in order)
1. stack/scripts/build_navtext_nav.py (BUILD_SPEC section 5) + T-NAVB tests (countdown, d_ann, UTURN none, bare text d=0, flag!=0 zeros; `+ travelled` mutation must FAIL G-CLOCK-NAV).
2. mutation runner tests/run_mutation_strat1.py (the in-file mutants already cover the spec's mutations; a runner may just call pytest -k mutation).
3. Real-data CPU smoke: pull join_train.npz, p10_prior.pt, native_tracks.pt, 4 train + 2 eval clips from the pod (pscp, filter lines), mini cache dir with 4-clip manifest, no --exclude; 3 steps per C0/N1/S1.
4. BUILD_REPORT.md (argv per arm: `strat1_check.py argv KIND SEED --root R --out O`), manifest with sha256, delete temp (build/tmp, hfcache, tree_base after compare).
