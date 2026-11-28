# HANDOFF - v8F paired TACTICAL / STRATEGIC families (2026-10-25)

State: BUILT, TESTED, and the MIXO s0 read RUN. Nothing staged, committed or pushed. Export root (the repo layout, 9 NEW files, 0 modified): see `FILES.txt` beside this file.

## What exists
* `stack/scripts/v8f_paired_families.py` - `--a <X eval npz> --b <B eval npz> --out <json> [--md <md>] [--label-a --label-b]`. Paired A - B for every tactical and strategic cell via
  `taniteval.ci.paired_episode_cluster_bootstrap` (B = 2,000, per-resample recompute of kappa / macro-F1 / the 22-token goal macro-F1). Plan-derived decisions (lat / lon / 5-way accuracy + kappa, goal point),
  the heads (lat / lon macro-F1, accuracy, kappa; goal macro-F1, token accuracy, per-token F1), STRATEGIC position + heading error at 30 / 50 / 80 m. Each cell carries n, episodes, and a reason counted from its mask.
* `stack/tests/test_v8f_paired_families.py` - 24 tests, ~25-50 s CPU, synthetic world (+ 1 test on the banked MIXO dumps, skipped when they are absent).
* `raw/2026-10-25-pairfam2/` (under the v8f-mini package): `mutate_pairfam.py` + `.out` (31 of 31 mutants RED, all on an assertion; unmutated GREEN before and after, tool bytes identical), `pairfam_mixo_s0.json`,
  `pairfam_mixo_s0_tables.md`, `pairfam_mixo_s0.out`, `PAIRFAM_MIXO.md` (+ `make_pairfam_mixo_md.py`, which writes it from the json so no number is retyped).

## Result (MIXO s0, X = M2b-mixo_s0 vs B = M2b_s0; B's own-code ADE6 2.1753 m confirmed)
No tactical or strategic cell is separated: 37 paired intervals (31 tactical, 6 strategic), 0 exclude 0; 8 goal tokens have point differences only (F1 undefined on some resamples); 5 UNAVAILABLE (3 tokens with no
positive label, anchor selection, selected-vs-executed). The 16 point differences RESULT_V8F_MIXO 5.3 / 5.4 print equal the paired read's. One training seed: descriptive, decides nothing.

## Controls (all read their known values; in the json under `controls` / `alignment`)
* identity X vs X: exactly 0.0 [0.0, 0.0] on 45 cells. Reproduction vs each eval JSON: 119 checks per arm, 0 failed (max abs diff 0 / 5.6e-17).
* alignment: identical `eid`, stored `ade6` = ADE6 recomputed from `pred` + grid GT (dev 0.0), `ck_pos_err` NaN mask = grid `rc_ok`. Misaligned windows (eid, length, other grid, within-episode reorder) -> rc 2, tested.
* the dump has no heading error: it is rebuilt from the raw tick-frame `ckpt` + held-age poses (cycle ages, hist 4) and gated on reproducing the stored `ck_pos_err` (dev 0.0 on both arms). Other cfgs: pass `--ages/--hist/--max-age-t/--max-age-s/--window-start`; a failed gate gives UNAVAILABLE with the deviation.

## Exact next step (FLAT, and every later eval-141 dump)
`CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=stack:taniteval python stack/scripts/v8f_paired_families.py --a <FLAT eval_*.npz> --b <B eval_*.npz> --out pairfam_flat.json --md pairfam_flat_tables.md --label-a FLAT --label-b B`
(~4-5 min CPU; `--no-identity-control` halves it; rc 0 ok, 2 refusal, 3 a control failed with the json still written). `v8f_flat_read.py` can call `compute_pairs(A, B, ctx)` instead (I did not touch it). Landing: `pytest stack/tests/test_v8f_paired_families.py` (2 tests import `scripts/v8f_mini_eval.py` to build eval-JSON blocks and skip if it is not importable).

## Open / caveats
* Heads' accuracy / kappa, token accuracy and per-token F1 are NOT in the eval JSON (no value to reproduce); verified by a brute-force bootstrap test and the mutants instead.
* Per-token F1 has no percentile interval for 8 sparse tokens (positives in too few episodes); not worked around, because any fix leaves `taniteval.ci`.
* The eval JSONs still stamp `raw.tier` T0 (RESULT_V8F_MIXO 9, instrument-labelling item); this tool stamps "T1-class open loop (own held outputs; no recorded future action)".
* Environment: C: dropped to 0 bytes free for a few minutes mid-session (not from this work; its footprint is ~120 MB after I deleted `taniteval/results` and `stack/experiments` from MY scratch export only). Files were re-verified intact afterwards.
