# BUILD_REPORT: ACTRSP lever (`--wm-rsp-w`), built on ACTSENS commit 81cee6c1 (2026-11-16)

Nothing was launched, committed or pushed. No GPU, no box 2, no pod. Root: `F:/TanitAD-scratch/actrsp/build/` (450 MB in total). All evidence classes are MEASURED on the dev-box CPU unless marked.

## Headline for the Master Mind
The term is built, reachable from argv to backward, and OFF equals the tip. Escalation: the diff `actrsp.diff` is NOT staged in any repo (the brief put E: out of bounds); it lands as the `raw/` diff on a branch from `81cee6c1` (ACTSENS freeze note F2). Three trainer files changed (3 lines edited, 183 added, 0 removed beyond that).
Build findings beyond the two blockers in `REACHABILITY_ACTRSP.md` section 1 (both resolved): (1) a THIRD hazard: `built_rollout` counted the term's own rollout call as a level-1 ladder free run (k read 14 instead of 4, anchors and context read from the term), so `bad_wm_ladder` refused the RSP argv; fixed by skipping 5-D calls there (MR10). (2) C3 was re-specified, see below.

## Manifest (sha256 first 16 hex; full list in `final.sha`)
| file | where | sha256 | note |
|---|---|---|---|
| `stack/tanitad/models/hjepa.py` | `build/tree/` (orig in `build/orig/`) | 2c386726c5cd92b5 | LevelConfig `rsp_w`, `rsp_delta`; `response_loss`; loss block; `built_levers` row; `built_rsp`; `built_rollout` fixes |
| `stack/tanitad/models/v8f_mini.py` | `build/tree/` | 17dbb463fe8beece | config field + validation, `RSP_DELTA`, level-1 kwargs, `WMAux.built_rsp` |
| `stack/scripts/v8f_mini_train.py` | `build/tree/` | cfb9bbafbd47cf79 | flag, FLAG_CHECKS, `make_config` line, `bad_wm_rsp`, `wm_rsp_readback`, G-LIVE (`expect`, `live.c3`), `live_first` |
| `stack/tests/test_v8f_actrsp.py` | `build/tree/` | 428e435970a22427 | 25 tests |
| `actrsp.diff` | `build/` | 6a4275751ca4c5fc | unified diff of the three files vs `build/orig/` |
| `box2/actrsp_check.py` | `build/box2/` | 27783b98b8de9cd6 | fork of `actsens_check.py`; token schema `actsens-gate-token-v1`, `written_by` `actrsp_check.py gate` |
| `box2/tests/test_actrsp_check.py` | `build/box2/tests/` | ef9c0aacfc3beec8 | 63 unit checks |
| `box2/argvs_actrsp.json`, `argv/argvs_actrsp.txt`, `argv/argv_<tag>.txt` (7) | `build/` | 890b95898e0af0f6 / aa7ba1ef96e0d70b | argv per arm; the pinned sha in the checker is that of the json |
| `make_argvs_actrsp.py` | `build/` | 425b61b0cd8ce4d4 | |
| `HOPS.md` | `build/` | 912b73b1ea9724a8 | hop table with file:line |
| `mut_rsp.json`, `mutate_rsp.py`, `mutation_results_rsp_ALL.json` | `build/` | 86ac4b658c824371 / 3638b47ed30ab2b2 / df7dc9e427fcaedf | |
| `fingerprint.py`, `base_fp.json`, `new_fp.json`, `new_fp2.json`, `gate_smoke.py`, `patch_*.py` | `build/` | | build instruments |
Everything exists in ONE place (this folder on F:). Bank it before the 17:00 window ends.

## What the term does (card 2.1, level 1 only)
Anchors `range(0, T-10, 10)` (9 at T = 91), K = 10, four contrasts +/- 1 sd (0.42, 0.24) in one batched `rollout(..., context=1)`; readout = level-1 IDM head with detached parameters; slope on real transitions (detached, floor 0.1); `G = diff / (2 delta s)`; loss `mean (G - I)^2`; `level/L1 += lambda * rsp/L1`. Six detached diagnostics logged. No parameter added, no random draw. Hop table: `HOPS.md`.

## Proofs (dev-box CPU)
* **Gradient (P1):** term alone, through the pooled-adapter path: `predictors.0` and `act_embed` non-zero; EXACTLY 0.0 on `idm_heads.0/1/2`, `proj`, `predictors.1/2`, `encoders`, `macro_encoders`, and on the adapter-output tensor. Trainer G-LIVE: control non-WM and adapter sums read exactly 0.0, no dead WM group, `wm/rsp/L1` present.
* **OFF equals the tip (MEASURED, `base_fp.json` taken on the unedited export vs `new_fp.json` and `new_fp2.json`):** for plain, ladder and `--wm a` configs: state-dict sha256, every loss term (float.hex), gradient sha256, `built_levers`, `built_rollout` all equal; and the trainer's flag-absent `--selfcheck-only` on the synthetic ladder recipe: live terms, key sets, per-group gradients all equal. Verified before AND after the mutation runs.
* **Dose identity (P3):** float64, lambda 1.0 and 0.3: level/L1 difference = lambda x rsp/L1 to 1e-9 relative; every unweighted term bit-equal.
* **Analytic (P4/C9, float64):** matched response gives loss 2e-33-class (< 1e-24) and gradient < 1e-12; B = 0 gives exactly 0.5; general linear case equals an independent recursion to 1e-6 (the slope is float32 by design); slope floor reads 0.1; slope equals cov/var; geometry 1 call / 4 contrasts / 9 anchors / K 10 / context 1 at hmax 91 and `w_in_loss_rsp` = 1.0 to 1e-6 (read on a float64 copy; in float32 the measurement is blurred to ~1e-5 by level/L1 ~ 100).
* **Pairing (P5/C4):** state dict equal flag on vs off at seeds 0 and 2; no RNG draw in the term; global RNG stream after the full loss equal on vs off.
* **C3 deviation (decision for the MM):** the card's C3 compares against R35's banked step-0 terms; R35's selfcheck has only the LAST step's terms (live_first was added after it). So the trainer compares step 0 against a deep copy of the SAME weights and batch with `rsp_w = 0` and writes `live.c3`; tolerance 1e-5 relative to the term or 8 float32 ulps of level/L1 (level/L1 ~ 100 at step 0).

## Mutations (`mutation_results_rsp_ALL.json`): 10 of 10 RED, 0 controls red
| id | mutation | RED tests (excerpt) | summary |
|---|---|---|---|
| MR1 | `make_config` line deleted | gate passes/readback | 1 failed, 24 passed |
| MR2 | default 0.0 -> 1.0 | default value, key set, flag-absent gate | 2 failed, 6 errors |
| MR3 | computed, not added | P3 (both doses), gate | 5 failed |
| MR4 | readout not detached | P1 | 1 failed |
| MR5 | `z0` not detached | P1 | 1 failed |
| MR6 | latm sign flipped | P4 matched and general | 3 failed |
| MR7 | one random draw | P5 x2, C3 | 3 failed |
| MR8 | `rsp_w` not passed to level 1 | gate + 5 more | 6 failed |
| MR9 | `built_rollout` `other` without the term | ladder reader, gate | 2 failed |
| MR10 | term's rollout call counted as a ladder run | ladder reader, gate | 2 failed |
Controls kept green in every row: `test_off_*`, flag-absent gate, P1 zero asserts (MR1-3, 8-10), P3 (MR2, 4, 5, 7), slope floor (MR6).

## Checker (`box2/actrsp_check.py`)
`argv_rsp_problems` refuses an RSP argv without `--wm-rsp-w 1.0` named exactly once and a NULL argv that names it; `gate_problems` reads `levers.wm_rsp_readback` (declared, built config, built loss weight 1e-6, geometry 1/4/10/1/9), `live.c3`, `live_first`, `wm/rsp/L1`, the absence of the ACTSENS lever readbacks; the log identity C2 is `rsp_from_rows`. `launch_gate.py` untouched. 63 of 63 unit checks pass, including 25 planted defects. NOT done: the stage/ship shell scripts, jobs, dry-run fakes and manifests of `actsens/box2_build` were not forked (MM package); they export `ACTSENS_*` env names that the fork renamed to `ACTRSP_*` (sed), so the package scripts need the same rename.

## Argv per arm (card section 3; files in `build/argv/`, not launched)
Order: PILOT_RSP_s0 (RSP_s0 + `--steps 200`, out moved), RSP_s0, RSP_s2, NULL_s4 (chassis 0, init M2b_s0); RSP_s1, RSP_s3, NULL_s5 (chassis 1, init M2b_s1). Base = `actsens_NULL_s2` / `NULL_s3` argv with seed and `--out D:\tanitad\runs\v8f_actrsp\<tag>`; RSP arms append `--wm-rsp-w 1.0`.

## Tests
* `test_v8f_actrsp.py`: 25 passed (final tree).
* Regression set (15 files that import the edited modules: test_hjepa, test_ra1_hjepa, test_v8f_mini, test_v8f_actsens, test_v8f_divrs, test_v8f_ladder, test_v8f_mrate, test_v8f_mrate_train, test_v8f_beta, test_v8f_flat, test_v8f_level_head, test_v8f_lead_head, test_text_encoding_is_explicit, test_v8f_wc, test_v8f_cz), one pytest process each: BASE (81cee6c1 files) 7 failed / 378 passed / 245 skipped; NEW 7 failed / 378 passed / 245 skipped. The same 7 fail on both (4 queue tests in test_ra1_hjepa, test_v8f_flat seed-state, test_text_encoding_is_explicit on `test_r9_e3_make_argv.py:497/502`, test_v8f_cz queue). **Regressions: 0.** The 7 are base failures of this partial export (UNVERIFIED cause: queue scripts and git absent from the export). Not run: the other 37 test files that import trainer or hjepa (time); the MM's pinned suite covers them.

## What remains for the MM before a box-2 launch
1. Freeze the card; fold in the C3 re-specification and the third hazard.
2. Land `actrsp.diff` (and the test) on ONE commit from `81cee6c1`; set `ACTRSP_COMMIT`.
3. B-CAL (card section 6): `z_t`-only vs pair readout skill on R35_s0 TRAIN latents, BEFORE the first run; if the `z_t`-only R2 exceeds 0.25 x the pair R2, the slope is replaced by amendment. Not done.
4. Read-side instruments (READ_PLAN N1-N3, C10) are not built.
5. Fork the box-2 package (stages, jobs, ship, manifests, dry run), pin `ARGVS_SHA256` after any argv edit, run the dry run, issue tokens (only via `actrsp_check.py gate`).
6. Pilot (200 steps): s/step, `max_memory_allocated`, `rsp/L1` trajectory. ESTIMATED 0.55-0.70 s/step per the card; bf16 (`--amp`) is off in the argv and the term was not tested under autocast.
7. Real-batch G-LIVE and the real `--selfcheck-only` on box 2 (only the synthetic recipe ran here).

## Deletions log
`build/tmp/` 6,198 MB (pytest basetemps and mutation scratch; own temporary data) and `build/newfiles/` 1 MB, deleted by literal path. Kept: `build/tree`, `build/orig`. ⚠ The footprint rule (500 MB) was exceeded transiently: tmp reached about 6.2 GB during the test and mutation runs before the delete; F: free read 51.6 GiB afterwards.
