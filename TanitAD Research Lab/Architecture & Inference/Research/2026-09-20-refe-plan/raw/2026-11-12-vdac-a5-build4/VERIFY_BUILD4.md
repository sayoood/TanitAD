# VERIFY_BUILD4: regression check of the VDAC build4 hold against the pristine tip

UTC 2026-11-12. CPU only (`CUDA_VISIBLE_DEVICES=-1`), interpreter `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `PYTHONPATH=<refe>;<eval>;<PK>`, each test run as `python <file>` with cwd = PK, timeout 1500 s, 6 in parallel. Evidence class: MEASURED. No GPU lock, no pod, nothing staged or committed.
Pristine = mirror commit `28db77fc` (`git -c core.autocrlf=false archive`, PK minus `media/`). Build = the same tree with every file of the hold's PK tree overlaid (verified by cmp: all 17 overlaid files identical to the hold, none identical to pristine).
`a3cc6a77` vs `28db77fc`: `git diff --name-only` lists 18 files: `Project Steering/GOALS_AND_CLAIMS.md`, `eval/stage2_sft.py`, `eval/stage3_full.py`, `eval/test_p7_read_yaw.py`, and `raw/2026-11-12-*` instrument files. **No `refe/` file differs** (empty `git diff --stat` on `refe/`). Two `eval/` files changed; `eval/s1_read.py` did not.

## Headline: NOT LANDABLE as built (1 REGRESSION), one-line fix verified

74 test files (the build agent's 7 + every test/selftest that mentions a modified file, + new test_vdac_hb): **72 SAME** (61 pass in both, 10 fail identically in both = ENV, 1 times out identically in both), **1 REGRESSION**, **1 NEW**.

### REGRESSION: `refe/test_dump_scores.py` (pristine PASS, build FAIL, rc 1)
Pristine: `ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage)`.
Build: `FAIL parser: exactly ONE new dest from the dump edit (dump_scores); every pre-edit dest is still there  ['aux_drv', ...` then `AssertionError: the unmutated source FAILS its own checks: ['parser: exactly ONE new dest ...']` (`test_dump_scores.py:470`).
Cause: the test whitelists every known added trainer dest in sets (`test_dump_scores.py:262-268`, `vdac_dests` at :268 = the two `aux_drv_exclude_*` dests); the build adds five dests (`--aux-drv-trunk-lora`, `--aux-drv-lora-lr`, `--aux-drv-cell`, `--aux-drv-bw`, `--aux-drv-bw-band-m`, `aux_hb.py:50-60`, called `scorer_finetune.py:1700`) that the check does not whitelist, so `dests - ... - vdac_dests == {"dump_scores"}` is false. It is not the `A5L.peel(TRAINER)` fix; it is a dest-whitelist drift of the same family. Not in the build agent's list, not run by them.
Fix verified in a COPY: `_verify_fixes/test_dump_scores.py` (one line, `vdac_dests` gains the five dest names), run in the build tree: `ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage)`. NOT applied to the hold's PK tree; the MM must apply it.

### NEW: `refe/test_vdac_hb.py`: build `ZZTEST_PASS 133` (pristine: file absent, rc 2).

### Masked, not verified (identical failure in both trees, so classed ENV, but the failure stops the file before it can see the build)
* `refe/test_launch_gate.py` (rc 1 both): `FileNotFoundError: D:\Projects\TanitAD\data\refe_consistent\targets_rank1.jsonl` from `diag_onpolicy_train.py:66 carve` (data absent). Everything after that line is UNVERIFIED on both trees; the build's launch_gate and the `G21_` exemption line are therefore not exercised past it. MM must run it where `D:\Projects\TanitAD\data` exists.
* `refe/test_vdac_stage0.py` and `refe/test_vdac_stage0_forward.py` (rc 1 both, 0 tests run): `REFUSED: ...eval\s1_read.py sha256 80a9f410... != the frozen s1_read.py 58aa3a77...`. The tip's `s1_read.py` (blob history: `421ffc30` = 80a9f410) does not match the frozen pin on BOTH trees, so the build agent's "Ran 33 / 22 OK" cannot be reproduced from an export of the tip; UNVERIFIED either way. (Cause not investigated: the pin vs the tip blob.)
* `eval/test_safeobj_outcome.py`: TIMEOUT 1500 s in both.

## Per-file table (P = pristine rc and last mark, B = build; last two numbers = seconds P, B; SAME = identical rc and marks)
Logs: `F:/tanitad_scratch/hbverify/logs` (deleted at the end; this table is the record).

```
SAME eval/test_aux_navtest_forward.py | P 1 [] | B 1 [] | 3 3
SAME eval/test_aux_s1_readout.py | P 0 ['ZZTEST_PASS 65  (29 checks + 0 real-store + 35 RED mutations + coverage)'] | B 0 ['ZZTEST_PASS 65  (29 checks + 0 real-store + 35 RED mutations + coverage)'] | 18 17
SAME eval/test_command_adherence.py | P 1 [] | B 1 [] | 17 17
SAME eval/test_hn_screen_read.py | P 1 [] | B 1 [] | 15 15
SAME eval/test_p6_read_gaps.py | P 0 [] | B 0 [] | 36 36
SAME eval/test_r0e_diagnostics.py | P 0 ['ZZTEST_PASS 44  (1 unmutated suite of 28 checks + the CLI end to end, 43 RED mutations)'] | B 0 ['ZZTEST_PASS 44  (1 unmutated suite of 28 checks + the CLI end to end, 43 RED mutations)'] 
SAME eval/test_safeobj_outcome.py | P TIMEOUT [] | B TIMEOUT [] | 1500 1500
SAME refe/selftest_measures.py | P 1 [] | B 1 [] | 7 7
SAME refe/test_aux_drivable_head.py | P 1 ["ZZTEST_FAIL  11 failing: ['E1 the label source accepts the pinned store and REFUSES one whose store_sha256 is "] | B 1 ["ZZTEST_FAIL  11 failing: ['E1 the label source accepts the pinned
SAME refe/test_aux_g12_bf16.py | P 0 [] | B 0 [] | 4 4
SAME refe/test_aux_s1_wiring.py | P 0 ['ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations)'] | B 0 ['ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations)'] | 433 434
SAME refe/test_carve_feed.py | P 0 ['ZZTEST_PASS 18  (2 unmutated suites + 16 RED mutations)'] | B 0 ['ZZTEST_PASS 18  (2 unmutated suites + 16 RED mutations)'] | 266 262
SAME refe/test_carve_feed_gate.py | P 0 ['ZZTEST_PASS 10  (1 unmutated suite + 9 RED mutations)'] | B 0 ['ZZTEST_PASS 10  (1 unmutated suite + 9 RED mutations)'] | 78 78
SAME refe/test_ctx_cache.py | P 0 ['ZZTEST_PASS 15  (1 unmutated suite + 14 RED mutations)'] | B 0 ['ZZTEST_PASS 15  (1 unmutated suite + 14 RED mutations)'] | 30 29
SAME refe/test_div_gate.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 24 23
SAME refe/test_dose_gate.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 25 23
DIFF refe/test_dump_scores.py | P 0 ['ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage)'] | B 1 [] | 91 9
SAME refe/test_ego_yaw.py | P 0 [] | B 0 [] | 377 378
SAME refe/test_gate_gpu_probes.py | P 0 ['ZZTEST_PASS 14  (SKIP_CUDA: the GPU check did NOT run)'] | B 0 ['ZZTEST_PASS 14  (SKIP_CUDA: the GPU check did NOT run)'] | 10 11
SAME refe/test_gate_suite_prerun.py | P 0 ['ZZTEST_PASS 144'] | B 0 ['ZZTEST_PASS 144'] | 566 554
SAME refe/test_hn_bank.py | P 0 ['ZZTEST_PASS 25  (1 unmutated suite + 24 RED mutations)'] | B 0 ['ZZTEST_PASS 25  (1 unmutated suite + 24 RED mutations)'] | 37 37
SAME refe/test_hn_content_fp.py | P 0 ['ZZTEST_PASS 81  (62 arms on the real tree + 19 RED mutations)'] | B 0 ['ZZTEST_PASS 81  (62 arms on the real tree + 19 RED mutations)'] | 140 140
SAME refe/test_hn_ctx_anchor.py | P 0 ['ZZTEST_PASS 72  (38 arms on the real tree + 34 RED mutations)'] | B 0 ['ZZTEST_PASS 72  (38 arms on the real tree + 34 RED mutations)'] | 300 301
SAME refe/test_hn_screen_wiring.py | P 0 ['ZZTEST_PASS 120'] | B 0 ['ZZTEST_PASS 120'] | 180 173
SAME refe/test_hnscreen_chain.py | P 0 ['ZZTEST_PASS 31  (1 unmutated suite of 14 checks + 30 RED mutations; the shared dev-box lock files untouched: ('] | B 0 ['ZZTEST_PASS 31  (1 unmutated suite of 14 checks + 30 RED mutations; 
SAME refe/test_hnscreen_ctx_chain.py | P 0 ['ZZTEST_PASS 22  (10 chain arms + 12 RED mutations of hnscreen_devbox.py)'] | B 0 ['ZZTEST_PASS 22  (10 chain arms + 12 RED mutations of hnscreen_devbox.py)'] | 210 214
SAME refe/test_label_dir_refusal.py | P 0 ['ZZTEST_PASS 6'] | B 0 ['ZZTEST_PASS 6'] | 9 9
SAME refe/test_launch_gate.py | P 1 [] | B 1 [] | 98 98
SAME refe/test_p2r_chain.py | P 0 ['ZZTEST_PASS 139'] | B 0 ['ZZTEST_PASS 139'] | 9 8
SAME refe/test_p2rep_chain.py | P 0 ['ZZTEST_PASS 353'] | B 0 ['ZZTEST_PASS 353'] | 277 278
SAME refe/test_p6_chain.py | P 0 [] | B 0 [] | 689 685
SAME refe/test_p6_cpu_probe.py | P 0 ['ZZTEST_PASS 111'] | B 0 ['ZZTEST_PASS 111'] | 246 284
SAME refe/test_p6_levers.py | P 0 [] | B 0 [] | 757 752
SAME refe/test_p6_overlap.py | P 0 ['ZZTEST_PASS 54'] | B 0 ['ZZTEST_PASS 54'] | 5 3
SAME refe/test_p7_chain.py | P 0 [] | B 0 [] | 647 612
SAME refe/test_p7_head_wiring.py | P 0 [] | B 0 [] | 18 19
SAME refe/test_p7_integration.py | P 0 [] | B 0 [] | 42 40
SAME refe/test_p7_ship_list.py | P 0 [] | B 0 [] | 19 18
SAME refe/test_pair_loss.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 9 9
SAME refe/test_pair_loss_p6.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 7 6
SAME refe/test_r0e_b9b10.py | P 0 ['ZZTEST_PASS 3638  (112 checks on the unmutated tree + 49 RED mutations)'] | B 0 ['ZZTEST_PASS 3638  (112 checks on the unmutated tree + 49 RED mutations)'] | 152 151
SAME refe/test_rep_gate.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 35 35
SAME refe/test_rl_epo.py | P 0 ['ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations)'] | B 0 ['ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations)'] | 241 241
SAME refe/test_rl_pickcon.py | P 0 ['ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations)'] | B 0 ['ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations)'] | 73 70
SAME refe/test_rl_r0c.py | P 0 ['ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations)'] | B 0 ['ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations)'] | 149 147
SAME refe/test_rl_s1b.py | P 0 ['ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations)'] | B 0 ['ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations)'] | 42 45
SAME refe/test_rl_safeobj.py | P 0 ['ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations)'] | B 0 ['ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations)'] | 470 469
SAME refe/test_rl_terms_label.py | P 0 ['ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage)'] | B 0 ['ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage)'] | 6 7
SAME refe/test_rls1_chain.py | P 1 [] | B 1 [] | 136 138
SAME refe/test_rls1_queue.py | P 1 [] | B 1 [] | 3 3
SAME refe/test_rls1e_chain.py | P 0 ['ZZTEST_PASS 936  (1 unmutated chain + 16 RED mutations)'] | B 0 ['ZZTEST_PASS 936  (1 unmutated chain + 16 RED mutations)'] | 64 64
SAME refe/test_s1aux_chain.py | P 0 ['ZZTEST_PASS 253'] | B 0 ['ZZTEST_PASS 253'] | 50 50
SAME refe/test_s1c.py | P 0 ['ZZTEST_PASS 220  (2 skipped, not counted)'] | B 0 ['ZZTEST_PASS 220  (2 skipped, not counted)'] | 250 246
SAME refe/test_s1z_gate.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 7 7
SAME refe/test_s1z_layer.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 0 0
SAME refe/test_s1z_loss.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 7 5
SAME refe/test_s1z_preflight.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 28 29
SAME refe/test_s1zsm_gate.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 8 6
SAME refe/test_s1zsm_layer.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 0 0
SAME refe/test_s1zsm_loss.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 4 4
SAME refe/test_s1zsm_preflight.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 39 41
SAME refe/test_scorer1_gate.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 14 12
SAME refe/test_scorer1_layer.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 0 0
SAME refe/test_scorer1_mutation.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 13 13
SAME refe/test_scorer1p6_gate.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 19 16
SAME refe/test_scorer1p6_preflight.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 31 30
SAME refe/test_session_reserve.py | P 0 ['OK'] | B 0 ['OK'] | 171 175
SAME refe/test_sft4r_chain.py | P 0 ['ZZTEST_PASS 31'] | B 0 ['ZZTEST_PASS 31'] | 6 5
SAME refe/test_sft4r_gate.py | P 0 ['ZZTEST_PASS 153'] | B 0 ['ZZTEST_PASS 153'] | 22 21
SAME refe/test_suite_nesting.py | P 0 ['ZZTEST_PASS 250'] | B 0 ['ZZTEST_PASS 250'] | 17 19
SAME refe/test_vdac_gate.py | P 0 ['ZZTEST_PASS'] | B 0 ['ZZTEST_PASS'] | 129 128
SAME refe/test_vdac_stage0.py | P 1 [] | B 1 [] | 0 0
SAME refe/test_vdac_stage0_forward.py | P 1 [] | B 1 [] | 4 5
DIFF refe/test_vdac_hb.py | P 2 [] | B 0 ['ZZTEST_PASS 133'] | 0 241
```

ENV (fail identically, same last line in both): `eval/test_aux_navtest_forward` ("the head module is not the D: copy"), `eval/test_command_adherence`, `eval/test_hn_screen_read` (taniteval dir missing), `refe/selftest_measures`, `refe/test_aux_drivable_head` (identical ZZTEST_FAIL lists, 11 failing: label store absent; the build note said 12, the run says 11, same on both), `refe/test_rls1_chain` (MODEL_REGISTRY.md not found), `refe/test_rls1_queue`, `refe/test_launch_gate`, `refe/test_vdac_stage0`, `refe/test_vdac_stage0_forward`. Rows showing `[]` marks with rc 0 are pytest-less scripts that print no token (test_ego_yaw, test_p6_chain, test_p6_levers, test_p7_*, eval/test_p6_read_gaps, test_aux_g12_bf16); rc 0 in both.
The DIV, S1Z, S1ZSM, SCORER1, SCORER1P6 drift-loop tests (test_div_gate, test_dose_gate, test_s1z_*, test_s1zsm_*, test_scorer1_*, test_scorer1p6_*) all pass on both trees with no change: no `A5L.peel(TRAINER)` fix was needed outside test_vdac_gate (which passes on the build with the build's own edit).

## Commands
Export: `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive 28db77fc "<PK>" | tar -x -C F:/tanitad_scratch/hbverify/pristine --exclude='*/media/*'`; build = copy of pristine + `cp -r <hold>/"TanitAD Research Lab"/. build/"TanitAD Research Lab"/`. Runner: per file per tree `python <file>` with the env above (`subprocess.run(..., encoding="utf-8")`). Fix check: `python refe/test_dump_scores_fixed.py` in the build tree.
Cleanup: see the hand-back (scratch `F:/tanitad_scratch/hbverify/` deleted by literal path; size logged there).
