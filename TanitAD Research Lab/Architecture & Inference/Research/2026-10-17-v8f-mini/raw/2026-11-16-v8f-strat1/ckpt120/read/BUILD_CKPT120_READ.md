# BUILD_CKPT120_READ: the CKPT120 read (PREREG_V8F_CKPT120, frozen 2026-11-16; MM rulings 1-12)

Package: `F:/TanitAD-scratch/strat1/read/ckpt120/`. Code of record `213250ce` (tree/). Nothing committed or pushed, no pod touched, no training, CPU only. Evidence class of every statement below: MEASURED (this build, test run on the dev box) unless marked ESTIMATED, INHERITED or UNVERIFIED.

## MM start

1. `bash F:/TanitAD-scratch/strat1/read/ckpt120/after_ckpt120.sh` (Git bash, dev box). It polls the pod log `/root/strat1/logs/ckpt120.log` for `ZZ STRAT1 CKPT120 DONE`, banks the 7 arms private on HF (`strat1_bank_pod_ckpt120.py` via `bank_ckpt120.sh`), pulls `ckpt_last.pt` + `argv.json` + manifest to `ckpts/<arm>/` (sha256 and commit `213250ce` asserted), runs `resolve_reach.py --controls` then every arm, and launches the read via `launch_wmi.ps1` unless `SKIP_LAUNCH=1`. Last line `ZZ AFTER CKPT120 DONE ZZ` or `ZZ AFTER CKPT120 FAIL ZZ <why>`.
2. Manual launch of the read only: `powershell -File F:\TanitAD-scratch\strat1\read\ckpt120\launch_wmi.ps1` (CreateFlags 0x08000200, `cmd /c job_ckpt120_read.cmd < NUL`, `C:\Program Files\Git\usr\bin\bash.exe`). Waiter: `bash .../ckpt120/wait_ckpt120_read.sh` (sentinel `ZZ STRAT1 CKPT120 READ DONE|FAIL ZZ` in `out/ckpt120_read.log`).
3. Result: `out/RESULT_CKPT120.json` and `out/RESULT_CKPT120.md`.

## Pieces (file:line)

| piece | file:line | what |
|---|---|---|
| constants, pins, bank / reach refusals | `c120_lib.py:66` `bank_refusal`, `:87` `reach_verdict`, `:52` `tree_problem` | an arm is refused without its bank manifest, tag, commit, checkpoint sha256 and a K-REACH-e PASS |
| passes | `c120_run.py:53` `t1_argv`, `:74` `t0_argv`, `:129` `cmd_dumps` (causal dump and K9b `--limit logged --batch 64`), `:162` `cmd_t0` | argv templates = the banked STRAT1 argv files; only script, ckpt, tag, out dir and the CKPT120 flags change |
| K-REACH-e off the GPU | `resolve_reach.py:29` `optimiser_names` (trainer order, `v8f_mini_train.py:3831-3844`, group sizes checked against the checkpoint's own `opt.param_groups`), `:68` `resolve`, `:95` `controls` | writes `reach/<arm>.json`; a group-size mismatch gives no names file, so the gate names by shape and may read INCONCLUSIVE (never a silent PASS) |
| labels (pure) | `c120_read.py:81` `route_label`, `:102` `far_label`, `:119` `driving_label`, `:138` `route_cost_cells`, `:145` `detect_cost_labels`, `:150` `echo_share`, `:157` `floors_over` | every committed outcome of card section 8 |
| read | `c120_read.py:251` `read` (ground `:172`, route block `:223`, strata `:209`), render `:511`, CLI `:547` | four families, controls, floors, JSON + markdown |
| shell | `after_ckpt120.sh`, `launch_ckpt120_read.sh`, `job_ckpt120_read.cmd`, `launch_wmi.ps1`, `wait_ckpt120_read.sh`, `bank_ckpt120.sh`, `strat1_bank_pod_ckpt120.py` | copies of the S1d set with paths, tags and commit changed |
| tests | `test_c120_read.py` | 27 tests, all green: `27 passed in 78.8 s` (dev box, CPU) |

Reused unchanged: `strat1_strategic.py` (`ground`, `route_cells`, `rcA_errors`, `rc_cells`, `route_setting`, `arm_block`), `strat1_read.py` (`load_arm_cells`, `slim`, `family_blocks`), `obd_read.py` (`CELLS` = the 24 driving cells, `cell_delta`, `floor_of`, `identity_control`), `strat1_ctx_read.py` (`label_rows`, `split_logits`), `tree/stack/scripts/ckpt120_cells.py` (`s_ckpt120`, `route_following`, `route_setting_120`, `strata`, `k7`, `k8`, `k9b`). The reader asserts `taniteval` is imported from the CKPT120 tree, and that its `ci.py` sha256 equals the STRAT1 tree's (528276c5, same bytes: MEASURED).

## What the tests establish

* PURE labels: S120-FOLLOWS, -STRONG, -PARTIAL (balanced fails; raw point <= 0; balanced below 0.5), -IGNORES (not separated; wrong side; below F120), NOT-YET, NOT-READ, `(FLOOR-PENDING)`; FAR-SETS / FAR-NO-GAIN; KEEPS, ADDS, COSTS-FAR, SEEDS-DISAGREE (both print), ROUTE-COST, the detected-info trigger. One mutation per rule (floor comparison dropped, sign test dropped, cost rule, balanced contrast) flips its planted case.
* K-RE against banked STRAT1: `s_ckpt120` at arc 80 equals `strat1_strategic.route_cells` bit for bit on the banked S1_s0 pass; levels and real - donor at 80 m reproduce `out/STRAT1_STRATEGIC.json` for S1_s0 and S1_s1 to 1e-12.
* K-LABEL on the eval side: 341 windows / 27 episodes, left 79 / right 262, strata 219/122, 244/97, 304/37 (card 2.2), identity with `eval141_grid_truth.npz` on rows, side, d_text, episode ids and the 120 m validity column.
* K-PERFECT reads +0.232 [+0.090, +0.406] and K-NAVINF raw -0.126, not separated (card values, B 2,000). K7 reads 0.0 / 1.0 / 0.0; K8 real 1.0 and donor = the label-computed agreement.
* K-REACH-e controls on the banked S1_s0 checkpoint (`reach/CONTROLS.json`): a trained head (ctx head, standing in for `s_far_out.*`) reads PASS, a deleted optimiser state and a zeroed optimiser state both read HEAD-DEAD.
* First execution of `c120_run.py` (no defect found): T0 flags-off pass of banked S1_s0 on the 48-window smoke grid equals the banked rows to within 1e-4 (float32 batching noise, about 1e-7, not bit identity: the subset batches differently); T1 flags-off pass on 6 windows runs and its tick 0 equals the banked tick 0.
* End-to-end reader smoke (synthetic far points planted as a perfect follower, banked dumps lent under CK120 names): H1 and H2 read S120-FOLLOWS-STRONG, FAR-SETS, R1 and R2 read as must-fail passes, markdown has none of the banned words, a HEAD-DEAD arm is skipped and its rung reads HEAD-DEAD, a missing reach file is never a PASS. A smoke is a code check, never a result.
* Not exercised (no CKPT120 checkpoint exists yet): `resolve_reach.resolve` on a real far-head arm (only its refusal path and the controls on S1_s0), the far-wrapper passes with `--far-out on` / `--nav-strat on` (argv checked by tests, execution needs a checkpoint carrying the heads), and K9b on real tick dumps (the function is tested on arrays).

## Wall time (ESTIMATED)

7 x (6 min T1 + 6 min K9b + 3 x 7 min T0) = 231 min = 3.9 h, plus the reader (3 to 5 min at B 2,000, ESTIMATED from the 33 s end-to-end smoke at B 100) and the unit tests (about 1.5 min). The handoff's "about 5.5 h" does not follow from its own unit times; those unit times are INHERITED (STRAT1 reads), not re-measured for the far wrappers. If the projection exceeds 4 h the stop rule of card section 9 applies.

## Deviations for the MM (each interpretation of the frozen card the build made)

1. **"Every effect > F120" applied to (a) and (b) only; raw needs a point > 0.** Card section 8: "(judged for H1 and for H2 separately, both seeds; every effect also > F120)" and the FOLLOWS row: "real − donor separated, balanced − 0.5 separated, and raw point > 0, at both seeds". Built: real - donor and balanced - 0.5 must each be separated above 0 and above F120; raw only needs a point > 0 (separated raw gives STRONG). Alternative: require the raw point > F120 as well, which makes PARTIAL more likely.
2. **F120 per contrast.** Card section 6: "F120 = |H1_rep − H1_s0| on the 120 m cell" ("STRATEGIC 120 m cells have only the first pair"). Built: F120 is taken on each contrast's own statistic (the real - donor delta, the balanced - 0.5 delta, and the arm - EXTRAP position error), each as |H1_rep - H1_s0|, applied to both rungs and both seeds. Alternative: one F120 on the S_CKPT120 real level for all three.
3. **Echo share, UNVERIFIED.** Ruling 10: "route-following rows are printed with the echo share beside them". The card gives no formula. Built: hits (S_CKPT120 = 1) on windows where the tactical token carries the side, divided by all hits. Alternative: the share of the real - donor effect that lies in the token-on stratum, or the hits that the donor nav also produces.
4. **K-LABEL counts on the release of record.** Card section 7: "K-LABEL: recomputed RC-A 30/50/80 vs the train release of record | 0 validity mismatches"; section 2.2: "341 windows / 27 episodes, left 79 / right 262". Built: the reader counts scorable windows from the shipped `rc120_table.npz` (built on the release of record, md5 83e30bd3) and the label tables: **341 / 27, left 79 / right 262, equal to the card** (MEASURED), so the counts measured on the 2026-10-04 release hold on the release of record. The train-side recompute (0 validity mismatches over the train release, needs the 254 MB release from box 2) is NOT in this reader; it belongs to the label build and is not re-run here (UNVERIFIED in this package). Side sign: +1 = left.
5. **Frames.** S_CKPT120 reads `ckpt_far_now` from the `_far.npz`; the 30/50/80 continuity and EXTRAP read the T0 `ckpt` of the same pass at the first tick (the STRAT1 convention). Card: "T0 at the window's first tick".
6. **Route labels need n >= 30 windows and 10 episodes per row.** A 120 m row below that reads NOT-READ with its n (STRATEGIC_PLAN n rule, as STRAT1).
7. **FAR-SETS floor.** Card: "H − EXTRAP on position error separated negative at both seeds with |Δ| > F120". Built: F120 = |H1_rep - H1_s0| on the arm - EXTRAP position cell (see 2). Each arm's EXTRAP is its own 80 m point carried 40 m; S1's EXTRAP is printed as the base row.
8. **Floors for driving, 30/50/80 and detected info.** Card section 6: "F per cell = max(|H1_rep − H1_s0|, |S1_s10 − S1_s0|) at seed 0, applied at both seeds". Built exactly so per cell, for the 24 driving cells, the six RC cells and the five detected-info variables; the pairs used are printed (`floors_*.pairs_used`). With no pair a cell reads without a verdict and the label carries `(FLOOR-PENDING)`.
9. **Driving labels.** Card section 8 gives the four labels per rung. Built as STRAT1's reading choice r1: when a cell costs at both seeds and another cell at one seed, both COSTS-FAR and SEEDS-DISAGREE print. H2 is compared with the same-seed H1 arm, H1 with the same-seed banked S1.
10. **R1 / R2 pass rules.** R1: "NOT S120-FOLLOWS, and H1_s0 − R1_s0 on far position error separated negative". R2: "real − donor ... on the stratum where the tactical token is off NOT separated, and H2_s0 − R2_s0 on S_CKPT120 real there separated positive". Built literally; R1 is read at its single seed with the label function fed that seed twice; a stratum that is n/a counts as not separated.
11. **Detected info.** "DETECTS in S1" is taken from the banked `out/STRAT1_CTX.json` outcome strings (both seeds); the COSTS-FAR trigger is a variable that costs at both seeds in H - S1 (H1 and H2 each against S1 of the same seed).
12. **Not computable here, named.** DK-VIOL, TTC, the tactical confusion matrix and goal selection (STRAT1 work items with no instrument in the frozen STRAT1 reader); visibility (not built); derived goals (later card). Each prints with its reason; headway and time-gap MAE are in LONGITUDINAL.
13. **K-REACH-e: HEAD-DEAD continues, INCONCLUSIVE stops.** Card: "every `s_far_out.*` and `e_nav_s.*` param has optimizer state; else HEAD-DEAD and nothing is read". Built: a HEAD-DEAD arm is INVALID (skipped by the passes and the read, its rung reads HEAD-DEAD); an INCONCLUSIVE arm stops `after_ckpt120.sh` before the launch.
14. **Chain steps do not stop each other.** `launch_ckpt120_read.sh` runs dumps, K9b and T0 for all arms even if one arm is refused, runs the reader, and then ends with a FAIL sentinel naming the failed steps (the result labels each arm). Reason: ruling 9, a stopped chain should still yield both rungs.
15. **Pod layout assumed from S1d (UNVERIFIED).** The bank script and pulls assume `/root/strat1/runs/<tag>/{train/ckpt_last.pt, argv.json, gate/gate_token.json, ...}` and the pod log `/root/strat1/logs/ckpt120.log`, as the S1d chain. The first pull would fail loud if the layout differs.
16. **PROVISIONAL.** Until R1 and R2 have read as must-fail passes and H1_rep exists, `verdict_state` reads PROVISIONAL and labels carry `(FLOOR-PENDING)` where F120 is missing (ruling 9). If R1 or R2 does not pass, the state is BROKEN READ and `problems` lists it.

## Deletion log

| path | size | why |
|---|---|---|
| `F:/TanitAD-scratch/strat1/read/ckpt120/tmp` | 200 MB | test basetemp dirs and synthetic far files made by this build's tests; re-creatable by the tests; no other owner |

Nothing else deleted.

## Manifest (all on the dev box; each single-location until the MM lands it)

| artifact | where |
|---|---|
| reader, passes, lib, K-REACH-e resolver | `F:/TanitAD-scratch/strat1/read/ckpt120/{c120_read.py, c120_run.py, c120_lib.py, resolve_reach.py}` (one copy) |
| tests | `.../ckpt120/test_c120_read.py` (one copy) |
| shell and launch set | `.../ckpt120/{after_ckpt120.sh, launch_ckpt120_read.sh, job_ckpt120_read.cmd, launch_wmi.ps1, wait_ckpt120_read.sh, bank_ckpt120.sh, strat1_bank_pod_ckpt120.py}` (one copy) |
| K-REACH-e control record | `.../ckpt120/reach/CONTROLS.json` |
| code tree of record, pinned inputs | `.../ckpt120/tree/`, `.../ckpt120/inputs/` (from the earlier builder) |
| this note | `.../ckpt120/BUILD_CKPT120_READ.md` |

Integration needed (headline for the MM): land the package under the study folder with the card; start `after_ckpt120.sh` once the pod chain is launched. The pod-side CKPT120 chain itself is not part of this package.
