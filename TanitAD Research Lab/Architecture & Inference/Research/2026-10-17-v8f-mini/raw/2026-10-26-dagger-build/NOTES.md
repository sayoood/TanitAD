# NOTES: PREREG_V8F_DAGGER, the DATA side (own-state bank, matched route point, swap)

> **INTEGRATION UPDATE (2026-10-26, build `di`; read `../2026-10-26-dagger-integration/NOTES.md` first).** This file is the DATA worker's hand-back. Since then `v8f_dagger_bank.py` carries the E1 patch (`window_start=cfg.window_start` spelled out at both `V8FMiniWindows` calls; behaviour identical, sha256 changed from the sha256 quoted below), and the proofs were re-run on the final bytes: `../2026-10-26-dagger-integration/FINAL.txt`. Hashes and run results below are the worker's.

Built 2026-10-26 from `PREREG_V8F_DAGGER.md` + `DESIGN_NOTES.md` + `probe/` (mirror commit `07dcb513`). STAGED, nothing committed or pushed. CPU only, no GPU, no box 2.
Evidence classes: MEASURED (this build, artifact path) / ESTIMATED / UNVERIFIED. File:line below are for the frozen deliverables
(`v8f_dagger.py` sha256 `831f64f8...`, `v8f_dagger_bank.py` sha256 `41a31de9...`, `test_v8f_dagger.py` sha256 `4a44431f...`).

## 0. Headline

* **Built:** the matcher, the bank record + loader + controls, the generator (reuses `v8f_t1_eval.rollout`, not a fork), the batch/item swap, 40 tests (35 at first build; 40 passed after the survivor kill, section 7b), a 57-mutant script, a CPU smoke.
* **Nothing existing was edited.** `v8f_t1_eval.py`, `v8f_mini_data.py`, `v8f_mini_train.py`, `v8f_mini.py` are byte-identical to `07dcb513` (sha256 compared).
* **Two decisions are needed from the Master Mind** (section 5): **A1** the tie-set (the probe's code and the PREREG's text disagree), **A2** nav progress on frames with no route.
* **Not done / UNVERIFIED** (section 6): the real TRAIN cache is not on the dev box, so no real bank exists; the smoke runs the REAL `M2b_s0` checkpoint on SYNTHETIC TRAIN clips (plumbing only, its numbers say nothing about real coverage).

## 1. What each rule became (PREREG section -> code -> test -> mutant)

| PREREG rule | where (file:line) | test (`test_v8f_dagger.py`) |
|---|---|---|
| constants 3.0 m / 30 deg / 0.05 m / T-31 / ticks (1,2) / m = 10 / COVERAGE 0.5 (THRESHOLDS_FIXED.txt) | `v8f_dagger.py:37-43` | `test_constants_are_the_preregs...` (parses THRESHOLDS_FIXED.txt, sha `438366d5`) |
| `p` is not hard-coded anywhere | `DAGGER_DEFAULTS` `:56`, `DaggerSwap.__init__` | same test (no `P_DAG*`, default 0.0, source scan) + `test_p_is_a_parameter...` (0.1/0.3/0.5/0.9 follow) |
| S2.1 candidates = logged rows [t0, T-1] | `match_route` `:129` | `test_kb4_cut_boundaries...` (rows before t0 are not candidates) |
| S2.2 project onto polyline segment by segment, e_lat = smallest distance | `:134-146` | K-B4 cases; `u = np.clip` `:140` |
| S2.3 r' = row whose cumulative arc is nearest s* | `:147` | `test_kb4_tie_set_...` |
| S2.4 ties within 0.05 m of that nearest arc; nearest in time to j, then the earlier row | `:148-150` | `test_kb4_tie_set_...`, probe case `stationary_tie_*` |
| S2.5 off-route cut e_lat > 3.0 or heading > 30 deg | `:152-155` | `test_kb4_cut_boundaries...` (inclusive on the kept side) |
| S2.6 end edge r' > T-31 | `:156` | same |
| every cut counted by reason | `REASONS` `:46`; `by_tick` in `v8f_dagger_bank.py:283` | `test_bank_holds_ticks_1_and_2_only...` |
| S1 generator: frozen M2b, one pass, `--replan-steps 10 --limit causal --nav deadreckon --scene logged --schedule periodic`, plan executor, PLAIN | `ROLLOUT` `v8f_dagger.py:45`; `ROLL` + `T1.rollout(...)` `v8f_dagger_bank.py:44,138` | `test_bank_rows_are_the_t1_rollouts_own_states...` (independent full `T1.rollout`, atol 1e-9), `test_the_generator_is_deterministic_and_reuses_the_t1_rollout` |
| S1 ticks 1 and 2 only, tick 0 never | hook `v8f_dagger_bank.py:103` | `test_bank_holds_ticks_1_and_2_only...` |
| S1 record: ci, t0, tick, j, p10f, pose, arc, kin3, r', e_lat, heading, reason; NaN before the clip | `RECORD_FIELDS` `v8f_dagger.py:204`; `snapshot` `v8f_dagger_bank.py:53-68` | `test_bank_rows_are_...` (rows <= t0 are the LOGGED rows exactly; rows after are executed; pre-clip rows NaN) |
| S1 header: ckpt sha256, rollout, window-list sha256, constants sha256 | `v8f_dagger_bank.py:275` | `test_bank_holds_...` |
| S1 TRAIN only, clip-sharded, `--limit-clips`, `--dry-run`, every path an argument | `cmd_gen` `:223`, `positions` `:247`, `train_list` `:203` | `test_dry_run_writes_nothing_limit_clips_marks_partial_and_merge_checks_the_shards` |
| S2 sample = logged item at (clip, r') with REBUILT ego tensors replaced; KEPT / RECORD / targets are the same objects | `swap_rows` `v8f_dagger.py:370`; `REBUILT/KEPT/RECORD` imported from ORC `:49` | `test_swap_reachability_differs_exactly_in_the_ego_tensors`, `test_the_partition_is_orcs...` |
| S2 p10f, kin3, p10_t, base rebuilt the way ORC does | `:386-389` | `test_the_rebuild_is_the_stored_own_state_and_agrees_with_orcs_rebuild_at_the_identity` (ORC `_rebuilt` at s=1: <= 1e-6; tick-0 swap bit-exact) |
| S2 poses re-anchored by the rigid motion own NOW -> logged pose at r' | `reanchor_rows` `:162-175`, `:392-398` | `test_kb5_*` (2 tests) |
| S2 nav: channels 0-2 and 4 kept; channel 3 = nav[r',3] + own arc behind NOW / 100 on frames sharing NOW's token and validity | `:400-406` | `test_nav_progress_follows_the_own_arc...` |
| S2 `lab_s.rc` re-expressed (same world points, `transform_ckpt`), identity at a_s = 0 | `rc_frame_change` `:182-190`, `:409` | `test_rc_is_re_expressed_not_changed` |
| S2 ages = the sampler's, as clamped by the logged item at r' | `batch["a_t"], batch["a_s"]` `:397-398` | `test_swap_reachability...` (a_t 1, a_s 3), `test_rc_is_...` (a_s 0 and 3) |
| S4 eligible = the window has >= 1 passed own state with r' = t; one drawn uniformly; private generator seeded by (seed, step) | `candidates` `:290`, `pick` `:300`, `apply` `:486-515` | `test_draws_are_a_function_of_seed_and_step...`, `test_the_draw_among_a_windows_eligible_own_states_is_uniform_and_only_eligible` |
| EgoPerturb contract: p = 0 returns the SAME object, no draw | `apply` off path | `test_p_zero_returns_the_same_batch...` |
| K-B0 tick-0 own state == logged item, bit for bit | `Capture._tick0` `v8f_dagger_bank.py:111-119` (every window of every chunk) | `test_kb0_tick0_identity_and_a_leaky_state_is_refused` (+ a leaky `WindowState` is refused) |
| K-B1 seeded 0.5 % replay, bit for bit | `run_windows` `v8f_dagger_bank.py:170-172`, `replay_chunks` `:253` | `test_kb1_replay_is_bit_exact_and_a_stochastic_planner_fails_it` (the control: noise -> 3 of 3 chunks fail) |
| K-B2 bank ckpt sha256 == `--init-ckpt` sha256 | `DaggerBank.check_on_policy` `v8f_dagger.py:321` | `test_kb2_on_policy_and_coverage` |
| K-B3 TRAIN only; eval clip refused; window-list sha | `check_train_only` `v8f_dagger.py:236`, generator `v8f_dagger_bank.py:238`, merge `:333`, `windows_sha256` `v8f_dagger.py:101` | `test_kb3_*` (2 tests; the control reads overlap exactly 0) |
| K-B4 matcher cases | `match_route` | `test_kb4_the_seven_probe_cases` (the probe's own data) + 2 more |
| K-B5 re-anchor: pose_now = logged pose at r', relative transforms <= 1e-4 m | `reanchor_rows` | `test_kb5_*` (float64 <= 1e-9, float32 tensors <= 1e-4 measured in-test) |
| COVERAGE >= 0.5, below it BLOCKED | `coverage` `:312`, `COVERAGE_MIN` `:41`; per shard `v8f_dagger_bank.py:282`, merged `:342-345` | `test_kb2_on_policy_and_coverage` (brute force; every record cut -> 0.0 / blocked) |
| a cut record is stored but never eligible | `ok = ...reason == OK` `v8f_dagger.py:271` | `test_an_off_route_own_state_is_stored_with_its_cut_counted_and_is_never_swapped` |
| the control that must read EXACTLY equal | | same loss twice, p = 0, tick-0 swap loss/grad diff `== 0.0`, frozen P10 prior grad diff `== 0.0`, WM (`act` kept) grad diff `== 0.0` (`test_the_swap_reaches_the_loss_gradient...`, `test_a_tick0_swap_is_the_identity_on_the_whole_batch...`) |

## 2. Evidence (MEASURED)

**pytest** (`CUDA_VISIBLE_DEVICES=-1`, one process, the frozen code): `35 passed in 55.06s`, full verbose output in `pytest_output.txt` next to this file.

**Differential vs the probe's matcher** (`diff_vs_probe.py`, `diff_vs_probe.log`): 20,000 random polylines (stops, waits, turns, loops).
`v8f_dagger == probe` on (r', reason): 19,575; different: 425 (2.1 %), reason differs in 33. With the ONE tie-set line replaced by the probe's, `v8f_dagger` reproduces the probe on **20,000/20,000**
(e_lat included): that line is the only difference (section 5, A1).

**Smoke, the generator end to end** (`smoke/smoke_run.log`, `smoke/PROVENANCE.json`): the REAL `M2b_s0` checkpoint (sha256 `601cb435...`, hmax 16, WM `b`) on 12 SYNTHETIC TRAIN clips (T = 100, `make_synthetic_world`), M2b's real
`argv.json` with the three path overrides, the real eval-141 grid for the K-B3 refusal, CPU 4 threads, batch 32: **804 windows, 1,608 records, 31 s** (39 ms per window), coverage 0.7214, K-B0 804 checked / 0 mismatch,
K-B1 3 chunks (96 windows) / 0 mismatch; cuts tick 1: ok 683, off_head 2, end_edge 119; tick 2: ok 437, off_lat 179, off_head 24, end_edge 164. `smoke/smoke_check.log`: the bank loads by manifest, K-B2 passes against the same
checkpoint, `--p 0.5` swaps 15 of 32 eligible rows, changed keys = base, kin3, p10_t, p10f, pose, pose_s, pose_t (nav did not move in these 15 rows: none had a valid route in the synthetic world), unselected rows unchanged, kept keys are the
same objects, the real model's loss moves 60.855 -> 62.323, the same loss twice differs by exactly 0.0, p = 0 returns the same object.
**These numbers are plumbing only: the features are synthetic, so the cut counts and the coverage say nothing about the real TRAIN set.**

**Mutation** (`mutate_dagger.py`, 57 mutants, one per rule, run on a COPY of the tree so the deliverables were never edited): see section 7.

## 3. API for the trainer worker (the part that is theirs)

```python
from tanitad.data import v8f_dagger as G
bank = G.DaggerBank.load(bank_dir, clip_names=J["clip"], expect_sha256=<--dagger-bank-sha256>)   # sha256 of bank.json; re-hashes every shard
bank.check_on_policy(init_ckpt_sha256)                                       # K-B2
bank.check_train_only(train_clips=clips, eval_clips=eval141, train_window_list_sha256=G.windows_sha256(J["clip"], wins))   # K-B3 (refuses PARTIAL banks)
bank.coverage(wins)                                                          # {"fraction", "blocked", ...}  (COVERAGE)
dag = G.DaggerSwap(bank, p=<--dagger-p>, seed=<--dagger-seed>, kin_mode="yaw", bank_sha256=<declared>)
dag.problems(); dag.built_levers()                                           # for bad_dagger / G-DVB
b = dag.apply(b, step)                                                       # EgoPerturb's contract; dag.last = n_rows / n_eligible / n_selected / dagger_nav_kept / sel / record_idx
```
`p` has default 0.0 and appears nowhere else. The batch must carry `ci` and `t` (the collate does) and the bank's `hmax` must equal the batch's history length (`swap_rows` raises otherwise).
For the G-LIVE probe: `G.REBUILT/KEPT/RECORD` are ORC's tuples; the rc check must use `G.rc_frame_change(rc, rc_ok, src, dst)` (see A4).

## 4. How to run

```
python stack/scripts/v8f_dagger_bank.py gen --ckpt <M2b_sK/ckpt_last.pt> --m2b-argv <M2b_sK/argv.json> --cache-dir <TRAIN cache> --tracks <native_tracks.pt> --join-train <join_train.npz> \
   --eval-grid <eval_grid_windows.json> --out-dir <bank dir> --shard K --n-shards 4 --batch 32 --expect-clips 4512 --expect-windows 306566 [--dry-run | --limit-clips N]
python stack/scripts/v8f_dagger_bank.py merge --out-dir <bank dir>      # after all shards (n-shards 1 merges by itself); prints the bank sha256
CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_v8f_dagger.py ; python <this dir>/mutate_dagger.py <repo root> <python> [--check | --list]
```
Bank layout: `shard_KKKofNNN.npz` (+ `.provenance.json`), `bank.json` (manifest; **its sha256 is the bank's sha256**), `PROVENANCE.json`. Shards are written with a fixed zip timestamp, so the same inputs give the same bytes (tested).

## 5. Ambiguities (the most literal reading was implemented; each is listed with what it would take to change)

* **A1 (DECISION NEEDED): the tie set.** PREREG S2.4 and `THRESHOLDS_FIXED.txt` say "rows whose arc lies within 0.05 m of **that nearest arc**" (`|arc - arc[r*]| <= 0.05`); the probe's CODE (`probe_dagger_cost.py`, `tied = da <= da.min() + ARC_TIE_M`)
  ties every row within 0.05 m of the minimum DISTANCE to s*, which can hold two rows 2 m apart that straddle s* and then lets j pick between them. I implemented the TEXT (`v8f_dagger.py:148`). MEASURED: they disagree on 2.1 % of
  random stop-and-go polylines (section 2); on real data the rate is UNKNOWN. An exact midpoint (two rows equally near, not an arc tie) takes the earlier row. The probe's cost numbers are unaffected. To change: that one line; mutant 10 pins the text.
* **A2 (DECISION NEEDED): nav progress on frames with no route.** The PREREG writes "channel 3 = nav[r',3] + own arc behind NOW / 100 on frames sharing NOW's token and validity (ORC's rule)". Read to the letter this also writes `arc / 100` onto
  frames whose route is INVALID (`nav[:,4] == 0`), where the join stores exactly 0.0 and the model has never seen anything else; ORC's rule leaves those at 0 (it scales a zero difference) and the T1-kin dead reckoning counts down only `if row[4] > 0`.
  I implemented ORC's/T1's behaviour (`NAV_REBUILD_INVALID = False`, `v8f_dagger.py:55,402`); `True` is the letter. The test pins the choice both ways (a mutant flips it and goes RED). The flag is in `built_levers()`.
* **A3: `frame_ok`.** The table says KEPT; the probe's prototype ANDed it with the own item's. Kept (literal). The only frames where they differ are own history rows before the clip start (j < 15 at tick 1), which no reader indexes;
  `check_readable_rows` (`v8f_dagger_bank.py:194`) REFUSES a configuration where a held S frame could reach row < 1, and the swap keeps the logged value on any own row that is not a real kinematic row (ORC's `_hist_valid` rule).
* **A4: `lab_s.rc`.** Entries with `rc_ok` False are zero placeholders (`rc * rc_ok` in the join) with no world point behind them and are left as they are; every other entry is moved with `transform_ckpt`; rows whose two frames coincide bit for bit (a_s = 0) are returned unchanged
  (the "identity at a_s = 0" is exact, not 1e-7). A probe that compares to `transform_ckpt` over ALL entries would flag the placeholders: use `rc_frame_change`.
* **A5: the re-anchor is the identity, bit for bit, when own NOW equals the logged NOW at float32 resolution** (as at tick 0) -- so K-B0 holds on `pose`, `pose_t`, `pose_s` and not only on p10f / kin3 / base. Yaw is not wrapped (as ORC and the probe).
* **A6: K-B1 replays whole CHUNKS** (batch composition identical), 0.5 % of the chunks, at least one, seeded. A lone window re-run is a different computation (a batched forward is not invariant to the batch's composition). The bank is bit-reproducible for the same (batch, device, threads).
* **A7: "the trainer's window-list sha256"** does not exist (`build_data` reports `train_sha256` over CLIPS only). I defined `windows_sha256(J["clip"], wins)` = sha256 of `clip:t` lines in list order; the arm's gate must call it on ITS `wins`.
* **A8: a non-finite own pose or logged row** is counted under a fifth reason `nonfinite` (not one of the PREREG's three cuts), never matched; provenance warns.
* **A9: the third forward is skipped** by a planner wrapper that still lets `v8f_t1_eval.rollout` run unmodified; the hook captures tick 2's state and raises a sentinel. `T1.CAUSAL_STATS` (a global counter) counts tick 2's item build and is recorded in provenance as it is.
* **A10: the draw.** One `[2, B]` uniform draw per step from the private (seed, step) generator regardless of eligibility: row b's selection depends on its position, not on which rows are eligible; the second uniform picks the own state.
* **A11: the bank stores cut records** (the PREREG's list of stored fields includes "the cut reason"). Only `reason == ok` is eligible. Bank size before cuts as the DESIGN_NOTES estimate (ESTIMATED).
* **A12: the bank's `ci` is remapped BY NAME** to the trainer's join at load (`clip_names=J["clip"]`); a clip the join lacks is refused.

## 6. Not verified (UNVERIFIED)

* The real TRAIN cache / `join_train.npz` / TRAIN `native_tracks.pt` are not on the dev box: **no real bank was generated**, real coverage and the real cut table are unknown, and `build_data` ran only through M2b's real `argv.json` with synthetic path overrides (its `--exclude 540:600` slice is empty on 12 clips).
* The CUDA path (the swap on device tensors, K-B1 bit-exactness under cuBLAS, four shards sharing one GPU) was not run; the code mirrors ORC's device handling but nothing was executed on a GPU.
* Box-2 wall time (the DESIGN_NOTES 1.1-1.5 h per seed is ESTIMATED; the smoke measured 39 ms per window on 4 CPU threads, vs 88 ms in the probe).
* Nothing connects the swap to the trainer (flags, `FLAG_CHECKS`, `bad_dagger`, `dagger_probe`, the loop line): out of scope. "Built, tested and unreachable from its caller" applies until that worker lands.

## 7. Mutation result

<!--MUT-->
**54 of 57 mutants killed; SURVIVORS: ['heading cut inclusive (>=)', 'stationary tie: the LATER row on equal time distance', 'nav: the token / validity test is dropped (rebuilt across a manoeuvre boundary)']**; the mutated copy was restored byte-identical to the originals: **True**; the delivered files changed since the snapshot taken before the run: **none**.
Log: `mutation_run.log` (run on a COPY of the tree, so no deliverable was edited while it ran). Each mutant is one rule bent in one place; RED = the named tests of `test_v8f_dagger.py` fail.

| result | mutant (file) and the test selector that must fail |
|---|---|
| RED | lateral cut dropped   -k 'kb4' |
| RED | lateral cut inclusive (>=)   -k 'kb4' |
| RED | heading cut dropped   -k 'kb4' |
| **SURVIVED** | heading cut inclusive (>=)   -k 'kb4' |
| RED | end edge off by one   -k 'kb4' |
| RED | end edge dropped   -k 'kb4' |
| RED | cut order: the end edge before the off-route cuts   -k 'kb4' |
| RED | stationary tie: the first row (the reader's first-minimum rule), not the row nearest in time   -k 'kb4' |
| **SURVIVED** | stationary tie: the LATER row on equal time distance   -k 'kb4' |
| RED | tie set = the probe's band around the nearest distance (not 'within 0.05 m of that nearest arc')   -k 'kb4' |
| RED | tie set = the nearest row only (no 0.05 m tolerance)   -k 'kb4' |
| RED | projection onto the infinite line, not the segment   -k 'kb4' |
| RED | candidate rows start at 0, not at t0   -k 'kb4' |
| RED | the replan step is not 10   -k 'constants' |
| RED | no re-anchoring: the own poses stay in the world frame   -k 'kb5 or swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | the identity short-circuit is dropped (own NOW == logged NOW is not bit-exact)   -k 'kb5 or tick0_swap' |
| RED | re-anchoring drops the heading   -k 'kb5' |
| RED | THE swap line is deleted: `out` is `batch`   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | kin3 is not replaced   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | base is not rebuilt from the own kin3   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | p10_t is the frame before NOW   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | p10f keeps the logged value on every frame   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | pose_t ignores the age   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | pose_s ignores the age   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | nav progress is not rebuilt   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| **SURVIVED** | nav: the token / validity test is dropped (rebuilt across a manoeuvre boundary)   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | nav: progress is rebuilt on frames WITHOUT a route too (the PREREG's letter; see NAV_REBUILD_INVALID)   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | nav: the distance behind NOW is dropped   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | lab_s.rc keeps the logged frame (no frame change)   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | rc: the identity at a_s = 0 is not exact (no coincident-frame short-circuit)   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | rc: placeholders (rc_ok False) are moved too   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | the natural-frame logic uses the OWN pose_s on both sides (rc moved by the wrong source frame)   -k 'swap_reachability or swap_item or rc_is or rebuild_is or nav_progress or tick0_swap' |
| RED | the draws come from the global RNG   -k 'draws_are or p_zero or p_is_a_parameter or draw_among' |
| RED | eligibility is ignored (every row may be selected)   -k 'draws_are or p_zero or p_is_a_parameter or draw_among or swap_reachability' |
| RED | p = 0 returns a copy, not the same batch   -k 'draws_are or p_zero or p_is_a_parameter or draw_among' |
| RED | the draw among a window's states always takes the first   -k 'draw_among' |
| RED | a cut record is eligible (only reason == ok may be swapped in)   -k 'off_route or kb2' |
| RED | K-B3: the eval-clip refusal is removed (both checks)   -k 'kb3' |
| RED | K-B3: the TRAIN-subset check is removed   -k 'kb3' |
| RED | K-B3: the window-list sha256 is not compared   -k 'kb3' |
| RED | a partial bank may train an arm   -k 'dry_run' |
| RED | K-B2: the on-policy check is removed   -k 'kb2' |
| RED | the shard's sha256 and content hash are not verified at load   -k 'shard_roundtrip' |
| RED | the declared bank sha256 is not compared   -k 'shard_roundtrip' |
| RED | COVERAGE counts states, not windows   -k 'kb2' |
| RED | the rollout configuration constant says nav = logged   -k 'constants' |
| RED | the off-route lateral constant is moved   -k 'constants or kb4' |
| RED | p is given a non-zero default   -k 'constants' |
| RED | tick 0 is stored as an own state   -k 'shard_roundtrip or bank_holds or bank_rows_are' |
| RED | the history window is shifted by one row   -k 'shard_roundtrip or bank_holds or bank_rows_are' |
| RED | K-B0 is not checked   -k 'tick0_identity' |
| RED | K-B1 never compares the replay   -k 'kb1' |
| RED | the eval-clip refusal is removed from the generator   -k 'kb3_the_generator' |
| RED | --limit-clips does not mark the bank partial   -k 'dry_run' |
| RED | every shard takes every clip   -k 'dry_run' |
| RED | the replay set is empty   -k 'kb1 or generator_is_deterministic' |
| RED | the rollout is not the decisive configuration (--limit logged instead of causal)   -k 'bank_rows_are' |
<!--/MUT-->

## 7b. The three survivors of run 1, and the tests that kill them (2026-10-26; TESTS ONLY, `v8f_dagger.py` and `v8f_dagger_bank.py` untouched)

Run 1 (section 7) killed 54 of 57. The three survivors named three tests that were too weak. Five tests were ADDED to `stack/tests/test_v8f_dagger.py` (35 -> 40, **40 passed in 75 s**, `out/pytest_full_after.txt`); one
existing assertion's MESSAGE was corrected (comment only). The test file's sha256 moved `4a44431f...` -> `a4f53cd1...`; the two code files are byte-identical (`831f64f8...`, `41a31de9...`, sha256 before = after).

| # | survivor | why it survived (MEASURED / proved) | added test(s) | on the copy (`out/survivor_check.log`) |
|---|---|---|---|---|
| 1 | heading cut inclusive (`>=`) | the old test probed 29.9 and 30.1 deg only. **30 deg EXACTLY is unreachable through `wrap()`**: it adds pi, so the result lies on a 2**-51 rad grid that does not contain radians(30); the nearest reachable errors are 29.999999999999993 and 30.000000000000018 deg. So `>` and `>=` differ only at the float the matcher COMPUTES, never at the literal 30.0 | `test_kb4_a_heading_error_EXACTLY_at_the_cut_is_kept_the_cut_is_strictly_greater`: (1) ulp-neighbour sweep of radians(30): kept iff the computed error <= the cut, and the sweep must straddle it; (2) `OFF_ROUTE_HEAD_DEG` moved onto the computed error (equal -> kept, one ulp below -> cut); (3) the lateral twin by the same device | RED |
| 2 | stationary tie: the LATER row on equal time distance | **EQUIVALENT mutant at an integer clock row, proved**: the tied rows are an arc stretch, and the arc is non-decreasing, so they are a CONTIGUOUS run [a, b]; for an integer j the nearest row is `clip(j, a, b)` and is unique (two rows equally far on both sides of j would put j inside the run, at distance 0). The clause "then the earlier row" (PREREG section 2 item 4) therefore never fires through `match_route`. The old assertion `j = 24.5` was no tie either: `int(j)` truncates it to 24 (its message is corrected) | `test_kb4_a_time_tie_cannot_occur_at_an_integer_clock_row_so_the_nearest_in_time_row_is_the_clamp` (all integer j x 4 stretch shapes; PASSES under the mutant, by design: it is the proof of equivalence) and `test_kb4_the_secondary_key_of_the_tie_break_is_the_earlier_row_pinned_through_a_forced_time_tie` (a **white-box SEAM**: `np.lexsort` is wrapped so its primary key, the time distance, is all zeros; the answer must be the EARLIER row; fails loudly if the matcher stops using `np.lexsort`) | RED, **by the seam only** |
| 3 | nav: the token / validity test dropped | the natural batch has no REAL, route-valid frame with ANOTHER token (the old test's `n_kept > 0` counted frames whose validity differed, which the `valid > 0` clause also keeps) | `test_nav_progress_is_rebuilt_only_on_frames_sharing_NOWs_token_and_validity[token, validity]`: a constructed batch (progress 2.0 logged, NOW 0.5): frames 3..5 on another token must keep exactly 2.0 while the rest follow the own arc; with NO route at NOW no frame may change | RED (both params) |

**Honest scope of survivor 2.** Its kill is by a forced-tie seam, not by reachable behaviour: no input can distinguish the mutant from the code. The code is NOT wrong against the PREREG (the rule is implemented as written; the secondary clause is
vacuous at an integer j). Consequence for the register: PREREG item 4 reads "then the earlier row" as if it could fire; it cannot. Nothing to change; it should not be read as a tested behaviour.
**Honest scope of survivor 1.** The test pins the STRICT comparator and the pinned constant (30.0, `test_constants_...`) separately; the literal "exactly 30 deg" state does not exist in float.

**Run 2 (the FULL proof, all 57 mutants on the COPY with the new test file)** -- `mutation_run_2.log`:
<!--MUT2-->
SUPERSEDED at integration (2026-10-26, build `di`): this worker-level run was not the proof of record. The proof of record is the run of the same 57 mutants on the FINAL integrated bytes (`v8f_dagger_bank.py` carries the E1 patch), `../2026-10-26-dagger-integration/FINAL.txt` and `p1b_mutate_dagger.log` next to it.
<!--/MUT2-->

## 8. Deliverable manifest

Paths relative to the tree `E:/Projects/TanitAD-artifacts/dg/tree` (NEW files only; no existing file was edited):

* `stack/tanitad/data/v8f_dagger.py`  -- matcher, re-anchor, bank format + loader + controls, swap
* `stack/scripts/v8f_dagger_bank.py`  -- the generator (gen / merge)
* `stack/tests/test_v8f_dagger.py`  -- 40 tests
* `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-dagger-build/`: `NOTES.md`, `mutate_dagger.py`, `pytest_output.txt`, `diff_vs_probe.py`, `diff_vs_probe.log`, `smoke_check.py`,
  `smoke/` (`bank.json`, `PROVENANCE.json`, `shard_000of001.npz`, `shard_000of001.provenance.json`, `smoke_run.log`, `smoke_check.log`), `mutation_run.log` (run 1: 54 of 57), `mutation_run_2.log` (run 2, after the survivor kill; section 7b, written by `finish_mutation2.py`), and `out/` (`survivor_check.py`, `survivor_check.log`, `pytest_full_after.txt`, `DELIVERABLE_SHA256.txt`)

Lives in ONE place only: `E:/Projects/TanitAD-artifacts/dg/mutation_run/mutate_run.log` (the detached mutation run's log), `dg/finish_mutation.py`, `dg/HANDOFF.md`, `dg/mut/` (the mutated COPY, disposable), `dg/smoke_world/` (the synthetic TRAIN world of the smoke).
