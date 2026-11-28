# HANDOFF E3 to E6 (T2 re-perception rollout, v8F-mini) -- 2026-11-06, tip 24b97ba9

Evidence class: MEASURED (ours; `logs/pytest_e3.log`, `logs/mutations_e3.log`; hashes in `PINS.txt`). CPU only, no network, no box 2, nothing committed or staged, no kit file edited
(`closedloop_drive.py` is untouched, so the historical arms are bit for bit; its sha in PINS is the exported copy's).

## DONE (E3)
`code/t2v8f_exec.py`: `PlaybackExecutor` (calls the T1 tool's `envelope`, `plan_to_world`, `hermite_frames`: the tool is loaded by path, sha256 checked), `run_t2_rollout`
(logged warm-up, 3 replans x 10 knots, `set_history(.., n_new=1)` on EVERY raw tick, render on grid ticks only; odd ticks carry `None` and the policy never reads them),
conditions `T2` (executed pose and time, z/roll/pitch from the logged surface), `RR` (tool's `reindex_rows` anchored at t0, tau = running max of min(i, tau*)), `logged` (control).
It returns `cl` (executed knots in the t0 ego frame, the tool's `exec_in_ego`), `g` (logged knots), `v0`: the T1 window arrays.
Gates `tests/test_t2v8f_e3.py` (3 passed) + `code/run_mutations_e3.py` (3 of 3 RED):
* T2g: logged plan by playback -> grid rows equal the logged track, max dev 1.4e-14 m (3 scenes); odd (Hermite midpoint) rows deviate up to 1.3e-3 m, reported not gated. Mutation: knot shift, 85.5 m.
* RR index: at the logged speed RR equals logged playback (pose dev 0.0, time dev 0 us); at half speed RR lags 16.9 m, tau monotone. Mutation: index = clock -> lag 0 -> RED.
* Live path: `plan(frames, intr)` with stub-renderer frames and an encoder returning the banked tokens of the logged frame equals the cached-feature plan bit for bit (2 windows). Mutation: tokens off by one step -> 0.068 m.

## LEFT (in this order; each is independent of the next)
1. E4 (`code/t2v8f_kit.py`, not written): register C0_s0, C0_s1, DAG-OACC_s0/_s1 (base ckpt U, see HANDOFF_E1E2), P10, ZERO, BLIND in `closedloop_drive.EXTRA_ARMS`
   (entry `{build, needs_ckpt, add_args}`, as `t2a_policies.register`); dispatch by replacing `cd.run_rollout` in the entry script (no kit edit): v8F arms -> `run_t2_rollout`, others -> original.
   * Nav: the harness' `nav_from_route` is NOT the T1 tool's nav. The tool reads `join_eval141.npz` rows built in `v8f_mini_data.load_join` (lines ~199-209: `nav_token`, `nav_args_valid`, `nav_d_next_m`
     -> one-hot(3), d/100 clipped, valid). Find the producer of those columns for a logged route, call it on `gt_T` rows, and test equality on a banked eval-141 window. `NAV_FROM_HARNESS` is UNVERIFIED.
   * BLIND = banked training mean of the scene tokens (`cache` stats, CPU). ZERO = policy arm `p10`. Starts from `t2_scenes/raw/SCREEN_TABLE.json` (`scenes[].launch_events[].t_c_s`; 10 Hz ticks; t_c-4 s, t_c-2 s;
     guards `default_rng(1)` uniform over the renderable span leaving 6 s). `f0` must satisfy `f0 >= 18` (window) and `f0 + 60 < n_ticks`.
2. E5 (`code/t2v8f_score.py`): the T1 window functions live in `dagorc_rebase/read/ref/readA_code/rr_lib.py` (`rows_of`, `m1_window`; sources AST-extracted from the pinned readers in the scratchpad `rd/`:
   `t1k_read.comps`, `t1klc_read.lat_rows`, `lcdiag_common.pair_err`), taking `cl`/`g` [N,30,2]. They give 10 comps/lateral rows + M1. The other 8 rows (DK-VIOL/HW/TG, TAC-LAT/LON, S-POS@30/50/80) need the lead block, the eval join
   and the ticks (`dag2_read.tacdk_context`, `T1KF.prep`): NOT available for NuRec scenes yet -> report them per family as NOT COMPUTED with the reason, as a work item. Bootstrap: `taniteval/ci.py` paired, B 2000, seed 0.
   Contrasts X1, X2 and the section-6 outcome as a pure function; tier stamp T2.
3. E6 remaining: T3 (ZERO == P10 exactly: policy `arm="p10"` is already pinned to the tool's floor in E1 G3; add the scorer side: attribution of P10 against itself reads exactly 0),
   T4 (RIC0_s0 pin: `ri_read/raw/arrays/RIC0_s0.npz` a_k/b_k vs the scorer on `F:/TanitAD-scratch/motion_c/ri/out/RIC0_s0/dump_RIC0_s0`, bit for bit on the labels the functions cover), each with a mutation.
4. Stage-1 package (`stage1/`, not written): `launch.ps1`-style job for the 2 scenes on hand: render gap (rendered frame at the logged pose vs the real frame, pooled-token cosine), T5 determinism,
   actor falsifier (`objects` draws the lead; planted no-lead scene reads absent), a plumbing run of every arm; pull folder with done/fail markers; safe mode (one job); rehearsal on CPU with
   `T2A_STUB=1` (`t2a_run.py`) and the stub renderer. Do NOT launch.
5. Re-run `pytest tests -q` (E1/E2 file too), then refresh `PINS.txt`.
Blockers: none for 1-5. The G2 encoder gate and the E1 gates still pass as before (not re-run here).
