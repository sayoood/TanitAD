# RESULT — S8 / T2-A: closed-loop (T2) evaluation of refcv7-50,400, prepared for box 2

**Verdict: the instrument is built and the dry path is proven on CPU; no T2 number exists yet, and none is claimed.** The GPU run is blocked on box 2
(unreachable, B1) and on two things only a GPU can answer (gsplat on Windows, B2; renderer + model on 8 GB, B3) — `preflight_box2.py` answers both in minutes.
Nothing here trains, downloads, touches Thor, or uses the dev box's GPU (`CUDA_VISIBLE_DEVICES=-1` and `assert not torch.cuda.is_available()` in every CPU entry point).

Tier T2. **The loop is closed on PERCEPTION, not on INTERACTION** (nothing responds to the ego; no map; no collision response).

## What exists (priority order of the brief)

1. **Inventory** — `INVENTORY.md`: inputs and the paths probed, checkpoint, dependencies, VRAM vs 8 GB, which metric families are emitted (evidence-classed).
2. **CPU dry path** — harness patch + stub renderer + policies + scorer + 24 tests (22 fast, all green; 2 slow, real checkpoint on CPU; `raw/pytest_fast_final.log`, and again from inside the shipped kit: `raw/pytest_fast_from_kit.log`).
   - Known-value smoke: the hold-speed floor drives a straight line the analytic distance (15 m/s × 49 intervals × 0.1 s = 73.5 m), steer and accel exactly 0.0.
   - **Deliberate-regression arms go RED**: a steer bias, an accel offset, a wrong tick length each fail the same checker (`test_deliberate_regression_*`); a `"kappa"`-units mutation over-rotates a circle ≈2.9× (R46) and does not reproduce the banked P10; the box-2 zero-control guard goes RED on a 1e-9 steer or accel.
   - Loop closed on perception: the pose handed to the renderer is the executed pose (full 4×4 compare).
   - Real model: refcv7-50,400 builds strictly on CPU through the launch gate's G-EVAL loader and plans one tick through the stub loop (`raw/refcv7_cpu_tick.json`: finite 4×2 plan, 15.08 m/s first-waypoint speed against the stub's 15 m/s, `observed_frac` 0.9035; 165-201 s per tick on CPU).
   - Kit entry point + scorer on 5 stub floor arms: `raw/dry_stub_scored/` (zero control reads exact zeros). **These stub numbers are plumbing checks, not driving results.**
   - **Reachability from the actual caller** (CLAUDE.md "built, tested, unreachable" class), argv -> `main()` -> policy -> model call, on the real checkpoint (CPU, one stub tick each; `raw/refcv7_argv_reach_*.json`): `--t2a-infer-seed 3` moves the plan 0.72 m against seed 0 (the seed reaches the sampler); `--t2a-vmax-mode logged_2_6s` and `fixed` deliver `v_max_valid` 1.0 and the right m/s to the model call; the box-2 render flags `--cull-scale-quantile 0.95 --sky-gain 0.3` reach the renderer (`test_box2_render_flags_reach_the_renderer_through_main`, with a no-flags control). **Not established: that the ceiling moves the emitted PLAN** (finding 1 below).
   - **The whole box-2 orchestration ran end to end on the stub** (`run_on_box2.py` -> `t2a_run.py` -> `t2a_score.py` -> pairs -> table): it exposed a real bug (the scorer could not import `refb_labels` from a clean interpreter), now fixed.
3. **Box-2 kit** — `RUN_ON_BOX2.md` (exact copy list with sizes; ≈ 2.96 GB kit, optional control and the R4 tree included), `code/preflight_box2.py`, `code/run_on_box2.py` (2 scenes × {refcv7 off/log/off-seed1, P10 held+live, `ha0_ext` held+live, zero control} + optional refc-base control; resumable; per-arm JSON stamped `"tier": "T2"`; `action_units="steer"` wherever a steer angle is integrated). On the dev box the preflight returns 26/29 with exactly the three expected failures (msgpack, gsplat, no visible GPU): `raw/preflight_devbox_expected_3_hard_failures.json`.
4. **Blockers** — `BLOCKERS.md` (10 items, each with what unblocks it).

## Findings worth the orchestrator's attention

1. **The speed ceiling is probably inert in the closed loop, and the dry path could not prove it either way.** `refcv7-50,400` is the pre-restart run `refcv7-r101-s0` (`--conflict-every 10`, launch `fec3a0dc`); SPEC_REFCV7 §26.1 (INHERITED) says on that code the ceiling never reaches the emitted plan. A 5 m/s ceiling left the plan bit-identical on `fec3a0dc` **and** on the R4 fix commit `0fcb1c1c` (MEASURED, CPU, 1 tick). That does not discriminate: with no compliant candidate `SpeedCeilingFilter` keeps the whole fan by design (`refcv6_selection.py:187-192`), and my first pre-registration (identical = dead) omitted that fallback — corrected in the test docstring, not hidden. Practical consequence: the two oracle-ceiling arms are demoted to low-yield, run last; scene 1's oracle bin (100 km/h, 27.8 m/s) cannot bind a ≤ 20 m/s plan at all. The headline arm withholds the ceiling. A discriminating control is a 0-GPU follow-up (BLOCKERS B5).

2. **The working-tree branch cannot build refcv7** (`agent/arch-inf-20260803` has no `kinematic_prior.py` and no `refcv7_loader`). The kit is a `git archive` of the launch commit `fec3a0dc` plus an overlay. MEASURED.
3. **`taniteval/taniteval/floors_kin.py` is untracked**; the P10 weights existed only in `C:/Users/Admin/s1_floors_scratch/` (now also in `assets/`). Three single-location artifacts.
4. **Per-tick echo floors are unit-gain feedback loops, so any bias integrates.** On the stub's exactly-straight world the live P10 reads ADE_0_2s +1.539 m against +0.107 m for the same P10 held from t0 (`raw/dry_stub_scored/T2A_SUMMARY.md`) — all of it is the floor's own bias fed back. Hence both variants run on box 2, and the live echo is the one that must be compared to the model, not the held one alone. (Dry-path plumbing evidence, not a driving claim.)
5. **R46 unit trap pinned**: recorded channel 1 is a steer ANGLE `arctan(2.9 κ)`. `ha0_ext_paths` and `const_control_paths` default to `action_units="kappa"` — every call in this package passes `"steer"` explicitly, and the circle test fails if a kappa-unit integration is used.
6. **The oracle ceiling is definition-sensitive on scene 2** (whole-clip max → 100 km/h bin; the [2 s, 6 s] window → 30 km/h bin, margin 17.6 km/h). The headline arm withholds the ceiling.
7. **Neither scene was seen in training** (`raw/leak_check.json`), so a T2 result is not a train-set echo.

## Not done, and why

- No GPU run (B1-B3). No T2 number. The pre-registered reading of the eventual numbers is committed in `RUN_ON_BOX2.md` §6 before any exists.
- Collision / off-road UNAVAILABLE (B8); STRATEGIC family PAUSED (PI ruling 2026-09-19, item 25 — carried in every per-arm JSON, never as a pass, never omitted).
- Not staged / not committed (brief): everything below is untracked or unstaged in the working tree for the orchestrator to land.

## Deliverable manifest (every artifact and where it lives)

Package (repo, untracked, ready to `git add`): `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-t2a-closedloop-prep/`
`RESULT.md` · `INVENTORY.md` · `RUN_ON_BOX2.md` · `BLOCKERS.md` · `HANDOFF.md` · `code/` (11 files incl. `tests/test_t2a_dry.py`) · `assets/` (P10 weights + known-value fixture, 65 KB) · `raw/` (evidence).

**Single-location items — land these:**
| item | where it lives now |
|---|---|
| harness patch (EXTRA_ARMS + `set_history`, +35/-1) | `stack/experiments/alpasim-gsplat/closedloop_drive.py` — **modified, unstaged** in the working tree |
| `floors_kin.py` | `taniteval/taniteval/floors_kin.py` — **untracked** |
| the assembled 2.96 GB kit (copy this to box 2) | `C:/Users/Admin/t2a_scratch/box2_kit/` — **C: only**; rebuildable with `code/make_kit.py <dest> --assets --control --md5` |
| launch-tree scratch + code-only kit used by the tests | `C:/Users/Admin/t2a_scratch/{launch_tree,kit}` — C: only, disposable |
| refcv7 checkpoint / scenes / refc-base | `E:/refcv7_eval_kit/ckpt/`, `E:/thor_archive/2026-09-14/...` — pre-existing, not moved |

Staged under 15 GB on C:: 3.42 GB in `C:/Users/Admin/t2a_scratch/` (MEASURED `du -sb`: `box2_kit` 2.96 GB + the scratch trees `launch_tree`, `kit`, `r4_tree`, dry outputs; all disposable except `box2_kit`).

Test status, exactly: 22 fast green (package and again from inside the kit). The two slow tests pass individually on the real checkpoint (CPU): `test_refcv7_one_cpu_tick_through_the_stub_loop`; and `test_refcv7_argv_flags_reach_the_model_through_main`, whose three components ran as `off_s3` + `log_s0` (together, before its final edit, which only REMOVED the plan assertion on the ceiling) and `fixed5` (final version, `raw/refcv7_argv_reach_final_test_fixed5.log`). The final 3-tag version has not been run as one ~11-minute piece.
