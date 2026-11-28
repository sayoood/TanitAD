# clamp_gpu -- the clamp-ON battery path + K2 / K4 (AMENDMENT A-R8-CLAMP-1), built and CPU-tested, NOT run

EvalFlyWheel, 2026-10-18. Single copy of this package: `C:/Users/Admin/qland/hold_r8batt/stage/.../2026-10-15-refcv8-milestone-battery/` (not staged, not committed).
Tier: T1 self-action open loop. These controls are deterministic identity checks; no interval, no sample. **Nothing here ran on the GPU. The live battery holds the lock.**

## Headline (decisions / findings for the Master Mind)

1. **The tree under test already carries the pose-shift fix, so `R8_NO_POSE_SYNC=1` alone does NOT give the banked `_unc` clock on `tree_de4e15b`.** `refcv7_loader.py:801-802` (de4e15b) calls
   `enable_pose_sync` whenever the argv names `--pose-sync-sidecar`; the fb47a58 loader never did. K2 compares against `step30000ema_unc` (standalone clock), so `r8clamp_patch.py` blanks the
   sidecar for the dataset build under `R8_NO_POSE_SYNC=1` and asserts the result is UNSHIFTED (and asserts SHIFTED on the corrected clock). Without this K2 would compare two different eval sets.
2. **Config-level parity is already MEASURED (CPU, no model):** from refcv8's own argv, fb47a58 and de4e15b build identical parsed args (278 fields) and an identical pinned model config
   (277 leaves) except exactly three ADDED, inert-by-default fields: `p10_prior`, `p10_prior_sha256`, `plan_vmax_clamp` (`CFG_PARITY_fb47a58_vs_de4e15b.json`). A K2 failure therefore cannot have a
   config cause; it would sit inside a forward (`refc.py` / `kinematic_prior.py` changed in the same commit) or be GPU nondeterminism. The optional control (below) separates those.
3. **K2 is a PREFIX roll, not "3 arbitrary episodes".** `refcv3_arm.py:1999` seeds the generator ONCE and draws sequentially, so a window's DDIM draw depends on every window before it. Only the first
   K episodes of the banked dump, rolled from window 0 with the same (full) arm set, see the same stream. `nav_shuffled.*` is excluded from the bit-parity verdict (the shuffle is a permutation
   over the whole window list, `refcv3_arm.shuffle_nav`); `nav_true.traj` is the primary key, every other shared key is reported beside it.
4. **K=3 would make K4 vacuous.** On `step30000ema_unc` the first three episodes hold 3 post-hoc-binding `nav_true` rows; K4 needs >= 5 (committed). The planner picks the smallest prefix with
   >= 10 binding rows: **K = 10 episodes, 342 windows, 12 binding rows** (`k2k4_prefix_plan_step30000ema.json`, episodes named by clip_id + sha12). Override with `K2K4_N_EPISODES`.
5. **K4's 1e-6 m tolerance is below float32 resolution for any waypoint beyond ~8 m** (1 ulp = 3.8e-6 m at 32-64 m). A 1-ulp GPU-vs-CPU difference on a binding row would read FAIL. I did NOT relax it:
   the verdict stays as coded, the max error in ULPs is printed beside it, and `--gpu-posthoc` runs the SAME vendored function on the GPU so device numerics are separable from logic.
   Relaxing the tolerance is a Master Mind amendment, not mine.
6. **Integration needed (nothing is wired into a live chain):** (a) pick when `queue_k2k4_r8.sh` runs (it waits on the shared lock; ~30-40 min GPU, 45-55 with the control); (b) after K2 AND K4 PASS,
   the final-checkpoint battery runs through `run_battery_r8clamp.py --clamp on` and `--clamp off` (commands below); (c) PYTHONPATH for those runs must name the clamp tree, never fb47a58 (the patch refuses a wrong tree).

## Commands (exact; none was started)

```
# K2 + K4 (+ optional control).  Waits for the shared GPU lock exactly as chain_battery_r8.sh does; never breaks a foreign lock.
nohup sh C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code/queue_k2k4_r8.sh 30000ema > E:/refcv8_eval_kit/chain/queue_k2k4.out 2>&1 &
#   K2K4_CONTROL=1  adds the fb47a58 control roll;  K2K4_N_EPISODES=12  overrides the prefix
#   markers: E:/refcv8_eval_kit/chain/k2k4_step30000ema.log  (ZZK2K4STARTZZ ... ZZK2K4VERDICTZZ K2=.. K4=.. OVERALL=.. ... ZZK2K4CHAINENDZZ)
#   result : raw/clamp_gpu/K2K4_step30000ema.json     rolls: E:/refcv8_eval_kit/clamp_gpu/step30000ema/{off,on,ctl}/dump_s0

# the battery itself, clamp ON (primary, A-R8-CLAMP-1 s2.1) and OFF (beside), AFTER K2+K4 pass; tag MUST contain clampon / clampoff
set R8_NO_POSE_SYNC=1  (standalone clock)  or unset (corrected clock, then --no-baselines --os-only-all as chain_battery_r8.sh does)
PYTHONPATH='E:/refcv8_eval_kit/tree_de4e15b/stack;E:/refcv8_eval_kit/tree_de4e15b/taniteval' \
  python -X utf8 code/run_battery_r8clamp.py --clamp on  --tag step<N>_clampon_unc  <the usual run_battery_r8.py args>
  python -X utf8 code/run_battery_r8clamp.py --clamp off --tag step<N>_clampoff_unc <the usual run_battery_r8.py args>

# OWED (needs >= 6 GB free commit; the box had 3.7 GB): the real-model CPU smoke of the whole path
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONPATH='E:/refcv8_eval_kit/tree_de4e15b/stack;E:/refcv8_eval_kit/tree_de4e15b/taniteval' \
  python -X utf8 code/run_battery_r8clamp.py --clamp on --smoke-cpu --max-clips 2 --max-windows-per-clip 3 --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt \
  --no-g0 --no-baselines --infer-seeds 0 --tag smoke_clampon --out-root E:/refcv8_eval_kit/battery_clampsmoke
```

## What each piece asserts

| piece | asserts | REFUSES / goes RED when |
|---|---|---|
| `r8clamp_common.check_provenance` | PROVENANCE.txt names de4e15b19a39...; 8 load-bearing files are LF-normalised byte-equal to `git cat-file blob de4e15b:<p>` (both reads non-empty, object name 40 chars); clamp markers (`traj_precap`, `plan_cap_binding`, `cap_plan_to_limit`) are IN the files | any mismatch, an empty/short git read (INCONCLUSIVE, never "match"), a missing marker or file |
| `r8clamp_patch.install` | `tanitad` + the loader come from the tree; pose clock as stated; ON sets `cfg.plan_vmax_clamp=True` after build and replicates the trainer's build refusals (needs `max_speed_onehot_v6`, no `refcv7_select`, `hier`, eval mode); child redirected to `roll_seed_r8clamp.py` | wrong tree; any refusal above; ON forward lacking `traj_precap`/`plan_cap_binding` (every forward); ON with a limit that is +inf on all of the first 30 rows (built and inert); OFF forward carrying either key |
| `k2k4_compare K2` | `nav_true.traj` bit-equal (`np.array_equal`) on every window, rows keyed `(clip_index, ws)` like `nr_matched_instrument`; >= 3 episodes; no window absent from the banked dump | one ulp in one row reads FAIL (tested); < 3 episodes / missing window / missing key = INCONCLUSIVE |
| `k2k4_compare K4` | per condition in {nav_true, nav_shuffled, nav_flipped, nav_zero}: ON `traj` vs vendored `cap_plan_to_limit(OFF traj)` within 1e-6 m, limits from `clamp_posthoc_r8.fed_ceiling`, v0 from the extras; >= 5 binding nav_true rows | > 1e-6 on any row; nothing binds = INCONCLUSIVE; secondary (reported): ON `traj_precap` == OFF `traj`, forward `plan_v_lim_ms` == `fed_ceiling`, forward `plan_cap_binding` == post-hoc set |
| `queue_k2k4_r8.sh` | provenance + plan BEFORE the lock; both rolls and the compare INSIDE it; the EXIT trap releases on every path | covered by 9 sandbox tests with stub scripts |

Clamp semantics already pinned independently (not an expression over the code under test): `test_vendored_function_matches_the_independent_analytic_clamp` compares the vendored function with an analytic
straight-plan expectation (`progress_i = min(v_plan, max(limit, v0 - 2.5 t_mid)) * dt`), and shows the strict (no-brake) variant differs.

## Verified (CPU) vs UNVERIFIED until the GPU run

* VERIFIED: `test_r8_clamp_k2k4.py` 22 passed, `test_queue_k2k4_r8.py` 9 passed (logs here); provenance of the real tree passes (`provenance_check_tree_de4e15b.json`); the install runs in a fresh process
  on the real tree (tanitad + loader resolve inside `tree_de4e15b`, build/dataset hooks present, clamp keys appended, child redirected); both wrappers import and reach argparse; a wrong tree is refused;
  config parity (headline 2); the prefix plan on the real banked dump with its clip<->index map proven by sha12.
* UNVERIFIED: that the real model's forward emits `traj_precap` / `plan_cap_binding` under the post-build flag (the hook refuses on the first forward if not); K2 itself; K4 itself; that
  `obey30`, `filter_off`, `oracle_sel` behave sensibly with the clamp ON (in ON mode `filter_off.traj` is the clamp applied to the filter-off plan and `obey30.traj` is clamped to 30 km/h; the oracle
  selects among UNclamped fan members) -- decide what the bars read before the final-checkpoint run; G-DVB on the clamp tree with refcv8's config (a refusal there is a real finding, not a kit bug);
  GPU bit-reproducibility of the rig (the optional control answers it); the ~15 min per-roll estimate (from the banked 1.8 s/window).

## Files (all NEW; in `code/` unless noted)

`r8clamp_common.py` `r8clamp_patch.py` `roll_seed_r8clamp.py` `run_battery_r8clamp.py` `k2k4_compare.py` `queue_k2k4_r8.sh` `cfg_parity_probe_r8.py` `test_r8_clamp_k2k4.py` `test_queue_k2k4_r8.py`;
here: `k2k4_prefix_plan_step30000ema.json`, `k2k4_prefix_windows_step30000ema.json`, `provenance_check_tree_de4e15b.json`, `CFG_PARITY_fb47a58_vs_de4e15b.json` (+ the two dumps), `pytest_*.log`.

## K4 criterion -- Master Mind decision 2026-10-18 ~20:40 Berlin, BEFORE any K4 number exists

A condition PASSES iff every value is finite AND (max |d| <= 1e-6 m OR max float32 ULP over all rows <= 4). Reason: 1e-6 m is below
float32 spacing beyond ~8 m, so a GPU forward vs a CPU post-hoc of the SAME arithmetic could FAIL on rounding alone; 4 ulp is <= 6e-5 m even at
256 m, while a wrong clamp moves a waypoint by centimetres. The 1e-6 m was the brief's number, never in A-R8-CLAMP-1 (which says "equals").
Pinned by `test_r8_clamp_k2k4.py::test_k4_ulp_band_is_the_registered_criterion` (1 ulp PASS, 8 ulp FAIL, 1 cm FAIL; mutation ULP_TOL=2 goes RED).
