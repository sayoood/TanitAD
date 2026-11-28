# HANDOFF -- P6 D7 fix (2026-10-16). State, artifacts, exact next steps. No secrets in this file.

**State.** Lane-rank twins are scored on their own route. Verified twice (pre-freeze and on the frozen tree), through the real `build_metric_cache.py build --lane-rank K`, on 74 twins (51/14/9 at K=1/2/3):
- twin centreline == PDM-Closed's search from DriveRL's own rank-K lane (all points), **74/74**; the verifier reproduces NAVSIM's logged centreline 74/74.
- route digest differs 71/74 (the 3 equal: PDM's own lane IS DriveRL's rank-1 lane). Clamp rate 0/74.
- rank 0 untouched: 12/12 parts identical modulo nuPlan `track_id`, score_plans equal.
- mutations: initializer removed -> build fails loud (rc 1, 0 files); initializer+guard removed (= P6 defect) -> (a) 0/6, (b) 0/6.
- suites on the pod, frozen tree: test_pdm_route_rank 79, test_d9_accept 16, test_p6_chain 340, test_p6_cpu_probe 111, all PASS (dev box: 64, no numpy).
- NOT passed: pre-registered semantic check (c), see RISK 1.

**Fixed code dir (pod):** `/workspace/ops/d7fix/pkg` = shipped c1dc5d3 tree + 6 files; hashes in `pkg/FIXED_FILES.sha256` (also in this package). Frozen. The same 6 files are in `PK/refe/` and `p6_d7_pdm_route_rank.patch` (`git apply --check` clean on c1dc5d3): `refe/{build_metric_cache.py, pdm_route_rank.py (new), p6_pod.sh, test_p6_chain.py, test_pdm_route_rank.py (new), diag_d7_twin_route.py (new)}`. To land: those 6, then regenerate SHIP_MANIFEST.

## STEPS (pod, standalone, outside the chain)
```bash
source /workspace/teacher_env.sh; PY=$DRIVERL_EVAL_PYTHON
FIX=/workspace/ops/d7fix/pkg; CB=/workspace/data/refe_navtrain/train_grow_cmd; MC=/workspace/data/refe_navtrain/metric_cache; DBS=/workspace/data/navtrain_dbs
cd $FIX && sha256sum -c FIXED_FILES.sha256 && cd refe
# 1. QUARANTINE the stale twins (a move on one volume). Pod today: 32,929 rank-1 files, 0 rank-2, 0 rank-3. (Command shape dry-run on a scratch cache: ok.)
for K in $($PY p6_prep.py lane-ranks --bank $CB | sed -n 's/^ZZLANE_RANKS//p'); do      # prints ZZLANE_RANKS 1 2 3
  $PY p6_prep.py quarantine --cache $MC --k $K --dst /workspace/data/refe_navtrain/metric_cache_twins_stale_pre_d7fix; done   # expect ZZD7_QUARANTINED 32929 / 0 / 0
# 2. BUILD (launch with setsid nohup), one rank after the other; resumable (an existing file is skipped, stat-only for a finished log)
for K in 1 2 3; do
  env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK -u REFE_D7_MUTATE -u REFE_PAUSE_FILE OMP_NUM_THREADS=1 nice -n 19 \
    $PY build_metric_cache.py build --targets $CB --out $MC --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers 6 --lane-rank $K \
    > /workspace/data/refe_p6/d7fix_lanes_$K.out 2>&1; done     # success: ZZMETRIC_CACHE_DONE built=<N> ... failed=0 ; the first line must read "pdm route rank K: _get_starting_lane PATCHED"
# 3. DISTINCT check afterwards (the chain's own command, a bigger sample)
for K in 1 2 3; do $PY p6_prep.py lane-distinct --bank $CB --cache $MC --k $K --n 400 --min-frac 0.9; done   # expect ~0.94 at K=1 (mini cache: 48/51); the defect reads 0.0
```
**Proof the chain's later skip is safe.** (i) Same code: the chain's lanes stage runs `build_metric_cache.py build ... --lane-rank "$k"` from the shipped tree; once re-shipped, `build_metric_cache.py`, `pdm_route_rank.py` must match `FIXED_FILES.sha256`. (ii) Same path scheme: both go through `online_labeller.parts_path` -> `<MC>/<log>/<token>__lane_rank_<K>.pkl.xz`. (iii) MEASURED with the chain's exact env (`DRIVERL_EVAL_ROUTE_LANE_RANK=k REFE_PAUSE_FILE=... nice 10`): its twins equal the standalone ones on 6/6 (all six parts modulo `track_id`, route digest, score_plans; `results/proof_chainstyle_vs_standalone.json`); over a complete cache it builds nothing (`built=51 failed=0`, 51 mtimes unmoved, 9 s vs 87 s to build); after deleting 3 of 6 files it rebuilds only those and they equal the independent build (`results/final_partial_vs_chainstyle.json`). Byte-identity is not available: 35/74 files are byte-identical between two builds, the rest differ only in `track_id` metadata. **Set `P6_D7_MIN_DISTINCT=0.8` for the chain relaunch**: it samples 40 keys at 0.9 and refuses a correct build with probability 8.4 % (rank-1 non-distinct rate 3/51).

## DISK / TIME
- Disk: the quarantined parts (~11 GB) stay on the volume after the move. The rebuild writes 44,409 parts (**rank 3 adds 3,390**, absent from `p6_start.env`'s note) = ~14.9 GB at 0.335 MB (INHERITED mean). Pod showed 23 GB free (df is not authoritative; the volume reclaims lazily). If space is short, delete the quarantine dir after step 1: the stale parts equal the logged ones (digest 40/40 in the chain's own refusal).
- Time: MEASURED 3.1 s/twin/worker (40 twins of one log, 1 worker, pod load 8.5) and 2.0 s (least squares over 5 runs). ESTIMATED 25-38 worker-hours for 44,409 twins -> **8 workers 3.1-4.8 h, 6 workers 4.1-6.4 h**. Rank 1 is 74 %. A partial cache is safe (the labeller masks a twin without parts).

## RISKS / DECISIONS NOT MADE HERE
1. **(c) FAILS, upstream of D7.** The banked rank-K twin trajectory does not follow its rank-K route: closer to the twin centreline than to the logged one on 1/34, 0/7, 0/4 (bar 0.75); mean distance to the teacher's own rank-K step-0 route 2.9 / 4.6 / 6.5 m vs 0.6 / 0.4 / 0.5 m to the rank-0 route. Probable cause (code-derived, UNVERIFIED): the teacher re-ranks lanes against the CURRENT anchor every step (`feature_builder.py:398-418`). Options: accept (SPEC D7), mask low-adherence twins (per-twin `d_traj_to_route_k` in `results/verify_k*.json`; ~12 % follow), or re-roll with a route frozen at step 0.
2. **Dead-end lanes:** twin centreline < 0.5x the logged one on 15/51, 3/14, 6/9; < 100 m on 4/51, 2/14, 5/9 (min 31 m). PDM's Dijkstra cannot jump lanes the way DriveRL's continuity rule does. Decide whether to mask short twins.
3. DriveRL's start roadblock differs from the ego's on 6/74 twins (block boundaries). PDM's lane0 == DriveRL's rank-0 lane on 70/74.
4. `diag_d7_twin_route.py verify` is not a chain stage; run it on ~50 twins after the rebuild (`select` makes the mini-bank; see `scripts/run_rest.sh`).
5. Design notes: the Pool initializer never raises (MEASURED: a raising one respawned workers 276 times in 25 s and hung the parent, `evidence/pool_init_hang.py`); it records the failure and every job then fails loudly. `REFE_D7_MUTATE` is a test-only arm.

## ARTIFACTS (PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`; dev-box working copy `C:/Users/Admin/AppData/Local/Temp/p6d7/<PK>`; nothing staged, committed or pushed)
Code: `PK/refe/` (6 files). Package `PK/raw/2026-10-16-p6-d7-pdm-route-rank/`: `p6_d7_pdm_route_rank.patch`, `FIXED_FILES.sha256`, `results/*.json`, `logs/`, `scripts/`, `banks/` (seeded mini-banks), `evidence/`. **ONE place only:** pod `/workspace/ops/d7fix/` (pkg = the frozen fixed tree; mc_test, mc_final and the other scratch caches; 179 MB in all).
