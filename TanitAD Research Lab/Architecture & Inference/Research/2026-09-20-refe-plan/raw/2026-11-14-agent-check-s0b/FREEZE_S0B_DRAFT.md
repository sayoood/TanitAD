# FREEZE_S0B_DRAFT (S0-B, built 2026-11-14; DRAFT until the MM freezes; no result was read)

## Exact commands (dev box, venv driverl-eval, one process each, CUDA_VISIBLE_DEVICES=-1, TMP/TEMP on C:)
1. `python test_s0b.py` and `python test_s0b.py --mutations` (15 tests green; 7 mutations all RED)
2. `python s0b_cid.py --n 64 --out <dir>`  -> cid_result.json (ESTIMATED 5-6 min: 5 s/token MEASURED on n=2)
3. `python s0b_run.py --out <dir> --cid-json <dir>/cid_result.json`  -> s0b_result.json, S0B_RESULT.md, s0b_flags.npz
ESTIMATED CPU: 0.26 h (smoke MEASURED 0.224 s/set + 2.1 s DB load/log, 3,958 sets, 47 logs; the card's 0.3-0.5 h).

## Decisions mirrored (MM, before data) -- see docstrings of s0b_raster.py / s0b_readout.py
Grid 160x128 at 0.5 m, rear-axle t0 frame, row=x, col=y; cell occupied iff its CENTRE lies in the rotated box (boundary inclusive), one
rasteriser for agents and ego. 10 Hz x 51 steps built as NAVSIM (2 Hz samples = lidar_pc rows i0+10k, per-track linear interpolation inside
[first,last] sample, single-sample tracks present at all steps). No confidence filter. ego_pose = rear axle (nuplan_scenario_queries.py:559-598).
Footprint read from get_pacifica_parameters (vehicle_parameters.py:125-138). Arms R-true, R-t0, R-cv, R-dyn, R-true-2Hz, C-SHUF (derangement
rng 20261114, one stream over sorted logs), C-REG-zero/one. Bootstrap: log clusters, 10,000 draws, seed 20260927. Pick = sel_common.agg_torch(lane=False).

## sha256
s0b_raster.py 1d2022c26eae593686eb80e5da2f47d5c84f230c59c46715d5a3510f723207de
s0b_readout.py f08e24846eab63556310605801b514b01e384b411f0d91021f34cd728b80ce80
s0b_cid.py f3480985b63a1201a4c9c584599a227402d99a222e6c7f70b6f41787547c2d85
s0b_run.py c9d9c6cc8d6417df5ecb2d9c019ff3c86f5defe2deb8b3777b9e61d8cd60b59b
test_s0b.py 439648cd61f16407955f999097cdd586567751174cb2510c87129ce2012141cf
covered_sets.txt 4e61d655fb523b14b968c573546eebbd80d577a5fd82f10d52aa52cfe71c0607  (3,958 sets, 47 logs, 1,343 held-out)
found_logs.txt 06cdec34bd0d0617a1933ab6c751b06eb7ac432db5a30a35e624558501a7c713 ; ref/sel_common.py 50727023... (tip 99ffcd3a export, verbatim)
fit pack e58653c5... (asserted at run start)

## Could NOT mirror in NAVSIM (the proxy is weaker than the scorer here)
NC: at-fault typing (active-front / stopped-track / lateral; static = 0.5 vs agent = 0), multi-lane / non-drivable conditions, already-collided track
ids, red-light polygons. TTC: is_agent_ahead/behind, intersection / lane conditions, collided ids. Both: the PDM simulator's re-tracking of the proposal
(proxy uses the proposal poses at 10 Hz, no A7 last-heading repair), exact polygons (0.5 m raster), proposal speed = own finite difference not the simulated state.
TTC proxy tests t=0..40 x delta {0,.3,.6,.9} except (0,0). Empty-set rules: out-of-grid cells are free.

## Open items for the MM at the freeze
1. Which column decides the outcome row: the code reports BOTH (all covered, held-out covered); card names the covered sets, MM Q-A calls held-out the clean read.
2. covered set is 3,958 / 47 logs, not 3,976 / 48 (see HANDOFF). 3. C-ID NOT RUN status leaves the row marked provisional; FAIL -> INVALID.
