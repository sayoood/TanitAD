# FREEZE_S0B: the agent check's CPU gate S0-B, frozen by the MM before any result exists (2026-11-14)

This freezes **S0-B** of `PK/raw/2026-11-14-agent-check-feas/PREREG_AGENT_CHECK_DRAFT.md` section 3:
- its arms and controls;
- its committed outcome table: S0B-GO / S0B-STATIC / S0B-READOUT / INVALID / NOT RUN, with c(R-true) ≥ 0.50, c(R-t0) ≥ 0.25, C-SHUF gain ≤ 0 and capture < 0.10, C-REG exactly 0.0, C-ID ≥ 0.99.

Stages 1 and 2 of that card stay DRAFT.
- **Build and smoke:** `FREEZE_S0B_DRAFT.md`, `HANDOFF.md`, `smoke_report.json` and `cid_smoke_n2.json`, all in this folder.
- **What has been seen:** no gain, capture or precision number has been computed or seen by anyone. The builder's smoke wrote counts and timings only, and its full-path self-test on one log was deleted unread.
- **Commit order:** this record and the scripts are committed BEFORE the run.

## Pins (sha256, MEASURED by the MM on the hold's files)

| file | sha256 |
|---|---|
| `s0b_raster.py` | `1d2022c26eae593686eb80e5da2f47d5c84f230c59c46715d5a3510f723207de` |
| `s0b_readout.py` | `f08e24846eab63556310605801b514b01e384b411f0d91021f34cd728b80ce80` |
| `s0b_cid.py` | `f3480985b63a1201a4c9c584599a227402d99a222e6c7f70b6f41787547c2d85` |
| `s0b_run.py` | `c9d9c6cc8d6417df5ecb2d9c019ff3c86f5defe2deb8b3777b9e61d8cd60b59b` |
| `test_s0b.py` | `439648cd61f16407955f999097cdd586567751174cb2510c87129ce2012141cf` |
| `covered_sets.txt` (3,958 sets, 47 logs, 1,343 held-out) | `4e61d655fb523b14b968c573546eebbd80d577a5fd82f10d52aa52cfe71c0607` |
| `found_logs.txt` | `06cdec34bd0d0617a1933ab6c751b06eb7ac432db5a30a35e624558501a7c713` |
| `ref/sel_common.py` (verbatim tip `99ffcd3a` export of `PK/raw/2026-11-07-selection-probe/sel_common.py`) | `50727023525c86b2be503db5c7f84d56de9b2be98f260aab700e4c9438f27d23` |
| fit pack `F:/tanitad_scratch/p6fit/fit_pack.npz` (asserted at run start) | `e58653c5…` (card section 2) |

## MM rulings at the freeze (each taken before any number exists)

1. **The deciding column is the HELD-OUT covered sets (1,343 sets).**
   - The all-covered column (3,958) is reported beside it and decides nothing.
   - **Why:**
     - The fit pack's held-out logs are the SFT held-out logs.
     - Whether P6 trained on the 77 fit-extra logs is UNVERIFIED (card Q-1), so P6's pick there may be in-sample.
     - The oracle the MM verified is the held-out one (`MM_VERIFY.md`).
   - **Controls bind in BOTH columns:** a C-SHUF, C-REG or C-ID failure in either column makes the read INVALID.
2. **Covered set: 3,958 sets over 47 logs, not 3,976 over 48.**
   - The 48th log's DB (`2021.06.23.14.54.32_veh-16_00862_01000`, 18 sets) existed only in a live download staging dir, which the build was told not to read.
   - The exclusion follows file availability. It is not a selection on outcomes.
   - The card's Q-A ruling (`MM_VERIFY.md`) is amended to this set.
3. **Statics are interpolated as NAVSIM's code does,** not held fixed as the brief said.
   - The label is defined as NAVSIM's own observation, so the code wins (`metric_cache_processor._interpolate_gt_observation`).
4. **C-ID runs on the dev box, not on pod tokens:** 64 covered tokens, seeded.
   - Environment: the driverl-eval venv's nuPlan (the DriveRL fork) plus NAVSIM `3e8291b`. The `F:` nuplan-devkit needs `pytest`, which that venv lacks.
   - The second half of C-ID (`score_plans` on the banked proposals reproduces the fit pack's nc and ttc) is what makes this environment admissible.
   - The n = 2 smoke reproduced both exactly, with cell agreement 0.9996 (MEASURED, `cid_smoke_n2.json`).
   - A C-ID FAIL makes the read INVALID. C-ID NOT RUN leaves the row provisional, as coded.
5. **C-SHUF reads as written:** a point-estimate gain > 0 OR capture ≥ 0.10 makes the read INVALID, in either column.
6. **What the proxy does NOT reproduce of NAVSIM's NC/TTC is listed in `FREEZE_S0B_DRAFT.md`.**
   - It covers at-fault typing, lane and drivable conditions, the PDM simulator's re-tracking, and exact polygons.
   - That gap is what S0-B measures, through c(R-true). It is not a defect to fix before the read.

## Run (dev box, one process, `CUDA_VISIBLE_DEVICES=-1`, venv driverl-eval; nothing keyed, navtrain only)
1. `python test_s0b.py` and `python test_s0b.py --mutations`: must give 15 tests green and 7 mutations RED.
2. `python s0b_cid.py --n 64 --out <run>/cid`
3. `python s0b_run.py --out <run> --cid-json <run>/cid/cid_result.json`

**ESTIMATED cost:** 0.26 CPU-h for the run plus ~6 min for C-ID.

**After the run, under Rule Zero, the committed successor of whatever row reads is run in the same turn** (card section 3):
- **S0B-GO:** S1, which needs the PI's pod time.
- **S0B-STATIC:** a motion-conditioned arm. Q-D goes to the PI.
- **S0B-READOUT:** R-sim (the true agent list through NAVSIM's `score_plans`, CPU) plus the learned fusion; if R-sim also reads < 0.50, the direct top-5 re-ranker.
- **INVALID:** fix and re-run.
