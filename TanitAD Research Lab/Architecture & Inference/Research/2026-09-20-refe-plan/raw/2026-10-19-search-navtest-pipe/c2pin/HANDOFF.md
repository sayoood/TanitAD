# HANDOFF c2pin (2026-10-20; tip fa7d856; nothing committed, pushed or git-added) -- STATE: DONE, staged

- Staged under `C:/Users/Admin/par18/c2pin/out/` (repo layout) with `out/paths.json` (13 rows; 7 edited existing files carry `base=<tip blob>`) and `out/RESULT.md` (copy of `.../search-navtest-pipe/c2pin/RESULT.md`).
- Counts (MEASURED): full navpipe suite 380 passed (baseline 357); mutation of 12 guards (8 new/changed + G-CONFIG-NONE, G-CONFIG-IS-C2CONFIG, G-LICENCE-ROW-AGREES, G-LICENCE-MOVED) 12 of 12 caught, control green.
- Integration (Master Mind): land the 7 edited files BEFORE running `seal_c2_amendment.py`; the draft's terms text changed, so any earlier seal dry run is stale. `dose_chain_sha256` stays the seal placeholder, expected `b41ccab5a24fa67c4d07c3aa4dce1b60175c97ad9787f513e91491193dccc217`.
- Finding: `select_config` gave the real `c2_config` no `inert` block (frac_moved NaN at R1, so no read licensed, C2_CONFIG_NONE every time). Fixed, pinned by G-C2CONFIG-INERT.
- Not done: full 125-guard mutation re-run; `g0_traingrow_scan2.json` and the 70 l2-stepB npz shards were not exported (disk).
- Local only, never stage: `tree/.../code/conftest.py`, `pt/`, `tmp/`, `npmutA/`, `tree/` (working copy).
- Re-run finalize only if code changes: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/par18/c2pin/finalize.py` (needs `mut_outA/mut.done`).
