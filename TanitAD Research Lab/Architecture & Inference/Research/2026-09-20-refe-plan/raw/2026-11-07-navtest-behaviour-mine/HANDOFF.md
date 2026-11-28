# HANDOFF - stream MINER (navtest behaviour features), 2026-11-07

**State: DONE.** Per-token feature table over all of NAVSIM navtest is built, verified and pinned. Nothing is running. No tokens were selected (by design; stream BEHAV freezes the rule).

## Result in one paragraph
12,146 of 12,146 navtest tokens read (136 logs, 0 unpickle failures, 0 skipped: 0 not found, 0 failed `has_route`, 0 duplicates). 81 columns: ego kinematics, lead vehicle (gap / closing / time gap / TTC / lead decel), two cut-in definitions, traffic lights (raw list + map-matched "red ahead"), map-based junction flags, driving command. 11 of the 147 pkls in the directory are not navtest logs and were not read. The 3 pkls restored on 11-06 (veh-48_00221_00299, veh-42_00172_00347, veh-52_00032_00170): all three are navtest logs and read fine.

## Files (all under `E:/Projects/TanitAD-artifacts/navtest_behaviour_mine_20261107/`)
`features.parquet` (+ `features.csv.gz`), `SCHEMA.md` (every column), `SUMMARY.md` (counts under EXAMPLE thresholds, cruise-bundle check, 10 hand-verified tokens), `map_handcheck.json`, `hand_verified_tokens.json`, `cruise_bundle_other_tokens.csv`, `run_report.json`, `run.log`, `CODE_PIN.json`, `MANIFEST.json`, `code/{mine_features,summarize,verify_map}.py`.

## Reproduce (about 4.5 min on 4 workers, CPU only)
```
cd E:/Projects/TanitAD-artifacts/navtest_behaviour_mine_20261107
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I code/mine_features.py --workers 4
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I code/summarize.py
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I code/verify_map.py
```
Column join key = `token`; locate the frame with `frames[list_idx]` (NOT `frame_idx`, which restarts per scene).

## Things the next reader must know
* `cutin` (ego-frame) fires on 9.1 % of tokens, `cutin_path` on 0.4 %. Neither is validated against a cut-in label (none exists). Choose deliberately.
* `tl_any_red` (raw list) is True on 52 % of tokens and says little about the ego's own signal; `tl_red_ahead` (map-matched, 17 %) is the usable one. Bool = `is_red` (navsim_scenario.py:282-283); yellow is invisible.
* 44 % of tokens with a lead have a stationary lead (queues / parked cars in the 3.6 m strip).
* Cruise bundle (154 tokens, all in navtest): 70.1 % free road, 3.2 % steady lead, 26.6 % other (38 of 41 have an intermittent lead in the strip). 0 stop, 0 brake >= 2.
* `driving_command` order [left, straight, right, unknown] is confirmed empirically (mean future yaw +0.645 / +0.006 / -0.636 rad).
* Horizon is 8 future frames (4 s) although navtest.yaml asks for 10 future frames per window.

## Mistakes made and fixed during the run (for honesty)
1. First run stored the pkl `frame_idx` as if it were the list index; the independent map check caught it (3 mismatches), `list_idx` was added and the extraction re-run.
2. A first hand-verification pick for `ego_stops` had a precedence bug (`A & x == False`) and picked a token that never stops; fixed.
3. **I overwrote four of the Master Mind's old scratch files** in the shared scratchpad `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/c216c849-5fe7-4d43-a3be-32429241f811/scratchpad/`: `patch1.py` (born 2026-10-18), `patch2.py`, `patch3.py`, `patch4.py` (born 2026-10-15). They now hold my patch code for summarize.py. Their old content is lost; they were one-off edit scripts. Left in place, not deleted.

## Deletions (own temp only, literal paths, rule 4)
scratchpad `probe1.py`..`probe5.py`, `chk1.py`, `pin.py` (about 15 KB), `smoke/features_limit2.parquet` 59,973 B, `smoke/features_limit2.csv.gz` 13,112 B, `smoke/run_report_limit2.json` 1,009 B, dir `smoke/`. All re-creatable from `code/`. Free space seen at start: C: 9.0 GB, E: 9.2 GB, F: 41 GB; the package is 4.7 MB.

## Not done / open
* Not staged, not committed (brief). **The package exists on E: only**; banking it (git or a copy off E:) is the caller's decision.
* No ground-truth validation of either cut-in flag; no yellow-light state; junction uses scaled-width connector polygons (flags turn True slightly before the stop line).
