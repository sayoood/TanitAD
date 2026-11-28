# HANDOFF_TOWN (2026-11-07 ~09:25 dev-box clock). Folder `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/town/` (`$PK`). Scratch rollouts `F:/TanitAD-scratch/motion_c/town/out/`.
Card `../PREREG_V8F_TOWN.md`, frozen prefix 4,676 bytes sha256 707e29f3... (verified at start; the chain re-checks it). Implementation choices frozen BEFORE any number in `IMPL_CHOICES.md` (+ `.sha256`, re-checked by the chain). Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, CUDA hidden, 4 threads. Nothing staged, committed, pushed, deleted.

## DONE
* `town_run.py`: thin wrapper over the pinned tool (sha 0417c838...), twin T on the LOGGED ego state per replan, `--zero-t-ego` regression arm, `--twin-mode mutant`. `town_gates.py` (G-ID, G-TWIN0, G-TWINL, G-OWN, G-REG), `town_read.py` (all statistics of the card), `chain_town.sh/.cmd`, `tests/test_town_controls.py`.
* 10-episode subset (170 windows, 4 threads, `raw/grid_first10.json`): **town run == bare tool run, tick file + 10 episode dumps bitwise, max |delta| 0.0** (`raw/SUB_GATE_ID_town_vs_bare.json`). Twin: tick-0 outputs bitwise equal (170 rows), 340/340 later rows differ in the swapped inputs, logged p10f mismatch 0 (`raw/SUB_GATE_TWIN.json`). Mutant twin (swaps nothing): 0/340 differ, G-TWINL FAILS as it must.
* ⚠️ The BANKED RIC0_s0 (12 threads) is NOT bitwise equal to the 4-thread bare tool on the subset (all 10 episodes differ, max |delta| 0.0024 m; `raw/SUB_GATE_ID_bare_vs_bank.json`). Thread count changes float order; the identity that decides is therefore against a 4-thread bare tool run on the FULL grid (stages BARE_s0/1 of the chain). The banked comparison is written as information (`raw/GATE_ID_INFO_*`). `raw/SUBSET_EXTRA.done` marks S_REG (regression smoke) and S_BARE2 (2 threads vs 4: thread dependence check) finished.

## RUNNING: detached chain (launched by `go_town.cmd` -> `run_subset_extra.sh` -> `chain_town.sh`), resumable, ~1.3 h
Stages: A BARE_s0 + C0_s0 + G-ID + twin gates | B same seed 1 | C REG_s0 | D REG_s1 | E reads -> `raw/READ_{C0,REG}_s{0,1}.json`. Log `logs/chain_town.out` (grep ZZTOWN). Markers in `raw/`: `CHAIN_STAGE_<A..E>.done`, **`CHAIN_TOWN.done`**, `.gidfail` (identity failed: report INVALID, read nothing; the first differing value is in `raw/GATE_ID_C0_s*.json`), `.gatefail`, `.fail`.
Relaunch after a failure: `powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\town\chain_town.cmd'}"`.

## NEXT (fresh agent, Sonnet, <= 100k context)
1. Wait for a marker with ONE background watcher (until-loop on `raw/CHAIN_TOWN.*`), not turns. Run `pytest -q -p no:cacheprovider tests` (needs the S_* subset outputs; they exist).
2. On `.done`: write `town_result.py` (not written yet) -> `RESULT_TOWN.md/.json` from `raw/READ_*.json` + gates, by the card: T-FEEDBACK needs slope CI hi < 0 (`shift_committed.slope`) at BOTH seeds AND `quality.all_replans.lon_macro_f1.worse_separated` at both seeds; T-STABLE needs the slope interval to contain 0 and lon F1 not worse-separated at both seeds; else MIXED. The regression arm (`READ_REG_*`) must have a slope interval containing 0 at both seeds, else the control FAILED and the outcome is not read. Report the beside rows (ticks 1-2 quality, through-origin slope, dv at T's frame, rank variants, share of slower-than-GT argmax, 26-cell constraint RMSE) as beside.
3. Append ONE dated entry to the card's Log (append-only; re-check the 4,676-byte prefix hash). Hand back <= 300 words with the manifest.

## Disclose
* Mechanism is a wrapper, not a file copy (a copy moves the tool's REPO root and drops the sha pin). The tool already has `--dump-ticks --dump-heads`; the wrapper adds the twin only.
* Primary quality population = all labelled replans (3 ticks), not only ticks 1-2 (the latter is beside); the card is silent. Constraint RMSE headline = the 5 m/s speed cells pooled (per-row mean squared error), 26 cells beside.
* The regression arm zeroes `e_ego` inside `run_t` only; T still sees ego through S's checkpoints (`ck_for_t`), the `limit` bucket and `nav`: if the slope survives it, that is the card's reading ("the probe is not reading T's use of ego"), and the other routes are named here.
* The twin keeps `nav` (dead-reckoned from the own arc) and the reindexed frames identical to the own batch; it swaps `p10f, base, pose_now, pose_t, pose_s, limit, limit_ok`.
* Disk: C: 6.9 GB, E: 5.0 GB, F: 2.0 GB free at start; each full run writes ~60 MB to F:.

## UPDATE 09:12
* Chain first start died on its own freeze check (a cwd bug in `sha256sum -c`; IMPL_CHOICES.md unchanged, `OK`); fixed, relaunched 09:11:04 (BARE_s0 running). Old marker kept as `raw/CHAIN_TOWN.fail.cwdbug1`.
* Subset extras finished: S_REG zeroed_calls 36 = 18 own + 18 twin forward calls (G-REG PASS in test); S_BARE2: bare tool at 2 threads vs 4 threads differs by max |delta| 0.00198 m on the tracks, i.e. thread count changes the bits, which explains the 0.0024 m against the 12-thread banked RIC0_s0. `logs/pytest_town.out`: 8 passed.
* Expected wall: ~10-12 min per rollout x 6 (BARE_s0, C0_s0, BARE_s1, C0_s1, REG_s0, REG_s1) + reads.
