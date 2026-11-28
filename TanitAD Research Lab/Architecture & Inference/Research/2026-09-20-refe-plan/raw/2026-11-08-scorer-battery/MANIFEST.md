# MANIFEST: BATTERY stream (2026-11-08). Nothing is staged in git; every file below lives ONLY in this folder (one place).

| artifact | where | note |
|---|---|---|
| `scorer_battery.py` | `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/battery/` | the battery (stats, bootstrap, arms, CLI); only here |
| `test_scorer_battery.py` | same | 19 tests: positive control, regression arms, determinism, mutation checks; only here |
| `battery_u400.json` | same | all numbers behind RESULT_u400.md (sha in CODE_PIN.json) |
| `RESULT_u400.md` | same | the read |
| `LANE_TRUTH.md` | same | lane-truth reachability |
| `CODE_PIN.json` | same | sha256 of code, outputs, inputs; env versions |
| `HANDOFF.md`, `MANIFEST.md` | same | |
| inputs read (not modified) | `TanitAD Research Lab/.../raw/2026-10-11-c3-selection-headroom/table.npz` (repo), `refe_ego_probe_20261106/full/results.npz`, `refe_selection_probe_20261107/sel_common.py`, `refe_behaviour_20261107/u400_reference.{py,json}` | |
Integration: if the Master Mind wants this in the repo it needs staging into `stack/` or the Research Lab package; I did not stage it (artifacts tree).
