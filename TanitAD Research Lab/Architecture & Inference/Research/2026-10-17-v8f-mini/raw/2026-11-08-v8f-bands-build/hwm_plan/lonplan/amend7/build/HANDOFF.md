# HANDOFF: AMEND_HWMPLAN_7 FIT read (build agent to a fresh agent or the MM), 2026-11-18

**State.** Build finished and tested (30 tests green, see `BUILD_AMEND7.md`). The FIT stage runs detached as `run_amend7.sh fit` (launched by `launch7.ps1` through WMI; the launch time and PID are in `amend7/logs/LAUNCH_FIT.txt`). Only the FIT stage was launched. The SCORE stage was built, not run.

**Where to look.**
* `amend7/read.log`: the last line is the sentinel: `ZZ AMEND7 FIT DONE ZZ <outcome>` | `ZZ AMEND7 FIT INVALID ZZ <control>` | `ZZ AMEND7 FIT FAIL ZZ <why>` | `ZZ AMEND7 FAIL ZZ spec`. Step logs: `amend7/logs/*.log` (`fit_<k>.log`, `fitctl_<k>.log`, `gate.log`, `tests_test_amend7.log`). Repair notes go in a separate file, never into `read.log`.
* Result: `amend7/readout/RESULT_AMEND7_FIT.json` + `.md` (a copy as `G7_FIT.json` + `.md`), stamped T0 / MEASURED / estimator; controls in `amend7/controls/CONTROLS_FIT.json`; per-arm outputs in `amend7/fit/`; the final readouts and their sha256 in `amend7/readout_in/` and `amend7/RA7_FIT_s{0,1}.json`.
* Expected wall: about 2.5-4 h (ESTIMATED), serial chassis. Steps whose output exists are skipped on a rerun (delete `amend7/fit/` and `RA7_FIT_<k>.json` to force a refit).

**Exact next step (the MM, from the FIT result).**
* R7-FIT-PASS: `launch7.ps1 -Mode score` (inline WMI Create from the PowerShell tool; script execution is disabled for `-File`). It checks the label, then runs controls, TRAIN stage, the six SCORE arms per chassis and the readout (about 1.3 h), ending `ZZ AMEND7 READ DONE ZZ <outcome>`.
* R7-STATE-ONLY or R7-FIT-FAIL: no SCORE read; the next lever is ACTRSP Option B (card section 4 and 5.2). Under R7-FIT-FAIL with TRANS-ONLY a SCORE read of RA7 is an MM decision.
* R7-FIT-INVALID: read `CONTROLS_FIT.json` for the red control; fix and rerun `launch7.ps1`; PLANT P1 alone means the grid is re-carded by an amendment before any rerun.
* The chain deletes `amend7/pairs/` and `amend7/work/` at its own end (re-creatable; sizes logged in `read.log`).

**Do not.** Edit the frozen card or anything under `amend5/` or `amend6/`; fit RA7 anywhere else; pass `AMEND7_ROOT` (it is only for the build's scratch smokes).
