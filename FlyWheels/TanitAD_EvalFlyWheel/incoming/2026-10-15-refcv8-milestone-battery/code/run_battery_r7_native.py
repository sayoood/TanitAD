"""run_battery_r7_native.py -- the NATIVE refcv7 battery (`run_battery_r7.py` of the launch tree) with ONLY the baseline paths re-pointed.

The alternative path of PREREG_R7_MATCHED_PHASE_TURN.md section 2 / the brief: no `r8_repoint`, no seam-guard / micro-batch / pose-sync patch,
the roll child is the unpatched `roll_seed_r7.py`, the loader runs with its OWN defaults (kit D:/refcv6_eval_kit == E: volume, no remap override).
The one edit: `r7_panel.BASELINES` named the pre-2026-10-15-cleanup location, so it is repointed by content (`r8_battery_patch.fix_baselines`,
which only edits that dict).  NEW file.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAT = Path("C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code")
sys.path.insert(0, str(BAT))
sys.path.insert(0, str(HERE))
import run_battery_r7 as RB  # noqa: E402  (module-level: loader import + bootstrap with the loader's own defaults)
import r7_panel as P  # noqa: E402
import r8_battery_patch as BPM  # noqa: E402  (import only: nothing is applied except fix_baselines below)

BPM.fix_baselines(P)
if __name__ == "__main__":
    RB.main()
