"""roll_seed_r8.py -- ONE inference-seed roll as a CHILD process, with the refcv8 re-pointing applied FIRST.

`run_battery_r7._run` spawns `roll_seed_r7.py` from the refcv7 battery directory; that child imports the loader directly,
so it would carry neither the dev-box remap nor the pose-sync policy (monkey-patches are per process).  This file is what
`r8_battery_patch.rewrite_child_cmd` substitutes: same argv, same script, after `r8_repoint.apply()`.

    python roll_seed_r8.py --ckpt ... --config ... --seed 0 --dump-dir ... --windows-json ... --out-json ... [--os-only]

The pose clock is chosen by the ENVIRONMENT, inherited from the battery parent: `R8_NO_POSE_SYNC=1` = the standalone
(refcv7-comparable) clock, unset = the corrected clock the run trained and evaluated on (SPEC_REFCV8 section 2.3 item 6).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_battery_patch as BP  # noqa: E402

BP.apply_child()
import roll_seed_r7 as RS  # noqa: E402  (from the launch tree's battery code; its module-level loader import is now re-pointed)

if __name__ == "__main__":
    print(f"[roll_seed_r8] pose_sync policy: {'OFF (R8_NO_POSE_SYNC)' if os.environ.get('R8_NO_POSE_SYNC') in ('1', 'true', 'yes') else 'ON'}",
          flush=True)
    RS.main()
