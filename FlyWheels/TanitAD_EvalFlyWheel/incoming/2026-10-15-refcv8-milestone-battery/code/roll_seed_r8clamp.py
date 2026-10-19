"""roll_seed_r8clamp.py -- ONE inference-seed roll as a CHILD process against the CLAMP-CARRYING tree (tree_de4e15b), flag ON or OFF.

Same as `roll_seed_r8.py` (re-pointing + pose-sync policy applied FIRST), except `r8clamp_patch.install()` runs before anything else:
it checks the tree's provenance (REFUSES otherwise), points `r8_repoint` at the clamp tree, and wires the flag + the per-forward assertions.
The mode comes from env `R8_CLAMP_MODE=on|off` (set by `run_battery_r8clamp.py` / the K2K4 queue).  NEW file (AMENDMENT A-R8-CLAMP-1).

    R8_CLAMP_MODE=off R8_NO_POSE_SYNC=1 python roll_seed_r8clamp.py --ckpt ... --config ... --seed 0 --dump-dir ... --windows-json ... --out-json ...

The pose clock follows the ENVIRONMENT exactly as for `roll_seed_r8.py` (R8_NO_POSE_SYNC=1 = standalone/uncorrected, unset = corrected).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8clamp_patch  # noqa: E402

r8clamp_patch.install()
import r8_battery_patch as BP  # noqa: E402

BP.apply_child()
import roll_seed_r7 as RS  # noqa: E402  (the launch tree's battery code; its module-level loader import is now re-pointed at the clamp tree)

if __name__ == "__main__":
    print(f"[roll_seed_r8clamp] clamp mode: {os.environ.get('R8_CLAMP_MODE')}; tree {r8clamp_patch.C.TREE}; pose_sync policy: "
          f"{'OFF (R8_NO_POSE_SYNC)' if os.environ.get('R8_NO_POSE_SYNC') in ('1', 'true', 'yes') else 'ON'}", flush=True)
    RS.main()
