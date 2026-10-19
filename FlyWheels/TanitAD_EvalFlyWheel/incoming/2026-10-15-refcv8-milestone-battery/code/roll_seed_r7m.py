"""roll_seed_r7m.py -- ONE inference-seed roll as a CHILD process, for a REFCV7 checkpoint (refcv7 remap; see r7m_patch.py).

Same as `roll_seed_r8.py` (re-pointing + pose-sync policy applied FIRST), except the eval-side remap is the loader defaults, because
this process must install the SAME `REMAP_OVERRIDES` as its parent (a disagreeing REFCV7_REMAP_OVERRIDES in the inherited environment is a hard
refusal in `r8_repoint.configure_env`).  With env `R7M_CEIL_OFF=1` the max-speed ceiling filter is switched OFF (r7m_patch.disable_ceiling); with `R7M_E9_NO_CEIL=1` only E9's final re-rank ignores it
(r7m_patch.disable_e9_ceiling). NEW file.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7m_patch  # noqa: E402

r7m_patch.install()
import r8_battery_patch as BP  # noqa: E402

BP.apply_child()
if os.environ.get("R7M_CEIL_OFF") == "1":
    r7m_patch.disable_ceiling()
if os.environ.get("R7M_E9_NO_CEIL") == "1":
    r7m_patch.disable_e9_ceiling()
import roll_seed_r7 as RS  # noqa: E402  (launch tree's battery code)

if __name__ == "__main__":
    print(f"[roll_seed_r7m] refcv7 remap (loader defaults); pose_sync policy: "
          f"{'OFF (R8_NO_POSE_SYNC)' if os.environ.get('R8_NO_POSE_SYNC') in ('1', 'true', 'yes') else 'ON'}; "
          f"ceiling filter: {'OFF (R7M_CEIL_OFF=1)' if os.environ.get('R7M_CEIL_OFF') == '1' else 'as built (ON for refcv7/8 argv)'}; "
          f"E9 ignores ceil_keep: {os.environ.get('R7M_E9_NO_CEIL') == '1'}", flush=True)
    RS.main()
