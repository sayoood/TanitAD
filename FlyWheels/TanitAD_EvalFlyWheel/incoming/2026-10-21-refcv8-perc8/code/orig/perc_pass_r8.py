"""perc_pass_r8.py -- the refcv7 PERCEPTION pass (`perc_pass_r7.py`: map at 10 cm + box heads, one window at a time, through the
trainer's own `compute_losses_v3`) re-pointed at refcv8: `r8_repoint.apply()` FIRST (dev-box remap, launch tree, pose-sync policy
from the env `R8_NO_POSE_SYNC`; the perception targets ride the corrected clock by default, as the run's own eval does), then the
refcv7 `main()` unchanged.  Same argv as perc_pass_r7.py.  UNVERIFIED beyond what `raw/step5000/smoke_perc_cpu_r8.json` shows.

    python perc_pass_r8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_5000.pt --config E:/refcv8_eval_kit/ckpt/config.json \
        --windows-json E:/refcv8_eval_kit/windows/s2_windows.json --out E:/refcv8_eval_kit/perc/step5000/perc [--prior-npz ...]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_battery_patch as BP  # noqa: E402

BP.apply_child()
import perc_pass_r7 as PP  # noqa: E402

if __name__ == "__main__":
    PP.main()
