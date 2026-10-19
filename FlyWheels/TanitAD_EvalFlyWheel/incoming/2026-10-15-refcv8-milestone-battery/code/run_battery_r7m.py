"""run_battery_r7m.py -- the refcv8 battery wrapper (`run_battery_r8.py`) pointed at a REFCV7 checkpoint (refcv7 remap; see r7m_patch.py).

    R8_NO_POSE_SYNC=1 R8_SEAM_TELE_DIR=<dir> python run_battery_r7m.py --ckpt E:/refcv7_eval_kit/ckpt/ckpt_20000.pt \
        --config E:/refcv7_eval_kit/ckpt/config.json --no-g0 --tag r7step20000_unc --infer-seeds 0,1 --lock-held \
        --out-root C:/Users/Admin/refcv8_eval_archive/battery

Identical CLI and outputs to `run_battery_r8.py` (the panels embed the banked refcv7-50,400 `os` as `b_refcv7_50400_s{s}`).
Only the eval-side file remap differs (refcv7's own files = the loader defaults).  NEW file (PREREG_R7_MATCHED_PHASE_TURN.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7m_patch  # noqa: E402

r7m_patch.install()
import run_battery_r8 as W  # noqa: E402

if __name__ == "__main__":
    W.main()
