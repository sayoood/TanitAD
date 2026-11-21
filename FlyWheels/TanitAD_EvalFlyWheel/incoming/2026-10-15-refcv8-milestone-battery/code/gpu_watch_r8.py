"""refcv8 wrapper around the refcv7 battery's `gpu_watch.py` that also counts THIS package's processes as ours.

Defect it fixes (MEASURED 2026-10-18, `E:/refcv8_eval_kit/chain/battery_gpu_watch_step30000.jsonl`): the refcv7
watcher's allow-list `OURS` names only the refcv6/refcv7 battery packages, so the refcv8 battery's own
`roll_seed_r8.py` was classified FOREIGN and 173 of 175 polls of the raw-30k battery read `"COLLISION": true`.
A collision flag that is true whenever our own roll runs cannot show a real collision.

The upstream file (in the fb47a58 export) is not edited: this module loads it by path, extends `OURS` (read at
call time by `snapshot()`), and runs its `main()`. Usage is identical:
    python gpu_watch_r8.py --out <jsonl> --stop-file <path> [--every 60]
"""
from __future__ import annotations

import importlib.util
import os
import sys

BAT = os.environ.get("R8_BAT", "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code")
R8_OURS = ("2026-10-15-refcv8-milestone-battery\\code", "2026-10-15-refcv8-milestone-battery/code")


def load(bat: str = BAT):
    spec = importlib.util.spec_from_file_location("gpu_watch_upstream", os.path.join(bat, "gpu_watch.py"))
    gw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gw)
    gw.OURS = tuple(gw.OURS) + R8_OURS
    return gw


if __name__ == "__main__":
    sys.exit(load().main())
