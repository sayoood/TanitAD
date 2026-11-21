"""run_battery_r8clamp.py -- the refcv8 battery wrapper (`run_battery_r8.py`) pointed at the CLAMP-CARRYING tree (tree_de4e15b).

    # PRIMARY (A-R8-CLAMP-1 section 2.1): clamp ON, the landed function, not a re-implementation
    R8_NO_POSE_SYNC=1 python run_battery_r8clamp.py --clamp on  --tag step48852_clampon_unc  <the usual run_battery_r8.py args>
    # the clamp-OFF rows beside, same tree, same windows, same DDIM draws per seed
    R8_NO_POSE_SYNC=1 python run_battery_r8clamp.py --clamp off --tag step48852_clampoff_unc <the usual run_battery_r8.py args>

`--clamp on|off` is REQUIRED and is stripped before run_battery_r8 sees argv; `--tag` MUST contain `clampon` / `clampoff` to match (a bank cannot mix).
Everything else (seeds, arms, S2/S6 surfaces, panels, baselines, G0 policy, --smoke-cpu ...) is `run_battery_r8.py` verbatim: this file
installs `r8clamp_patch` (provenance REFUSAL, clamp tree, flag, per-forward assertions, roll child -> roll_seed_r8clamp.py) and calls its main().
After the run the summary gets an `r8_clamp` stamp.  PYTHONPATH must point at the CLAMP tree (see queue_k2k4_r8.sh), never at fb47a58.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _val(flag, argv):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def split_clamp(argv):
    """Pure (tested): -> (mode, argv without --clamp).  Refuses a missing/invalid --clamp or a --tag that does not name the mode."""
    argv = list(argv)
    mode = None
    for i, a in enumerate(argv):
        if a == "--clamp" and i + 1 < len(argv):
            mode = argv[i + 1]
            del argv[i:i + 2]
            break
        if a.startswith("--clamp="):
            mode = a.split("=", 1)[1]
            del argv[i]
            break
    if mode not in ("on", "off"):
        raise SystemExit("[run_battery_r8clamp] --clamp on|off is REQUIRED")
    if "-h" in argv or "--help" in argv:
        return mode, argv
    tag = _val("--tag", argv)
    if not tag or f"clamp{mode}" not in tag or (mode == "on" and "clampoff" in tag) or (mode == "off" and "clampon" in tag):
        raise SystemExit(f"[run_battery_r8clamp] --tag {tag!r} must contain 'clamp{mode}' (and not the other mode's name)")
    return mode, argv


def main() -> None:
    mode, rest = split_clamp(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    os.environ["R8_CLAMP_MODE"] = mode
    import r8clamp_patch
    r8clamp_patch.install(mode)
    import run_battery_r8 as W
    out_root, tag = _val("--out-root", rest) or f"{W.KIT}/battery", _val("--tag", rest)
    try:
        W.main()
    finally:
        sp = Path(out_root) / (tag or "_") / "battery_summary.json"
        if sp.exists():
            j = json.load(open(sp, encoding="utf-8"))
            j["r8_clamp"] = {"wrapper": "run_battery_r8clamp.py", "mode": mode, "amendment": "A-R8-CLAMP-1",
                             "note": ("clamp ON: out['traj'] is the plan after cap_plan_to_limit (landed function, inference only); `traj_precap` is the "
                                      "same-draw clamp-OFF plan" if mode == "on" else "clamp OFF on the clamp-carrying tree (flag False; K2 parity read)"),
                             "patch": r8clamp_patch.describe()}
            json.dump(j, open(sp, "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    main()
