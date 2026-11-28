"""run_battery_r8.py -- the refcv7 four-family battery (`run_battery_r7.py`), re-pointed at the refcv8 run.

    python run_battery_r8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_5000.pt --g0-json E:/refcv8_eval_kit/battery/step5000/g0.json \
        --tag step5000_unc --infer-seeds 0,1 --lock-held [--g0-nonblocking]        # env R8_NO_POSE_SYNC=1 = refcv7-comparable clock

What it adds on top of `run_battery_r7` (all in `r8_battery_patch.py`, no repo file edited; see its docstring):
  1. `r8_repoint.apply()` -- dev-box remap, launch tree, and the pose-sync policy (env `R8_NO_POSE_SYNC=1` turns it OFF = the
     standalone, refcv7-comparable clock; unset = the corrected clock the run trained / was evaluated on, SPEC_REFCV8 2.3.6);
  2. the moved refcv5cmp baselines are re-pointed; refcv7-50,400 `os` (per inference seed) is registered as a paired baseline;
  3. the roll CHILD process is `roll_seed_r8.py` (so it carries (1)); the parent's G0 stop may be made NON-blocking with the
     explicit `--g0-nonblocking` (stamped PROVISIONAL into the summary) for the step-5,000 early read;
  4. `--g0-json` (from `g0_refcv8.py`) or `--no-g0` is REQUIRED -- run_battery_r7 would otherwise spawn the UNPATCHED
     `g0_refcv7.py`; `--tag` is REQUIRED (the stamp is written into `<out-root>/<tag>/battery_summary.json["r8"]`);
  5. `--smoke-cpu` (CPU, no GPU lock, no GPU gate; refuses more than 2 clips x 3 windows) -- code-path check only;
  6. `--os-only-all` -- every seed's roll is `os` + model-free controls only (the corrected-clock pass; saves ~1.5 h);
  7. `--no-baselines` -- REQUIRED on the corrected pose clock (env `R8_NO_POSE_SYNC` unset) and refused otherwise: the banked
     refcv4b / refcv5v2 / refcv7-50,400 dumps are on the UNCORRECTED clock, `build_s6_dump` raises on a GT mismatch (hours
     into the run), and the pairing would compare two different eval sets.  No `--no-baselines` on the standalone clock.
UNVERIFIED beyond what `raw/step5000/smoke_battery_cpu_r8.json` and the logs show; no battery number counts unless G0 passes.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KIT = "E:/refcv8_eval_kit"


def _has(flag: str, argv: list[str]) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _val(flag: str, argv: list[str]):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _pop_switch(flag: str, argv: list[str]) -> bool:
    if flag in argv:
        argv.remove(flag)
        return True
    return False


def prepare_argv(argv: list[str]) -> tuple[list[str], dict]:
    """Pure function (tested): strip the r8-only switches, apply the refusals and the defaults."""
    argv = list(argv)
    smoke = _pop_switch("--smoke-cpu", argv)
    nonblock = _pop_switch("--g0-nonblocking", argv)
    nobase = _pop_switch("--no-baselines", argv)
    osall = _pop_switch("--os-only-all", argv)
    if "-h" in argv or "--help" in argv:
        return argv, {"smoke": smoke, "nonblock": nonblock, "nobase": nobase, "osall": osall, "help": True}
    synced = os.environ.get("R8_NO_POSE_SYNC", "") not in ("1", "true", "yes")
    if synced and not nobase:
        raise SystemExit("[run_battery_r8] refusing: the corrected pose clock (R8_NO_POSE_SYNC unset) cannot be paired with the "
                         "banked baselines (uncorrected clock; S6 would raise on the GT mismatch hours in). Pass --no-baselines, "
                         "or set R8_NO_POSE_SYNC=1 for the refcv7-comparable pass.")
    if (not synced) and nobase:
        raise SystemExit("[run_battery_r8] refusing --no-baselines on the standalone clock (R8_NO_POSE_SYNC=1): that pass IS the "
                         "paired read against refcv7-50,400")
    if not (_has("--g0-json", argv) or _has("--no-g0", argv)):
        raise SystemExit("[run_battery_r8] refusing: pass --g0-json <g0.json from g0_refcv8.py> (or --no-g0 and say "
                         "why). run_battery_r7 would spawn the unpatched g0_refcv7.py child otherwise.")
    if not _has("--tag", argv):
        raise SystemExit("[run_battery_r8] refusing: pass --tag (the r8 stamp is written under <out-root>/<tag>)")
    if smoke:
        mc, mw = _val("--max-clips", argv), _val("--max-windows-per-clip", argv)
        if not mc or not mw or int(mc) > 2 or int(mw) > 3:
            raise SystemExit("[run_battery_r8] --smoke-cpu needs --max-clips <=2 and --max-windows-per-clip <=3 "
                             "(a CPU roll of the full grid would take days)")
        for flag, val in (("--device", "cpu"), ("--n-boot", "50")):
            if not _has(flag, argv):
                argv += [flag, val]
        if _val("--device", argv) != "cpu":
            raise SystemExit("[run_battery_r8] --smoke-cpu with a non-cpu --device")
        if "--lock-held" not in argv:
            argv.append("--lock-held")                      # a CPU smoke never takes the GPU lock
    for flag, val in (("--config", f"{KIT}/ckpt/config.json"), ("--out-root", f"{KIT}/battery"),
                      ("--yield-log", f"{KIT}/chain/gpu_yields.jsonl")):
        if not _has(flag, argv):
            argv += [flag, val]
    return argv, {"smoke": smoke, "nonblock": nonblock, "nobase": nobase, "osall": osall, "help": False}


def main() -> None:
    argv, opt = prepare_argv(sys.argv[1:])
    import r8_battery_patch as BP                          # imported AFTER the refusals: apply() loads torch / the loader
    L, P = BP.apply()
    RB = BP.load_battery_module(g0_nonblocking=opt["nonblock"])
    if opt["help"]:
        sys.argv = [sys.argv[0]] + argv
        RB.main()
        return
    kit_ids = set(RB._clip_ids_by_index(RB.KIT_EVAL))
    windows = RB.RR.s2_window_list(P.BASELINES["b_refcv4b"], kit_ids)    # the window source stays refcv4b either way
    if opt["nobase"]:
        P.baselines_available = lambda: {}                  # the panels / S6 are built with NO banked baseline
        print("[run_battery_r8] --no-baselines: panels and S6 are built without any banked baseline", flush=True)
    else:
        BP.add_refcv7_baselines(P, RB)
        dropped = BP.drop_unusable_optional(P, windows)
        if dropped:
            print(f"[run_battery_r8] optional baselines DROPPED (cannot pair on the S2 windows): {dropped}", flush=True)
    RB.subprocess = BP.SubprocessShim(opt["smoke"], opt["osall"])
    if opt["smoke"]:
        RB.gate_wait = lambda out_json, min_free_ram_gb=6.0: {"ok": True, "smoke_cpu_bypass": True}
    out_root, tag = _val("--out-root", argv), _val("--tag", argv)
    sys.argv = [sys.argv[0]] + argv
    t0 = time.time()
    try:
        RB.main()
    finally:
        sp = Path(out_root) / tag / "battery_summary.json"
        if sp.exists():
            j = json.load(open(sp, encoding="utf-8"))
            j["r8"] = {"wrapper": "run_battery_r8.py", "argv": argv, "smoke_cpu": opt["smoke"],
                       "g0_nonblocking": opt["nonblock"], "no_baselines": opt["nobase"], "os_only_all": opt["osall"],
                       "pose_sync_policy": ("OFF (R8_NO_POSE_SYNC): the standalone, refcv7-comparable clock"
                                            if os.environ.get("R8_NO_POSE_SYNC", "") in ("1", "true", "yes") else
                                            "ON: the corrected clock the run trained and was evaluated on"),
                       "wall_s": round(time.time() - t0, 1), "patch": BP.describe(),
                       "repoint": __import__("r8_repoint").describe(L)}
            json.dump(j, open(sp, "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    main()
