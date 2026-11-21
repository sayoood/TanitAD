"""navfeed_roll.py -- ONE (nav-feed, inference-seed) os-only roll as a CHILD process (EvalFlyWheel 2026-10-20).

The sibling of ``roll_seed_r8.py`` + ``roll_seed_r7.py`` for the nav-feed experiment.  It imports -- and never edits -- the live battery
package (``r8_battery_patch`` -> ``r8_repoint``: dev-box remap, pose-clock policy, seam-guard / microbatch patches) and the launch tree's
``run_battery_r7.roll_one`` (UNCHANGED, called with ``full=False`` = os + the model-free controls only, so ``--no-navshuf --no-navzero``),
then swaps in the navfeed-PATCHED ``refcv3_arm`` (``navfeed_overlay.install_into_r7_roll``) and appends ``--nav-feed`` to the argv that
``roll_one`` builds.

    R8_NO_POSE_SYNC=1 python navfeed_roll.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt --config E:/refcv8_eval_kit/ckpt/config.json \
        --nav-feed perframe_full --seed 0 --dump-dir <dir> --windows-json <{clip_id: [ws,..]}> --out-json <rec.json>

The pose clock is the ENVIRONMENT's (``R8_NO_POSE_SYNC=1`` = the standalone clock the ``step40000ema_unc`` tree was rolled on: the archived
roll log's first line reads ``pose_sync policy: OFF (R8_NO_POSE_SYNC)``).  Assert on the ARTIFACTS (``manifest.json`` + ``--out-json``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R8_CODE = Path(os.environ.get("NF_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
sys.path.insert(0, str(R8_CODE))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass

import r8_battery_patch as BP  # noqa: E402  (the live battery package, READ-ONLY import)

BP.apply_child()                                   # r8_repoint.apply(): env + sys.path + loader patch; the overlay goes FIRST below
sys.path.insert(0, str(HERE))
import navfeed_overlay as NO  # noqa: E402


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nav-feed", required=True, choices=NO.NAV_FEEDS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expect-windows", type=int, default=0, help="assert the dump holds exactly this many windows (0 = len of the json)")
    a = ap.parse_args(argv)
    t0 = time.time()
    import run_battery_r7 as RB                      # launch-tree battery code (on sys.path via r8_repoint)
    import r7_roll as RR
    R = NO.install_into_r7_roll(RR)                  # r7_roll.ra3() now returns the PATCHED refcv3_arm
    meta = dict(R.__navfeed__)
    _orig_main = R.main

    def main_with_feed(argv_=None):
        av = list(argv_ or [])
        if "--nav-feed" not in av:
            av += ["--nav-feed", a.nav_feed]
        return _orig_main(av)
    R.main = main_with_feed
    windows = json.load(open(a.windows_json, encoding="utf-8"))
    n_req = sum(len(v) for v in windows.values())
    pose = "OFF (R8_NO_POSE_SYNC)" if os.environ.get("R8_NO_POSE_SYNC") in ("1", "true", "yes") else "ON"
    print(f"[navfeed_roll] nav_feed={a.nav_feed} seed={a.seed} windows={n_req} episodes={len(windows)} pose_sync={pose} "
          f"patched_sha256={meta['patched_sha256'][:16]} device={a.device}", flush=True)
    rec = RB.roll_one(a.ckpt, a.config, a.seed, a.dump_dir, windows, None, device=a.device, full=False)
    man = json.load(open(os.path.join(a.dump_dir, "manifest.json"), encoding="utf-8"))
    nf = man.get("nav_feed")
    want = a.expect_windows or n_req
    problems = []
    if a.nav_feed == "clip" and nf is not None:
        problems.append("clip mode but the manifest carries a nav_feed block (default must be byte-identical)")
    if a.nav_feed != "clip" and (nf or {}).get("mode") != a.nav_feed:
        problems.append(f"manifest nav_feed {nf!r} != {a.nav_feed}")
    if int(man["grid"]["n_windows"]) != want:
        problems.append(f"dump holds {man['grid']['n_windows']} windows, expected {want}")
    rec.update({"nav_feed": a.nav_feed, "nav_feed_manifest": nf, "navfeed_overlay": meta, "problems": problems,
                "argv": sys.argv[1:], "ckpt": a.ckpt, "ckpt_md5": md5_file(a.ckpt),
                "windows_json_md5": md5_file(a.windows_json), "n_windows_requested": n_req,
                "r8_code_md5": {k: md5_file(R8_CODE / k) for k in ("r8_battery_patch.py", "r8_repoint.py")},
                "env": {k: os.environ.get(k) for k in ("R8_NO_POSE_SYNC", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
                "total_wall_s": round(time.time() - t0, 1), "done": not problems})
    json.dump(rec, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[navfeed_roll] {a.nav_feed} seed {a.seed} -> {a.out_json} done={not problems} problems={problems}", flush=True)
    return 0 if not problems else 3


if __name__ == "__main__":
    sys.exit(main())
