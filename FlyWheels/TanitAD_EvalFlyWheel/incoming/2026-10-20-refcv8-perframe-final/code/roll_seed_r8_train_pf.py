"""roll_seed_r8_train_pf.py -- the train-fit roll of refcv8 (A6 train139, os-only) with the PER-FRAME nav feed (SHIP-R8 AMENDMENT 3 (ii)).

NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  It changes exactly ONE thing against the registered clip-fed train roll: the nav input.
  * the roll itself is ``roll_seed_r8_train.py`` (battery package, READ-ONLY import, UNCHANGED): TRAIN remap, TRAIN v9 md5, the standalone
    pose clock (R8_NO_POSE_SYNC=1), the t % 5 == 0 window rule over the 139 A6 train clips, os + model-free controls, its own argv;
  * BEFORE its ``main()`` runs, ``navfeed_overlay.install_into_r7_roll`` (nav-feed hold, sha256 registered in PREREG_NAV_FEED.md) makes
    ``r7_roll.ra3()`` return the PATCHED refcv3_arm, and ``--nav-feed perframe_full`` is appended to the argv refcv3_arm's ``main`` receives:
    the per-frame v9 token at the window's NOW row + ``r8_nav = scale_nav(args, known)`` exactly as training builds it.
After the roll the record (``--out-json``) gains a ``pf`` block, asserted on the ARTIFACT (manifest.nav_feed + the decisions sidecar):
``pf.ok`` is True only if the manifest says perframe_full and every decisions file carries nav_cmd / nav_cmd_clip / r8_nav_known / r8_nav_fed.
The independent feed-parity check (v9 npz looked up directly, per-clip token from the CLIP-FED train dump) is ``feed_parity_pf.py``.

    python roll_seed_r8_train_pf.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 \
        --dump-dir C:/Users/Admin/refcv8_pf/trainfit/step30000ema_train139_pf/dump_s0 --out-json .../roll_s0.json --device cuda
    (+ --max-clips N --max-windows-per-clip M --smoke-cpu-fp32-trunk --device cpu = SMOKE ONLY, as in the parent script)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R8_CODE = Path(os.environ.get("PF_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
NF_HOLD = Path(os.environ.get("PF_NF_HOLD", "C:/Users/Admin/qland/hold_r8navfix/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-20-refcv8-nav-feed-fix"))
NF_OVERLAY = NF_HOLD / "code" / "overlay"
NAV_FEED = "perframe_full"
#: the overlay file this roll must run (== the sha256 PREREG_NAV_FEED.md commits; re-checked against the prereg TEXT at start)
OVERLAY_SHA256 = "98293dd7411270160cb47d070bb1538babe1ff3a0f8ab85da61dcc84129ba72f"
SIDE_KEYS = ("nav_cmd", "nav_cmd_clip", "r8_nav_known", "r8_nav_fed")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_overlay() -> dict:
    """The overlay must be the registered file: its sha256 equals OVERLAY_SHA256 AND appears in the sealed prereg text."""
    got = sha256_file(NF_OVERLAY / "navfeed_overlay.py")
    pre = (NF_HOLD / "PREREG_NAV_FEED.md").read_text(encoding="utf-8")
    if len(got) != 64 or got != OVERLAY_SHA256 or got not in pre:
        raise SystemExit(f"[train_pf] navfeed_overlay.py sha256 {got!r} is not the registered {OVERLAY_SHA256} (or not in PREREG_NAV_FEED.md)")
    return {"navfeed_overlay_sha256": got, "prereg_nav_feed_sha256": sha256_file(NF_HOLD / "PREREG_NAV_FEED.md")}


def sidecar_census(dump_dir: str) -> dict:
    """Artifact assertion: the manifest names the feed, every decisions file carries the four per-frame keys with one row per window."""
    import numpy as np
    d = Path(dump_dir)
    man = json.load(open(d / "manifest.json", encoding="utf-8"))
    nf = man.get("nav_feed") or {}
    n_files = n_win = n_known = n_diff = 0
    missing = []
    for e in man["episodes"]:
        z = np.load(d / "decisions" / f"ep{int(e['file_index']):03d}.npz", allow_pickle=False)
        n = len(z["ws"])
        for k in SIDE_KEYS:
            if k not in z.files or z[k].shape[0] != n:
                missing.append(f"ep{int(e['file_index']):03d}:{k}")
        if not missing:
            n_known += int(np.asarray(z["r8_nav_known"], bool).sum())
            n_diff += int((z["nav_cmd"].astype(np.int64) != z["nav_cmd_clip"].astype(np.int64)).sum())
        n_files += 1
        n_win += n
    ok = (nf.get("mode") == NAV_FEED and nf.get("args_fed") is True and not missing and n_win == int(man["grid"]["n_windows"]))
    return {"ok": bool(ok), "manifest_nav_feed": nf, "n_decision_files": n_files, "n_windows": n_win, "n_args_known": n_known,
            "n_token_differs_from_clip": n_diff, "missing_keys_first": missing[:5], "n_missing": len(missing)}


def main() -> int:
    ov = check_overlay()
    sys.path.insert(0, str(R8_CODE))
    import roll_seed_r8_train as T                     # READ-ONLY import: TRAIN remap + loader patch applied at import (UNCHANGED)
    sys.path.insert(0, str(NF_OVERLAY))                # the patched refcv3_arm does `import navfeed_overlay`
    import navfeed_overlay as NO
    R = NO.install_into_r7_roll(T.RR)                  # r7_roll.ra3() now returns the PATCHED refcv3_arm (BEFORE install_patches runs)
    meta = dict(R.__navfeed__)
    _orig = R.main

    def main_with_feed(argv_=None):
        av = list(argv_ or [])
        if "--nav-feed" in av:
            raise SystemExit("[train_pf] the parent argv already carries --nav-feed")
        return _orig(av + ["--nav-feed", NAV_FEED])
    R.main = main_with_feed
    print(f"[train_pf] nav_feed={NAV_FEED} overlay={ov['navfeed_overlay_sha256'][:16]} patched_refcv3_arm={meta['patched_sha256'][:16]}", flush=True)
    T.main()                                           # parses sys.argv exactly as the clip-fed train roll does
    a = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
    out_json, dump_dir = a["--out-json"], a["--dump-dir"]
    rec = json.load(open(out_json, encoding="utf-8"))
    cen = sidecar_census(dump_dir)
    rec["pf"] = {"nav_feed": NAV_FEED, "ok": cen["ok"], "census": cen, "overlay": ov, "patched_refcv3_arm": meta,
                 "parent_roll_script": str(R8_CODE / "roll_seed_r8_train.py"),
                 "parent_roll_script_sha256": sha256_file(R8_CODE / "roll_seed_r8_train.py"),
                 "this_script_sha256": sha256_file(__file__),
                 "spec": "SHIP-R8 AMENDMENT 3 (ii): the train-fit procedure re-run on PER-FRAME-fed train rolls"}
    rec["tool"] = "roll_seed_r8_train_pf.py (wraps roll_seed_r8_train.py)"
    json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[train_pf] pf.ok={cen['ok']} windows={cen['n_windows']} args_known={cen['n_args_known']} token_differs_from_clip={cen['n_token_differs_from_clip']} "
          f"{'ZZTRAINPFROLLOKZZ' if cen['ok'] else 'ZZTRAINPFROLLBADZZ'}", flush=True)
    return 0 if cen["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
