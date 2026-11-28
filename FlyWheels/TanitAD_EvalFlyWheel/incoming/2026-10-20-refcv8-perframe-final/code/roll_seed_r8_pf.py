"""roll_seed_r8_pf.py -- ONE inference-seed EVAL battery roll of refcv8 with the PER-FRAME nav feed (SHIP-R8 AMENDMENT 3 (i)).

NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  It is ``roll_seed_r8.py`` (battery package) plus exactly ONE change, the nav input:
  * ``r8_battery_patch.apply_child()`` (READ-ONLY import, UNCHANGED): dev-box remap, the pose-clock policy from the ENVIRONMENT
    (R8_NO_POSE_SYNC=1 = the standalone clock every 48,852 battery pass uses), the seam / microbatch patches;
  * BEFORE ``roll_seed_r7.main()`` (launch tree, UNCHANGED, same argv, same ``--os-only`` rule) builds anything,
    ``navfeed_overlay.install_into_r7_roll`` (nav-feed hold, sha256 98293dd7... registered in PREREG_NAV_FEED.md) makes ``r7_roll.ra3()``
    return the PATCHED refcv3_arm, and ``--nav-feed perframe_full`` is appended to the argv refcv3_arm's ``main`` receives.
After the roll the record (``--out-json``) gains a ``pf`` block asserted on the ARTIFACT (manifest.nav_feed + decisions sidecar, see
``roll_seed_r8_train_pf.sidecar_census``).  Exit 3 if the artifact does not show the feed.  The independent clause-(1b) check is
``feed_parity_pf.py`` (v9 npz looked up directly; per-clip token from a CLIP-FED dump of the same windows).

KNOWN, STATED: seed 0 runs the full SPEC 3.2 arm set.  The nav-condition arms (shuffled / flipped / zero) are derived from the per-frame
``nav_true`` by the overlay (perturbed rows carry the UNKNOWN r8_nav row, by design of PREREG_NAV_FEED).  The speed-obedience /
vmax-zero proxy arms of r7_roll call the model through their own path and carry NO r8_nav (= the unknown row); they are BESIDE arms of
the battery and enter no SHIP-R8 clause.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import roll_seed_r8_train_pf as TP  # noqa: E402  (overlay check + artifact census; no side effects at import)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass

OV = TP.check_overlay()
sys.path.insert(0, str(TP.R8_CODE))
import r8_battery_patch as BP  # noqa: E402  (live battery package, READ-ONLY)

BP.apply_child()
sys.path.insert(0, str(TP.NF_OVERLAY))
import navfeed_overlay as NO  # noqa: E402
import r7_roll as RR  # noqa: E402  (launch tree, on sys.path via r8_repoint)

R = NO.install_into_r7_roll(RR)                     # BEFORE roll_one -> RR.install_patches() -> RR.ra3()
META = dict(R.__navfeed__)
_orig_main = R.main


def _main_with_feed(argv_=None):
    av = list(argv_ or [])
    if "--nav-feed" in av:
        raise SystemExit("[roll_pf] the parent argv already carries --nav-feed")
    return _orig_main(av + ["--nav-feed", TP.NAV_FEED])


R.main = _main_with_feed
import roll_seed_r7 as RS  # noqa: E402  (launch tree: loader bootstrap at import, re-pointed by apply_child)


def main() -> int:
    print(f"[roll_pf] nav_feed={TP.NAV_FEED} overlay={OV['navfeed_overlay_sha256'][:16]} patched_refcv3_arm={META['patched_sha256'][:16]} "
          f"pose_sync={'OFF (R8_NO_POSE_SYNC)' if os.environ.get('R8_NO_POSE_SYNC') in ('1', 'true', 'yes') else 'ON'}", flush=True)
    RS.main()
    a = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
    out_json, dump_dir = a["--out-json"], a["--dump-dir"]
    rec = json.load(open(out_json, encoding="utf-8"))
    cen = TP.sidecar_census(dump_dir)
    rec["pf"] = {"nav_feed": TP.NAV_FEED, "ok": cen["ok"], "census": cen, "overlay": OV, "patched_refcv3_arm": META,
                 "parent_child_script": str(TP.R8_CODE / "roll_seed_r8.py"), "parent_child_script_sha256": TP.sha256_file(TP.R8_CODE / "roll_seed_r8.py"),
                 "this_script_sha256": TP.sha256_file(__file__), "spec": "SHIP-R8 AMENDMENT 3 (i): every clause read on rolls fed perframe_full",
                 "os_only": "--os-only" in sys.argv,
                 "beside_arms_note": "speed-obedience / vmax-zero proxy arms carry no r8_nav (unknown row); beside arms, in no SHIP-R8 clause"}
    json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[roll_pf] pf.ok={cen['ok']} windows={cen['n_windows']} args_known={cen['n_args_known']} "
          f"token_differs_from_clip={cen['n_token_differs_from_clip']} {'ZZROLLPFOKZZ' if cen['ok'] else 'ZZROLLPFBADZZ'}", flush=True)
    if not cen["ok"]:
        os.replace(out_json, out_json + ".pf_bad")       # the parent treats a missing record as a failed roll: no un-fed dump is analysed
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
