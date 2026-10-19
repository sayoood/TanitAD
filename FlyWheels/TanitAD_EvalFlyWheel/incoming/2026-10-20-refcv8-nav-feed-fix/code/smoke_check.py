"""smoke_check.py -- assert, on the ARTIFACTS of the two mini-rolls (clip, perframe_full; 6 windows of one episode), that the feed is wired.

Run by the wrapper under the GPU lock BEFORE the 75-minute rolls.  Any failed assertion -> ``ok: false`` in ``--out`` and the wrapper stops without
spending the rest of the slot.  numpy only.

  clip dump  : decisions.nav_cmd == the clip token the battery fed (sealed ``mismatch_fed`` / the control token); manifest carries NO nav_feed block
  full dump  : manifest.nav_feed.mode == perframe_full; nav_cmd == the v9 per-frame token (sealed ``mismatch_v9`` / ``control_v9``);
               nav_cmd_clip == the clip token; r8_nav_fed has a non-zero row exactly where r8_nav_known; every plan finite
  reach      : on the MISMATCH windows the two plans DIFFER (a feed that never reaches the model leaves them identical)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(dump):
    d = Path(dump)
    man = json.load(open(d / "manifest.json", encoding="utf-8"))
    out = {}
    for e in man["episodes"]:
        fi = int(e["file_index"])
        dec = np.load(d / "decisions" / f"ep{fi:03d}.npz", allow_pickle=False)
        for i, w in enumerate(dec["ws"].astype(np.int64)):
            out[(int(e["episode_index"]), int(w))] = {k: dec[k][i] for k in dec.files if k not in ("ep_poses",)}
    return man, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dump", required=True)
    ap.add_argument("--full-dump", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--smoke-episode-index", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    win = json.load(open(a.windows, encoding="utf-8"))
    ep = next(e for e in win["episodes"] if int(e["episode_index"]) == a.smoke_episode_index)
    mism = {w: (f, v) for w, f, v in zip(ep["mismatch_ws"], ep["mismatch_fed"], ep["mismatch_v9"])}
    ctrl = dict(zip(ep["control_ws"], ep["control_v9"]))
    want_ws = sorted(sorted(mism)[:3] + sorted(ctrl)[:3])
    r = {"checks": {}, "ok": False}
    try:
        mc, dc = load(a.clip_dump)
        mf, df = load(a.full_dump)
        ei = a.smoke_episode_index
        ck = lambda name, cond: r["checks"].__setitem__(name, bool(cond))      # noqa: E731
        ck("windows_equal_the_smoke_set", sorted(w for (_e, w) in dc) == want_ws and sorted(w for (_e, w) in df) == want_ws)
        fed = {w: (mism[w][0] if w in mism else ctrl[w]) for w in want_ws}
        v9 = {w: (mism[w][1] if w in mism else ctrl[w]) for w in want_ws}
        ck("clip_nav_cmd_equals_banked_clip_token", all(int(dc[(ei, w)]["nav_cmd"]) == fed[w] for w in want_ws))
        ck("clip_manifest_has_no_nav_feed_block", "nav_feed" not in mc)
        ck("full_manifest_nav_feed_mode", (mf.get("nav_feed") or {}).get("mode") == "perframe_full")
        ck("full_nav_cmd_equals_v9_per_frame_token", all(int(df[(ei, w)]["nav_cmd"]) == v9[w] for w in want_ws))
        ck("full_nav_cmd_clip_equals_banked_clip_token", all(int(df[(ei, w)]["nav_cmd_clip"]) == fed[w] for w in want_ws))
        known = {w: bool(df[(ei, w)]["r8_nav_known"]) for w in want_ws}
        nz = {w: bool(np.abs(df[(ei, w)]["r8_nav_fed"]).sum() > 0) for w in want_ws}
        ck("full_r8_nav_fed_nonzero_exactly_where_known", all(known[w] == nz[w] for w in want_ws))
        ck("plans_finite", all(np.isfinite(dc[(ei, w)]["plan_full_nav_true"]).all() and np.isfinite(df[(ei, w)]["plan_full_nav_true"]).all()
                               for w in want_ws))
        diffs = {w: float(np.abs(dc[(ei, w)]["plan_full_nav_true"] - df[(ei, w)]["plan_full_nav_true"]).max()) for w in want_ws}
        ck("feed_reaches_the_model_plans_differ_on_mismatch_windows", all(diffs[w] > 1e-4 for w in mism if w in want_ws))
        r["max_abs_plan_diff_m_by_ws"] = diffs
        r["n_args_known_in_smoke"] = int(sum(known.values()))
        r["ok"] = all(r["checks"].values())
    except Exception as exc:      # noqa: BLE001 -- recorded; the wrapper asserts on `ok`
        r["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"ok": r["ok"], "failed": [k for k, v in r["checks"].items() if not v], "error": r.get("error")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
