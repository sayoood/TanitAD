#!/usr/bin/env python
"""How many windows ever carry a tactical LABEL — in the TRAINING shape and the
EVAL shape. 0 GPU, loader metadata only.

WHY THIS EXISTS. `RefAV1Windows` attaches the v7.2 lat/lon label to a window
ONLY when the window's NOW falls inside the clip's tactical band
(`refav1_loader.py:443`: ``row[3] <= t*dt <= row[4]``, the band being
``t0_s ± (tac_hi - tac_lo)/2``). Every other window emits ``-100`` and
``F.cross_entropy``'s ``ignore_index`` drops it. The number of gradient-carrying
rows per epoch is therefore NOT the number of windows, and the two shapes differ
because ``reach`` differs:

  * TRAIN  ``str_ext_steps=2`` -> reach = round((6 + 2*3)/0.2) = 60
  * EVAL   ``str_ext_steps=0`` -> reach = K (30 here)

and the window range is ``t in [W-1, T_c - reach - 2]`` (`:211`). A shorter
reach admits LATER windows, and the labelled band sits late (t0_s = 8.0 s for
every v7.2 record), so the two shapes do not have the same labelled fraction.

⚠️ The episodes here are the EVAL slice. The TRAIN corpus is a different (much
larger) episode set on Thor and is NOT contacted; the train-shaped number below
is therefore the shape's arithmetic evaluated on THESE episode lengths —
label it INFERRED-for-train, MEASURED-for-this-slice.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys


def census(cache, episodes, labels, nav, op_steps, str_ext_steps, lru, tag):
    from tanitad.data.refav1_loader import RefAV1Windows
    ld = RefAV1Windows(cache, episodes, op_window=4, op_steps=op_steps,
                       op_dt=0.2, str_dt=3.0, str_ext_steps=str_ext_steps,
                       lru=lru, seed=0, labels_path=labels, nav_path=nav)
    rep = getattr(ld, "_join_report", None) or getattr(ld, "report", None) or {}
    per_ep = {}
    n_in = 0
    for ei, t in ld.windows:
        nm = ld.names[ei]
        row = ld._lab.get(ld.clip_id[nm])
        ok = row is not None and row[3] <= t * ld.dt <= row[4]
        d = per_ep.setdefault(nm, {"n": 0, "n_in_band": 0, "t_min": 10**9,
                                   "t_max": -1})
        d["n"] += 1
        d["n_in_band"] += int(ok)
        d["t_min"] = min(d["t_min"], t)
        d["t_max"] = max(d["t_max"], t)
        n_in += int(ok)
    tot = len(ld.windows)
    per = [v["n"] for v in per_ep.values()]
    perb = [v["n_in_band"] for v in per_ep.values()]
    return {
        "shape": tag, "op_steps": op_steps, "str_ext_steps": str_ext_steps,
        "reach": int(ld.reach), "W": int(ld.W), "dt": float(ld.dt),
        "n_episodes": len(ld.names),
        "n_windows": tot, "n_windows_in_band": n_in,
        "frac_in_band": round(n_in / tot, 6) if tot else None,
        "windows_per_episode_mean": round(statistics.mean(per), 3),
        "in_band_per_episode_mean": round(statistics.mean(perb), 3),
        "in_band_per_episode_min": min(perb), "in_band_per_episode_max": max(perb),
        "t_range_over_episodes": [min(v["t_min"] for v in per_ep.values()),
                                  max(v["t_max"] for v in per_ep.values())],
        "band_seconds_from_first_clip": (
            [round(next(iter(ld._lab.values()))[3], 3),
             round(next(iter(ld._lab.values()))[4], 3)] if ld._lab else None),
        "expected_batches_with_ZERO_labelled_rows_pct": {
            str(bs): round(100.0 * (1.0 - n_in / tot) ** bs, 4)
            for bs in (1, 4, 8, 16)} if tot else None,
        "per_episode": per_ep,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--nav", required=True)
    ap.add_argument("--lru", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"tool": "window_band_census.py", "tier": "T0",
           "evidence_class": "MEASURED (this episode slice) / "
                             "INFERRED (train corpus, same shape arithmetic)",
           "inputs": {"cache": a.cache, "episodes": a.episodes,
                      "labels": a.labels},
           "rule": ("label attached iff |t*0.2 - t0_s| <= (tac_hi-tac_lo)/2 "
                    "(refav1_loader.py:443); all other rows are -100 and are "
                    "dropped by cross_entropy's ignore_index"),
           "shapes": []}
    for tag, op_steps, ext in (("train (str_ext_steps=2)", 30, 2),
                               ("eval  (str_ext_steps=0)", 30, 0)):
        res["shapes"].append(census(a.cache, a.episodes, a.labels, a.nav,
                                    op_steps, ext, a.lru, tag))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for s in res["shapes"]:
        print(f"{s['shape']}: reach={s['reach']} windows={s['n_windows']} "
              f"in-band={s['n_windows_in_band']} ({100*s['frac_in_band']:.2f}%) "
              f"per-ep {s['windows_per_episode_mean']} / "
              f"{s['in_band_per_episode_mean']} t={s['t_range_over_episodes']}")
        print(f"   P(batch has ZERO labelled rows): "
              f"{s['expected_batches_with_ZERO_labelled_rows_pct']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
