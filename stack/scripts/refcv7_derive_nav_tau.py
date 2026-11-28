"""refcv7 — derive the nav-compliance tolerance ``tau_rad`` on the TRAIN split.

The refcv7 scorer's ``nav`` sub-score (``refcv7_oracle``) uses the programme's own predicate
``refc_selector_targets.compliance_target``: a candidate complies when its terminal heading
has the commanded sign and a magnitude of at least ``tau``. ``tau`` has NO default anywhere
(an invented tolerance is a number with no evidence class), so the trainer refuses
``--w-r7-scorer > 0`` without ``--r7-nav-tau-rad``. This script produces that number.

⛔ It reuses the TRAINER'S OWN objects, never a re-implementation: the v2 providers
(``build_v2_providers``), ``V3Dataset`` and its window index, the v7/v8 label join and
``enable_nav_from_v7`` (the SAME nav the trainer feeds), ``refb_labels.waypoint_targets``
(the SAME GT the loss uses) and ``compliance_target``'s terminal-heading rule (last segment,
0.05 m stall). The rule is ``taniteval.nav_compliance.derive_tolerance`` (Youden's J of
|terminal heading| on LEFT/RIGHT windows vs FOLLOW windows) — GT + label only.

⛔ Run it on the split the model TRAINS on and pass the printed ``tau`` as
``--r7-nav-tau-rad``. A tau derived on the scored split is tuning on the scored split.

Usage:
  PYTHONPATH=<repo>/stack;<repo>/stack/scripts;<repo>/taniteval \\
  python refcv7_derive_nav_tau.py --v2-cache <train cache> --labels <s2_labels_*.jsonl.gz> \\
      [--window 8] [--stride 5] [--json out.json]
Only POSES are read (no frames are decoded), so it is cheap per clip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent.parent / "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refb_labels                                          # noqa: E402
import refc_v3_train as T                                   # noqa: E402
from tanitad.data.v2_dataset import build_v2_providers, stable_episode_id  # noqa: E402
from tanitad.refs import refc_v3 as v3                      # noqa: E402
from tanitad.data import v7_labels as v7l                   # noqa: E402


def terminal_heading(xy: np.ndarray, stall_m: float = 0.05) -> float:
    """``compliance_target``'s rule, on one GT path [S, 2]: the heading of the LAST
    segment, 0.0 when that segment is shorter than ``stall_m``."""
    d = xy[-1] - xy[-2]
    return 0.0 if float(np.hypot(*d)) < stall_m else float(np.arctan2(d[1], d[0]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--channels", type=int, default=9,
                    help="frame channels AS STORED in the cache (3-frame stack = 9); "
                         "only poses are read, so this must merely agree with the cache")
    ap.add_argument("--stride", type=int, default=5,
                    help="subsample windows (every k-th); tau is a distribution "
                         "statistic, and neighbouring windows are near-duplicates")
    ap.add_argument("--json")
    a = ap.parse_args(argv)

    from taniteval.nav_compliance import derive_tolerance

    eps = build_v2_providers(a.v2_cache, lru_size=2)
    ds = T.V3Dataset(eps, window=a.window, max_horizon=20, channels=a.channels)
    labels, manifest = v7l.load_v7_labels(a.labels, allow_oracle_nav=True)
    ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in labels}
    ds.v7_dt = 0.1
    stats = ds.enable_nav_from_v7(manifest)
    nav_by_sid = ds._nav_by_sid
    left = T.refb.NAV_COMMANDS.index("left")
    right = T.refb.NAV_COMMANDS.index("right")
    follow = T.refb.NAV_COMMANDS.index("follow")
    hz = tuple(v3.V3_HORIZONS)
    pos, neg = [], []
    n_win = 0
    for i, (e_i, t) in enumerate(ds.index):
        if i % a.stride:
            continue
        ep = ds.episodes[e_i]
        nav = nav_by_sid.get(int(ep.episode_id))
        if nav is None:
            continue
        f = t + a.window - 1
        T_ = int(ep.poses.shape[0])
        if f + max(hz) > T_ - 1:
            continue                          # GT must cover the full 6 s
        idx = torch.arange(f + 1, f + 1 + max(hz))
        fut = ep.poses[idx][None]
        wp = refb_labels.waypoint_targets(ep.poses[f][None], fut, hz)[0].numpy()
        th = terminal_heading(wp)
        n_win += 1
        if nav == left:
            pos.append(th)                    # signed; derive_tolerance thresholds |.|
        elif nav == right:
            pos.append(th)
        elif nav == follow:
            neg.append(th)
    res = derive_tolerance(pos, neg)
    res.update({"split_v2_cache": a.v2_cache, "labels": a.labels,
                "windows_scored": n_win, "stride": a.stride, "window": a.window,
                "nav_from_v7_stats": stats,
                "rule_source": "taniteval.nav_compliance.derive_tolerance",
                "heading_rule": "compliance_target: last-segment heading, 0.05 m stall"})
    print(json.dumps(res, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0 if res.get("status") == "OK" else 3


if __name__ == "__main__":
    raise SystemExit(main())
