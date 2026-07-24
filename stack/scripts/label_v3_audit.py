"""v2.1 -> v3 LABEL MIGRATION audit on real PhysicalAI-AV poses.

Answers, on a real corpus rather than on synthetic arcs, the two questions the
v3 label set was built for:

  ROUTE     how many windows that v2.1 calls `left`/`right`/`follow` does v3
            call `roundabout` / `exit_left` / `exit_right` / `merge` / `u_turn`,
            and how far ahead is the maneuver (the DISTANCE half, which is the
            part a planner can actually act on)?
  TACTICAL  how many windows does the shipped 5-way `classify_maneuver_v2`
            label with a LATERAL class while a longitudinal mode is live — i.e.
            how much longitudinal information the single 5-way softmax
            structurally destroys before any model ever sees it?

Reads ONLY poses out of an episode cache (mmap, no frames touched, no GPU, no
pod). Selects nothing: it walks whatever episodes the given cache contains, in
sorted order, with the canonical eval window protocol, so it cannot perturb
corpus parity.

Protocols
  eval   window=8, stride=8, K_MAX=20 -> starts = range(0, T-28, 8), the label
         is taken at the LAST window pose (start + 7). This is byte-identical to
         taniteval.refc_rerank.dump / refc_eval.collect, so a window index here
         is the same window index as in results/fan_refc-*.pt.
  dense  every ``--stride`` steps from 0 (broader statistics, not fan-aligned).

Usage
  PYTHONPATH=<repo>/stack python scripts/label_v3_audit.py \
      --val <epcache-dir> --protocol eval --json <out.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refb_labels as R                                            # noqa: E402

WINDOW, STRIDE, K_MAX = 8, 8, 20            # the canonical eval protocol
MAN5_NAMES = {R.LANE_KEEP: "lane_keep", R.TURN_LEFT: "turn_left",
              R.TURN_RIGHT: "turn_right", R.ACCELERATE: "accelerate",
              R.BRAKE_STOP: "brake_stop"}
MAN5_LATERAL = (R.LANE_KEEP, R.TURN_LEFT, R.TURN_RIGHT)


def _window_starts(T: int, protocol: str, stride: int):
    """(t_label, ...) — the pose index each window's label is taken at."""
    if protocol == "eval":
        return [s + WINDOW - 1 for s in range(0, T - WINDOW - K_MAX, STRIDE)]
    return list(range(0, T - 1, stride))


def audit(val_dir: str, protocol: str = "eval", stride: int = 8,
          episodes: int | None = None, horizon: int = R.NAV_HORIZON_STEPS
          ) -> dict:
    files = sorted(glob.glob(os.path.join(val_dir, "ep_*.pt")))
    if episodes:
        files = files[:episodes]
    assert files, f"no ep_*.pt under {val_dir}"

    route_mig = Counter()      # (token_v21, token_v3)
    v3_rule = Counter()
    dist_band = Counter()
    dist_band_by_token = {}
    rb_candidate = 0
    uturn_confounded = 0
    ep_with_token = {}          # token -> set(episode files): the EVENT count.
                                # Window counts over-report: one 180 deg arc is
                                # seen by ~8 consecutive stride-8 windows.
    dists_m = []
    lat_tok, lon_tok, man5_tok = Counter(), Counter(), Counter()
    man5_x_lon = Counter()
    collapsed = 0
    lon_active = 0
    stop_bands = Counter()
    stop_dists = []
    n = 0
    per_ep = []

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        poses = d["poses"].float()
        eid = int(d["episode_id"])
        T = poses.shape[0]
        ts = _window_starts(T, protocol, stride)
        ep_up = 0
        for t in ts:
            r = R.route_from_future_v3(poses, t, horizon)
            tac = R.tactical_from_future_v3(poses, t)
            n += 1
            route_mig[(r["token_v21"], r["token"])] += 1
            v3_rule[r["v3_rule"]] += 1
            dist_band[r["dist_band"]] += 1
            dist_band_by_token.setdefault(r["token"], Counter())[r["dist_band"]] += 1
            rb_candidate += int(r["roundabout_candidate"])
            uturn_confounded += int(r.get("uturn_roundabout_confounded", False))
            if r["upgraded"]:
                ep_with_token.setdefault(r["token"], set()).add(os.path.basename(f))
            if r["roundabout_candidate"]:
                ep_with_token.setdefault("roundabout_candidate", set()).add(
                    os.path.basename(f))
            ep_up += int(r["upgraded"])
            if r["dist_m"] is not None:
                dists_m.append(round(float(r["dist_m"]), 2))
            lat_tok[tac["lat"]["token"]] += 1
            lon_tok[tac["lon"]["token"]] += 1
            lon_active += int(tac["lon"]["active"])
            m5 = MAN5_NAMES.get(tac["man5"], "none")
            man5_tok[m5] += 1
            man5_x_lon[(m5, tac["lon"]["token"])] += 1
            collapsed += int(tac["collapsed"])
            stop_bands[tac["lon"]["stop_dist_band"]] += 1
            if tac["lon"]["stop_dist_m"] is not None:
                stop_dists.append(round(float(tac["lon"]["stop_dist_m"]), 2))
        per_ep.append({"file": os.path.basename(f), "episode_id": eid, "T": T,
                       "windows": len(ts), "upgraded": ep_up})
        del d, poses

    def _pct(x):
        return round(100.0 * x / max(n, 1), 3)

    upgraded = sum(c for (a, b), c in route_mig.items() if a != b)
    return {
        "val_dir": val_dir, "n_episodes": len(files), "n_windows": n,
        "reproduce": (f"PYTHONPATH=<repo>/stack python scripts/label_v3_audit.py "
                      f"--val {val_dir} --protocol {protocol}"
                      + (f" --stride {stride}" if protocol != "eval" else "")),
        "protocol": protocol, "stride": stride if protocol != "eval" else STRIDE,
        "route_horizon_steps": horizon,
        "labeler": "scripts/refb_labels.py route_from_future_v3 / "
                   "tactical_from_future_v3 (v2.1 base UNCHANGED)",
        "route": {
            "upgraded": upgraded, "upgraded_pct": _pct(upgraded),
            "roundabout_candidates_not_promoted": rb_candidate,
            "u_turn_windows_confounded_with_roundabout": uturn_confounded,
            "episodes_with_token": {k: len(v) for k, v in
                                    sorted(ep_with_token.items())},
            "episode_files_with_token": {k: sorted(v) for k, v in
                                         sorted(ep_with_token.items())},
            "v3_rule": dict(v3_rule),
            "migration": {f"{a}->{b}": c for (a, b), c in
                          sorted(route_mig.items(), key=lambda kv: -kv[1])},
            "token_v3": dict(Counter({b: 0 for (a, b) in route_mig})
                             | {b: sum(c for (x, y), c in route_mig.items() if y == b)
                                for (a, b) in route_mig}),
            "dist_band": dict(dist_band),
            "dist_band_by_token": {k: dict(v) for k, v in
                                   dist_band_by_token.items()},
            "dist_m_quantiles": _quant(dists_m),
            "n_with_distance": len(dists_m),
        },
        "tactical": {
            "man5": dict(man5_tok), "lat": dict(lat_tok), "lon": dict(lon_tok),
            "lon_active": lon_active, "lon_active_pct": _pct(lon_active),
            "collapsed": collapsed, "collapsed_pct": _pct(collapsed),
            "collapsed_definition": "5-way label is LATERAL (lane_keep/"
                                    "turn_left/turn_right) while the factorized "
                                    "LONMODE is active (not free_cruise)",
            "man5_x_lon": {f"{a}|{b}": c for (a, b), c in
                           sorted(man5_x_lon.items(), key=lambda kv: -kv[1])},
            "stop_dist_band": dict(stop_bands),
            "stop_dist_m_quantiles": _quant(stop_dists),
            "n_with_stop_distance": len(stop_dists),
        },
        "per_episode": per_ep,
    }


def _quant(xs):
    if not xs:
        return None
    t = torch.tensor(sorted(xs))
    q = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {f"p{int(p*100)}": round(float(v), 2)
            for p, v in zip(q.tolist(), torch.quantile(t, q).tolist())}


def main():
    ap = argparse.ArgumentParser("label_v3_audit")
    ap.add_argument("--val", required=True, help="episode cache dir (ep_*.pt)")
    ap.add_argument("--protocol", choices=("eval", "dense"), default="eval")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=R.NAV_HORIZON_STEPS)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = audit(a.val, a.protocol, a.stride, a.episodes, a.horizon)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
    r, tc = out["route"], out["tactical"]
    print(f"[audit] {out['n_episodes']} episodes, {out['n_windows']} windows, "
          f"protocol={out['protocol']}")
    print(f"  ROUTE   upgraded {r['upgraded']} ({r['upgraded_pct']} %)  "
          f"roundabout candidates not promoted {r['roundabout_candidates_not_promoted']}"
          f"  u_turn windows confounded w/ roundabout "
          f"{r['u_turn_windows_confounded_with_roundabout']}")
    print(f"  events (DISTINCT EPISODES of {out['n_episodes']}) "
          f"{r['episodes_with_token']}")
    for k, v in sorted(r["migration"].items(), key=lambda kv: -kv[1]):
        if k.split("->")[0] != k.split("->")[1]:
            print(f"            {k:34s} {v}")
    print(f"  dist    {r['dist_band']}")
    print(f"  TACTICAL man5 {tc['man5']}")
    print(f"           lon  {tc['lon']}")
    print(f"           lat  {tc['lat']}")
    print(f"           collapsed {tc['collapsed']} ({tc['collapsed_pct']} %)  "
          f"lon_active {tc['lon_active']} ({tc['lon_active_pct']} %)")
    print(f"           stop bands {tc['stop_dist_band']}")
    if a.json:
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
