"""Route-label regression harness: v2 (shipped) vs v2.1 (adaptive) on REAL poses.

Originally a one-shot diagnostic that lived only on pod3; promoted here as the
standing regression harness for the route labeler because it is the only check
that runs on REAL trajectories — tests/test_refb_labels_v2*.py pin the physics
on synthetic arcs, this measures what the corpus actually contains.

WHAT IT MEASURES (per labeler version)
  coverage            share of windows carrying a usable route label (valid)
  class balance       left / straight / right / unknown, split by why
  false-straight      windows whose future turns >= FALSE_TURN_DEG of net
                      heading yet are labelled straight AND valid — these TRAIN
                      the head to say "straight" through a real turn
  unlearnable turns   false-straight + masked (valid=False) real turns: the
                      share of genuine turns the strategic head can never learn
  band x label        |net heading| band crossed with the emitted label

GROUND TRUTH is the future ego track itself: |cumulative net heading change|
over the available future. That is a weak but honest referee — it cannot tell a
junction turn from a long road sweep, which is exactly the distinction v2 was
built for, so read `false_straight` together with the radius column rather than
as a verdict on its own. The VLM cross-validation (Part 2) is the independent
referee.

Usage:
  PYTHONPATH=<repo>/stack python scripts/route_label_audit.py \
      --val /workspace/pai_epcache/physicalai-val-f1b378f295ae \
      --stride 20 --json /workspace/route_audit.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refb_labels as R  # noqa: E402

FALSE_TURN_DEG = 30.0                        # >= this net heading = a real turn
BANDS = [(0, 15), (15, 30), (30, 45), (45, 90), (90, 1e9)]
V2_NAMES = {R.ROUTE_LEFT: "left", R.ROUTE_STRAIGHT: "straight",
            R.ROUTE_RIGHT: "right"}


def _band(deg: float):
    return next(b for b in BANDS if b[0] <= deg < b[1])


def _blank():
    return {"n": 0, "valid": 0, "cls": {}, "reason": {}, "band": {},
            "false_straight": [], "masked_turns": 0, "real_turns": 0,
            "unknown": 0, "masked_deg": []}


def _tally(acc, name, route, valid, deg, reason, extra):
    acc["n"] += 1
    acc["valid"] += int(valid)
    key = f"{name}/{'valid' if valid else 'masked'}"
    acc["cls"][key] = acc["cls"].get(key, 0) + 1
    acc["reason"][reason] = acc["reason"].get(reason, 0) + 1
    acc["unknown"] += int(route == R.ROUTE_UNKNOWN)
    bk = f"{_band(deg)[0]}"
    acc["band"].setdefault(bk, {"left": 0, "straight": 0, "right": 0,
                                "unknown": 0, "masked": 0})
    acc["band"][bk][name] += 1
    if not valid:
        acc["band"][bk]["masked"] += 1
        acc["masked_deg"].append(round(deg, 1))
    if deg >= FALSE_TURN_DEG:
        acc["real_turns"] += 1
        if name == "straight" and valid:
            acc["false_straight"].append(extra)
        elif not valid:
            acc["masked_turns"] += 1


def audit(val_dir: str, stride: int = 20, limit: int | None = None) -> dict:
    eps = sorted(glob.glob(os.path.join(val_dir, "ep_*.pt")))
    if limit:
        eps = eps[:limit]
    if not eps:
        raise SystemExit(f"no ep_*.pt under {val_dir}")
    out = {"v2": _blank(), "v21": _blank(), "v21_strict": _blank(),
           "episodes": len(eps), "stride": stride, "windows": 0, "gt_deg": [],
           "config": {
               "NAV_HORIZON_STEPS": R.NAV_HORIZON_STEPS,
               "NAV_MIN_STEPS": R.NAV_MIN_STEPS,
               "MIN_ARC_ROUTE_M": R.MIN_ARC_ROUTE_M,
               "CONC_ARC_M": R.CONC_ARC_M,
               "TRANSIENCE_MIN_ARC_M": R.TRANSIENCE_MIN_ARC_M,
               "NET_DYAW_TURN_DEG": math.degrees(R.NET_DYAW_TURN_RAD),
               "R_TURN_M": 1.0 / R.CURV_TURN_PER_M,
               "R_ROAD_M": 1.0 / R.CURV_ROAD_PER_M,
               "FALSE_TURN_DEG": FALSE_TURN_DEG}}
    for p in eps:
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:                       # corrupt shard: say so
            print(f"  skip {os.path.basename(p)}: {type(e).__name__}")
            continue
        poses = d.get("poses") if isinstance(d, dict) else None
        if poses is None:
            continue
        ep = os.path.basename(p)
        for t in range(0, poses.shape[0], stride):
            out["windows"] += 1
            r21 = R.route_from_future_v21(poses, t)
            r21s = R.route_from_future_v21(poses, t, use_net_dyaw=False)
            r2 = R.route_from_future(poses, t)
            # ONE ground-truth heading for all three, from the adaptive read
            # (v2 returns 0.0 whenever it bails, which would hide the turns).
            deg = abs(math.degrees(r21["net_dyaw"]))
            out["gt_deg"].append(round(deg, 1))
            ex = {"ep": ep, "t": t, "deg": round(deg, 1),
                  "peak_kappa": round(r21["peak_kappa"], 5),
                  "R_m": round(1.0 / max(r21["peak_kappa"], 1e-9), 0),
                  "conc": round(r21["concentration"], 2),
                  "arc_m": round(r21["arc_m"], 1), "h": r21["h_steps"]}
            _tally(out["v2"], V2_NAMES[r2["route"]], r2["route"], r2["valid"],
                   deg, "ambiguous" if r2["ambiguous"] else
                   ("judged" if r2["valid"] else "no_future"), ex)
            _tally(out["v21"], R.ROUTE_V21_NAMES[r21["route"]], r21["route"],
                   r21["valid"], deg, r21["reason"], ex)
            _tally(out["v21_strict"], R.ROUTE_V21_NAMES[r21s["route"]],
                   r21s["route"], r21s["valid"], deg, r21s["reason"], ex)
    return out


def _pct(a, b):
    return 100.0 * a / max(b, 1)




def report(a: dict) -> None:
    n = a["windows"]
    print(f"episodes={a['episodes']}  windows={n}  stride={a['stride']}")
    print("config: " + json.dumps(a["config"]))
    print(f"\n{'':22}{'v2 (shipped)':>16}{'v2.1':>16}{'v2.1 strict':>16}")
    rows = [
        ("coverage (valid)", lambda s: f"{_pct(s['valid'], n):.1f}%"),
        ("unknown emitted", lambda s: f"{_pct(s['unknown'], n):.1f}%"),
        ("turn labels", lambda s: f"{_pct(sum(v for k, v in s['cls'].items() if k.startswith(('left', 'right')) and k.endswith('valid')), n):.1f}%"),
        ("straight+valid", lambda s: f"{_pct(s['cls'].get('straight/valid', 0), n):.1f}%"),
    ]
    for label, fn in rows:
        print(f"  {label:20}{fn(a['v2']):>16}{fn(a['v21']):>16}"
              f"{fn(a['v21_strict']):>16}")
    print(f"\n=== genuine turns (|net heading| >= {a['config']['FALSE_TURN_DEG']} deg) ===")
    for k in ("v2", "v21", "v21_strict"):
        s = a[k]
        rt = s["real_turns"]
        fs, mt = len(s["false_straight"]), s["masked_turns"]
        print(f"  {k:12} real_turns={rt:5d}  false_straight={fs:5d} "
              f"({_pct(fs, rt):5.1f}%)  masked={mt:5d} ({_pct(mt, rt):5.1f}%)"
              f"  UNLEARNABLE={_pct(fs + mt, rt):5.1f}%")
    for k in ("v2", "v21"):
        print(f"\n=== {k}: class balance ===")
        for kk, v in sorted(a[k]["cls"].items(), key=lambda x: -x[1]):
            print(f"    {kk:22s} {v:6d}  {_pct(v, n):5.1f}%")
        print(f"    -- reasons: " + json.dumps(a[k]["reason"]))
    print("\n=== v2.1: |net heading| band x label ===")
    print(f"  {'band(deg)':>12} {'left':>7} {'straight':>9} {'right':>7} "
          f"{'unknown':>8} {'masked':>7}")
    for b in BANDS:
        bk = f"{b[0]}"
        row = a["v21"]["band"].get(bk, {})
        lbl = f"{b[0]}-{'inf' if b[1] > 1e8 else int(b[1])}"
        print(f"  {lbl:>12} {row.get('left', 0):7d} {row.get('straight', 0):9d} "
              f"{row.get('right', 0):7d} {row.get('unknown', 0):8d} "
              f"{row.get('masked', 0):7d}")
    # The "real turn" referee threshold is a CHOICE, and the false-straight rate
    # moves with it. Show the sweep so the residual is read as a band, not tuned
    # to zero: the labeler's own net-heading rule fires at NET_DYAW_TURN_DEG, so
    # false-straights can only survive BELOW that (a gentle drift the labeler
    # deliberately still calls road-following).
    print(f"\n=== sensitivity: false-straight vs the referee threshold "
          f"(labeler net rule fires at {a['config']['NET_DYAW_TURN_DEG']:.0f} deg) ===")
    print(f"  {'referee>=':>10} {'real':>7} {'v2 false':>10} {'v2.1 false':>11} "
          f"{'v2.1 masked':>12}")
    for thr in (20.0, 30.0, 45.0, 60.0, 90.0):
        real = sum(1 for d in a["gt_deg"] if d >= thr)
        f2 = sum(1 for e in a["v2"]["false_straight"] if e["deg"] >= thr)
        f21 = sum(1 for e in a["v21"]["false_straight"] if e["deg"] >= thr)
        m21 = sum(1 for d in a["v21"]["masked_deg"] if d >= thr)
        print(f"  {thr:>9.0f}  {real:7d} {f2:10d} {f21:11d} {m21:12d}")

    print("\n=== v2 worst false-straights (real turn -> straight+valid) ===")
    for e in sorted(a["v2"]["false_straight"], key=lambda x: -x["deg"])[:12]:
        print(f"  {e['ep']} t={e['t']:4d}  net={e['deg']:6.1f}deg  "
              f"R={e['R_m']:7.0f}m  conc={e['conc']:.2f}  arc={e['arc_m']:.0f}m")
    print("\n=== v2.1 remaining false-straights ===")
    fs21 = sorted(a["v21"]["false_straight"], key=lambda x: -x["deg"])[:12]
    for e in fs21:
        print(f"  {e['ep']} t={e['t']:4d}  net={e['deg']:6.1f}deg  "
              f"R={e['R_m']:7.0f}m  conc={e['conc']:.2f}  arc={e['arc_m']:.0f}m")
    if not fs21:
        print("  (none)")


def main():
    ap = argparse.ArgumentParser("route_label_audit")
    ap.add_argument("--val", default="/workspace/pai_epcache/"
                                     "physicalai-val-f1b378f295ae")
    ap.add_argument("--stride", type=int,
                    default=int(os.environ.get("STRIDE", "20")))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    a = audit(args.val, args.stride, args.limit)
    report(a)
    if args.json:
        slim = json.loads(json.dumps(a))
        for k in ("v2", "v21", "v21_strict"):
            slim[k]["false_straight"] = sorted(
                slim[k]["false_straight"], key=lambda x: -x["deg"])[:50]
            slim[k].pop("masked_deg", None)
        slim.pop("gt_deg", None)
        with open(args.json, "w") as f:
            json.dump(slim, f, indent=1)
        print(f"\n[json] {args.json}")
    print("ROUTE_LABEL_AUDIT_DONE")


if __name__ == "__main__":
    main()
