"""Cross-validate VLM route labels against the kinematic labeler.

The kinematic labeler and the VLM are the only two route-label sources we have,
and neither is ground truth: the kinematics restate the ego track (they cannot
see the junction), the VLM reads the scene (it cannot measure the heading). This
script measures where they agree and — the part that actually teaches us
something — dissects where they do not.

ONLY PASS A COUNTS. Pass A saw the future FRAMES and an ego-motion block built
from the PAST only; it never saw our computed future heading, so its ROUTE is
independent evidence. Pass B was handed the numeric future track precisely so it
could interpret the scene richly, which makes its ROUTE an echo of ours — it is
excluded here by construction, and the script FAILS LOUDLY if a Pass B record
tries to enter the statistics.

Outputs: agreement vs v2 and vs v2.1 (both restricted to windows the kinematic
side calls valid, plus a coverage-honest view over ALL windows), the 3x3+unknown
confusion, and the disagreement dossier with the VLM's own cited evidence beside
the kinematic numbers so a human can adjudicate.

Usage:
  PYTHONPATH=<repo>/stack python scripts/vlm_kin_crossval.py \
      --vlm /workspace/vlm_passA \
      --val /workspace/pai_epcache/physicalai-val-f1b378f295ae \
      --json /workspace/vlm_crossval.json
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

NAMES = {R.ROUTE_LEFT: "left", R.ROUTE_STRAIGHT: "straight",
         R.ROUTE_RIGHT: "right", R.ROUTE_UNKNOWN: "unknown"}
VLM_TO_KIN = {"left": R.ROUTE_LEFT, "straight": R.ROUTE_STRAIGHT,
              "right": R.ROUTE_RIGHT, "unknown": R.ROUTE_UNKNOWN,
              # a u-turn is a route decision the 3-class CE cannot express;
              # counted separately rather than folded into left/right.
              "u_turn": R.ROUTE_UNKNOWN}
ORDER = [R.ROUTE_LEFT, R.ROUTE_STRAIGHT, R.ROUTE_RIGHT, R.ROUTE_UNKNOWN]


def collect(vlm_dir: str, val_dir: str) -> list[dict]:
    poses_cache: dict[str, torch.Tensor] = {}
    rows = []
    for f in sorted(glob.glob(os.path.join(vlm_dir, "*.json"))):
        rec = json.load(open(f))
        a = rec.get("pass_A")
        if not a:
            continue
        if a.get("pass") != "A" or a.get("future_track_given") is not False:
            raise SystemExit(
                f"REFUSING to score {os.path.basename(f)}: pass="
                f"{a.get('pass')} future_track_given={a.get('future_track_given')}"
                " — only frames-only Pass A records may enter the agreement "
                "statistics (see module docstring)")
        ep, t = rec["episode"], int(rec["t"])
        if ep not in poses_cache:
            p = os.path.join(val_dir, ep + ".pt")
            d = torch.load(p, map_location="cpu", weights_only=False)
            poses_cache[ep] = d["poses"].float()
        poses = poses_cache[ep]
        r2 = R.route_from_future(poses, t)
        r21 = R.route_from_future_v21(poses, t)
        vr = VLM_TO_KIN.get(a.get("ROUTE", "unknown"), R.ROUTE_UNKNOWN)
        rows.append({
            "ep": ep, "t": t,
            "vlm": vr, "vlm_raw": a.get("ROUTE"),
            "vlm_conf": a.get("route_confidence"),
            "vlm_event_s": a.get("route_event_time_s"),
            "vlm_geom": a.get("road_geometry"),
            "vlm_junction": a.get("sees_junction_ahead"),
            "evidence": (a.get("route_evidence") or "")[:220],
            "v2": r2["route"], "v2_valid": bool(r2["valid"]),
            # v2 emits ROUTE_STRAIGHT when it cannot judge; the label FILE
            # carries that class, so score what a consumer would actually read.
            "v2_emitted": r2["route"],
            "v21": r21["route"], "v21_valid": bool(r21["valid"]),
            "v21_reason": r21["reason"],
            "net_deg": round(math.degrees(r21["net_dyaw"]), 1),
            "peak_R_m": round(1.0 / max(r21["peak_kappa"], 1e-9)),
            "arc_m": round(r21["arc_m"], 1), "h": r21["h_steps"],
            "n_future_frames": rec.get("n_future_frames"),
        })
    return rows


def _agree(rows, key, valid_key=None):
    sub = [r for r in rows if (valid_key is None or r[valid_key])]
    if not sub:
        return 0.0, 0
    n = sum(1 for r in sub if r[key] == r["vlm"])
    return 100.0 * n / len(sub), len(sub)


def _confusion(rows, key, valid_key=None):
    m = {a: {b: 0 for b in ORDER} for a in ORDER}
    for r in rows:
        if valid_key is not None and not r[valid_key]:
            continue
        m[r[key]][r["vlm"]] += 1
    return m


def _print_conf(m, title, kin_label):
    print(f"\n=== {title} ===")
    print(f"  {'kin \\\\ vlm':>12}" + "".join(f"{NAMES[b]:>10}" for b in ORDER)
          + f"{'row n':>8}")
    for a in ORDER:
        row = m[a]
        tot = sum(row.values())
        print(f"  {kin_label + ':' + NAMES[a]:>12}"
              + "".join(f"{row[b]:>10}" for b in ORDER) + f"{tot:>8}")


def main():
    ap = argparse.ArgumentParser("vlm_kin_crossval")
    ap.add_argument("--vlm", default="/workspace/vlm_passA")
    ap.add_argument("--val", default="/workspace/pai_epcache/"
                                     "physicalai-val-f1b378f295ae")
    ap.add_argument("--json", default=None)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    rows = collect(args.vlm, args.val)
    if not rows:
        raise SystemExit(f"no pass-A records under {args.vlm}")
    n = len(rows)
    print(f"windows with a Pass-A VLM label: {n}  "
          f"(episodes {len({r['ep'] for r in rows})})")
    print("Pass B is EXCLUDED by construction — it was shown the numeric future "
          "track and would echo the kinematics.\n")

    print("=== VLM class balance (independent evidence) ===")
    bal = {}
    for r in rows:
        bal[r["vlm_raw"]] = bal.get(r["vlm_raw"], 0) + 1
    for k, v in sorted(bal.items(), key=lambda x: -x[1]):
        print(f"  {k:10s} {v:5d}  {100 * v / n:5.1f}%")

    print("\n=== agreement ===")
    for label, key, vk in (
            ("v2 as EMITTED (what a label consumer reads)", "v2_emitted", None),
            ("v2 where v2 says valid", "v2", "v2_valid"),
            ("v2.1 as emitted (unknown counts as its own class)", "v21", None),
            ("v2.1 where v2.1 says valid", "v21", "v21_valid")):
        acc, m = _agree(rows, key, vk)
        print(f"  {label:52s} {acc:5.1f}%  (n={m})")

    both = [r for r in rows if r["v21_valid"] and r["vlm"] != R.ROUTE_UNKNOWN]
    acc = 100.0 * sum(1 for r in both if r["v21"] == r["vlm"]) / max(len(both), 1)
    print(f"  {'v2.1 vs VLM, both committed (no unknowns)':52s} {acc:5.1f}%"
          f"  (n={len(both)})")

    _print_conf(_confusion(rows, "v2_emitted"), "v2 EMITTED x VLM", "v2")
    _print_conf(_confusion(rows, "v21"), "v2.1 x VLM", "v2.1")

    # DETECTION vs DIRECTION. These are different competences and conflating
    # them hides which source to trust where: "a route event happens here" is a
    # scene-understanding question the VLM can answer from what it sees, while
    # "the heading went left" is a measurement the ego track owns. Score them
    # apart before concluding anything about who is right.
    print("\n=== detection vs direction (the two competences, scored apart) ===")
    com = [r for r in rows if r["v21_valid"] and r["vlm"] != R.ROUTE_UNKNOWN]
    turn = lambda x: x in (R.ROUTE_LEFT, R.ROUTE_RIGHT)          # noqa: E731
    det = [r for r in com if turn(r["v21"]) == turn(r["vlm"])]
    print(f"  IS-A-TURN agreement (turn vs straight):        "
          f"{100 * len(det) / max(len(com), 1):5.1f}%  (n={len(com)})")
    bt = [r for r in com if turn(r["v21"]) and turn(r["vlm"])]
    same = [r for r in bt if r["v21"] == r["vlm"]]
    print(f"  DIRECTION agreement, both call it a turn:      "
          f"{100 * len(same) / max(len(bt), 1):5.1f}%  (n={len(bt)})")
    print(f"    -> of the {len(bt)} shared turns, {len(bt) - len(same)} are "
          f"OPPOSITE-direction calls. 50 % here is chance: a direction score at "
          f"or below chance means the VLM is not reading direction at all.")
    tight = [r for r in bt if r["peak_R_m"] <= 60]
    ts = [r for r in tight if r["v21"] == r["vlm"]]
    print(f"  DIRECTION agreement on TIGHT turns (R<=60 m):  "
          f"{100 * len(ts) / max(len(tight), 1):5.1f}%  (n={len(tight)})")
    lb = sum(1 for r in rows if r["vlm"] == R.ROUTE_LEFT)
    rb = sum(1 for r in rows if r["vlm"] == R.ROUTE_RIGHT)
    kl = sum(1 for r in rows if r["v21"] == R.ROUTE_LEFT)
    kr = sum(1 for r in rows if r["v21"] == R.ROUTE_RIGHT)
    print(f"  left:right ratio — VLM {lb}:{rb} "
          f"({lb / max(rb, 1):.2f}) vs kinematic {kl}:{kr} "
          f"({kl / max(kr, 1):.2f})  <- a skew here is a VLM prior, not a road")

    dis = [r for r in rows if r["v21_valid"] and r["vlm"] != R.ROUTE_UNKNOWN
           and r["v21"] != r["vlm"]]
    print(f"\n=== DISAGREEMENTS (v2.1 valid vs VLM committed): {len(dis)} "
          f"of {len(both)} ===")
    for r in sorted(dis, key=lambda x: -abs(x["net_deg"]))[:args.show]:
        print(f"  {r['ep']} t={r['t']:3d}  kin={NAMES[r['v21']]:8s}"
              f"({r['v21_reason']:15s}) net={r['net_deg']:+7.1f}° "
              f"R={r['peak_R_m']:>8}m arc={r['arc_m']:6.0f}m  "
              f"VLM={r['vlm_raw']:8s} conf={r['vlm_conf']} "
              f"geom={r['vlm_geom']}")
        print(f"      evidence: {r['evidence']}")

    # Where the VLM rescues what v2 masked: v2 emitted `straight` (unlabeled),
    # the VLM saw a turn, and v2.1's adaptive read confirms it.
    rescue = [r for r in rows if not r["v2_valid"]
              and r["vlm"] in (R.ROUTE_LEFT, R.ROUTE_RIGHT)]
    conf = [r for r in rescue if r["v21"] == r["vlm"]]
    print(f"\n=== windows v2 could not judge, VLM called a TURN: {len(rescue)} "
          f"— v2.1 independently agrees on {len(conf)} "
          f"({100 * len(conf) / max(len(rescue), 1):.0f}%) ===")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"n": n, "rows": rows}, f, indent=1)
        print(f"\n[json] {args.json}")
    print("VLM_KIN_CROSSVAL_DONE")


if __name__ == "__main__":
    main()
