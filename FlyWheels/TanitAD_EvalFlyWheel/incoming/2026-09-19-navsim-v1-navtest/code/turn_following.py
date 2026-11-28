#!/usr/bin/env python3
"""Does a plan turn where the road turns? Model vs logged-human NET heading, by NavSim command.

    python code/turn_following.py --seam raw/bridge_navtest/seam_A1_ego_cmd.npz --label A1
    python code/turn_following.py --seam raw/bridge_navtest_diag/seam_A4_blind_ego_cmd.npz \
        --label A4 --tokens raw/A4diag_tokens.json --key curve_tokens

NET heading = the direction of travel of the LAST path segment longer than 0.25 m, measured from
POSITIONS only (never the pose's heading channel: E2's spline-tangent headings can spin through
several turns on erratic knots, and a heading-channel read would inherit that). Relative to the
ego's t0 heading, which is 0 in the ego frame.

For every token whose logged human path turns more than ``--min-turn-deg`` net, report the median
model/human ratio and the fraction turning the same way. The command order is NavSim's one-hot
(left, straight, right, unknown) — verified on the logs by E2's ``tests/test_command_order.py``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
CMD = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT", 3: "UNKNOWN"}
MIN_SEG_M = 0.25


def net_heading(poses) -> float | None:
    q = np.vstack([[0.0, 0.0], np.asarray(poses, np.float64)[:, :2]])
    d = np.diff(q, axis=0)
    ok = np.where(np.hypot(d[:, 0], d[:, 1]) > MIN_SEG_M)[0]
    if len(ok) == 0:
        return None
    j = ok[-1]
    return math.atan2(d[j, 1], d[j, 0])


def turn_following(model_by_tok: dict, human_by_tok: dict, cmd_by_tok: dict,
                   min_turn_deg: float = 10.0) -> dict:
    out = {}
    for c in ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"):
        ratios, same = [], []
        for t, cc in cmd_by_tok.items():
            if cc != c or t not in model_by_tok:
                continue
            hh, mh = net_heading(human_by_tok[t]), net_heading(model_by_tok[t])
            if hh is None or mh is None or abs(math.degrees(hh)) <= min_turn_deg:
                continue
            ratios.append(mh / hh)
            same.append(1.0 if (mh > 0) == (hh > 0) else 0.0)
        if ratios:
            out[c] = {"n_turning": len(ratios),
                      "median_model_over_human": round(st.median(ratios), 4),
                      "same_sign_pct": round(100.0 * sum(same) / len(same), 2)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--key", default="tokens", help="which list in the tokens JSON")
    ap.add_argument("--min-turn-deg", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    doc = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    z = np.load(a.seam, allow_pickle=False)
    model = {str(t): z["poses"][i] for i, t in enumerate(z["token"].tolist())}
    keep = None
    if a.tokens:
        d = json.load(open(a.tokens, encoding="utf-8"))
        keep = set(d[a.key] if isinstance(d, dict) else d)
    toks = [t for t in model if keep is None or t in keep]
    res = turn_following({t: model[t] for t in toks},
                         {t: doc[t]["human_future_poses"] for t in toks},
                         {t: CMD[int(np.argmax(doc[t]["ego_statuses"][-1]["driving_command"]))]
                          for t in toks}, a.min_turn_deg)
    rec = {"label": a.label, "seam": a.seam, "n_tokens": len(toks),
           "token_set": (f"{a.tokens}:{a.key}" if a.tokens else "all tokens in the seam"),
           "min_turn_deg": a.min_turn_deg, "evidence": "MEASURED", "by_command": res}
    out = Path(a.out or f"raw/turn_following_{a.label}.json")
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
