#!/usr/bin/env python
"""v8f_t1_eval_rule_qcal.py -- card PREREG_V8F_PROOF_ADD31 (QCAL): the distance rule of card ADD26 (note N1) fed by the vision head, with

  --head-p-min P     the head calls a lead present at probability >= P
  --gap-map JSON     the rule takes g_used = min(g_head, Q(g_head)), Q = the piecewise-linear map of the file (qcal_map.py)

Everything else is v8f_t1_eval_rule_mix.py (imported, not edited), which imports the unedited wrapper v8f_t1_eval_rule.py; the pinned eval tool stays pinned.
With --head-p-min 0.5, the identity map and the mix `head head head` the run is the all-head mix of card ADD29 (control q1).

  python v8f_t1_eval_rule_qcal.py --head-p-min P --gap-map JSON --tree TREE --mix-presence head --mix-gap head --mix-closing head
         [--rule-log NPZ] [--floor {card,cap-only}] run <the argv of the pinned tool's `run`>

Tier T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). STRATEGIC: PAUSED (PI item 25).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import qcal_map as QM  # noqa: E402
import v8f_t1_eval_rule_mix as X  # noqa: E402

W = X.W
MAP = {"xs": None, "ys": None}
_ORIG_HEAD_VALUES = W.head_values


def head_values_qcal(model, feats, frame_ok):
    """The wrapper's head read with the gap passed through the map (never above the head read; NaN stays NaN)."""
    p, g, c, ok = _ORIG_HEAD_VALUES(model, feats, frame_ok)
    return p, QM.apply_map(g, MAP["xs"], MAP["ys"]).astype(g.dtype), c, ok


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--head-p-min", type=float, required=True)
    ap.add_argument("--gap-map", required=True)
    mine, rest = ap.parse_known_args(argv)
    if not (0.0 < mine.head_p_min < 1.0):
        raise SystemExit("REFUSED: --head-p-min must lie strictly between 0 and 1")
    if "--mix-presence" not in rest or "--mix-gap" not in rest:
        raise SystemExit("REFUSED: name the mix (--mix-presence, --mix-gap, --mix-closing)")
    xs, ys = QM.load_map(mine.gap_map)
    u = QM.unit_check(xs, ys)
    print("q5 unit check of the map %s: %s" % (mine.gap_map, u), flush=True)
    if not u["pass"]:
        raise SystemExit("REFUSED: the map fails its unit check")
    MAP.update(xs=xs, ys=ys)
    W.HEAD_P_MIN = float(mine.head_p_min)          # read by the mix planner at call time
    W.head_values = head_values_qcal               # read by the mix planner at call time
    print("qcal constants: head_p_min=%.4f knots=%d" % (W.HEAD_P_MIN, len(xs)), flush=True)
    return X.main(rest)


if __name__ == "__main__":
    sys.exit(main())
