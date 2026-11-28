#!/usr/bin/env python
"""v8f_t1_eval_rule_margin.py -- the distance rule of card ADD26 (note N1) fed by the vision head, with two fixed safety constants on the head read:

  --head-p-min P     the head calls a lead present at probability >= P (the wrapper's value is 0.5)
  --gap-margin-m M   the rule takes the head gap minus M metres (not below 0); M = 0 leaves the gap as read

Everything else is v8f_t1_eval_rule_mix.py (imported, not edited), which itself imports the unedited wrapper v8f_t1_eval_rule.py; the pinned eval tool stays pinned.
With --head-p-min 0.5 --gap-margin-m 0 and the mix `head head head` the run is the all-head mix of card ADD29 (its control m2), bit for bit.

  python v8f_t1_eval_rule_margin.py --tree TREE --head-p-min P --gap-margin-m M --mix-presence head --mix-gap head --mix-closing {head,zero}
         [--rule-log NPZ] [--floor {card,cap-only}] run <the argv of the pinned tool's `run`>

The margin is applied to the HEAD gap only (a label gap is never changed), before the mix, so the logged g_src of a head-gap run is the gap the rule used.
Tier T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). STRATEGIC: PAUSED (PI item 25).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import v8f_t1_eval_rule_mix as X  # noqa: E402

W = X.W
CONST = {"head_p_min": 0.5, "gap_margin_m": 0.0}
_ORIG_HEAD_VALUES = W.head_values


def head_values_margin(model, feats, frame_ok):
    """The wrapper's head read with the gap lowered by the margin (NaN stays NaN; never below 0)."""
    p, g, c, ok = _ORIG_HEAD_VALUES(model, feats, frame_ok)
    m = float(CONST["gap_margin_m"])
    if m != 0.0:
        g = np.where(np.isfinite(g), np.maximum(g - m, 0.0), g).astype(g.dtype)
    return p, g, c, ok


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--head-p-min", type=float, required=True)
    ap.add_argument("--gap-margin-m", type=float, required=True)
    mine, rest = ap.parse_known_args(argv)
    if not (0.0 < mine.head_p_min < 1.0):
        raise SystemExit("REFUSED: --head-p-min must lie strictly between 0 and 1")
    if not (0.0 <= mine.gap_margin_m <= 30.0):
        raise SystemExit("REFUSED: --gap-margin-m must lie in [0, 30]")
    if "--mix-presence" not in rest or "--mix-gap" not in rest:
        raise SystemExit("REFUSED: name the mix (--mix-presence, --mix-gap, --mix-closing)")
    CONST.update(head_p_min=mine.head_p_min, gap_margin_m=mine.gap_margin_m)
    W.HEAD_P_MIN = float(mine.head_p_min)          # read by the mix planner at call time
    W.head_values = head_values_margin             # read by the mix planner at call time
    print("margin constants: head_p_min=%.4f gap_margin_m=%.3f" % (W.HEAD_P_MIN, CONST["gap_margin_m"]), flush=True)
    return X.main(rest)


if __name__ == "__main__":
    sys.exit(main())
