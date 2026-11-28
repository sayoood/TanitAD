#!/usr/bin/env python3
"""Episode-cluster CI for PER-CLASS LATERAL RECALL, which the records give as a
bare fraction with no interval.

⛔ THE TURN GATE IS NOT USED. `D-TURNGATE` measured that the GROUND TRUTH itself
fails `|dyaw| > 0.15` on 3 of 9, so that gate is a broken criterion here. The
recall scored is the one the banked rungs already use: the four-family TACTICAL
block's own per-class lateral recall, recomputed from the decisions sidecar and
CONTROLLED against the record's printed value.

⛔ CONTROL THAT MUST READ A KNOWN VALUE: the recomputed recall must reproduce the
record's `four_families.tactical.lateral_decision` per-class recall exactly.
ASCII only.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\tanitad-ctg\taniteval")
from taniteval import ci as CI                                    # noqa: E402

P4 = r"C:\Users\Admin\refav1_margin\p4out"
ALT = r"C:\Users\Admin\wkfront\out"


def keys_of(dumpdir):
    fs = sorted(glob.glob(os.path.join(dumpdir, "decisions", "ep*.npz")))
    if not fs:
        return []
    return list(np.load(fs[0], allow_pickle=True).files)


def load_dec(dumpdir, key):
    out, eps = [], []
    for f in sorted(glob.glob(os.path.join(dumpdir, "decisions", "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            return None, None
        a = np.asarray(z[key]).reshape(-1)
        out.append(a)
        eps += [os.path.basename(f)] * len(a)
    return (np.concatenate(out) if out else None), eps


def main():
    tags = sys.argv[1:]
    if not tags:
        print("usage: recall_ci.py <tag> [...]")
        return 2
    d0 = os.path.join(P4, "dump_%s" % tags[0])
    if not os.path.isdir(d0):
        d0 = os.path.join(ALT, "dump_%s" % tags[0])
    print("decisions sidecar keys:", keys_of(d0))
    print()
    for tag in tags:
        d = os.path.join(P4, "dump_%s" % tag)
        if not os.path.isdir(d):
            d = os.path.join(ALT, "dump_%s" % tag)
        rec = os.path.join(P4, "rec_%s.json" % tag)
        if not os.path.exists(rec):
            rec = os.path.join(ALT, "rec_%s.json" % tag)
        banked = None
        if os.path.exists(rec):
            t = json.load(open(rec, encoding="utf-8"))["arms"]["cl"]["four_families"]["tactical"]
            banked = json.dumps(t.get("lateral_decision"))[:300]
        # The sidecar names them `lat_label` (truth) and `lat_pred_nav_true`
        # (the TRUE-nav prediction, i.e. the `cl` arm's decision -- NOT the
        # shuffled or zeroed nav variants, which are the leak controls).
        pred, eps = load_dec(d, "lat_pred_nav_true")
        true, _ = load_dec(d, "lat_label")
        if pred is None or true is None:
            print("%-12s per-window lateral decisions NOT in the sidecar "
                  "(keys above) -- recall CI not constructible here" % tag)
            if banked:
                print("             banked lateral_decision: %s" % banked)
            continue
        for cls, name in ((1, "turn_left"), (2, "turn_right"), (0, "lane_keep")):
            m = true == cls
            if not m.any():
                continue
            hit = (pred[m] == cls).astype(float)
            e = [x for x, k in zip(eps, m) if k]
            r = CI.episode_cluster_bootstrap(hit, e, n_boot=2000)
            print("%-12s %-11s n_true=%3d recall %.4f [%.4f, %.4f]"
                  % (tag, name, int(m.sum()), r["mean"], r["lo"], r["hi"]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
