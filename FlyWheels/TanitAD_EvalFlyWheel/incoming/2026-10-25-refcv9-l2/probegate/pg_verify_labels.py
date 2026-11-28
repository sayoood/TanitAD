"""pg_verify_labels.py -- INDEPENDENT 0-GPU verification that the forced windows are exactly what PREREG_PROBEGATE.md section 3 says.

Re-implements the policy as a per-window PYTHON LOOP (no shared code with pg_policy.py: no numpy argsort/argmax helpers) from the three raw inputs --
the head's raw logits (L2 oracle trace), the saved probe scores, and the label npz files -- and requires window-by-window equality.  Writes LABELS_VERIFY.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L2 = HERE.parent
LON6 = [0, 1, 3, 4, 5, 7]            # v7 ids in LON6 order
DEC, CRU = [3, 4, 5], [0, 1]          # v7 ids of the decel group / cruise group (LON6 idx 2,3,4 / 0,1)


def first_max(vals, ids):
    """v7 id (from ids) with the highest logit; ties -> the earlier id in the list order."""
    best = ids[0]
    for i in ids[1:]:
        if vals[i] > vals[best]:
            best = i
    return best


def group(vals):
    top = first_max(vals, LON6)
    return "D" if top in DEC else ("C" if top in CRU else "A")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="directory holding probe_scores_s0.npz + the label npz files (default: this directory); LABELS_VERIFY.json is written there")
    a = ap.parse_args()
    D = Path(a.dir) if a.dir else HERE
    z = np.load(D / "probe_scores_s0.npz")
    tr = np.load(L2 / "bank" / "trace_oracle_s0.npz")
    lp, lr = np.load(D / "labels_probegate_s0.npz"), np.load(D / "labels_randgate_s0.npz")
    n = len(z["score"])
    raw = [[float(x) for x in tr["raw_lon_logits"][i]] for i in range(n)]
    score = [float(x) for x in z["score"]]
    assert np.array_equal(lp["clip_index"], z["clip_index"]) and np.array_equal(lp["ws"], z["ws"]) and np.array_equal(lr["ws"], z["ws"]) and np.array_equal(lr["clip_index"], z["clip_index"])
    assert np.array_equal(tr["e_i"], z["clip_index"]) and np.array_equal(tr["ws"], z["ws"])
    grp = [group(raw[i]) for i in range(n)]
    k = sum(1 for g in grp if g == "D")
    ranked = sorted(range(n), key=lambda i: (-score[i], i))          # descending score, ties by window index
    pdec = set(ranked[:k])
    tau = score[ranked[k - 1]]
    exp_p = [-100] * n
    n_up = n_down = 0
    for i in range(n):
        if i in pdec and grp[i] != "D":
            exp_p[i] = first_max(raw[i], DEC)
            n_up += 1
        elif i not in pdec and grp[i] == "D":
            exp_p[i] = first_max(raw[i], CRU)
            n_down += 1
    got_p = [int(x) for x in lp["lon_v7"]]
    bad_p = [i for i in range(n) if exp_p[i] != got_p[i]]
    # RANDGATE: counts per direction, eligibility and forced class (the draw itself is the prereg's numpy-seed-0 contract, re-drawn here)
    got_r = [int(x) for x in lr["lon_v7"]]
    r_up = [i for i in range(n) if got_r[i] != -100 and grp[i] != "D"]
    r_dn = [i for i in range(n) if got_r[i] != -100 and grp[i] == "D"]
    bad_r = [i for i in r_up if got_r[i] != first_max(raw[i], DEC)] + [i for i in r_dn if got_r[i] != first_max(raw[i], CRU)]
    rng = np.random.default_rng(0)
    ue = [i for i in range(n) if grp[i] != "D"]
    de = [i for i in range(n) if grp[i] == "D"]
    redraw_up = sorted(int(x) for x in rng.choice(np.asarray(ue), n_up, replace=False))
    redraw_dn = sorted(int(x) for x in rng.choice(np.asarray(de), n_down, replace=False))
    out = {"n": n, "head_decel_k": k, "tau": tau, "probegate": {"n_up": n_up, "n_down": n_down, "n_forced": n_up + n_down, "mismatching_windows": len(bad_p),
                                                              "forced_v7_counts": {str(c): sum(1 for x in exp_p if x == c) for c in sorted(set(exp_p) - {-100})}},
           "probegate_labels_in_file": {"n_forced": sum(1 for x in got_p if x != -100)},
           "randgate": {"n_up": len(r_up), "n_down": len(r_dn), "forced_class_violations": len(bad_r), "redraw_equal": bool(redraw_up == r_up and redraw_dn == r_dn),
                        "forced_v7_counts": {str(c): sum(1 for x in got_r if x == c) for c in sorted(set(got_r) - {-100})}},
           "all_labels_are_lon6_or_none": bool(all(x in LON6 or x == -100 for x in got_p + got_r))}
    out["ALL_PASS"] = bool(len(bad_p) == 0 and out["randgate"]["n_up"] == n_up and out["randgate"]["n_down"] == n_down and len(bad_r) == 0
                           and out["randgate"]["redraw_equal"] and out["all_labels_are_lon6_or_none"] and n_up == n_down and n_up > 0)
    (D / "LABELS_VERIFY.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("ZZPGVERIFYDONEZZ", out["ALL_PASS"])
    return 0 if out["ALL_PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
