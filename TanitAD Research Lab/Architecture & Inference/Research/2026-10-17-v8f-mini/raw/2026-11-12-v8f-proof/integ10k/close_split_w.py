# close_split_w.py : window-level part of the split of section 5 of card PREREG_V8F_PROOF_ADD48 (outcome CUTS-UNSAFE with READER-NO-GAIN). Descriptive; names nothing.
# Question: the windows that are unsafe under the reader-fed rule with the label closing speed (RQ50c) and safe under the label-fed rule with it (RLc): what did the
# reader do on the in-range label ticks of those windows? Classes per window (label in range = lead present with a gap of at most 60 m at the tick):
#   missed   : the reader is present at none of the in-range ticks
#   late     : absent at the first in-range tick, present at a later one
#   gap_over : present at the first in-range tick and the fed gap is more than 5 m above the label gap at an in-range tick where both are present
#   on_time  : present at the first in-range tick and never more than 5 m above
#   no_tick  : the window has no in-range label tick in the rule log (the rule could not act on a label there either)
# The same classes are counted over all scored lead windows (the base rate) and over the reverse set (unsafe under RLc, safe under RQ50c).
# Tier T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.
# Run from the headway folder (numpy; roots from PS_ROOT / PS_OUT): python ../close_split_w.py
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import headway_read_r2 as H  # noqa: E402

ROOT = os.environ.get("PS_ROOT", "F:/TanitAD-scratch/v8proof/integ10k/pull")
OUT = os.environ.get("PS_OUT", "F:/TanitAD-scratch/v8proof/integ10k/read_close/CLOSE_SPLIT_W.json")
ARMS, SEEDS, G_MAX, OVER_M = ("I10R0", "I10RLb", "I10RLc", "I10RQ50", "I10RQ50c"), (0, 1), 60.0, 5.0
CLASSES = ("missed", "late", "gap_over", "on_time", "no_tick")


def rlog(arm, k):
    return np.load(os.path.join(ROOT, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d.npz" % (arm, k)))


Wd, missing, where = H.load_world([ROOT], [(a, s) for a in ARMS for s in SEEDS], 141)
assert not missing, missing
J = H.load_join()
res = {"card": "PREREG_V8F_PROOF_ADD48", "what": "window-level split of the reader's behaviour on the windows unsafe under RQ50c and safe under RLc; names nothing", "per_seed": {}}
for k in SEEDS:
    ref = Wd[("I10R0", k)]
    Lw = H.label_windows(J, ref["clip_index"], ref["ws"])
    U = {a: np.asarray(H.window_cells(Wd[(a, k)]["g"], Wd[(a, k)]["cl"], Lw)["U"], np.float64) for a in ARMS}
    a, b = rlog("I10RLc", k), rlog("I10RQ50c", k)
    assert all(np.array_equal(a[c], b[c]) for c in ("ci", "t0", "tick"))
    lp, rp = np.asarray(a["present"]).astype(bool), np.asarray(b["present"]).astype(bool)
    gl, gf = np.asarray(a["g_src"], np.float64), np.asarray(b["g_src"], np.float64)
    rng = lp & np.isfinite(gl) & (gl <= G_MAX)
    key = np.asarray(a["ci"], np.int64) * 1000000 + np.asarray(a["t0"], np.int64)
    tick = np.asarray(a["tick"], np.int64)
    cls_of = {}
    for w in np.unique(key):
        m = key == w
        o = np.argsort(tick[m], kind="stable")
        r_in, p_in, over = rng[m][o], rp[m][o], ((gf - gl)[m][o] > OVER_M)
        idx = np.flatnonzero(r_in)
        if len(idx) == 0:
            cls_of[int(w)] = "no_tick"
        elif not p_in[idx].any():
            cls_of[int(w)] = "missed"
        elif not p_in[idx[0]]:
            cls_of[int(w)] = "late"
        elif (over[idx] & p_in[idx]).any():
            cls_of[int(w)] = "gap_over"
        else:
            cls_of[int(w)] = "on_time"
    wkey = np.asarray(ref["clip_index"], np.int64) * 1000000 + np.asarray(ref["ws"], np.int64)
    known = np.array([int(x) in cls_of for x in wkey])
    scored = np.isfinite(U["I10RLc"]) & np.isfinite(U["I10RQ50c"])
    assert known[scored].all(), "scored windows without rule-log ticks: %d" % int((scored & ~known).sum())
    wcls = np.array([cls_of.get(int(x), "no_tick") for x in wkey])

    def count(mask):
        return {c: int((mask & (wcls == c)).sum()) for c in CLASSES} | {"n": int(mask.sum())}

    S = {"n_scored": int(scored.sum()), "U_levels": {a_: float(np.nanmean(U[a_])) for a_ in ARMS},
         "base_all_scored": count(scored),
         "unsafe_RQ50c_safe_RLc": count(scored & (U["I10RQ50c"] == 1) & (U["I10RLc"] == 0)),
         "unsafe_RLc_safe_RQ50c": count(scored & (U["I10RLc"] == 1) & (U["I10RQ50c"] == 0)),
         "unsafe_both": count(scored & (U["I10RLc"] == 1) & (U["I10RQ50c"] == 1)),
         "unsafe_RQ50_safe_RQ50c": count(scored & (U["I10RQ50"] == 1) & (U["I10RQ50c"] == 0)),
         "unsafe_RLb_safe_RLc": count(scored & (U["I10RLb"] == 1) & (U["I10RLc"] == 0))}
    res["per_seed"][str(k)] = S
    print("s%d scored %d | U levels %s" % (k, S["n_scored"], json.dumps({x: round(v, 4) for x, v in S["U_levels"].items()})))
    for name in ("base_all_scored", "unsafe_RQ50c_safe_RLc", "unsafe_RLc_safe_RQ50c", "unsafe_both", "unsafe_RQ50_safe_RQ50c", "unsafe_RLb_safe_RLc"):
        print("s%d %-24s %s" % (k, name, json.dumps(S[name])))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print("ZZ CLOSE-SPLIT-W DONE")
