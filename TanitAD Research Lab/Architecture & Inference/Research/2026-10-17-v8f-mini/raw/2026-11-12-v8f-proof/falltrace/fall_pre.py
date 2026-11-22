# fall_pre.py -- lead tool, pod side, CPU only. Second count on banked T1-kin rollouts (kinematic rollout on the model's own actions): the nine looks of the rule per window
# (three of the pre-roll, three ticks, three later looks) of a camera-fed arm against the label gap form RLc in the same window. Definitions: FALLPRE_DEFS.md (fixed before this file).
# Usage: python fall_pre.py --pull LINKS --grid-dir DIR --join JOIN --code CODE --out OUTDIR [--arms VD5c,F2,...] [--no-expect]     |     python fall_pre.py --selftest
# Last line: ZZ FALLPRE DONE <n arms> <n excess arm-windows>     or     ZZ FALLPRE CONTROL FAIL <why>
import argparse
import io
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import export_fall as X  # noqa: E402  (loaders, label reads and run names of the first count)

KEYS = ("ci", "t0", "j", "present", "changed", "sat_exec", "g_src", "c_src")
LOGS = (("pre", "rule_log_pre_%s.npz"), ("tick", "rule_log_%s.npz"), ("mid", "rule_log_mid_%s.npz"))
ROWS_PER_LOG = 3
PRE_OFFSETS = range(0, 61)
EXPECT_EXCESS = {"VD5c": 34, "F2": 24, "F3": 26, "F4": 38, "E5c": 29}
EXPECT_NEITHER = {"VD5c": 297, "F2": 307, "F3": 305, "F4": 293, "E5c": 302}
CLASSES = ("NO-CUT-BY-X", "MORE-CUT-IN-PRE-ROLL", "MORE-CUT-IN-WINDOW", "NOT-MORE-LOOKS")
TAGS = ("FALSE-PRESENCE", "GAP-LOW", "SLOPE", "OTHER")
SHARE_MIN = 0.5
GAP_LOW_M = 5.0


def classify(a_pre, a_win, r_pre, r_win):
    """DEFS section 3, first match"""
    if a_pre + a_win == 0:
        return CLASSES[0]
    if a_pre > r_pre and (a_pre - r_pre) >= (a_win - r_win):
        return CLASSES[1]
    if a_win > r_win:
        return CLASSES[2]
    return CLASSES[3]


def tag_of(n_cut, n_nolab, n_lab, n_low, n_slope):
    """DEFS section 4, first match; for a window with at least one cut look of X"""
    if n_cut <= 0:
        return None
    if n_nolab >= SHARE_MIN * n_cut:
        return TAGS[0]
    if n_lab > 0 and n_low >= SHARE_MIN * n_lab:
        return TAGS[1]
    if n_slope >= SHARE_MIN * n_cut:
        return TAGS[2]
    return TAGS[3]


def read_log(pull, key, nm):
    p = os.path.join(pull, key, "ticks", nm % key)
    if not os.path.isfile(p):
        raise SystemExit("MISSING-RULE-LOG %s" % p)
    with np.load(p, allow_pickle=False) as z:
        for k in KEYS:
            if k not in z.files:
                raise SystemExit("MISSING-RULE-LOG-FIELD %s in %s" % (k, nm % key))
        return {k: np.asarray(z[k]) for k in KEYS}


def match(rows, ci, ws, offset=0):
    """window index of every row (-1: none); a row belongs to the window with its clip index and window start = t0 + offset"""
    wk = np.asarray(ci, np.int64) * (1 << 32) + np.asarray(ws, np.int64)
    if len(set(wk.tolist())) != len(wk):
        raise SystemExit("WINDOW-KEYS-NOT-DISTINCT")
    order = np.argsort(wk, kind="stable")
    sk = wk[order]
    rk = rows["ci"].astype(np.int64) * (1 << 32) + rows["t0"].astype(np.int64) + int(offset)
    pos = np.minimum(np.searchsorted(sk, rk), len(sk) - 1)
    hit = sk[pos] == rk
    return np.where(hit, order[pos], -1)


def pre_offset(rows, ci, ws):
    """the one constant offset between the t0 of the pre-roll rows and the window start that gives every window three rows; stops when there is none or more than one"""
    good = []
    n = len(ci)
    for d in PRE_OFFSETS:
        w = match(rows, ci, ws, d)
        if (w >= 0).all() and (np.bincount(w, minlength=n) == ROWS_PER_LOG).all():
            good.append(d)
    if len(good) != 1:
        raise SystemExit("PRE-ROLL-OFFSET-NOT-SINGLE %s" % good)
    return good[0]


def form_numbers(pull, run, seed, ci, ws, J, R, lead):
    """per window: cut looks and summed sat_exec of the pre-roll and of the window; the per-look arrays of the cut looks (for the tags). Stops when a lead window lacks a row."""
    key = R.key_of(run, seed)
    n = len(ci)
    out = {"cut_pre": np.zeros(n, int), "cut_win": np.zeros(n, int), "sum_pre": np.zeros(n, float), "sum_win": np.zeros(n, float),
           "n_nolab": np.zeros(n, int), "n_lab": np.zeros(n, int), "n_low": np.zeros(n, int), "n_slope": np.zeros(n, int)}
    off = None
    for part, nm in LOGS:
        rows = read_log(pull, key, nm)
        if part == "pre":
            off = pre_offset(rows, ci, ws)
            w = match(rows, ci, ws, off)
        else:
            w = match(rows, ci, ws, 0)
        cnt = np.bincount(w[w >= 0], minlength=n)
        if not (cnt[lead] == ROWS_PER_LOG).all():
            raise SystemExit("LEAD-WINDOW-WITHOUT-THREE-ROWS %s %s" % (key, part))
        A = X.row_arrays(rows, J)
        present = rows["present"].astype(bool)
        se = rows["sat_exec"].astype(np.float64)
        cut = present & (se > 0)
        ok = w >= 0
        tgt = "pre" if part == "pre" else "win"
        np.add.at(out["cut_" + tgt], w[ok & cut], 1)
        np.add.at(out["sum_" + tgt], w[ok & present], se[ok & present])
        with np.errstate(invalid="ignore"):
            low = A["lab_ok"] & (A["diff"] < -GAP_LOW_M)
        np.add.at(out["n_nolab"], w[ok & cut & ~A["lab_ok"]], 1)
        np.add.at(out["n_lab"], w[ok & cut & A["lab_ok"]], 1)
        np.add.at(out["n_low"], w[ok & cut & low], 1)
        np.add.at(out["n_slope"], w[ok & cut & A["closing_fed"] & A["lab_not_closing"]], 1)
    out["pre_offset"] = off
    return out


def count_arm(dx, dref, nx, nref):
    """-> rows of one arm and seed against one reference: list of dicts for the lead windows"""
    rows = []
    for i in np.flatnonzero(dx["lead"]).tolist():
        fx, fr = dx["F"][i] == 1.0, dref["F"][i] == 1.0
        grp = "excess" if (fx and not fr) else ("neither" if (not fx and not fr) else ("both" if (fx and fr) else "ref_only"))
        a_pre, a_win, r_pre, r_win = int(nx["cut_pre"][i]), int(nx["cut_win"][i]), int(nref["cut_pre"][i]), int(nref["cut_win"][i])
        cls = classify(a_pre, a_win, r_pre, r_win)
        n_cut = a_pre + a_win
        tag = tag_of(n_cut, int(nx["n_nolab"][i]), int(nx["n_lab"][i]), int(nx["n_low"][i]), int(nx["n_slope"][i]))
        rows.append({"i": i, "group": grp, "class": cls, "tag": tag, "a_pre": a_pre, "a_win": a_win, "r_pre": r_pre, "r_win": r_win, "n_cut": n_cut, "n_nolab": int(nx["n_nolab"][i]),
                     "n_lab": int(nx["n_lab"][i]), "n_low": int(nx["n_low"][i]), "n_slope": int(nx["n_slope"][i]),
                     "dsum_pre": float(nx["sum_pre"][i] - nref["sum_pre"][i]), "dsum_win": float(nx["sum_win"][i] - nref["sum_win"][i])})
    return rows


def tally(rows, group):
    S = [r for r in rows if r["group"] == group]
    c = {k: sum(1 for r in S if r["class"] == k) for k in CLASSES}
    T = [r for r in S if r["class"] != CLASSES[0]]
    t = {k: sum(1 for r in T if r["tag"] == k) for k in TAGS}
    n_cut = sum(r["n_cut"] for r in S)
    return {"n": len(S), "classes": c, "n_tagged": len(T), "tags": t, "cut_looks": n_cut, "cut_looks_no_label_lead": sum(r["n_nolab"] for r in S),
            "mean_dsum_pre": (float(np.mean([r["dsum_pre"] for r in S])) if S else None), "mean_dsum_win": (float(np.mean([r["dsum_win"] for r in S])) if S else None)}


def line(prefix, t):
    cl = " ".join("%s %d" % (k, t["classes"][k]) for k in CLASSES)
    tg = " ".join("%s %d" % (k, t["tags"][k]) for k in TAGS)
    sh = (t["cut_looks_no_label_lead"] / t["cut_looks"]) if t["cut_looks"] else float("nan")
    return "%s n %d | %s | tags of %d: %s | cut looks %d, no label lead %.2f | mean dsum pre %s win %s" % (
        prefix, t["n"], cl, t["n_tagged"], tg, t["cut_looks"], sh, ("%.2f" % t["mean_dsum_pre"]) if t["mean_dsum_pre"] is not None else "-",
        ("%.2f" % t["mean_dsum_win"]) if t["mean_dsum_win"] is not None else "-")


def run(a, arms, expect=True):
    R, H = X.get_engine(a.code)
    J = H.load_join(a.join)
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    if g30 is None:
        raise SystemExit("MISSING-GRID")
    clips = [str(c) for c in J["clip"]]
    keys = R.grid_keys(g30, clips)
    forms = ["RR", "RLc"] + list(arms)
    data = {f: {s: X.load_form(R, H, a.pull, J, X.RUNS[f], s, keys) for s in X.SEEDS} for f in forms}
    nums = {}
    for f in forms:
        for s in X.SEEDS:
            d = data[f][s]
            nums[(f, s)] = form_numbers(a.pull, X.RUNS[f], s, d["ci"], d["ws"], J, R, d["lead"])
    per = {x: {"RLc": [], "RR": []} for x in arms}
    table = []
    for x in arms:
        for s in X.SEEDS:
            for ref in ("RLc", "RR"):
                rows = count_arm(data[x][s], data[ref][s], nums[(x, s)], nums[(ref, s)])
                per[x][ref] += rows
                if ref == "RLc":
                    for r in rows:
                        i = r["i"]
                        table.append(dict({k: v for k, v in r.items() if k != "i"}, arm=x, seed=int(s), clip=clips[int(data[x][s]["ci"][i])], ws=int(data[x][s]["ws"][i])))
    # control
    bad = []
    if expect:
        for x in arms:
            ne, nn = sum(1 for r in per[x]["RLc"] if r["group"] == "excess"), sum(1 for r in per[x]["RLc"] if r["group"] == "neither")
            if x in EXPECT_EXCESS and (ne != EXPECT_EXCESS[x] or nn != EXPECT_NEITHER[x]):
                bad.append("%s excess %d neither %d" % (x, ne, nn))
    if bad:
        return 3, ["ZZ FALLPRE CONTROL FAIL " + "; ".join(bad)], None
    offs = sorted(set(int(v["pre_offset"]) for v in nums.values()))
    lines = ["FALLPRE CONTROL PASS arms %d pre-roll offset %s" % (len(arms), offs)]
    rec = {"tier": X.TIER, "arms": list(arms), "pre_roll_offsets": offs, "classes": list(CLASSES), "tags": list(TAGS), "constants": {"SHARE_MIN": SHARE_MIN, "GAP_LOW_M": GAP_LOW_M,
           "CLOSE_EPS_MPS": X.CLOSE_EPS_MPS}, "per_arm": {}, "pooled": {}, "windows": table}
    for ref in ("RLc", "RR"):
        allrows = []
        for x in arms:
            te, tn = tally(per[x][ref], "excess"), tally(per[x][ref], "neither")
            rec["per_arm"].setdefault(x, {})[ref] = {"excess": te, "neither": tn}
            allrows += per[x][ref]
            if ref == "RLc":
                lines.append(line("FALLPRE %s excess " % x, te))
                lines.append(line("FALLPRE %s neither" % x, tn))
        te, tn, tb = tally(allrows, "excess"), tally(allrows, "neither"), tally(allrows, "both")
        rec["pooled"][ref] = {"excess": te, "neither": tn, "both": tb}
        lines.append(line("FALLPRE pooled vs %s excess " % ref, te))
        lines.append(line("FALLPRE pooled vs %s neither" % ref, tn))
        lines.append(line("FALLPRE pooled vs %s both   " % ref, tb))
    ex = [r for r in table if r["group"] == "excess"]
    nd = len(set((r["seed"], r["clip"], r["ws"]) for r in ex))
    rec["excess_arm_windows"], rec["excess_distinct_windows"] = len(ex), nd
    lines.append("FALLPRE excess arm-windows %d distinct windows %d" % (len(ex), nd))
    lines.append("ZZ FALLPRE DONE %d %d" % (len(arms), len(ex)))
    return 0, lines, rec


def selftest():
    ok = True
    ok &= classify(0, 0, 2, 1) == "NO-CUT-BY-X"
    ok &= classify(2, 0, 1, 0) == "MORE-CUT-IN-PRE-ROLL"
    ok &= classify(2, 3, 1, 2) == "MORE-CUT-IN-PRE-ROLL"            # equal surplus: the pre-roll class comes first
    ok &= classify(2, 4, 1, 2) == "MORE-CUT-IN-WINDOW"
    ok &= classify(1, 2, 1, 1) == "MORE-CUT-IN-WINDOW"
    ok &= classify(1, 1, 1, 1) == "NOT-MORE-LOOKS" and classify(1, 0, 2, 3) == "NOT-MORE-LOOKS"
    ok &= tag_of(0, 0, 0, 0, 0) is None
    ok &= tag_of(4, 2, 2, 2, 4) == "FALSE-PRESENCE"                 # order: presence before gap before slope
    ok &= tag_of(4, 1, 3, 2, 4) == "GAP-LOW"
    ok &= tag_of(4, 1, 3, 1, 2) == "SLOPE"
    ok &= tag_of(4, 1, 3, 1, 1) == "OTHER"
    # row matching with an offset, on planted rows
    ci, ws = np.array([0, 0, 1]), np.array([40, 50, 40])
    rows = {"ci": np.repeat(ci, 3), "t0": np.repeat(ws - 30, 3)}
    ok &= pre_offset(rows, ci, ws) == 30
    w = match(rows, ci, ws, 30)
    ok &= w.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    try:
        pre_offset({"ci": np.repeat(ci, 2), "t0": np.repeat(ws - 30, 2)}, ci, ws)
        ok = False
    except SystemExit:
        pass
    print("ZZ SELFTEST-FALLPRE ok: %s" % bool(ok))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for n in ("pull", "grid_dir", "join", "code", "out"):
        ap.add_argument("--" + n.replace("_", "-"), default="")
    ap.add_argument("--arms", default=",".join(X.ARMS_DEFAULT))
    ap.add_argument("--no-expect", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    arms = [x for x in a.arms.split(",") if x]
    for x in arms:
        if x not in X.RUNS or x in X.REFS:
            raise SystemExit("UNKNOWN-ARM %s" % x)
    if os.path.exists(a.out):
        raise SystemExit("OUT-EXISTS %s" % a.out)
    rc, lines, rec = run(a, arms, expect=not a.no_expect)
    if rec is not None:
        os.makedirs(a.out)
        with io.open(os.path.join(a.out, "FALLPRE.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
    print("\n".join(lines))
    return rc


if __name__ == "__main__":
    sys.exit(main())
