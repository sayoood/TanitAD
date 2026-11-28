# export_fall.py -- pod side, CPU only. A COUNT, no card, no name: in which windows does a vision-fed arm fall behind (F: more than 10 m at the last scored step) while the label-fed form RLc does not,
# and what does the rule log say about why. Tier: T1-kin (kinematic rollout on the model's own actions), hb0.2, pre-roll of 30 steps (6 s); a diagnostic, not driving performance.
# Flags C, U, F per window are the ENGINE'S OWN (headway_read_r2.window_cells), loaded as export_traces.py of the rolltrace kit loads them. The rule log is read from <pull>/I10<run>_s<k>/ticks/ (tick log + later-look log).
# Definitions are fixed in FALLTRACE_DEFS.md; every threshold is a module constant below.
# Control (fails => no FALLTRACE.json, nothing else written, exit 3): same windows in all dumps, 1271 windows per seed, 650 windows with a lead pooled, contact rates of RR, RLc, VD5c, F2, F3, F4, E5c equal the banked values to
# four decimals, and C and F of every form equal the banked result file where one is given (--banked).
# Usage: python export_fall.py --pull <folder with I10<run>_s<k>> --grid-dir <folder with grid_p30.json> --join <join npz> --code <engine code folder> --out <new folder> [--arms VD5c,F2,F3,F4,E5c] [--banked a.json,b.json]
#        python export_fall.py --selftest
# Last line: ZZ FALLTRACE DONE <n arms> <n excess windows>   or   ZZ FALLTRACE CONTROL FAIL <which>   or   ZZ FALLTRACE STOP <why>
import argparse
import json
import math
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("SR_ESTREAD_DIR"), os.path.join(_HERE, "..", "integ10k")):
    if _p and os.path.isfile(os.path.join(_p, "read_longroll10k.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
import numpy as np  # noqa: E402

BASE = "RQ50KF5P30Hb20"
RUNS = {"RR": BASE, "RLc": "RLcKF5P30Hb20", "VD5c": BASE + "vVD5c", "F2": BASE + "fF2", "F3": BASE + "fF3", "F4": BASE + "fF4", "E5c": BASE + "eE5c",
        "G2": BASE + "gG2", "G3": BASE + "gG3", "G4": BASE + "gG4", "GEc": BASE + "gGEc", "E3c": BASE + "gE3c"}
REFS = ("RR", "RLc")
ARMS_DEFAULT = ("VD5c", "F2", "F3", "F4", "E5c")
ARMS_KNOWN = tuple(k for k in RUNS if k not in REFS)
EXPECT_C = {"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "F2": 0.0138, "F3": 0.0185, "F4": 0.0123, "E5c": 0.0138}
TOL_RATE = 1e-4
N_LEAD = 650
N_WIN = 1271
SEEDS = (0, 1)
DT = 0.2                                   # s per step (engine DT)
GAP_MIN_M, GAP_MAX_M, STEP_MAX_M = 0.0, 60.0, 3.0       # label presence and track step bound (LabelSource._ok, headway STEP_MAX_M)
NO_ACT_MIN = 5                             # class 1: the rule acted in fewer than this many steps
SHARE_MIN = 0.5                            # classes 2 to 4: this share of the steps (or more)
GAP_LOW_M = 5.0                            # class 3: fed gap below the label gap by more than this
CLOSE_EPS_MPS = 0.1                        # class 4: the gap "closes" when its rate is below minus this; "does not close" when the label rate is at or above minus this
CLASSES = ("NO-RULE-ACTION", "FALSE-PRESENCE", "GAP-LOW", "SLOPE", "MIXED")
RL_KEYS = ("ci", "t0", "j", "present", "changed", "g_src", "c_src")
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s)"


def fin(x):
    x = float(x)
    return round(x, 4) if math.isfinite(x) else None


def get_engine(code):
    import read_longroll10k as R                                           # the reader's own module (pod: SR_ESTREAD_DIR)
    if code:
        H = R.engine(types.SimpleNamespace(code=code))[0]
    else:                                                                  # selftest on the dev box: the local copy of the engine file
        sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "headway")))
        import headway_read_r2 as H
    return R, H


# ============================================================================================ rollouts and rule logs
def load_form(R, H, pull, J, run, seed, keys):
    """one run and seed -> per-window arrays in the order of `keys` (the engine's own cells)"""
    raw = R.load_dump_dir(R.dump_dir(pull, run, seed))
    if raw is None:
        raise SystemExit("MISSING-DUMP %s" % R.key_of(run, seed))
    idx, rep = R.restrict_idx(raw["clip_index"], raw["ws"], keys)
    if rep["missing"] != 0:
        raise SystemExit("WINDOWS-MISSING %s %d" % (R.key_of(run, seed), rep["missing"]))
    D = R.cut(raw, idx)
    L = H.label_windows(J, D["clip_index"], D["ws"])
    W = H.window_cells(D["g"], D["cl"], L)
    return {"C": W["C"], "U": W["U"], "F": W["F"], "lead": np.asarray(L["lead"], bool), "ws": np.asarray(D["ws"], np.int64), "ci": np.asarray(D["clip_index"], np.int64), "g": np.asarray(D["g"])}


def rule_rows(R, pull, run, seed):
    """the tick log and the later-look log of one rollout, concatenated; only the fields in RL_KEYS. Pre-roll rows live in another file and are not read."""
    key = R.key_of(run, seed)
    zs = []
    for nm in ("rule_log_%s.npz", "rule_log_mid_%s.npz"):
        p = os.path.join(pull, key, "ticks", nm % key)
        if not os.path.isfile(p):
            raise SystemExit("MISSING-RULE-LOG %s" % p)
        with np.load(p, allow_pickle=False) as z:
            for k in RL_KEYS:
                if k not in z.files:
                    raise SystemExit("MISSING-RULE-LOG-FIELD %s in %s" % (k, nm % key))
            zs.append({k: np.asarray(z[k]) for k in RL_KEYS})
    return {k: np.concatenate([z[k] for z in zs]) for k in RL_KEYS}


def load_banked(paths):
    """{arm: {"C": x, "F": y}} from the first dict named `levels` found (searching nested dicts) in each existing json file; missing files and files without it are skipped and listed."""
    out, used = {}, []

    def find(o):
        if isinstance(o, dict):
            lv = o.get("levels")
            if isinstance(lv, dict) and any(isinstance(v, dict) and "F" in v for v in lv.values()):
                return lv
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v)
                if r is not None:
                    return r
        return None
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                lv = find(json.load(f))
        except (OSError, ValueError):
            lv = None
        if lv is not None:
            used.append(os.path.basename(p))
            for k, v in lv.items():
                if isinstance(v, dict) and k not in out:
                    out[k] = {"C": v.get("C"), "F": v.get("F")}
    return out, used


# ============================================================================================ label and rule-log arithmetic
def label_at(J, ci, j):
    """label lead at (clip index, frame): ok [N] bool (inside the clip, valid, finite, 0 to 60 m as LabelSource._ok), gap [N] (NaN where not ok)"""
    ci, j = np.asarray(ci, np.int64), np.asarray(j, np.int64)
    inb = (j >= 0) & (j < J["n"][ci])
    r = np.where(inb, J["off"][ci] + j, 0)
    g = J["lead_gap"][r].astype(np.float64)
    with np.errstate(invalid="ignore"):
        ok = inb & (J["lead_valid"][r] == 1) & np.isfinite(g) & (g >= GAP_MIN_M) & (g <= GAP_MAX_M)
    return ok, np.where(ok, g, np.nan)


def row_arrays(rows, J):
    """row-level quantities. acted = present and changed (the rule changed the plan at that look). lab_ok/lab_gap: the label at the RECORDED frame j. slope: label gap forward difference over the step j to j+1 (m/s, NaN where either
    frame has no lead or the step exceeds 3 m)."""
    ci, j = rows["ci"].astype(np.int64), rows["j"].astype(np.int64)
    present, changed = rows["present"].astype(bool), rows["changed"].astype(bool)
    g, c = rows["g_src"].astype(np.float64), rows["c_src"].astype(np.float64)
    ok0, g0 = label_at(J, ci, j)
    ok1, g1 = label_at(J, ci, j + 1)
    with np.errstate(invalid="ignore"):
        sv = ok0 & ok1 & (np.abs(g1 - g0) <= STEP_MAX_M)
        slope = np.where(sv, (g1 - g0) / DT, np.nan)
        diff = np.where(ok0, g - g0, np.nan)
        closing_fed = c < -CLOSE_EPS_MPS
        lab_not_closing = np.isfinite(slope) & (slope >= -CLOSE_EPS_MPS)
    return {"present": present, "acted": present & changed, "lab_ok": ok0, "diff": diff, "slope": slope, "c": c, "closing_fed": closing_fed, "lab_not_closing": lab_not_closing}


def _med(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return fin(np.median(x)) if len(x) else None


def numbers_of(A, idx):
    """the per-window numbers from the row arrays A over the rows idx of one window"""
    a = A["acted"][idx]
    lab = A["lab_ok"][idx]
    n_acted = int(a.sum())
    lead_a = a & lab
    d = A["diff"][idx][lead_a]
    n_low = int((d[np.isfinite(d)] < -GAP_LOW_M).sum())
    n_sl = int((a & A["closing_fed"][idx] & A["lab_not_closing"][idx]).sum())
    return {"n_steps": int(len(idx)), "n_fed": int(A["present"][idx].sum()), "n_acted": n_acted, "n_acted_no_label_lead": int((a & ~lab).sum()), "n_acted_label_lead": int(lead_a.sum()),
            "n_gap_low": n_low, "n_closing_fed": int((a & A["closing_fed"][idx]).sum()), "n_slope_mismatch": n_sl,
            "gap_diff_median_m": _med(d), "closing_input_median_mps": _med(A["c"][idx][a]), "label_slope_median_mps": _med(A["slope"][idx][a])}


def classify(n):
    """first matching class (order fixed): NO-RULE-ACTION, FALSE-PRESENCE, GAP-LOW, SLOPE, MIXED. n: numbers_of output."""
    a = n["n_acted"]
    if a < NO_ACT_MIN:
        return CLASSES[0]
    if n["n_acted_no_label_lead"] >= SHARE_MIN * a - 1e-9:
        return CLASSES[1]
    lead = n["n_acted_label_lead"]
    if lead > 0 and n["n_gap_low"] >= SHARE_MIN * lead - 1e-9:
        return CLASSES[2]
    if n["n_slope_mismatch"] >= SHARE_MIN * a - 1e-9:
        return CLASSES[3]
    return CLASSES[4]


def window_numbers(rows, J, ci, ws):
    """-> list (one entry per window, order of ci/ws) of numbers_of dicts; a window with no rule-log row gets zeros. Rows are matched on (clip index, t0 = window start)."""
    n = len(ci)
    wk = np.asarray(ci, np.int64) * (1 << 32) + np.asarray(ws, np.int64)
    order = np.argsort(wk, kind="stable")
    sk = wk[order]
    rk = rows["ci"].astype(np.int64) * (1 << 32) + rows["t0"].astype(np.int64)
    pos = np.minimum(np.searchsorted(sk, rk), max(n - 1, 0))
    hit = (sk[pos] == rk) if n else np.zeros(len(rk), bool)
    widx = np.where(hit, order[pos], -1)
    A = row_arrays(rows, J)
    groups = {}
    for i in np.flatnonzero(widx >= 0).tolist():
        groups.setdefault(int(widx[i]), []).append(i)
    empty = numbers_of(A, np.zeros(0, np.int64))
    return [numbers_of(A, np.asarray(groups[w], np.int64)) if w in groups else dict(empty) for w in range(n)], int(len(rk) - hit.sum())


# ============================================================================================ rates, control, tables
def rates(data, forms, seeds):
    out = {}
    for f in forms:
        r = {}
        for c in ("C", "F"):
            pooled = np.concatenate([data[f][s][c][data[f][s]["lead"]] for s in seeds])
            per = [float(np.nanmean(data[f][s][c])) for s in seeds]
            r[c] = float(np.nanmean(pooled))
            r[c + "_seed_mean"] = float(np.mean(per))
        out[f] = r
    return out


def control(data, forms, seeds, expect_c, banked, n_lead, n_win, tol=TOL_RATE):
    """-> (ok, failing clause or '', record). Clauses: same-windows, n-windows, n-lead, rate-C-<form>, banked-<C|F>-<form>."""
    rec = {}
    ref = data[forms[0]][seeds[0]]
    for f in forms:
        for s in seeds:
            d = data[f][s]
            if not (np.array_equal(d["ws"], ref["ws"]) and np.array_equal(d["ci"], ref["ci"]) and np.array_equal(d["g"], ref["g"]) and np.array_equal(d["lead"], ref["lead"])):
                return False, "same-windows %s s%d" % (f, s), rec
    nw = int(len(ref["ws"]))
    rec["windows_per_seed"] = nw
    if nw != n_win:
        return False, "n-windows %d != %d" % (nw, n_win), rec
    nl = int(sum(int(data[forms[0]][s]["lead"].sum()) for s in seeds))
    rec["windows_with_lead_pooled"] = nl
    if nl != n_lead:
        return False, "n-lead %d != %d" % (nl, n_lead), rec
    rt = rates(data, forms, seeds)
    rec["rates"] = {f: {k: round(v, 6) for k, v in r.items()} for f, r in rt.items()}
    rec["expected_contact"] = {f: expect_c[f] for f in forms if f in expect_c}
    for f in forms:
        if f in expect_c:
            exp = expect_c[f]
            if not (abs(round(rt[f]["C"], 4) - exp) < 1e-9 and abs(round(rt[f]["C_seed_mean"], 4) - exp) < 1e-9):
                return False, "rate-C-%s %.4f vs %.4f" % (f, rt[f]["C"], exp), rec
    checked = []
    for f in forms:
        b = (banked or {}).get(f)
        if not b:
            continue
        for c in ("C", "F"):
            if b.get(c) is None:
                continue
            checked.append("%s-%s" % (f, c))
            if abs(rt[f][c + "_seed_mean"] - float(b[c])) > tol:
                return False, "banked-%s-%s %.4f vs %.4f" % (c, f, rt[f][c + "_seed_mean"], float(b[c])), rec
    rec["banked_checked"] = checked
    return True, "", rec


def table2x2(data, x, ref, seeds):
    out = {}
    for s in list(seeds) + ["pooled"]:
        ss = list(seeds) if s == "pooled" else [s]
        t = {"both": 0, "x_only": 0, "ref_only": 0, "neither": 0}
        for m in ss:
            lead = data[x][m]["lead"]
            a, b = data[x][m]["F"] == 1.0, data[ref][m]["F"] == 1.0
            t["both"] += int((lead & a & b).sum())
            t["x_only"] += int((lead & a & ~b).sum())
            t["ref_only"] += int((lead & ~a & b).sum())
            t["neither"] += int((lead & ~a & ~b).sum())
        out[str(s)] = t
    return out


def group_of(fx, fr):
    return "both" if (fx and fr) else ("x_only" if fx else ("ref_only" if fr else "neither"))


def analyse(data, J, clips, rules, arms, seeds):
    """-> (windows table rows, class counts {arm: {group: {class: n}}}, extra record)"""
    table, counts, extra = [], {}, {"windows_without_rule_rows": {}, "rule_rows_outside_windows": {}, "acted_rows_without_label_lead": {}}
    for x in arms:
        counts[x] = {"excess": {c: 0 for c in CLASSES}, "neither": {c: 0 for c in CLASSES}}
        for s in seeds:
            dx, dl, dr = data[x][s], data["RLc"][s], data["RR"][s]
            nums, outside = window_numbers(rules[(x, s)], J, dx["ci"], dx["ws"])
            extra["rule_rows_outside_windows"]["%s_s%d" % (x, s)] = outside
            A = row_arrays(rules[(x, s)], J)
            extra["acted_rows_without_label_lead"]["%s_s%d" % (x, s)] = {"acted_rows": int(A["acted"].sum()), "of_which_no_label_lead": int((A["acted"] & ~A["lab_ok"]).sum())}
            miss = 0
            for i in np.flatnonzero(dx["lead"]).tolist():
                fx, fl, fr = dx["F"][i] == 1.0, dl["F"][i] == 1.0, dr["F"][i] == 1.0
                n = nums[i]
                miss += int(n["n_steps"] == 0)
                cls = classify(n)
                grp = group_of(fx, fl)
                if grp == "x_only":
                    counts[x]["excess"][cls] += 1
                elif grp == "neither":
                    counts[x]["neither"][cls] += 1
                table.append(dict({"arm": x, "seed": int(s), "clip": clips[int(dx["ci"][i])], "ws": int(dx["ws"][i]), "C": fin(dx["C"][i]), "U": fin(dx["U"][i]), "F": fin(dx["F"][i]),
                                   "F_RLc": fin(dl["F"][i]), "F_RR": fin(dr["F"][i]), "group_vs_RLc": grp, "group_vs_RR": group_of(fx, fr), "class": cls}, **n))
            extra["windows_without_rule_rows"]["%s_s%d" % (x, s)] = miss
    return table, counts, extra


def count_lines(data, counts, arms, seeds):
    lines = []
    for x in arms:
        for s in list(seeds) + ["pooled"]:
            a, b = table2x2(data, x, "RLc", seeds)[str(s)], table2x2(data, x, "RR", seeds)[str(s)]
            lines.append("FALLTRACE 2x2 %s s%s vsRLc both %d x_only %d rlc_only %d neither %d | vsRR both %d x_only %d rr_only %d neither %d"
                         % (x, s, a["both"], a["x_only"], a["ref_only"], a["neither"], b["both"], b["x_only"], b["ref_only"], b["neither"]))
    pooled = {"excess": {c: 0 for c in CLASSES}, "neither": {c: 0 for c in CLASSES}}
    for x in arms:
        for g in pooled:
            for c in CLASSES:
                pooled[g][c] += counts[x][g][c]
    for name, cc in [(x, counts[x]) for x in arms] + [("pooled", pooled)]:
        for g in ("excess", "neither"):
            n = sum(cc[g].values())
            lines.append("FALLTRACE CLASS %s %s n %d | %s" % (name, g, n, " ".join("%s %d (%.2f)" % (c, cc[g][c], (cc[g][c] / n) if n else 0.0) for c in CLASSES)))
    return lines, pooled


def run(R, H, a, arms, expect_c=None, banked=None, n_lead=None, n_win=None, seeds=SEEDS):
    """-> (rc, lines to print, record or None). Control first; nothing else is computed or written when it fails."""
    expect_c = EXPECT_C if expect_c is None else expect_c
    n_lead = N_LEAD if n_lead is None else n_lead
    n_win = N_WIN if n_win is None else n_win
    arms = list(arms)
    forms = list(REFS) + [x for x in arms if x not in REFS]
    J = H.load_join(a.join)
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    if g30 is None:
        raise SystemExit("MISSING-GRID %s" % os.path.join(a.grid_dir, "grid_p30.json"))
    clips = [str(c) for c in J["clip"]]
    keys = R.grid_keys(g30, clips)
    data = {f: {s: load_form(R, H, a.pull, J, RUNS[f], s, keys) for s in seeds} for f in forms}
    ok, why, rec = control(data, forms, list(seeds), expect_c, banked, n_lead, n_win)
    if not ok:
        return 3, ["ZZ FALLTRACE CONTROL FAIL %s" % why], None
    rules = {(x, s): rule_rows(R, a.pull, RUNS[x], s) for x in arms for s in seeds}
    table, counts, extra = analyse(data, J, clips, rules, arms, list(seeds))
    lines, pooled = count_lines(data, counts, arms, list(seeds))
    n_exc = sum(sum(counts[x]["excess"].values()) for x in arms)
    tabs = {x: {"vsRLc": table2x2(data, x, "RLc", seeds), "vsRR": table2x2(data, x, "RR", seeds)} for x in arms}
    out = {"tier": TIER, "arms": arms, "model_seeds": list(seeds), "control": dict(rec, PASS=True), "table_2x2_F": tabs, "class_counts": counts, "class_counts_pooled_over_arms": pooled,
           "constants": {"NO_ACT_MIN": NO_ACT_MIN, "SHARE_MIN": SHARE_MIN, "GAP_LOW_M": GAP_LOW_M, "CLOSE_EPS_MPS": CLOSE_EPS_MPS, "GAP_MAX_M": GAP_MAX_M, "STEP_MAX_M": STEP_MAX_M, "DT_S": DT},
           "rule_log_coverage": extra, "classes": list(CLASSES), "windows": table}
    return 0, ["FALLTRACE CONTROL PASS forms %d" % len(forms)] + lines + ["ZZ FALLTRACE DONE %d %d" % (len(arms), n_exc)], out


def parse_arms(text):
    arms = []
    for t in (text or "").split(","):
        t = t.strip()
        if t and t not in arms:
            arms.append(t)
    return arms or list(ARMS_DEFAULT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for n in ("pull", "grid_dir", "join", "code", "out", "arms", "banked"):
        ap.add_argument("--" + n.replace("_", "-"), default="")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.code)
    for n in ("pull", "grid_dir", "join", "code", "out"):
        if not getattr(a, n):
            print("ZZ FALLTRACE STOP missing --%s" % n.replace("_", "-"))
            return 2
    arms = parse_arms(a.arms)
    bad = [x for x in arms if x not in ARMS_KNOWN]
    if bad:
        print("ZZ FALLTRACE STOP unknown arm %s" % ",".join(bad))
        return 2
    if os.path.exists(a.out):
        print("ZZ FALLTRACE STOP output exists")
        return 2
    banked, _ = load_banked([p.strip() for p in a.banked.split(",")] if a.banked else [])
    R, H = get_engine(a.code)
    rc, lines, rec = run(R, H, a, arms, banked=banked)
    if rec is not None:
        os.makedirs(a.out)
        with open(os.path.join(a.out, "FALLTRACE.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
    for ln in lines:
        print(ln)
    return rc


# ============================================================================================ selftest on tiny synthetic data
SYN_CLIPS = ("aaaaaaaa-0000", "bbbbbbbb-1111")
SYN_VALID_BELOW = 45                                     # label frames of clip 0 at or above this have no lead


def syn_speed(form, w):
    """rolled-ego speed (m/s) per window of the lead clip; the logged ego is 10 m/s, the lead gap 15 m. v = 1 gives F (not contact), v = 4 gives contact, v = 2 neither."""
    if form == "RLc":
        return 1.0 if w in (5, 6) else 2.0
    if form == "RR":
        return 4.0 if w in (8, 9) else (1.0 if w in (0, 5, 6) else 2.0)
    return 4.0 if w == 10 else (1.0 if w in (0, 1, 2, 3, 4, 5) else 2.0)


def syn_plan(w):
    """rule-log plan of a window of an arm: list of (step k, frame override or None, g_src, c_src) for the steps where the rule acted"""
    neutral = [(k, None, 15.0, 0.0) for k in range(10, 20)]
    if w == 0:
        return [(k, None, 15.0, 0.0) for k in (5, 6, 7)]                                                             # NO-RULE-ACTION (3 steps)
    if w == 1:
        return [(k, 50 + k - 10, 10.0, -1.0) for k in range(10, 18)] + [(20, None, 15.0, 0.0), (21, None, 15.0, 0.0)]  # FALSE-PRESENCE: 8 of 10 acted steps at frames with no label lead
    if w == 2 or w == 7:
        return [(k, None, 8.0, 0.0) for k in range(10, 20)]                                                          # GAP-LOW: fed 8 m against label 15 m (w 7 is a window where neither falls behind)
    if w == 3:
        return [(k, None, 15.0, -2.0) for k in range(10, 20)]                                                        # SLOPE: closing input -2 m/s, label gap constant
    if w == 4:
        return [(k, None, 8.0, 0.0) for k in range(10, 14)] + [(k, None, 15.0, -2.0) for k in range(14, 18)] + [(18, None, 15.0, 0.0), (19, None, 15.0, 0.0)]    # MIXED: 4 and 4 of 10
    if w in (5, 6):
        return neutral
    return []


def make_synth(root, arms, fault=""):
    """2 clips x 20 windows (clip 0 has a lead on frames below 45, clip 1 none), 30 steps. Returns (expected contact rates, expected banked levels, n_lead, n_win). fault: '' | 'contact' | 'banked'."""
    os.makedirs(os.path.join(root, "grid"))
    nclip = 60
    lead_valid = np.zeros(2 * nclip, np.int64)
    lead_valid[:SYN_VALID_BELOW] = 1
    np.savez(os.path.join(root, "join.npz"), clip=np.asarray(SYN_CLIPS), off=np.array([0, nclip]), n=np.array([nclip, nclip]), lead_valid=lead_valid, lead_gap=np.full(2 * nclip, 15.0))
    json.dump({"order": list(SYN_CLIPS), "windows": {c: list(range(20)) for c in SYN_CLIPS}}, open(os.path.join(root, "grid", "grid_p30.json"), "w"))
    kk = np.arange(1, 31)[:, None]
    g1 = np.concatenate([2.0 * kk, 0.0 * kk], axis=1)
    ci = np.repeat([0, 1], 20)
    ws = np.tile(np.arange(20), 2)
    forms = list(REFS) + list(arms)
    for f in forms:
        for s in SEEDS:
            run_n = RUNS[f]
            key = "I10%s_s%d" % (run_n, s)
            d = os.path.join(root, "pull", key, "dump_%s_m10_causal_heads" % key)
            os.makedirs(d)
            cl = np.zeros((40, 30, 2))
            for i in range(40):
                w = i % 20
                v = syn_speed(f, w) if i < 20 else 2.0
                if fault == "contact" and f == "VD5c" and s == 1 and i == 11:
                    v = 4.0                                    # the planted fault: one more contact in VD5c, seed 1
                cl[i] = np.concatenate([v * kk, 0.0 * kk], axis=1)
            np.savez(os.path.join(d, "ep0.npz"), g=np.tile(g1, (40, 1, 1)), cl=cl, ws=ws, eid=ci, clip_index=ci, v0=np.zeros(40))
            if f in REFS:
                continue
            cols = {k: [] for k in RL_KEYS}
            for w in range(20):
                plan = {p[0]: p for p in syn_plan(w)}
                for k in range(30):
                    p = plan.get(k)
                    cols["ci"].append(0)
                    cols["t0"].append(w)
                    cols["j"].append(w + k if (p is None or p[1] is None) else p[1])
                    cols["present"].append(True)
                    cols["changed"].append(p is not None)
                    cols["g_src"].append(15.0 if p is None else p[2])
                    cols["c_src"].append(0.0 if p is None else p[3])
            n = len(cols["ci"])
            os.makedirs(os.path.join(root, "pull", key, "ticks"))
            for nm, sel in (("rule_log_%s.npz" % key, slice(0, n, 2)), ("rule_log_mid_%s.npz" % key, slice(1, n, 2))):
                np.savez(os.path.join(root, "pull", key, "ticks", nm), **{k: np.asarray(v)[sel] for k, v in cols.items()}, form=np.asarray("RR"))
    exp_c = {"RR": 0.1, "RLc": 0.0}
    lv = {"RR": {"C": 0.1, "F": 0.15}, "RLc": {"C": 0.0, "F": 0.1}}
    for x in arms:
        exp_c[x] = 0.05
        lv[x] = {"C": 0.05, "F": 0.3}
    if fault == "banked":
        lv["RR"]["F"] = 0.2
    return exp_c, lv, 40, 40


def selftest(code=""):
    import tempfile
    base = os.path.join(_HERE, "tmp_pt")
    os.makedirs(base, exist_ok=True)
    R, H = get_engine(code)
    arms = ["VD5c", "F2"]
    res = {}
    for fault in ("", "contact", "banked"):
        root = tempfile.mkdtemp(prefix="selftest_fall_", dir=base)
        exp, lv, nl, nw = make_synth(root, arms, fault)
        a = types.SimpleNamespace(pull=os.path.join(root, "pull"), grid_dir=os.path.join(root, "grid"), join=os.path.join(root, "join.npz"))
        res[fault] = run(R, H, a, arms, expect_c=exp, banked=lv, n_lead=nl, n_win=nw)
    rc, lines, rec = res[""]
    good = rc == 0 and rec is not None and lines[-1] == "ZZ FALLTRACE DONE 2 20"
    if good:
        t = rec["table_2x2_F"]["VD5c"]
        good = t["vsRLc"]["0"] == {"both": 1, "x_only": 5, "ref_only": 1, "neither": 13} and t["vsRR"]["pooled"] == {"both": 4, "x_only": 8, "ref_only": 2, "neither": 26}
        want = {"NO-RULE-ACTION": 1, "FALSE-PRESENCE": 1, "GAP-LOW": 1, "SLOPE": 1, "MIXED": 1}
        good = good and all(rec["class_counts"][x]["excess"][c] == 2 * want[c] for x in arms for c in CLASSES)
        good = good and rec["class_counts"]["VD5c"]["neither"]["GAP-LOW"] == 2 and rec["class_counts"]["VD5c"]["neither"]["NO-RULE-ACTION"] == 24
    good = good and res["contact"][0] == 3 and res["contact"][2] is None and res["contact"][1][0].startswith("ZZ FALLTRACE CONTROL FAIL rate-C-VD5c") and len(res["contact"][1]) == 1
    good = good and res["banked"][0] == 3 and res["banked"][2] is None and res["banked"][1][0].startswith("ZZ FALLTRACE CONTROL FAIL banked-F-RR")
    print("selftest: clean -> %s | planted contact fault -> %s | planted banked F fault -> %s" % (res[""][1][-1], res["contact"][1][0], res["banked"][1][0]))
    print("ZZ SELFTEST-FALLTRACE ok: %s" % good)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
