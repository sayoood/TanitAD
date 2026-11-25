#!/usr/bin/env python
"""read_contactrh10k.py -- read tool of card PREREG_V8F_PROOF_ADD141 (CONTACTRH10K): in the reader form, does RH make its extra contacts where the
vision-only reader misreads the gap, or does it follow closer on the same read?
Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, logged camera frames; the lead does not react); not driving performance.
CPU only, banked runs (dumps) and banked rule logs of cards ADD140 / ADD139. No new rollout.

  python read_contactrh10k.py read --root PULL_DIR --join JOIN_NPZ --out-dir DIR --tests-out DEV_PYTEST_OUT [--seeds 0,1,2]

Run folders: <root>/<TAG>_s<m> with ticks/rule_log_<TAG>_s<m>.npz and dump_<TAG>_s<m>_m10_causal_heads/ep*.npz, TAG = H_RQ50K (RH) and U_RQ50K (UE).
Order: controls c1..c4 are computed and written (CONTACTRH10K_CONTROLS.json) BEFORE any value is computed; if one fails, the result file carries values = null
and no name, and the verdict says so. Exit codes: 0 read done, 2 a control failed (values withheld), 3 refused (inputs missing), 1 error.

Window and contact definitions: headway_read_r2.py (HEADWAY_DEFS.md), imported unchanged (label_windows, window_cells, logged_arc, progress).
Rule log fields (LOG_FIELDS of v8f_t1_eval_rule.py): ci, t0, j, k, tick, active, present, src_bad, sat, sat_exec, lift, changed, g_src, c_src, p_src, v0, v_l, v_L, a_l, a_m, g0.
  j = t0 + k is the cache step of the row; ci indexes the label join; (ci, t0) is the window key of the dump (clip_index, ws).
  c_src is the reader closing read as the rule used it: v_L = max(0, v_l + c_src), so c_src = lead speed minus ego speed (positive = the gap opens);
  the closing rate of the card is -c_src (positive = the gap shrinks).
Decisions the card leaves open (fixed here, stated in the verdict):
  * the tick of a window = the row with the largest k at or before its first contact step kc (kc in 1..30, window frame), among the rows of the tick log of that run;
  * gap error = g_src - label gap at the row's cache step, on windows where the reader presence (field `present`) is 1 and the label has a valid lead; presence miss = present 0
    while the label has a lead, over windows where the label has a lead;
  * Q3 uses each arm's own tick (its own first contact step), paired by window; primary closing = -c_src; beside it v0 - v_L (rolled ego speed minus the rule's lead speed).
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD141"
PRE = "CONTACTRH10K"
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions, logged camera frames; the lead does not react; not driving performance)"
TAG_H, TAG_U = "H_RQ50K", "U_RQ50K"
HEADWAY_DIR = os.environ.get("CONTACTRH_HEADWAY", "F:/TanitAD-scratch/v8proof/headway")
N_BOOT, BOOT_SEED = 2000, 141
MIN_X = 10
PLANT_M = 2.0
TOL_G = 1e-3
N_SAMPLE = 20
NEED = ("ci", "t0", "j", "k", "present", "g_src", "c_src", "v0", "v_L", "g0", "a_l", "a_m")
N_MISREAD, N_SAME, N_UNRES = PRE + "-MISREAD", PRE + "-SAME-READ", PRE + "-UNRESOLVED"
CONTROLS_NAME, RESULT_NAME, VERDICT_NAME = PRE + "_CONTROLS.json", PRE + "_RESULT.json", PRE + "_VERDICT.md"
FAMILIES = ["LONGITUDINAL: read here (where the contacts of RH under readout K are made, reader form, %s)" % "T1-kin/hb0.2",
            "LATERAL: not read by this card; n = 0", "TACTICAL: not read by this card; n = 0", "STRATEGIC: not read by this card; n = 0"]
LIMITS = ("T1-kin only: the lead moves as logged, the reader reads logged frames. Contact windows of three model seeds over one evaluation set, so X is small and the intervals are wide. "
          "The interval answers what another draw of evaluation episodes would say, not what another training run or another inference run would say. "
          "A gap error is defined on windows where the reader reports a lead; a missed lead is counted in Q2 and not in Q1.")

HW = None


class LoadError(Exception):
    pass


def bind_headway(path=None):
    """imports headway_read_r2.py (the window and contact definitions) unchanged"""
    global HW
    if HW is not None:
        return HW
    d = path or HEADWAY_DIR
    if not os.path.isfile(os.path.join(d, "headway_read_r2.py")):
        raise LoadError("the landed headway reader cannot be found")
    if d not in sys.path:
        sys.path.insert(0, d)
    import headway_read_r2 as mod
    HW = mod
    return HW


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


def write_new(path, text):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================================ loading
def run_folder(root, tag, m):
    return os.path.join(root, "%s_s%d" % (tag, m))


def dump_folder(root, tag, m):
    return os.path.join(run_folder(root, tag, m), "dump_%s_s%d_m10_causal_heads" % (tag, m))


def log_path(root, tag, m):
    return os.path.join(run_folder(root, tag, m), "ticks", "rule_log_%s_s%d.npz" % (tag, m))


def read_log(path):
    if not os.path.isfile(path):
        raise LoadError("a rule log file is missing")
    with np.load(path, allow_pickle=False) as z:
        for k in NEED:
            if k not in z.files:
                raise LoadError("a rule log lacks a field")
        return {k: np.asarray(z[k]) for k in NEED}


def index_log(L):
    """{(ci, t0, k): row}; rows with one key and differing fields map to -1 (ambiguous)"""
    ci, t0, k = (np.asarray(L[x], np.int64) for x in ("ci", "t0", "k"))
    out = {}
    for r in range(len(ci)):
        key = (int(ci[r]), int(t0[r]), int(k[r]))
        if key not in out:
            out[key] = r
        elif out[key] != -1:
            a = out[key]
            same = all((float(L[f][a]) == float(L[f][r])) or (math.isnan(float(L[f][a])) and math.isnan(float(L[f][r]))) for f in NEED)
            if not same:
                out[key] = -1
    return out


def load_run(root, tag, m, J, expect_files):
    """dump, labels, contact cells and the rule log of one run"""
    H = bind_headway()
    dd = dump_folder(root, tag, m)
    if not os.path.isdir(dd):
        raise LoadError("a dump folder is missing")
    D = H.RH.load_dump(dd, expect_files)
    L = H.label_windows(J, D["clip_index"], D["ws"])
    C = H.window_cells(D["g"], D["cl"], L)
    gm = C["_gm"]
    with np.errstate(invalid="ignore"):
        ct = np.isfinite(gm) & (gm <= 0.0)
    contact = ct.any(1) & L["lead"]
    kc = np.where(contact, np.argmax(ct, axis=1) + 1, 0)
    log = read_log(log_path(root, tag, m))
    return {"D": D, "L": L, "C": C, "contact": contact, "kc": kc, "log": log, "idx": index_log(log), "tag": tag, "m": m}


# ============================================================================================ per-window tables
def tick_row(idx, ci, t0, kc):
    """the row with the largest k at or before kc (None when absent, -1 when ambiguous)"""
    for k in range(int(kc), -1, -1):
        r = idx.get((int(ci), int(t0), k))
        if r is not None:
            return r, k
    return None, -1


def label_at(J, ci, j):
    """label validity and gap of the join row (clip, step j)"""
    H = bind_headway()
    if j < 0 or j >= int(J["n"][ci]):
        return False, float("nan")
    row = int(J["off"][ci]) + int(j)
    g = float(J["lead_gap"][row])
    ok = bool(H.valid_frames(np.array([J["lead_valid"][row]]), np.array([g]))[0])
    return ok, g


def build_table(run, log, sel, J, inv):
    """rows over the windows of sel (bool over the dump windows) -> dict of arrays and a list of problems (texts without values)"""
    D, kc = run["D"], run["kc"]
    recs, problems = [], set()
    for i in np.flatnonzero(sel):
        ci, t0 = int(D["clip_index"][i]), int(D["ws"][i])
        r, k = tick_row(run["idx"], ci, t0, kc[i])
        if r is None:
            problems.add("a window has no rule tick at or before its first contact step")
            continue
        if r == -1:
            problems.add("a rule tick has several differing rows")
            continue
        if int(log["j"][r]) != t0 + k:
            problems.add("a rule row has j different from t0 + k")
        lv, lg = label_at(J, ci, t0 + k)
        pres = bool(log["present"][r])
        gs = float(log["g_src"][r])
        err = gs - lg if (pres and lv and math.isfinite(gs)) else float("nan")
        miss = float(not pres) if lv else float("nan")
        recs.append({"i": int(i), "ep": int(inv[i]), "ci": ci, "t0": t0, "k": int(k), "row": int(r), "lab_valid": lv, "lab_gap": lg, "err": err, "miss": miss,
                     "clos": -float(log["c_src"][r]), "clos2": float(log["v0"][r]) - float(log["v_L"][r])})
    T = {"problems": sorted(problems)}
    for f, dt in (("i", np.int64), ("ep", np.int64), ("ci", np.int64), ("t0", np.int64), ("k", np.int64), ("row", np.int64), ("lab_valid", bool),
                  ("lab_gap", np.float64), ("err", np.float64), ("miss", np.float64), ("clos", np.float64), ("clos2", np.float64)):
        T[f] = np.array([x[f] for x in recs], dtype=dt)
    return T


def episodes_of(runs):
    H = bind_headway()
    ref = next(iter(runs.values()))
    uniq, inv = H.RH.episode_units(ref["D"]["eid"])
    return uniq, inv


def seed_sets(rH, rU):
    """X (contact for RH, none for UE), S (both), Y (UE contact, none for RH), bool over windows"""
    cH, cU = rH["contact"], rU["contact"]
    return {"X": cH & ~cU, "S": cH & cU, "Y": cU & ~cH}


def all_tables(runs, seeds, J, inv, log_override=None):
    """tabs[m] = {'X': tableH, 'S': tableH, 'S_U': tableU, 'Y_U': tableU, 'sets': masks}; log_override[m] replaces the RH log fields (planted check)"""
    tabs = {}
    for m in seeds:
        rH, rU = runs[("H", m)], runs[("U", m)]
        st = seed_sets(rH, rU)
        lh = (log_override or {}).get(m, rH["log"])
        tabs[m] = {"sets": st,
                   "X": build_table(rH, lh, st["X"], J, inv), "S": build_table(rH, lh, st["S"], J, inv),
                   "S_U": build_table(rU, rU["log"], st["S"], J, inv), "Y_U": build_table(rU, rU["log"], st["Y"], J, inv)}
    return tabs


def pool(tabs, key, field):
    ep = np.concatenate([tabs[m][key]["ep"] for m in sorted(tabs)]) if tabs else np.zeros(0, np.int64)
    v = np.concatenate([tabs[m][key][field] for m in sorted(tabs)]) if tabs else np.zeros(0)
    return ep, v


# ============================================================================================ estimator (pattern of read_rulelog10k.py / read_labelunsafe10k.py)
def make_draws(n_ep, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n_ep, size=(n_boot, n_ep))


def _sums(ep, v, n_ep):
    v = np.asarray(v, np.float64)
    f = np.isfinite(v)
    return np.bincount(ep[f], weights=v[f], minlength=n_ep), np.bincount(ep[f], minlength=n_ep).astype(np.float64)


def mean_diff(epA, vA, epB, vB, n_ep, idx):
    """mean of A minus mean of B (B None: the mean of A), cluster bootstrap over the evaluation episodes with the shared draws idx, 95 percent percentile interval.
    -> dict(value, interval, n_a, n_b); value None when a group is empty"""
    sA, cA = _sums(epA, vA, n_ep)
    na = int(cA.sum())
    nb = None
    if vB is not None:
        sB, cB = _sums(epB, vB, n_ep)
        nb = int(cB.sum())
    if na == 0 or (vB is not None and nb == 0):
        return {"value": None, "interval": [None, None], "n_a": na, "n_b": nb}
    val = sA.sum() / cA.sum()
    a, ca = sA[idx].sum(1), cA[idx].sum(1)
    ok = ca > 0
    stat = np.full(len(idx), np.nan)
    stat[ok] = a[ok] / ca[ok]
    if vB is not None:
        val = val - sB.sum() / cB.sum()
        b, cb = sB[idx].sum(1), cB[idx].sum(1)
        ok = ok & (cb > 0)
        stat = np.where(ok, stat - np.where(cb > 0, b / np.maximum(cb, 1), np.nan), np.nan)
    if not ok.any():
        return {"value": float(val), "interval": [None, None], "n_a": na, "n_b": nb}
    lo, hi = np.percentile(stat[ok], [2.5, 97.5])
    return {"value": float(val), "interval": [float(lo), float(hi)], "n_a": na, "n_b": nb}


def paired_closing(tabs, field, n_ep, idx):
    """Q3: in S, RH minus UE closing rate, paired by window (both ticks finite)"""
    eps, ds = [], []
    for m in sorted(tabs):
        a, b = tabs[m]["S"], tabs[m]["S_U"]
        ka = {int(i): n for n, i in enumerate(a["i"])}
        for n, i in enumerate(b["i"]):
            q = ka.get(int(i))
            if q is None:
                continue
            d = a[field][q] - b[field][n]
            if math.isfinite(d):
                eps.append(int(b["ep"][n]))
                ds.append(d)
    return mean_diff(np.array(eps, np.int64), np.array(ds, np.float64), None, None, n_ep, idx)


def read_values(tabs, n_ep):
    idx = make_draws(n_ep, N_BOOT, BOOT_SEED)
    out = {}
    for q, field in (("Q1", "err"), ("Q2", "miss")):
        ex, vx = pool(tabs, "X", field)
        es, vs = pool(tabs, "S", field)
        out[q] = mean_diff(ex, vx, es, vs, n_ep, idx)
    out["Q3"] = paired_closing(tabs, "clos", n_ep, idx)
    eu, vu = pool(tabs, "Y_U", "err")
    esu, vsu = pool(tabs, "S_U", "err")
    out["beside_Q1_UE_contact_only"] = mean_diff(eu, vu, esu, vsu, n_ep, idx)
    out["beside_Q3_ego_minus_imagined_lead_speed"] = paired_closing(tabs, "clos2", n_ep, idx)
    out["counts"] = {str(m): {"X": int(tabs[m]["sets"]["X"].sum()), "S": int(tabs[m]["sets"]["S"].sum()), "Y": int(tabs[m]["sets"]["Y"].sum())} for m in sorted(tabs)}
    out["n_X"] = int(sum(tabs[m]["sets"]["X"].sum() for m in tabs))
    out["n_S"] = int(sum(tabs[m]["sets"]["S"].sum() for m in tabs))
    out["n_Y"] = int(sum(tabs[m]["sets"]["Y"].sum() for m in tabs))
    return out


# ============================================================================================ names
def _pos(c):
    lo = c["interval"][0]
    return lo is not None and lo > 0


def _notsep(c):
    lo, hi = c["interval"]
    return lo is not None and hi is not None and lo <= 0 <= hi


def derive_name(v):
    q1, q2, q3 = v["Q1"], v["Q2"], v["Q3"]
    if v["n_X"] < MIN_X:
        return N_UNRES, "|X| = %d over the model seeds, below %d windows: too small to read" % (v["n_X"], MIN_X)
    if _pos(q1) or _pos(q2):
        return N_MISREAD, "Q1 or Q2 above zero with an interval excluding zero"
    if _notsep(q1) and _notsep(q2) and _pos(q3):
        return N_SAME, "Q1 and Q2 not separated, Q3 above zero with an interval excluding zero"
    return N_UNRES, "no name condition met; |X| = %d" % v["n_X"]


# ============================================================================================ controls
def ctl_c1(runs, seeds):
    """contact windows recomputed independently equal the contact cells of the landed headway reader, per run; same windows in the two runs of a seed"""
    H = bind_headway()
    bad, n_cells = [], 0
    for key, r in sorted(runs.items()):
        D, L = r["D"], r["L"]
        G, ns, lead = L["G"], L["ns"], L["lead"]
        kk = np.arange(1, H.K + 1)[None, :]
        M = lead[:, None] & (kk <= ns[:, None])
        gm = np.where(M, G[:, 1:], np.nan) + H.logged_arc(D["g"]) - H.progress(D["g"], D["cl"])
        with np.errstate(invalid="ignore"):
            ct = M & (gm <= 0.0)
        cell = r["C"]["C"]
        by_cell = np.isfinite(cell) & (cell == 1.0)
        n_cells += int(by_cell.sum())
        first2 = np.where(ct.any(1), np.argmax(ct, axis=1) + 1, 0)
        if not (np.array_equal(ct.any(1), by_cell) and np.array_equal(r["contact"], by_cell) and np.array_equal(first2, r["kc"])):
            bad.append("%s_s%d" % key)
    for m in seeds:
        a, b = runs[("H", m)]["D"], runs[("U", m)]["D"]
        if not all(np.array_equal(a[k], b[k]) for k in ("g", "ws", "eid", "clip_index", "v0")):
            bad.append("windows of seed %d differ between the two runs" % m)
    return {"PASS": not bad, "n_contact_cells": n_cells, "runs_differing": bad}


def sample_rows(tabs_by_key, seed=BOOT_SEED, n=N_SAMPLE):
    """a planted sample of ticks of X and S windows (key m, set, position); deterministic"""
    pool_ = [(m, s, q) for (m, s), T in sorted(tabs_by_key.items()) for q in range(len(T["i"]))]
    rng = np.random.default_rng(seed)
    if len(pool_) <= n:
        return pool_
    sel = rng.choice(len(pool_), size=n, replace=False)
    return [pool_[int(i)] for i in sorted(sel)]


def ctl_c2(runs, seeds, tabs, J):
    """every X and S window has a tick; the logged g_src of a planted sample of 20 ticks equals g0 - a_l + a_m with a_l, a_m recomputed from the dump, within 1e-3 m"""
    H = bind_headway()
    probs = []
    n_win = 0
    tk = {}
    for m in seeds:
        for s in ("X", "S"):
            T = tabs[m][s]
            probs += T["problems"]
            if len(T["i"]) != int(tabs[m]["sets"][s].sum()):
                probs.append("a window of %s has no usable tick" % s)
            n_win += len(T["i"])
            tk[(m, s)] = T
    rows = sample_rows(tk)
    worst = 0.0
    n_ok = 0
    for (m, s, q) in rows:
        T, r = tk[(m, s)], runs[("H", m)]
        i, row, k = int(T["i"][q]), int(T["row"][q]), int(T["k"][q])
        lg = r["log"]
        D = r["D"]
        g1 = D["g"][i:i + 1]
        a_l = 0.0 if k == 0 else float(H.logged_arc(g1)[0, min(max(k - 1, 0), 29)])
        a_m = 0.0 if k == 0 else float(H.progress(g1, D["cl"][i:i + 1])[0, k - 1])
        g0, gs = float(lg["g0"][row]), float(lg["g_src"][row])
        if not (math.isfinite(g0) and math.isfinite(gs)):
            continue
        dev = max(abs((g0 - a_l + a_m) - gs), abs(float(lg["a_m"][row]) - a_m), abs(float(lg["a_l"][row]) - a_l))
        worst = max(worst, dev)
        n_ok += 1
    if n_ok == 0:
        probs.append("no tick of the planted sample could be recomputed")
    if worst > TOL_G:
        probs.append("a planted tick differs from the recomputed gap by more than %g m" % TOL_G)
    return {"PASS": not probs, "n_windows_X_S": n_win, "n_sample": len(rows), "n_sample_checked": n_ok, "max_abs_dev_m": worst, "tol_m": TOL_G, "problems": sorted(set(probs))}


def plant_logs(runs, seeds, tabs):
    """copies of the RH rule logs with g_src at the tick of every X window = label gap + 2.0 m and at the tick of every S window = label gap (gap error 0)"""
    out = {}
    for m in seeds:
        lg = {k: np.array(v) for k, v in runs[("H", m)]["log"].items()}
        lg["g_src"] = lg["g_src"].astype(np.float64)
        lg["present"] = lg["present"].astype(bool)
        for s, add in (("X", PLANT_M), ("S", 0.0)):
            T = tabs[m][s]
            for q in range(len(T["i"])):
                if T["lab_valid"][q]:
                    lg["g_src"][T["row"][q]] = T["lab_gap"][q] + add
                    lg["present"][T["row"][q]] = True
        out[m] = lg
    return out


def ctl_c3(runs, seeds, tabs, J, inv, n_ep, q1_main):
    """planted: gap error +2.0 m in X and 0 in S reads Q1 = +2.0 within 1e-6; the unchanged logs read the main Q1"""
    if sum(int(tabs[m]["sets"]["X"].sum()) for m in seeds) == 0 or sum(int(tabs[m]["sets"]["S"].sum()) for m in seeds) == 0:
        return {"PASS": False, "reason": "X or S is empty, the planted check cannot run"}
    idx = make_draws(n_ep, N_BOOT, BOOT_SEED)
    pt = all_tables(runs, seeds, J, inv, plant_logs(runs, seeds, tabs))
    ex, vx = pool(pt, "X", "err")
    es, vs = pool(pt, "S", "err")
    q = mean_diff(ex, vx, es, vs, n_ep, idx)
    nc = all_tables(runs, seeds, J, inv)
    ex2, vx2 = pool(nc, "X", "err")
    es2, vs2 = pool(nc, "S", "err")
    q0 = mean_diff(ex2, vx2, es2, vs2, n_ep, idx)
    planted_ok = q["value"] is not None and abs(q["value"] - PLANT_M) <= 1e-6
    same_ok = (q0["value"] is None and q1_main["value"] is None) or (q0["value"] is not None and q1_main["value"] is not None and abs(q0["value"] - q1_main["value"]) <= 1e-12)
    return {"PASS": bool(planted_ok and same_ok), "planted_reads_plus_2": bool(planted_ok), "unchanged_reads_main_value": bool(same_ok),
            "planted_deviation": None if q["value"] is None else abs(q["value"] - PLANT_M)}


def ctl_c4(tests_out):
    """the output file of the tool tests on the dev box: passed, nothing failed or erroring"""
    if not tests_out or not os.path.isfile(tests_out):
        return {"PASS": False, "reason": "no tests output file given"}
    with open(tests_out, encoding="utf-8", errors="replace") as f:
        t = f.read()
    mp = re.search(r"(\d+) passed", t)
    bad = re.search(r"(\d+) (failed|error)", t, re.I) is not None or re.search(r"^(FAILED|ERROR) ", t, re.M) is not None
    return {"PASS": bool(mp and not bad), "n_passed": int(mp.group(1)) if mp else 0, "tests_out_sha256": sha_file(tests_out)}


# ============================================================================================ stages
def fmt(x, nd=4):
    return "n/a" if x is None else ("%+.*f" % (nd, x))


def fmt_iv(c):
    return "%s [%s, %s] (n_a=%s, n_b=%s)" % (fmt(c["value"]), fmt(c["interval"][0]), fmt(c["interval"][1]), c["n_a"], c["n_b"])


def verdict_md(res):
    L = ["# %s verdict (card %s)" % (PRE, CARD), "", "Tier %s." % TIER, "",
         "Controls: " + ", ".join("%s %s" % (k, "PASS" if v["PASS"] else "FAIL") for k, v in res["controls"].items()), ""]
    if res["values"] is None:
        L += ["A control failed: the values are withheld and no name is given. See %s." % CONTROLS_NAME, ""]
    else:
        v = res["values"]
        L += ["Name: **%s** (%s)" % (res["name"], res["name_reason"]), "",
              "Counts: |X| = %d, |S| = %d, |Y| = %d; per seed %s" % (v["n_X"], v["n_S"], v["n_Y"], json.dumps(v["counts"])), "",
              "* Q1 misread (mean signed gap error X(RH) minus S(RH), m, positive = reader sees the lead farther than it is): %s" % fmt_iv(v["Q1"]),
              "* Q2 presence misses (share X(RH) minus S(RH)): %s" % fmt_iv(v["Q2"]),
              "* Q3 closing rate S, RH minus UE, paired by window (m/s, positive = RH closes faster; -c_src): %s" % fmt_iv(v["Q3"]),
              "* beside, Q1 where UE makes contact and RH does not (UE reader, Y minus S): %s" % fmt_iv(v["beside_Q1_UE_contact_only"]),
              "* beside, Q3 with ego speed minus the rule lead speed: %s" % fmt_iv(v["beside_Q3_ego_minus_imagined_lead_speed"]), ""]
    L += ["Estimator: episode-cluster bootstrap, %d draws, seed %d, 95 percent percentile interval; it answers what another draw of episodes would say." % (N_BOOT, BOOT_SEED), "",
          "Four-family line:"] + ["* " + f for f in FAMILIES] + ["", "Limits: " + LIMITS, ""]
    return "\n".join(L)


def stage_read(a):
    seeds = [int(x) for x in a.seeds.split(",")]
    bind_headway(a.headway_dir)
    for p in (a.join, a.root):
        if not os.path.exists(p):
            print("REFUSED: an input path is missing")
            return 3
    os.makedirs(a.out_dir, exist_ok=True)
    cpath, rpath, vpath = (os.path.join(a.out_dir, n) for n in (CONTROLS_NAME, RESULT_NAME, VERDICT_NAME))
    for p in (cpath, rpath, vpath):
        if os.path.exists(p):
            print("REFUSED: an output file exists and is never overwritten")
            return 3
    try:
        J = HW.load_join(a.join)
        runs = {}
        for m in seeds:
            runs[("H", m)] = load_run(a.root, TAG_H, m, J, a.expect_files)
            runs[("U", m)] = load_run(a.root, TAG_U, m, J, a.expect_files)
    except LoadError as e:
        print("REFUSED: %s" % e)
        return 3
    uniq, inv = episodes_of(runs)
    n_ep = len(uniq)
    tabs = all_tables(runs, seeds, J, inv)
    ctl = {"c1": ctl_c1(runs, seeds), "c2": ctl_c2(runs, seeds, tabs, J)}
    q1_main = mean_diff(*pool(tabs, "X", "err"), *pool(tabs, "S", "err"), n_ep, make_draws(n_ep, N_BOOT, BOOT_SEED))
    ctl["c3"] = ctl_c3(runs, seeds, tabs, J, inv, n_ep, q1_main)
    ctl["c4"] = ctl_c4(a.tests_out)
    ok = all(v["PASS"] for v in ctl.values())
    write_new(cpath, json.dumps(clean({"card": CARD, "all_pass": ok, "controls": ctl, "seeds": seeds, "join_sha256": sha_file(a.join)}), indent=1))
    for k, v in ctl.items():
        print("ctl %s %s" % (k, "PASS" if v["PASS"] else "FAIL"))
    res = {"card": CARD, "tier": TIER, "controls": ctl, "values": None, "name": None, "name_reason": None, "families": FAMILIES, "limits": LIMITS}
    if ok:
        vals = read_values(tabs, n_ep)
        res["values"] = vals
        res["name"], res["name_reason"] = derive_name(vals)
    write_new(rpath, json.dumps(clean(res), indent=1))
    write_new(vpath, verdict_md(res))
    print("ZZ READ-CONTACTRH10K all_pass %s name %s" % (ok, res["name"]))
    return 0 if ok else 2


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    r = s.add_parser("read")
    r.add_argument("--root", required=True)
    r.add_argument("--join", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--tests-out", default=None)
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--expect-files", type=int, default=141)
    r.add_argument("--headway-dir", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return stage_read(a)
    except Exception as e:  # fail loud, one line
        print("ERROR: %s: %s" % (type(e).__name__, e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
