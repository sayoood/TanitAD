"""read_longroll10k.py -- card PREREG_V8F_PROOF_ADD70 (LONGROLL10K) with note N1 (wrapper of record rzgfhq): the read. numpy; the engines of the earlier cards are imported unchanged and driven in memory
(headway_read_r2.py: window_cells, label_windows, pair_cells, spread_of, run_controls; q3_read_r3.py: pair_cells, cell_rule; hierrep_read.py: comps, seed_arrays, read_cell; and strat_roll_read.py of
card ADD66 for the STRATEGIC family). The stock command lines of the headway and family readers are NOT used: they hold the window count of the full grid (2,399) in their own controls.
RESTRICTION: in memory. Every run is cut to the (clip, t0) windows of the grid of the pair (dump arrays per episode file, tick-file rows, rule-log rows by the (ci, t0) columns), in the order of the
grid; a window missing or present twice in a run is a control failure (l4); a run that is a P run must hold exactly the windows of its grid. Nothing is written but the result file and its log.
Names: LONGROLL10K-<PAIR>-<FAMILY>-BETTER|WORSE|MIXED|NOT-SEPARATED and LONGROLL10K-<PAIR>-U-LOWER|HIGHER|NOT-SEPARATED (pairs of section 5, at P = 30 only). The P = 60 pairs, the difference of the two
differences, the rule under K at age, the offsets, the levels and the counts name nothing.
Controls e1, l1 to l12 first; if one fails nothing is named and no level or difference is printed (the controls are printed as written). No training, no rollout. Never run on data by the tool builder.
Tier line: T1-kin (kinematic rollout on the model's own actions, logged frames); never driving performance.
Usage: python read_longroll10k.py --read-dir D --pull PULL --reused LONGROLL10K_REUSED.json --code CODEDIR --grid-dir GDIR --grid-full G.json --wrapper W.py --parent-wrapper PW.py --test-wrapper W2.py
       --check-preroll C.txt --e1 ENV_E1.json --add67-json READ_HEADBOUND10K.json [--add67-hw RESULT_HEADWAY.json --add67-q3 RESULT_Q3.json] [--pod-hashes H.txt] --w W --tree TREE --out OUT.json
       python read_longroll10k.py --selftest      (synthetic arrays only, no data file; last line ZZ SELFTEST-LONGROLL ok: True|False)"""
import argparse
import glob
import hashlib
import io
import itertools
import json
import math
import os
import re
import sys

try:
    import numpy as np
except ImportError:                                                                          # --selftest needs numpy too, but a clear message is better than a traceback
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CARD, PRE = "PREREG_V8F_PROOF_ADD70", "LONGROLL10K"
SEEDS = (0, 1)
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; logged camera frames; the lead does not react); never driving performance"
TOL = 1e-6                                                                                   # tolerance of l3, l10, l12 (fixed by the smoke: d = 0, tolerance = max(10 d, 1e-6))
TOL_ID = 1e-9                                                                                # control l6
CKPT_SHA_PREFIX = {0: "2fd4b747532f55b2", 1: "f13df8ce026eed31"}                       # card ADD46: UE_s0, UE_s1
PARENT_SHA_PREFIX = "373ffe72f2655c35"                                                 # v8f_t1_eval_reader_rule_rzgfh.py, unchanged since card ADD67
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
NAMED = {"LONGITUDINAL": ["LONGITUDINAL:speed_mae_mps", "LONGITUDINAL:along_mae_m"], "LATERAL": ["LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps"], "TACTICAL": ["TACTICAL:ade_tac_m"]}
CARD_COUNTS = {"30": 1271, "60": 284}
# run (without the prefix I10) -> form, readout label, P, nav, grid key, preroll exec, rule interval, kappa (None = no bound)
RUNS = {}
for _p, _g in ((30, "30"), (60, "60")):
    RUNS["R0P%dHb20" % _p] = ("R0", "pinned", _p, "deadreckon", _g, "plan", None, 0.2)
    RUNS["R0KP%dHb20" % _p] = ("R0", "K", _p, "deadreckon", _g, "plan", None, 0.2)
    RUNS["RLcKF5P%dHb20" % _p] = ("RLc", "K", _p, "deadreckon", _g, "plan", 5, 0.2)
    RUNS["RQ50KF5P%dHb20" % _p] = ("RR", "K", _p, "deadreckon", _g, "plan", 5, 0.2)
RUNS["R0KNlP30Hb20"] = ("R0", "K", 30, "logged", "30", "plan", None, 0.2)
RUNS["R0KNlHb20"] = ("R0", "K", 0, "logged", "30", "plan", None, 0.2)
CTRL_RUNS = {"R0LrID": ("R0", "pinned", 0, "deadreckon", "full", "plan", None, 0.2), "R0LrCarry": ("R0", "pinned", 30, "logged", "30", "logged", None, 0.2),
             "R0LrFresh": ("R0", "pinned", 0, "logged", "30", "plan", None, 0.2)}
BANKED = {"R0Hb20": ("R0", "pinned", 0, "deadreckon", "full", "plan", None, 0.2), "R0KHb20": ("R0", "K", 0, "deadreckon", "full", "plan", None, 0.2),
          "RLcKF5Hb20": ("RLc", "K", 0, "deadreckon", "full", "plan", 5, 0.2), "RQ50KF5Hb20": ("RR", "K", 0, "deadreckon", "full", "plan", 5, 0.2),
          "R0": ("R0", "pinned", 0, "deadreckon", "full", "plan", None, None)}
ALL_RUNS = dict(RUNS)
ALL_RUNS.update(CTRL_RUNS)
ALL_RUNS.update(BANKED)
NEW_RUN_LIST = [(r, k) for r in RUNS for k in SEEDS] + [(r, 0) for r in CTRL_RUNS]            # 20 outcome runs + 3 control runs
# pair key -> (A, B, grid); named at P = 30
NAMED_PAIRS = {"AGE-CONV": ("R0P30Hb20", "R0Hb20"), "AGE-K": ("R0KP30Hb20", "R0KHb20"), "AGE-LABEL-CAND": ("RLcKF5P30Hb20", "RLcKF5Hb20"), "AGE-READER-CAND": ("RQ50KF5P30Hb20", "RQ50KF5Hb20"),
               "K-AT-AGE": ("R0KP30Hb20", "R0P30Hb20"), "LABEL-CAND-AT-AGE": ("RLcKF5P30Hb20", "R0P30Hb20"), "READER-CAND-AT-AGE": ("RQ50KF5P30Hb20", "R0P30Hb20"),
               "AGE-K-NAVLOGGED": ("R0KNlP30Hb20", "R0KNlHb20")}
P60_PAIRS = {k + "@P60": (A.replace("P30", "P60"), B.replace("P30", "P60") if "P30" in B else B) for k, (A, B) in NAMED_PAIRS.items() if k != "AGE-K-NAVLOGGED"}
FRESH_PAIRS = {"K-AT-FRESH": ("R0KHb20", "R0Hb20"), "LABEL-CAND-AT-FRESH": ("RLcKF5Hb20", "R0Hb20"), "READER-CAND-AT-FRESH": ("RQ50KF5Hb20", "R0Hb20")}
DOD = {"K-AT-AGE": ("K-AT-AGE", "K-AT-FRESH"), "LABEL-CAND-AT-AGE": ("LABEL-CAND-AT-AGE", "LABEL-CAND-AT-FRESH"), "READER-CAND-AT-AGE": ("READER-CAND-AT-AGE", "READER-CAND-AT-FRESH")}
RULE_AT_AGE = {"LABEL-RULE-UNDER-K-AT-AGE": ("RLcKF5P30Hb20", "R0KP30Hb20"), "READER-RULE-UNDER-K-AT-AGE": ("RQ50KF5P30Hb20", "R0KP30Hb20")}
SEC6 = {"A": "the convention's errors grow with rollout age and the candidate's do not; this goes into the candidate specification (recommendation A of decision card D16), and reporting the read at age 6 to 12 s "
             "beside the 6 s windows is proposed to the PI and the other session",
        "C": "the model's reading of its own rolled history costs at 2.0 s too; the AGE read is then the acceptance check for the training line of recommendation C: an arm trained on its own rolled history "
             "must read NOT-SEPARATED or BETTER on every family and U NOT-SEPARATED or LOWER where the line models read WORSE or HIGHER here",
        "HOLDS": "the candidate's advantage holds at age 6 to 12 s", "LIMIT": "a family reads WORSE at age: the specification states that limit",
        "NAV_ROUTE": "that cost is the held route input (a route supply during the rollout is then the lever, not the state)", "NAV_STATE": "WORSE in both: the own-motion state",
        "NONE": "no cost of rollout age is measured at 6 to 12 s on the windows of the grid and two model seeds; stated as such, with the intervals",
        "BETTER_NOTE": "a pre-rolled window that reads BETTER than a fresh one is reported as written and decides nothing"}


# ============================================================================================ small helpers
def jl(p):
    try:
        return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None
    except Exception:                                                                           # noqa: BLE001
        return None


def sha(p):
    if not p or not os.path.isfile(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def txt(p):
    return io.open(p, encoding="utf-8", errors="replace").read() if p and os.path.exists(p) else ""


def conv(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, np.ndarray):
        return conv(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def key_of(run, k):
    return "I10%s_s%d" % (run, k)


def run_dir(pull, run, k):
    return os.path.join(pull, key_of(run, k))


def tick_dir(pull, run, k):
    return os.path.join(run_dir(pull, run, k), "ticks")


def dump_dir(pull, run, k):
    return os.path.join(run_dir(pull, run, k), "dump_%s_m10_causal_heads" % key_of(run, k))


def npz_load(path):
    return np.load(path, allow_pickle=True) if os.path.isfile(path) else None


def maxdiff(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return float("inf")
    if x.dtype.kind in "OUS" or y.dtype.kind in "OUS":
        return 0.0 if np.array_equal(x, y) else float("inf")
    x, y = x.astype(np.float64), y.astype(np.float64)
    if not np.array_equal(np.isfinite(x), np.isfinite(y)):
        return float("inf")
    f = np.isfinite(x)
    return float(np.max(np.abs(x[f] - y[f]))) if f.any() else 0.0


def cmp_npz_maps(za, zb):
    """largest absolute difference of every array two npz objects hold; (max over keys, keys in one only, per-key dict)"""
    ka, kb = set(za.files), set(zb.files)
    per = {k: maxdiff(za[k], zb[k]) for k in sorted(ka & kb)}
    return (max(per.values()) if per else float("inf")), sorted(ka ^ kb), per


def cmp_dump_dirs(da, db):
    """the dump files of two runs compared array by array: (n files a, n files b, files differing, largest difference, names differing)"""
    fa, fb = sorted(glob.glob(os.path.join(da, "ep*.npz"))), sorted(glob.glob(os.path.join(db, "ep*.npz")))
    if [os.path.basename(x) for x in fa] != [os.path.basename(x) for x in fb]:
        return len(fa), len(fb), None, float("inf"), ["file lists differ"]
    bad, mx = [], 0.0
    for x, y in zip(fa, fb):
        d, only, _ = cmp_npz_maps(np.load(x, allow_pickle=True), np.load(y, allow_pickle=True))
        if only:
            d = float("inf")
        mx = max(mx, d)
        if d > 0.0:
            bad.append(os.path.basename(x))
    return len(fa), len(fb), len(bad), mx, bad


# ============================================================================================ restriction (in memory)
def grid_keys(grid, clip_names):
    """the (clip index, t0) windows of a grid in the grid's order: [N, 2] int64; refuses a clip the join does not know"""
    idx = {str(c): i for i, c in enumerate(clip_names)}
    out = []
    for c in grid["order"]:
        if str(c) not in idx:
            raise ValueError("clip %s of the grid is not in the clip list" % c)
        out += [(idx[str(c)], int(t)) for t in grid["windows"][c]]
    return np.asarray(out, np.int64).reshape(-1, 2)


def restrict_idx(ci, ws, keys):
    """rows of (ci, ws) that are the windows of `keys`, in the order of `keys`. Returns (idx[N] with -1 where a window is missing, report). A window present twice takes its first row
    and is counted; rows of windows not in `keys` are counted as extra."""
    ci, ws = np.asarray(ci, np.int64), np.asarray(ws, np.int64)
    pos = {}
    for r, (c, w) in enumerate(zip(ci.tolist(), ws.tolist())):
        pos.setdefault((c, w), []).append(r)
    kset = {(int(c), int(w)) for c, w in keys.tolist()}
    idx = np.full(len(keys), -1, np.int64)
    dup = 0
    for i, (c, w) in enumerate(keys.tolist()):
        rows = pos.get((c, w))
        if rows:
            idx[i] = rows[0]
            dup += int(len(rows) > 1)
    extra = int(sum(len(v) for k, v in pos.items() if k not in kset))
    rep = {"n_keys": int(len(keys)), "n_rows": int(len(ci)), "found": int((idx >= 0).sum()), "missing": int((idx < 0).sum()), "duplicated_windows": int(dup), "extra_rows": extra}
    rep["exact"] = bool(rep["missing"] == 0 and rep["duplicated_windows"] == 0)
    rep["holds_exactly"] = bool(rep["exact"] and extra == 0)
    return idx, rep


def restrict_ticks(T, keys):
    """tick-file rows (fields of strat_roll_read.FIELDS) of the windows of `keys`: every row-aligned field cut by the (ci, t0) of its row; `win` renumbered to the position of the window in `keys`
    (so that the file reads like a file of the grid). Returns (T2, report). A window present twice at a tick is not removed: its win repeats and the structure control fails."""
    ci, t0 = np.asarray(T["ci"], np.int64), np.asarray(T["t0"], np.int64)
    posn = {(int(c), int(w)): i for i, (c, w) in enumerate(keys.tolist())}
    n = len(ci)
    p = np.array([posn.get((int(c), int(w)), -1) for c, w in zip(ci, t0)], np.int64)
    keep = p >= 0
    T2 = {}
    for k, v in T.items():
        T2[k] = v[keep] if (k != "clip_names" and np.ndim(v) >= 1 and len(v) == n) else v
    T2["win"] = p[keep]
    return T2, {"rows_before": int(n), "rows_after": int(keep.sum())}


def keep_rows_of_log(z, keys, fields=("ci", "t0")):
    """a rule log (npz) cut by its (ci, t0) columns: dict of the row-aligned arrays kept, and the number of rows before and after"""
    ci, t0 = np.asarray(z[fields[0]], np.int64), np.asarray(z[fields[1]], np.int64)
    kset = {(int(c), int(w)) for c, w in keys.tolist()}
    keep = np.array([(int(c), int(w)) in kset for c, w in zip(ci, t0)], bool)
    return {k: np.asarray(z[k])[keep] for k in z.files if np.ndim(z[k]) >= 1 and len(z[k]) == len(ci)}, int(len(ci)), int(keep.sum())


# ============================================================================================ rule logic: families, names, consequences
def family_of(vs):
    if all(v == "BETTER" for v in vs):
        return "BETTER"
    if all(v == "WORSE" for v in vs):
        return "WORSE"
    if not any(v in ("BETTER", "WORSE") for v in vs):
        return "NOT-SEPARATED"
    return "MIXED"


def reading(cells, fam):
    return family_of([cells[c]["verdict"] for c in NAMED[fam]])


def below(d):
    return bool(all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0)


def above(d):
    return bool(all(x > 0 for x in d["per_seed"]) and d["interval"][0] > 0)


def u_name(u):
    return "LOWER" if below(u) else ("HIGHER" if above(u) else "NOT-SEPARATED")


def is_bad(rd):
    """a family WORSE or U HIGHER (MIXED counts as neither)"""
    return bool(any(rd["fam"][f] == "WORSE" for f in FAMS) or rd["U"] == "HIGHER")


def consequences(R):
    """section 6 of the card, by fixed rules. R[pair] = {"fam": {family: reading}, "U": LOWER|HIGHER|NOT-SEPARATED} for the eight named pairs. Returns a list of (case, sentence)."""
    out = []
    conv_bad = is_bad(R["AGE-CONV"])
    cands = ("AGE-LABEL-CAND", "AGE-READER-CAND")
    cand_bad = [c for c in cands if is_bad(R[c])]
    if conv_bad and not cand_bad:
        out.append(("A", "AGE-CONV with a family WORSE or U HIGHER, and neither candidate's AGE pair: " + SEC6["A"]))
    for c in cand_bad:
        out.append(("C", "%s with a family WORSE or U HIGHER: %s" % (c, SEC6["C"])))
    for p, lab in (("LABEL-CAND-AT-AGE", "LABEL-CAND"), ("READER-CAND-AT-AGE", "READER-CAND")):
        rd = R[p]
        worse = [f for f in FAMS if rd["fam"][f] == "WORSE"]
        better = [f for f in FAMS if rd["fam"][f] == "BETTER"]
        if not worse and better and rd["U"] != "HIGHER":
            out.append(("HOLDS", "%s: %s (BETTER on %s, no family WORSE, U %s)" % (p, SEC6["HOLDS"], ",".join(better), rd["U"])))
        elif worse:
            out.append(("LIMIT", "%s: %s (WORSE on %s)" % (p, SEC6["LIMIT"], ",".join(worse))))
    k, nl = R["AGE-K"], R["AGE-K-NAVLOGGED"]
    for f in FAMS:
        if k["fam"][f] == "WORSE":
            if nl["fam"][f] == "NOT-SEPARATED":
                out.append(("NAV_ROUTE", "family %s: AGE-K WORSE, AGE-K-NAVLOGGED NOT-SEPARATED: %s" % (f, SEC6["NAV_ROUTE"])))
            elif nl["fam"][f] == "WORSE":
                out.append(("NAV_STATE", "family %s: %s" % (f, SEC6["NAV_STATE"])))
    ages = ("AGE-CONV", "AGE-K", "AGE-LABEL-CAND", "AGE-READER-CAND")
    if all(R[p]["U"] == "NOT-SEPARATED" and all(R[p]["fam"][f] == "NOT-SEPARATED" for f in FAMS) for p in ages):
        out.append(("NONE", "all AGE pairs NOT-SEPARATED on every family and U: " + SEC6["NONE"]))
    for p in ages:
        if any(R[p]["fam"][f] == "BETTER" for f in FAMS):
            out.append(("BETTER_NOTE", "%s has a family BETTER: %s" % (p, SEC6["BETTER_NOTE"])))
    return out


def off_stats(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def dod_arrays(Xp, Rp, Xf, Rf):
    """difference of two differences per window: (Xp - Rp) - (Xf - Rf), as the two lists the engine subtracts (x - y): x = Xp - Rp, y = Xf - Rf (NaN wherever any of the four is NaN)"""
    x = np.asarray(Xp, np.float64) - np.asarray(Rp, np.float64)
    y = np.asarray(Xf, np.float64) - np.asarray(Rf, np.float64)
    bad = ~(np.isfinite(x) & np.isfinite(y))
    return np.where(bad, np.nan, x), np.where(bad, np.nan, y)


def g0_identity(L, tol=TOL):
    """first-tick rows (k == 0) of a rule log: a_m and a_l are the definition 0 and g0 equals g_src (NaN where g_src is NaN): (n rows, largest |g0 - g_src|, largest |a_m|, ok)"""
    k = np.asarray(L["k"], np.int64)
    m = k == 0
    g0, gs = np.asarray(L["g0"], np.float64)[m], np.asarray(L["g_src"], np.float64)[m]
    am, al = np.asarray(L["a_m"], np.float64)[m], np.asarray(L["a_l"], np.float64)[m]
    d = maxdiff(g0, gs)
    ok = bool(m.any() and d <= tol and (np.abs(am[np.isfinite(am)]).max() if np.isfinite(am).any() else 0.0) <= tol and (np.abs(al[np.isfinite(al)]).max() if np.isfinite(al).any() else 0.0) <= tol)
    return int(m.sum()), d, ok


# ============================================================================================ selftest (synthetic arrays only)
def selftest():
    res = []

    def chk(name, ok):
        res.append((name, bool(ok)))
        print("   %-100s %s" % (name, bool(ok)))
    print(TIER_LINE)
    clips = ["c0", "c1", "c2", "c3"]
    full = {"order": clips, "windows": {"c0": [3, 7, 11, 15], "c1": [3, 7, 11], "c2": [3, 7], "c3": [3]}}
    g30 = {"order": ["c0", "c1"], "windows": {"c0": [7, 11, 15], "c1": [7, 11]}}
    keys = grid_keys(g30, clips)
    chk("grid_keys: 5 windows, order of the grid, clip index kept", keys.tolist() == [[0, 7], [0, 11], [0, 15], [1, 7], [1, 11]])
    try:
        grid_keys({"order": ["zz"], "windows": {"zz": [3]}}, clips)
        chk("grid_keys refuses a clip the clip list does not know", False)
    except ValueError:
        chk("grid_keys refuses a clip the clip list does not know", True)
    fk = grid_keys(full, clips)
    ci, ws = fk[:, 0], fk[:, 1]
    idx, rep = restrict_idx(ci, ws, keys)
    chk("restriction of the full run: 5 found, 0 missing, 0 twice, 5 extra rows, exact but not 'holds_exactly'", rep["found"] == 5 and rep["missing"] == 0 and rep["duplicated_windows"] == 0 and rep["extra_rows"] == 5 and rep["exact"] and not rep["holds_exactly"])
    chk("restriction returns the rows in the order of the grid", ci[idx].tolist() == [0, 0, 0, 1, 1] and ws[idx].tolist() == [7, 11, 15, 7, 11])
    ci2, ws2 = ci[idx], ws[idx]
    _, rep2 = restrict_idx(ci2, ws2, keys)
    chk("a P run holding exactly its grid: holds_exactly", rep2["holds_exactly"] and rep2["extra_rows"] == 0)
    _, rep3 = restrict_idx(ci2[:-1], ws2[:-1], keys)
    chk("PLANTED: a window dropped is detected (missing 1, not exact)", rep3["missing"] == 1 and not rep3["exact"])
    _, rep4 = restrict_idx(np.r_[ci2, ci2[0]], np.r_[ws2, ws2[0]], keys)
    chk("PLANTED: a window present twice is detected (duplicated 1, not exact)", rep4["duplicated_windows"] == 1 and not rep4["exact"])
    _, rep5 = restrict_idx(np.r_[ci2, 3], np.r_[ws2, 3], keys)
    chk("PLANTED: a window of another grid in a P run is detected (extra 1, not holds_exactly)", rep5["extra_rows"] == 1 and not rep5["holds_exactly"])
    T = {"ci": np.tile(fk[:, 0], 3), "t0": np.tile(fk[:, 1], 3), "tick": np.repeat(np.arange(3), len(fk)), "win": np.tile(np.arange(len(fk)), 3), "clip_names": np.array(clips),
         "pose": np.arange(3 * len(fk) * 4, dtype=float).reshape(-1, 4)}
    T2, st = restrict_ticks(T, keys)
    chk("tick rows: 3 x 5 kept of 3 x 10, row-aligned field cut with the rows, clip_names untouched", st["rows_after"] == 15 and len(T2["pose"]) == 15 and len(T2["clip_names"]) == 4)
    chk("tick rows: win renumbered 0..4 at every tick, each window once per tick", all(sorted(T2["win"][T2["tick"] == k].tolist()) == list(range(5)) for k in range(3)))
    Td = {k: (np.r_[v, v[1:2]] if (k != "clip_names") else v) for k, v in T.items()}
    Td2, _ = restrict_ticks(Td, keys)
    chk("PLANTED: a tick row present twice leaves a repeated win (the structure control sees it)", sorted(Td2["win"][Td2["tick"] == 0].tolist()) != list(range(5)) or len(Td2["win"]) != 15)
    # rule-log cut and the first-tick identity
    L = {"ci": np.array([0, 0, 1, 1]), "t0": np.array([7, 7, 7, 3]), "k": np.array([0, 10, 0, 0]), "g_src": np.array([5.0, 6.0, np.nan, 2.0]), "g0": np.array([5.0, 9.0, np.nan, 2.0]),
         "a_m": np.zeros(4), "a_l": np.zeros(4)}
    zz = type("Z", (), {"files": list(L), "__getitem__": lambda s, k: L[k]})()
    kept, nb, na = keep_rows_of_log(zz, keys)
    chk("rule log cut by (ci, t0): 3 of 4 rows kept (the row of clip 1, t0 3, is outside)", nb == 4 and na == 3 and len(kept["g0"]) == 3)
    chk("g0 identity at first-tick rows (k == 0): g0 == g_src, NaN rows equal, a_m == 0", g0_identity(L)[2])
    L2 = dict(L)
    L2["g0"] = np.array([4.5, 9.0, np.nan, 2.0])
    chk("PLANTED: g0 shifted by an offset at a first-tick row (as a rule log that stored the used value) is detected", not g0_identity(L2)[2])
    # tolerances
    a, b = np.array([1.0, 2.0, np.nan]), np.array([1.0 + 5e-7, 2.0, np.nan])
    chk("maxdiff: 5e-7 inside the tolerance 1e-6, NaN positions equal", maxdiff(a, b) <= TOL)
    chk("PLANTED: maxdiff 2e-6 outside the tolerance", maxdiff(a, np.array([1.0 + 2e-6, 2.0, np.nan])) > TOL)
    chk("PLANTED: a NaN where the other has a number is infinite", maxdiff(a, np.array([1.0, np.nan, np.nan])) == float("inf"))
    # difference of two differences
    rng = np.random.default_rng(1)
    Xp, Rp, Xf, Rf = [rng.normal(size=50) for _ in range(4)]
    Xp[3] = np.nan
    x, y = dod_arrays(Xp, Rp, Xf, Rf)
    want = (Xp - Rp) - (Xf - Rf)
    got = x - y
    chk("difference of differences: x - y equals (Xp - Rp) - (Xf - Rf) to 1e-15 on every finite window", np.nanmax(np.abs(got - want)) < 1e-15)
    chk("difference of differences: NaN in any of the four makes the window NaN in both lists", np.isnan(x[3]) and np.isnan(y[3]) and np.isfinite(x[4]) and np.isfinite(y[4]))
    Xp2 = Xp.copy()
    Xp2[10] += 7.0
    x2, y2 = dod_arrays(Xp2, Rp, Xf, Rf)
    chk("PLANTED: a regression of 7 in the pre-rolled arm shows as 7 in the difference of differences", abs((x2 - y2)[10] - want[10] - 7.0) < 1e-12)
    # offsets
    o = off_stats([0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
    chk("offset statistics: n 5, mean 2, median 2, p90 3.6, max 4, NaN dropped", o["n"] == 5 and o["mean"] == 2.0 and o["median"] == 2.0 and abs(o["p90"] - 3.6) < 1e-12 and o["max"] == 4.0)
    # family rule and names
    chk("family_of: all BETTER / all WORSE / none / mixed", [family_of(["BETTER", "BETTER"]), family_of(["WORSE", "WORSE"]), family_of(["OPEN", "INSIDE-SPREAD"]), family_of(["BETTER", "OPEN"])] ==
        ["BETTER", "WORSE", "NOT-SEPARATED", "MIXED"])
    ud = {"per_seed": [-0.02, -0.03], "interval": [-0.05, -0.001]}
    uh = {"per_seed": [0.02, 0.03], "interval": [0.001, 0.05]}
    uo = {"per_seed": [-0.02, 0.03], "interval": [-0.05, 0.05]}
    chk("U names: LOWER / HIGHER / NOT-SEPARATED (a seed on the other side blocks the name)", [u_name(ud), u_name(uh), u_name(uo), u_name({"per_seed": [-0.02, 0.03], "interval": [-0.05, -0.001]})] ==
        ["LOWER", "HIGHER", "NOT-SEPARATED", "NOT-SEPARATED"])

    def mk(**kw):
        d = {"fam": {f: "NOT-SEPARATED" for f in FAMS}, "U": "NOT-SEPARATED"}
        for k, v in kw.items():
            if k == "U":
                d["U"] = v
            else:
                d["fam"][k] = v
        return d
    base = {p: mk() for p in NAMED_PAIRS}
    cs = [c for c, _ in consequences(base)]
    chk("consequences: every pair NOT-SEPARATED -> exactly the NONE sentence", cs == ["NONE"])
    R = dict(base)
    R["AGE-CONV"] = mk(LATERAL="WORSE")
    cs = [c for c, _ in consequences(R)]
    chk("consequences: AGE-CONV WORSE, candidates clean -> case A, not NONE", cs == ["A"])
    R["AGE-LABEL-CAND"] = mk(U="HIGHER")
    cs = [c for c, _ in consequences(R)]
    chk("consequences: a candidate AGE pair with U HIGHER -> case C (named for that candidate), no case A", cs == ["C"] and "AGE-LABEL-CAND" in consequences(R)[0][1])
    R = dict(base)
    R["LABEL-CAND-AT-AGE"] = mk(TACTICAL="BETTER")
    R["READER-CAND-AT-AGE"] = mk(TACTICAL="BETTER", LATERAL="WORSE")
    cs = [c for c, _ in consequences(R)]
    chk("consequences: AT-AGE BETTER and none WORSE -> HOLDS; AT-AGE with a family WORSE -> LIMIT; (the NONE sentence concerns the AGE pairs only)", "HOLDS" in cs and "LIMIT" in cs)
    R = dict(base)
    R["AGE-K"] = mk(LONGITUDINAL="WORSE", LATERAL="WORSE")
    R["AGE-K-NAVLOGGED"] = mk(LATERAL="WORSE")
    cs = [(c, s[:9]) for c, s in consequences(R)]
    chk("consequences: AGE-K WORSE on two families, NAVLOGGED NOT-SEPARATED on one, WORSE on the other -> NAV_ROUTE and NAV_STATE", [c for c, _ in cs] == ["NAV_ROUTE", "NAV_STATE"] and "LONGITUDINAL" in consequences(R)[0][1])
    R = dict(base)
    R["AGE-CONV"] = mk(LONGITUDINAL="BETTER")
    cs = [c for c, _ in consequences(R)]
    chk("consequences: a BETTER AGE pair -> the note that it decides nothing, and not the NONE sentence", cs == ["BETTER_NOTE"])
    R = dict(base)
    R["AGE-K"] = mk(TACTICAL="MIXED")
    chk("consequences: MIXED counts as neither WORSE nor BETTER (no case A/C; NONE does not apply because MIXED is not NOT-SEPARATED)", [c for c, _ in consequences(R)] == [])
    chk("tables: 20 outcome runs + 3 control runs; 10 outcome arms x 2 seeds; every pair member is a known run",
        len(NEW_RUN_LIST) == 23 and len(RUNS) == 10 and all(a_ in ALL_RUNS and b_ in ALL_RUNS for a_, b_ in list(NAMED_PAIRS.values()) + list(P60_PAIRS.values()) + list(FRESH_PAIRS.values()) +
                                                                list(RULE_AT_AGE.values())))
    chk("tables: eight named pairs, seven P = 60 pairs, names unique", len(NAMED_PAIRS) == 8 and len(P60_PAIRS) == 7 and len({"%s-%s-%s-%s" % (PRE, p, f, w) for p in NAMED_PAIRS for f in FAMS for w in ("BETTER", "WORSE", "MIXED", "NOT-SEPARATED")}) == 8 * 16)
    ok = all(v for _, v in res)
    print("ZZ SELFTEST-LONGROLL ok: %s (%d checks)" % (ok, len(res)))
    return 0 if ok else 1


# ============================================================================================ loading
def load_dump_dir(d, keys6=("g", "cl", "ws", "eid", "clip_index", "v0")):
    """all episode files of a dump folder concatenated: dict of the six arrays the readers use, and the number of files; None when the folder is absent or empty"""
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    if not fs:
        return None
    out = {n: [] for n in keys6}
    for f in fs:
        with np.load(f) as z:
            for n in keys6:
                out[n].append(np.asarray(z[n]))
    o = {n: np.concatenate(v) for n, v in out.items()}
    o["n_files"] = len(fs)
    return o


def cut(D, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in D.items()}


def engine(a):
    """imports the earlier readers (their environment first); returns (H headway module, Q3 module, RH hierrep module)"""
    code = a.code
    os.environ.setdefault("HEADWAY_Q3DIR", os.path.join(code, "q3"))
    os.environ.setdefault("HIERREP_STAGEB", os.path.join(code, "stageB"))
    os.environ.setdefault("TANITEVAL_ROOT", os.path.join(code, "tev"))
    for p in (os.path.join(code, "headway"), os.path.join(code, "q3"), os.path.join(code, "stageB")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import headway_read_r2 as H                                                              # noqa: E402
    import q3_read_r3 as Q3                                                                  # noqa: E402
    return H, Q3, Q3.RH


# ============================================================================================ one grid: headway and family reads on restricted windows
def read_grid(a, E, grid_name, keys, runs, spread_run, pairs, dod_defs, log):
    """runs: list of run names (both seeds). Returns dict with restriction reports, controls of the two engines, cell reads per pair, levels, counts. Everything in memory."""
    H, Q3, RH = E
    import numpy as _np                                                                      # noqa: F401
    J = H.load_join()
    out = {"restriction": {}, "controls": {}, "pairs": {}, "levels": {}, "counts": {}, "dod": {}}
    Dr, bad = {}, []
    for r in runs:
        for s in SEEDS:
            raw = load_dump_dir(dump_dir(a.pull, r, s))
            if raw is None:
                out["restriction"]["%s_s%d" % (r, s)] = {"missing_dump": True}
                bad.append(key_of(r, s))
                continue
            idx, rep = restrict_idx(raw["clip_index"], raw["ws"], keys)
            rep["n_files"] = raw["n_files"]
            out["restriction"][key_of(r, s)] = rep
            if rep["missing"] == 0:
                Dr[(r, s)] = cut(raw, idx)
            else:
                bad.append(key_of(r, s))
    out["missing_or_incomplete"] = bad
    if bad or spread_run not in {r for r, _ in Dr}:
        out["controls"] = {"ALL_PASS": False, "why": "runs missing or a window missing in a run: %s" % bad}
        return out
    ref = Dr[(spread_run, 0)]
    same = all(all(np.array_equal(D[k], ref[k]) for k in ("g", "ws", "eid", "clip_index", "v0")) for D in Dr.values())
    out["controls"]["same_windows_every_dump"] = same
    uniq, inv = RH.episode_units(ref["eid"])
    n_ep = len(uniq)
    L = H.label_windows(J, ref["clip_index"], ref["ws"])
    out["counts"] = {"n_windows": int(len(ref["g"])), "n_episodes": int(n_ep), "n_lead_windows": int(L["lead"].sum()), "n_episodes_with_lead": int(len(np.unique(np.asarray(ref["eid"])[L["lead"]])))}
    W = {k: H.window_cells(D["g"], D["cl"], L) for k, D in Dr.items()}
    W["_dumps"] = [D for D in Dr.values()]
    out["counts"]["n_scored_U_windows"] = {r: int(np.isfinite(W[(r, 0)]["U"]).sum()) for r in runs}
    V = {}
    for k, D in Dr.items():
        c = RH.comps(D["cl"], D["g"])
        c["_lon"], c["_lat"] = RH.classify(*RH.kinematics(D["cl"]))
        V[k] = c
    glon, glat = RH.classify(*RH.kinematics(ref["g"]))
    joint_g = glon * 3 + glat
    crit = joint_g != RH.KEEP_STRAIGHT
    masks = {"critical": crit, "bulk": ~crit, "all": np.ones(len(crit), bool)}
    for c in V.values():
        c["lon_err_rate"] = (c["_lon"] != glon).astype(float)
        c["lat_err_rate"] = (c["_lat"] != glat).astype(float)
        c["joint_err_rate"] = ((c["_lon"] * 3 + c["_lat"]) != joint_g).astype(float)
    keys_all = [(k, "all") for _, k in RH.FAM_ROWS] + list(Q3.EXTRA)
    lv = {k: {kk + "|" + m: float(np.nanmean(c[kk][masks[m]])) for kk, m in keys_all} for k, c in V.items()}
    fam_spread = {}
    for kk, m in keys_all:
        lvs = [lv[(spread_run, s)][kk + "|" + m] for s in SEEDS]
        d = [abs(x - y) for x, y in itertools.combinations(lvs, 2)]
        fam_spread[kk + "|" + m] = {"max": float(max(d)), "median": float(np.median(d)), "n_pairs": len(d), "levels": lvs}
    hw_spread = {cell: H.spread_of([H.level(W[(spread_run, s)][cell]) for s in SEEDS]) for cell in H.CELLS}
    out["spread"] = {"arm": spread_run, "seeds": list(SEEDS), "family": fam_spread, "headway": hw_spread}
    # ---- controls of the two engines on the restricted windows
    pr0 = next(iter(pairs.values()))
    sb = {(pr0[0], pr0[1]): list(SEEDS)}
    hctrl = H.run_controls(ref, L, J, H.bank_speeds(), W, inv, n_ep, hw_spread["U"], list(SEEDS), a.n_boot_control, [(pr0[0], pr0[1])], sb, hw_spread)
    out["controls"]["headway_h1_h6"] = hctrl
    A0, B0 = pr0
    f_self, e_self = Q3.pair_cells(V, masks, inv, n_ep, A0, A0, list(SEEDS), a.n_boot_control, len(SEEDS), fam_spread)
    f_fw, _ = Q3.pair_cells(V, masks, inv, n_ep, A0, B0, list(SEEDS), a.n_boot_control, len(SEEDS), fam_spread)
    f_sw, _ = Q3.pair_cells(V, masks, inv, n_ep, B0, A0, list(SEEDS), a.n_boot_control, len(SEEDS), fam_spread)
    mx = max(abs(c["D"]) for c in list(f_self.values()) + list(e_self.values()))
    dev = max(abs(f_sw[k]["D"] + f_fw[k]["D"]) for k in f_fw)
    key = "speed_mae_mps"
    smax = fam_spread[key + "|all"]["max"]
    plant = max(3.0 * smax, 0.05)
    Vp = dict(V)
    for s in SEEDS:
        cp = dict(V[(A0, s)])
        cp[key] = V[(A0, s)][key] + plant
        Vp[("__PLANT__", s)] = cp
    f_pl, _ = Q3.pair_cells(Vp, masks, inv, n_ep, "__PLANT__", A0, list(SEEDS), a.n_boot_control, len(SEEDS), fam_spread)
    cpl = f_pl["LONGITUDINAL:" + key]
    named = Q3.q3_name(f_pl, True)
    qc = {"q2_self_is_zero": {"PASS": bool(mx == 0.0), "max_abs_D": float(mx)}, "q3_swap_is_minus": {"PASS": bool(dev < 1e-12), "max_abs_dev": float(dev)},
          "q4_planted_regression": {"PASS": bool(cpl["verdict"] == "WORSE" and abs(cpl["D"] - plant) < 1e-9 and named["cost_families"] == ["LONGITUDINAL"] and not named["gain_families"]),
                                    "planted_mps": float(plant), "D_read": cpl["D"], "verdict": cpl["verdict"]}}
    out["controls"]["family_q2_q3_q4"] = qc
    out["controls"]["ALL_PASS"] = bool(same and hctrl.get("ALL_PASS") is True and all(v["PASS"] for v in qc.values()))
    # ---- the pairs
    for nm, (A, B) in pairs.items():
        hc = H.pair_cells(W, inv, n_ep, A, B, list(SEEDS), a.n_boot, len(SEEDS), hw_spread, True)
        fc, ex = Q3.pair_cells(V, masks, inv, n_ep, A, B, list(SEEDS), a.n_boot, len(SEEDS), fam_spread, True)
        out["pairs"][nm] = {"A": A, "B": B, "headway": hc, "family": fc, "family_beside": ex}
    # ---- difference of the two differences (same bootstrap draws: the engine's rng is fixed by the shapes; one combined window array per seed)
    for nm, (age, fresh) in dod_defs.items():
        if age not in pairs or fresh not in pairs:
            continue
        (A1, B1), (A0_, B0_) = pairs[age], pairs[fresh]
        res = {"headway": {}, "family": {}}
        ones = np.ones(len(inv), bool)
        for cell in H.CELLS:
            xs, ys = zip(*[dod_arrays(W[(A1, s)][cell], W[(B1, s)][cell], W[(A0_, s)][cell], W[(B0_, s)][cell]) for s in SEEDS])
            sE, cE = RH.seed_arrays(list(xs), list(ys), ones, inv, n_ep)
            c = RH.read_cell(sE, cE, a.n_boot, len(SEEDS))
            res["headway"][cell] = {"D": c["D"], "per_seed": c["d_s"], "interval": [c["A_lo"], c["A_hi"]]}
        for kk, m in [(k, "all") for _, k in RH.FAM_ROWS]:
            xs, ys = zip(*[dod_arrays(V[(A1, s)][kk], V[(B1, s)][kk], V[(A0_, s)][kk], V[(B0_, s)][kk]) for s in SEEDS])
            sE, cE = RH.seed_arrays(list(xs), list(ys), masks[m], inv, n_ep)
            c = RH.read_cell(sE, cE, a.n_boot, len(SEEDS))
            res["family"][kk] = {"D": c["D"], "per_seed": c["d_s"], "interval": [c["A_lo"], c["A_hi"]]}
        out["dod"][nm] = {"age_pair": age, "fresh_pair": fresh, **res}
    # ---- levels of U, C, F per run and seed
    for r in runs:
        out["levels"][r] = {c: [H.level(W[(r, s)][c]) for s in SEEDS] for c in ("U", "C", "F")}
    return out


# ============================================================================================ STRATEGIC on restricted rows (instrument of card ADD66, unchanged; module constants set per grid)
def strategic_grid(a, keys, runs, p_of, log, ctx_cache):
    """three ticks per window. l9 (structure on the restricted file; the raw file of a P run must hold exactly the grid), s1 own transform vs the tool's s_ckpt_w0 (the R3 mapping), s2 and s5 on the
    runs with no pre-roll (the first tick of a pre-rolled run is not the logged state by construction), s3 on the restricted R0 first tick, s4 synthetic known values. -> (out, cells, eid, S)"""
    import strat_roll_read as S
    S.N_WIN, S.N_ROWS, S.N_EP = int(len(keys)), 3 * int(len(keys)), int(len(np.unique(keys[:, 0])))
    STATS = {}

    def load_sel(path):
        z = np.load(path, allow_pickle=True)
        T = {k: z[k] for k in S.FIELDS}
        T2, st = restrict_ticks(T, keys)
        STATS[os.path.basename(os.path.dirname(os.path.dirname(path)))] = st
        return T2
    S.load_ticks = load_sel
    sa = argparse.Namespace(w=a.w, tracks=S.DEF_TRACKS, join=S.DEF_JOIN, tree=a.tree)
    if "ctx" not in ctx_cache:
        ctx = S.Ctx(sa)
        ctx.load_static()
        ctx_cache["ctx"] = ctx
    ctx = ctx_cache["ctx"]
    C = {"s4": S.run_s4()}
    s1 = {"max_dxy_m": 0.0, "max_dpsi_rad": 0.0, "per_file": {}}
    s6 = {"per_file": {}, "pass": True}
    s2 = {"max_first_tick_anchor_dist_m": 0.0, "n_rows_anchor_not_t0": 0, "files": []}
    inputs, missing, ref = {}, [], None
    for run in runs:
        for k in SEEDS:
            nm = "%s_s%d" % (run, k)
            p = S.tick_path(ctx.W, run, k)
            if not os.path.isfile(p):
                missing.append(nm)
                continue
            inputs[nm] = {"path": p, "sha256": sha(p)}
            T = S.load_ticks(p)
            rbt = S.window_order(T)
            if ref is None:
                ref = {"ci": T["ci"][rbt[0]], "t0": T["t0"][rbt[0]]}
            ok6, det6 = S.structure(T, nm, ref)
            raw_rows = int(len(np.load(p, allow_pickle=True)["tick"]))
            exact = bool(raw_rows == 3 * len(keys)) if p_of[run] > 0 else True                  # a P run holds exactly the windows of its grid
            s6["per_file"][nm] = bool(ok6 and exact)
            s6["pass"] &= bool(ok6 and exact)
            dxy, dps = S.s1_row_check(T)
            s1["per_file"][nm] = [dxy, dps]
            s1["max_dxy_m"], s1["max_dpsi_rad"] = max(s1["max_dxy_m"], dxy), max(s1["max_dpsi_rad"], dps)
            if p_of[run] == 0:
                an = S.anchors_for(ctx, T, rbt, ticks=(0,))
                st, ds = an[0]
                s2["n_rows_anchor_not_t0"] += int((st != T["t0"][rbt[0]]).sum())
                s2["max_first_tick_anchor_dist_m"] = max(s2["max_first_tick_anchor_dist_m"], float(ds.max()))
                s2["files"].append(nm)
    s1["pass"] = bool(s1["max_dxy_m"] <= 1e-3 and s1["max_dpsi_rad"] <= 1e-3 and s1["per_file"])
    s2["pass"] = bool(s2["max_first_tick_anchor_dist_m"] <= 1e-6 and s2["n_rows_anchor_not_t0"] == 0 and s2["files"])
    C["s1"], C["s2"], C["s6"] = s1, s2, s6
    C["s9"] = {"pass": bool(inputs) and not missing, "n_files": len(inputs), "missing": missing}
    if "R0_s0" in inputs:
        T0 = S.load_ticks(S.tick_path(ctx.W, "R0", 0))
        C["s3"] = S.s3_check(ctx, T0, S.window_order(T0))
    else:
        C["s3"] = {"pass": False, "computable": False, "reason": "R0_s0 tick file absent"}
    out = {"controls": C, "inputs": inputs, "stage_A_pass": bool(C["s4"]["pass"] and s1["pass"] and s2["pass"] and s6["pass"] and C["s9"]["pass"] and C["s3"].get("pass") is True), "windows": int(len(keys))}
    if not out["stage_A_pass"]:
        return out, None, None, S
    cells, anc = {}, {}
    T0 = None
    for run in runs:
        for k in SEEDS:
            T = S.load_ticks(S.tick_path(ctx.W, run, k))
            T0 = T if T0 is None else T0
            c, st, ds = S.compute_cells(ctx, T)
            cells[(run, k)], anc[(run, k)] = c, ds
            log("cells %s_s%d done" % (run, k))
    eid, n_map = S.episode_map(ctx.W, T0)
    out["episode_map"] = {"all_windows_mapped": bool((eid >= 0).all()), "n_episodes": int(len(np.unique(eid))), "n_windows": int(len(eid))}
    d5 = 0.0
    for run in runs:
        if p_of[run] != 0 or run == "R0" or ALL_RUNS[run][3] != "deadreckon":                 # a run with a logged route input has another first-tick reading by construction
            continue
        for k in SEEDS:
            for cell in S.CELLS:
                x, y = cells[(run, k)][cell][0], cells[("R0", k)][cell][0]
                if not (np.isnan(x) == np.isnan(y)).all():
                    d5 = float("inf")
                else:
                    ok = np.isfinite(x)
                    d5 = max(d5, float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0)
    C["s5"] = {"pass": bool(d5 == 0.0), "largest_difference": d5, "note": "first-tick cells of every run with no pre-roll equal those of R0 at the same model seed (pre-rolled runs excluded by construction)"}
    out["stage_B_pass"] = bool(C["s5"]["pass"] and out["episode_map"]["all_windows_mapped"] and out["episode_map"]["n_episodes"] == S.N_EP)
    return out, (cells if out["stage_B_pass"] else None), (eid if out["stage_B_pass"] else None), S


# ============================================================================================ controls on the files of the runs
def run_record(pull, run, k):
    t = txt(os.path.join(tick_dir(pull, run, k), "run_%s.txt" % key_of(run, k)))
    mk, me, mc, mw = re.search(r"kappa (\S+)", t), re.search(r"rule every (\d+)", t), re.search(r"ckpt sha256 ([0-9a-f]{64})", t), re.search(r"wrapper sha256 ([0-9a-f]{64})", t)
    return {"kappa": mk.group(1) if mk else None, "rule_every": int(me.group(1)) if me else None, "ckpt": mc.group(1) if mc else None, "wrapper": mw.group(1) if mw else None, "text_present": bool(t)}


def file_controls(a, keys_by_grid, names, log):
    """l1 l2 l3 l4 l5 l7 l8 l10 l11 l12 on the files of the runs; every entry {"PASS": True|False|None, ...}"""
    C = {}
    e1 = jl(a.e1)
    C["e1_pod_read_reproduces_bank"] = {"file": os.path.basename(a.e1) if e1 else None, "PASS": bool(e1 and e1.get("PASS") is True)}
    # l1
    t = txt(a.check_preroll)
    lines = t.strip().splitlines()
    pw = re.findall(r"wrapper \S* ?sha256 ([0-9a-f]{64})", t)
    w1 = sha(a.wrapper)
    C["l1_check_preroll_b"] = {"last_line": lines[-1][:80] if lines else None, "printed_wrapper_sha": [x[:16] for x in pw][-1:] or None, "wrapper_sha256": w1[:16],
                               "PASS": bool(lines and lines[-1].strip() == "ZZ CHECK-PREROLL-B PASS" and w1 and (not pw or pw[-1] == w1))}
    # l2: R0LrID seed 0 equals the dump of R0Hb20 seed 0 (card ADD67), 141 files, 0 differing
    na, nb, nd, mx, bad = cmp_dump_dirs(dump_dir(a.pull, "R0LrID", 0), dump_dir(a.pull, "R0Hb20", 0))
    C["l2_P0_identity_with_banked_R0Hb20"] = {"files_R0LrID": na, "files_R0Hb20": nb, "differing": nd, "largest_difference": mx, "PASS": bool(na == 141 and nb == 141 and nd == 0)}
    # l3: carry against fresh on the P = 30 grid
    na, nb, nd, mx, bad = cmp_dump_dirs(dump_dir(a.pull, "R0LrCarry", 0), dump_dir(a.pull, "R0LrFresh", 0))
    C["l3_state_carry"] = {"files_carry": na, "files_fresh": nb, "files_differing_at_all": nd, "largest_difference": mx, "tolerance": TOL, "PASS": bool(na > 0 and na == nb and mx <= TOL)}
    # l12: tick-file arrays of carry against fresh
    pc = os.path.join(tick_dir(a.pull, "R0LrCarry", 0), "ticks_%s_m10_causal_heads.npz" % key_of("R0LrCarry", 0))
    pf = os.path.join(tick_dir(a.pull, "R0LrFresh", 0), "ticks_%s_m10_causal_heads.npz" % key_of("R0LrFresh", 0))
    zc, zf = npz_load(pc), npz_load(pf)
    if zc is None or zf is None:
        C["l12_recorder_frame"] = {"PASS": None, "why": "tick file missing"}
    else:
        skip = {"ckpt"}                                                                    # the checkpoint hash/path scalar is not an array of the rollout
        per = {k: maxdiff(zc[k], zf[k]) for k in sorted(set(zc.files) & set(zf.files)) if k not in skip}
        only = sorted(set(zc.files) ^ set(zf.files))
        mxx = max(per.values()) if per else float("inf")
        C["l12_recorder_frame"] = {"largest_difference": mxx, "tolerance": TOL, "keys_in_one_only": only, "worst_keys": sorted(per, key=lambda k: -per[k])[:5], "PASS": bool(per and not only and mxx <= TOL)}
    # l4 / l5 / l8 / l10 / l11 per run
    l4, l5, l8, l10, l11, rec = {"runs": {}, "grids": {}}, {"runs": {}}, {"runs": {}}, {"runs": {}}, {"runs": {}}, {}
    for g, keys in keys_by_grid.items():
        l4["grids"][g] = {"n_windows": int(len(keys)), "card_count": CARD_COUNTS.get(g)}
    ok4, ok5, ok8, ok10, ok11 = [], [], [], [], []
    for run, k in NEW_RUN_LIST:
        form, ro, P, nav, g, ex, every, kap = ALL_RUNS[run]
        keys = keys_by_grid[g]
        N = int(len(keys))
        key = key_of(run, k)
        td = tick_dir(a.pull, run, k)
        raw = load_dump_dir(dump_dir(a.pull, run, k), keys6=("ws", "clip_index"))
        e4, e5, e8, e10, e11 = [], [], [], [], []
        # l4: windows
        if raw is None:
            e4.append("dump missing")
        else:
            _, rp = restrict_idx(raw["clip_index"], raw["ws"], keys)
            if (not rp["holds_exactly"]) or (g != "full" and raw["n_files"] != len(np.unique(keys[:, 0]))):
                e4.append("dump windows: %s files %d" % (rp, raw["n_files"]))
            if g == "full" and raw["n_files"] != 141:
                e4.append("full grid: %d dump files, not 141" % raw["n_files"])
        zt = npz_load(os.path.join(td, "ticks_%s_m10_causal_heads.npz" % key))
        if zt is None or len(zt["tick"]) != 3 * N:
            e4.append("tick rows %s, expected %d" % (None if zt is None else len(zt["tick"]), 3 * N))
        L, Lm = npz_load(os.path.join(td, "rule_log_%s.npz" % key)), npz_load(os.path.join(td, "rule_log_mid_%s.npz" % key))
        if form != "R0":
            if L is None or len(L["ci"]) != 3 * N:
                e4.append("rule-log rows %s, expected %d" % (None if L is None else len(L["ci"]), 3 * N))
            if every is not None and every < 10:
                nm_ = N * 3 * (10 // every - 1)
                if Lm is None or len(Lm["q"]) != nm_:
                    e4.append("later-look rows %s, expected %d" % (None if Lm is None else len(Lm["q"]), nm_))
        # l5 / l11: pre-roll logs
        pl, po = jl(os.path.join(td, "preroll_log_%s.json" % key)), npz_load(os.path.join(td, "preroll_offsets_%s.npz" % key))
        rl_pre = npz_load(os.path.join(td, "rule_log_pre_%s.npz" % key))
        info = {"P": P}
        if P == 0:
            if pl is not None or po is not None or rl_pre is not None:
                e5.append("a pre-roll file exists for a P = 0 run")
        else:
            if pl is None or po is None:
                e5.append("preroll log or offsets file missing")
            else:
                kinds = sorted(pl.get("windows", {}))
                info["windows"], info["ticks_pre_rolled"] = pl.get("windows"), pl.get("ticks_pre_rolled")
                if pl.get("preroll_steps") != P or pl.get("segments_per_window") != P // 30:
                    e5.append("preroll_steps %s / segments %s" % (pl.get("preroll_steps"), pl.get("segments_per_window")))
                for kd in kinds:
                    if pl["windows"][kd] != N or pl["ticks_pre_rolled"][kd] != N * P // 10:
                        e5.append("kind %s: windows %s ticks %s, expected %d and %d" % (kd, pl["windows"][kd], pl["ticks_pre_rolled"][kd], N, N * P // 10))
                kind = np.asarray(po["kind"]).astype(str)
                mdl = kind == "model"
                if not mdl.any():
                    e5.append("no model rows in the offsets file")
                else:
                    dpos = np.asarray(po["dpos"], np.float64)
                    info["dpos_model"] = off_stats(dpos[mdl])
                    if ex == "plan" and not (np.median(dpos[mdl]) > 0.0):
                        e5.append("median position offset is not above 0 for the plan executor")
                    if ex == "logged":
                        info["dpos_max_all_kinds"] = float(np.abs(dpos).max())
                        if np.abs(dpos).max() > TOL:
                            e5.append("carry run: largest position offset %.3e above %.0e" % (np.abs(dpos).max(), TOL))
                # l10: a_m0 per window exists; the rule-log first-tick rows keep the definition
                am = np.asarray(po["a_m0_first_tick"], np.float64)
                if len(am) != len(po["ci"]) or not np.isfinite(am).all():
                    e10.append("a_m0_first_tick missing or not finite")
                elif form != "R0" and (am[mdl].shape[0] != N):
                    e10.append("a_m0 rows for the model kind: %d, expected %d" % (am[mdl].shape[0], N))
                if ex == "logged" and np.abs(am).max() > TOL:
                    e10.append("carry run: largest |a_m0| %.3e above %.0e" % (np.abs(am).max(), TOL))
                info["a_m0_max_abs"] = float(np.abs(am).max()) if len(am) else None
                used = pl.get("a_m0_used_in_rule") or {}
                info["a_m0_used_in_rule"] = used
                if form != "R0":
                    nrow, d, okg = g0_identity(L)
                    info["first_tick_rows"], info["g0_minus_gsrc_max"] = nrow, d
                    if not okg or nrow != N:
                        e10.append("rule-log first-tick rows: %d (expected %d), g0 - g_src max %.3e" % (nrow, N, d))
                    if rl_pre is None or len(rl_pre["ci"]) < N * P // 10:
                        e5.append("rule_log_pre rows %s, expected at least %d" % (None if rl_pre is None else len(rl_pre["ci"]), N * P // 10))
                # l11
                if P == 60:
                    nseg = len(po["seg_ci"])
                    cl = pl.get("clipped_windows_in_preroll") or {}
                    info["seg_rows"], info["clipped_windows_in_preroll"] = int(nseg), cl
                    if pl.get("segments_per_window") != 2 or any(v for v in cl.values()) or nseg != N * len(kinds):
                        e11.append("segments %s, seg rows %d (expected %d), clipped %s" % (pl.get("segments_per_window"), nseg, N * len(kinds), cl))
        # l8: readout logs and the run record
        rr = run_record(a.pull, run, k)
        RL = jl(os.path.join(td, "readout_log_%s.json" % key))
        if RL is None or not rr["text_present"]:
            e8.append("readout log or run record missing")
        else:
            kp = rr["kappa"]
            if kap is None or kp is None or kp == "none" or abs(float(kp) - kap) > 1e-12 or RL.get("kappa_max") is None or abs(float(RL["kappa_max"]) - kap) > 1e-12:
                e8.append("kappa: run record %s readout log %s, declared %s" % (kp, RL.get("kappa_max"), kap))
            if not ((RL.get("frames_total") or 0) > 0):
                e8.append("no frames counted")
            if RL.get("replan_steps") != 10:
                e8.append("replan_steps %s" % RL.get("replan_steps"))
            if every is not None and (rr["rule_every"] != every):
                e8.append("rule every %s, declared %s" % (rr["rule_every"], every))
            if RL.get("form") is not None and RL.get("form") != form:
                e8.append("form %s, declared %s" % (RL.get("form"), form))
            if ro == "pinned":
                if not (RL.get("readout") == "pinned" and RL.get("acc_gain") is None and RL.get("path_exec") is False):
                    e8.append("readout is not the pinned one")
            else:
                dv, dg = RL.get("max_abs_speed_minus_last_step"), RL.get("max_abs_acceleration_minus_gain_executed")
                if not (RL.get("readout") == "causal_g" and RL.get("acc_gain") == 0.0 and RL.get("path_exec") is True and dv is not None and dg is not None and dv <= TOL and dg <= TOL):
                    e8.append("readout is not K (causal_g, gain 0, path-exec)")
            if not (RL.get("max_excess_rad") is not None and RL["max_excess_rad"] <= 1e-9 and RL.get("max_abs_diff_where_not_acted") == 0.0 and RL.get("max_abs_diff_xy") == 0.0 and RL.get("max_abs_diff_v") == 0.0):
                e8.append("the bound did something other than bound the curvature")
        info["run_record"] = rr
        rec[key] = rr
        for dct, e, okl in ((l4["runs"], e4, ok4), (l5["runs"], e5, ok5), (l8["runs"], e8, ok8), (l10["runs"], e10, ok10), (l11["runs"], e11, ok11)):
            dct[key] = {"problems": e}
            okl.append(not e)
        l5["runs"][key]["info"] = info
    # per grid: the grid files themselves (the per-clip rule) and the restriction of every banked run (in read_grid; recorded there)
    pinned = jl(a.grid_full)
    for g, P in (("30", 30), ("60", 60)):
        ks = keys_by_grid[g]
        pr = []
        if pinned is not None:
            from_full = sum(1 for c in pinned["order"] for t in pinned["windows"][c] if int(t) - P >= 3)
            pr.append("restriction of the pinned grid by t0 - P >= 3 gives %d windows, grid file holds %d" % (from_full, len(ks)) if from_full != len(ks) else None)
        l4["grids"][g]["per_clip_rule_min_t0_minus_P"] = int(ks[:, 1].min() - P)
        l4["grids"][g]["per_clip_rule_ok"] = bool(ks[:, 1].min() - P >= 3 and (pinned is None or not [x for x in pr if x]))
        l4["grids"][g]["equals_card_count"] = bool(len(ks) == CARD_COUNTS[g])
    l4["PASS"] = bool(all(ok4) and all(l4["grids"][g]["per_clip_rule_ok"] for g in ("30", "60")))
    l4["note"] = "the counts of the grid tool replace the card's where they differ (the per-clip rule is checked); banked-run restrictions are recorded under restriction in each grid read and gate the read there"
    C["l4_window_counts"] = l4
    l5["PASS"] = bool(all(ok5))
    C["l5_preroll_logs"] = l5
    l8["PASS"] = bool(all(ok8))
    C["l8_readout_logs"] = l8
    l10["PASS"] = bool(all(ok10))
    C["l10_first_tick_offset_term"] = l10
    l11["PASS"] = bool(all(ok11))
    C["l11_segments"] = l11
    # l7: hashes
    ru = jl(a.reused) or {"files": {}, "dump_digests": {}}
    root = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    badr = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(root, p)) or sha(os.path.join(root, p)) != h]
    baddd = []
    for run_key, dg in (ru.get("dump_digests") or {}).items():
        d = os.path.join(root, "integ10k", "pull", run_key, "dump_%s_m10_causal_heads" % run_key)
        h = hashlib.sha256()
        for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
            h.update(("%s %s\n" % (os.path.basename(f), sha(f))).encode())
        if h.hexdigest() != dg:
            baddd.append(run_key)
    wsh = {key: rec[key]["wrapper"] for key in rec}
    csh = {key: bool(rec[key]["ckpt"] and rec[key]["ckpt"].startswith(CKPT_SHA_PREFIX[int(key[-1])])) for key in rec}
    pw, tw = sha(a.parent_wrapper), sha(a.test_wrapper)
    one = {w1, tw, *wsh.values()}
    podh = {}
    for ln in txt(a.pod_hashes).splitlines():
        m = re.match(r"^([0-9a-f]{64})  (\S+)$", ln.strip())
        if m:
            podh[m.group(2)] = m.group(1)
    C["l7_hashes"] = {"n_reused_files": len(ru["files"]), "reused_files_differing": badr, "dump_digests_differing": baddd, "parent_wrapper_sha256": pw[:16], "parent_prefix_card": PARENT_SHA_PREFIX,
                      "wrapper_sha256": w1[:16], "test_wrapper_sha256": tw[:16], "one_wrapper_sha_everywhere": bool(len(one) == 1 and None not in one and w1),
                      "ckpt_not_equal": [k for k, v in csh.items() if not v], "pod_side_hashes_recorded": podh, "pod_hashes_note": "pinned tool, grids, check, qmap: recorded at read time (the queue's run records carry the wrapper and checkpoint hashes)",
                      "PASS": bool(ru["files"] and len(ru["files"]) == ru.get("n_files", len(ru["files"])) and not badr and not baddd and pw.startswith(PARENT_SHA_PREFIX) and len(one) == 1 and None not in one and w1 and all(csh.values()))}
    return C


# ============================================================================================ control l6 and formatting
def cells_of(fc):
    return {c: {"D": float(v["D"]), "interval": [float(v["A_lo"]), float(v["A_hi"])], "per_seed": [float(x) for x in v["d_s"]], "verdict": v["verdict"]} for c, v in fc.items()}


def ucf(hc):
    return {c: {"per_seed": [float(x) for x in hc[c]["d_s"]], "pooled": float(hc[c]["D"]), "interval": [float(hc[c]["A_lo"]), float(hc[c]["A_hi"])]} for c in ("U", "C", "F")}


def r4(x):
    return round(float(x), 4)


def l6_identity(a, g_full):
    """the pair R0Hb20 - R0 on all windows reproduces the pair of card ADD67 (ten family cells and U, C, F) to 1e-9. The reference values: READ_HEADBOUND10K.json (rounded to four decimals, so compared
    after the same rounding) and, when given, the unrounded RESULT_HEADWAY.json / RESULT_Q3.json of that read (compared unrounded)."""
    pr = g_full["pairs"].get("L6")
    ref = jl(a.add67_json)
    if pr is None or ref is None:
        return {"PASS": False, "why": "no pair read or no ADD67 read file"}
    ru, rc = ref["pairs"]["HP"]["headway"], ref["cells"]["HP"]
    mine_u, mine_c = ucf(pr["headway"]), cells_of(pr["family"])
    worst, miss = 0.0, []
    for c in ("U", "C", "F"):
        for f, g in (("pooled", "pooled"), ("interval", "interval"), ("per_seed", "per_seed")):
            x, y = np.asarray(mine_u[c][f], np.float64), np.asarray(ru[c][g], np.float64)
            worst = max(worst, float(np.abs(np.round(x, 4) - y).max()))
    for c, v in mine_c.items():
        if c not in rc:
            miss.append(c)
            continue
        worst = max(worst, abs(r4(v["D"]) - rc[c]["D"]), float(np.abs(np.round(v["interval"], 4) - np.asarray(rc[c]["interval"])).max()), float(np.abs(np.round(v["per_seed"], 4) - np.asarray(rc[c]["per_seed"])).max()))
    verd = [c for c, v in mine_c.items() if c in rc and v["verdict"] != rc[c]["verdict"]]
    raw = {"headway_compared": False, "family_compared": False, "largest_unrounded": None}
    hw, q3 = jl(a.add67_hw), jl(a.add67_q3)
    wr = 0.0
    if hw is not None:
        h0 = hw["pairs"]["I10R0Hb20_minus_I10R0"]["cells"]
        for c in ("U", "C", "F"):
            wr = max(wr, abs(float(pr["headway"][c]["D"]) - h0[c]["D"]), abs(float(pr["headway"][c]["A_lo"]) - h0[c]["A_lo"]), abs(float(pr["headway"][c]["A_hi"]) - h0[c]["A_hi"]))
        raw["headway_compared"] = True
    if q3 is not None:
        q0 = q3["pairs"]["I10R0Hb20_minus_I10R0"]["families"]
        for c, v in pr["family"].items():
            wr = max(wr, abs(float(v["D"]) - q0[c]["D"]), abs(float(v["A_lo"]) - q0[c]["A_lo"]), abs(float(v["A_hi"]) - q0[c]["A_hi"]))
        raw["family_compared"] = True
    raw["largest_unrounded"] = wr if (hw is not None or q3 is not None) else None
    ok = bool(worst <= TOL_ID and not miss and not verd and (raw["largest_unrounded"] is None or raw["largest_unrounded"] <= TOL_ID))
    return {"largest_difference_rounded_to_4_decimals": worst, "cells_missing_in_reference": miss, "verdicts_differing": verd, "unrounded": raw, "tolerance": TOL_ID,
            "reference": "READ_HEADBOUND10K.json pair HP (rounded 4 decimals)" + (" and the unrounded RESULT files" if raw["headway_compared"] or raw["family_compared"] else " (no unrounded RESULT file given: weaker)"),
            "PASS": ok}


def line(v):
    return "  ".join("%s %+.4f [%+.4f, %+.4f] %s" % (c, v[c]["pooled"], v[c]["interval"][0], v[c]["interval"][1], v[c]["per_seed"]) for c in ("U", "C", "F"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        if np is None:
            print("ZZ SELFTEST-LONGROLL ok: False (numpy missing)")
            return 1
        return selftest()
    ap = argparse.ArgumentParser()
    for n in ("read-dir", "pull", "reused", "code", "grid-dir", "grid-full", "wrapper", "parent-wrapper", "test-wrapper", "check-preroll", "e1", "add67-json", "w", "tree", "out"):
        ap.add_argument("--" + n, required=True)
    for n in ("add67-hw", "add67-q3", "pod-hashes"):
        ap.add_argument("--" + n, default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-control", type=int, default=300)
    a = ap.parse_args(argv)
    a.read_dir_ = a.read_dir

    def log(s):
        print(s, flush=True)
    g30, g60, gfull = (jl(os.path.join(a.grid_dir, "grid_p30.json")), jl(os.path.join(a.grid_dir, "grid_p60.json")), jl(a.grid_full))
    if g30 is None or g60 is None or gfull is None:
        print("STOP: a grid file is missing (%s, %s)" % (a.grid_dir, a.grid_full))
        return 3
    E = engine(a)
    H = E[0]
    J = H.load_join()
    names_ = [str(c) for c in J["clip"]]
    keys_by_grid = {"30": grid_keys(g30, names_), "60": grid_keys(g60, names_), "full": grid_keys(gfull, names_)}
    C = file_controls(a, keys_by_grid, names_, log)
    P_OF = {r: v[2] for r, v in ALL_RUNS.items()}
    # ---- grid reads (headway + family): P = 30, P = 60, all windows (l6)
    pairs30 = {k: v for k, v in NAMED_PAIRS.items()}
    pairs30.update(FRESH_PAIRS)
    pairs30.update(RULE_AT_AGE)
    runs30 = sorted({x for p in pairs30.values() for x in p})
    G30 = read_grid(a, E, "30", keys_by_grid["30"], runs30, "R0Hb20", pairs30, DOD, log)
    runs60 = sorted({x for p in P60_PAIRS.values() for x in p})
    G60 = read_grid(a, E, "60", keys_by_grid["60"], runs60, "R0Hb20", dict(P60_PAIRS), {}, log)
    GF = read_grid(a, E, "full", keys_by_grid["full"], ["R0Hb20", "R0"], "R0", {"L6": ("R0Hb20", "R0")}, {}, log)
    C["l4_window_counts"]["restriction_of_every_run_read"] = {g: G["restriction"] for g, G in (("30", G30), ("60", G60), ("full", GF))}
    rs_ok = all(not G["missing_or_incomplete"] and all(r.get("exact") for r in G["restriction"].values()) for G in (G30, G60, GF))
    rs_p_ok = all(r.get("holds_exactly") for G in (G30, G60) for k_, r in G["restriction"].items() if re.search(r"P(30|60)", k_))
    C["l4_window_counts"]["PASS"] = bool(C["l4_window_counts"]["PASS"] and rs_ok and rs_p_ok)
    C["l6_reader_identity"] = l6_identity(a, GF) if GF.get("pairs") else {"PASS": False, "why": "the all-windows read did not run"}
    for g, G in (("30", G30), ("60", G60), ("full", GF)):
        C["l_reader_controls_headway_family_grid_" + g] = {"PASS": bool(G["controls"].get("ALL_PASS") is True), "detail": G["controls"]}
    # ---- strategic
    ctx_cache = {}
    SR30 = SR60 = None
    cells30 = cells60 = eid30 = eid60 = None
    try:
        r30 = sorted({"R0"} | {x for p in list(NAMED_PAIRS.values()) for x in p})
        SR30, cells30, eid30, S = strategic_grid(a, keys_by_grid["30"], r30, P_OF, log, ctx_cache)
        r60 = sorted({"R0"} | {x for p in P60_PAIRS.values() for x in p})
        SR60, cells60, eid60, S = strategic_grid(a, keys_by_grid["60"], r60, P_OF, log, ctx_cache)
        C["l9_strategic_rows_three_ticks_per_window"] = {"PASS": bool(SR30["stage_A_pass"] and SR60["stage_A_pass"] and SR30.get("stage_B_pass") and SR60.get("stage_B_pass")),
                                                         "P30": dict({k: v for k, v in SR30["controls"].items() if k != "s3"}, s3=SR30["controls"]["s3"].get("pass")),
                                                         "P60": dict({k: v for k, v in SR60["controls"].items() if k != "s3"}, s3=SR60["controls"]["s3"].get("pass"))}
    except SystemExit as ex:
        C["l9_strategic_rows_three_ticks_per_window"] = {"PASS": False, "why": "SystemExit: %s" % ex}
    except Exception as ex:                                                                   # noqa: BLE001
        C["l9_strategic_rows_three_ticks_per_window"] = {"PASS": False, "why": "%s: %s" % (type(ex).__name__, str(ex)[:300])}
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": CARD, "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "names": None, "pairs": None, "cells": None, "families": None, "strategic": None, "beside": None, "reported_no_bar": None}
    print(TIER_LINE)
    print("== controls (e1, l1 to l12)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(conv(v))[:700]))
    if ok:
        names, R, fams, strat, pairs_out, cells_out = {}, {}, {}, {}, {}, {}
        for nm, (A, B) in NAMED_PAIRS.items():
            p = G30["pairs"][nm]
            pairs_out[nm] = {"A": A, "B": B, "headway": ucf(p["headway"])}
            cells_out[nm] = cells_of(p["family"])
            fams[nm] = {f: reading(cells_out[nm], f) for f in NAMED}
            pr = S.pair_read(cells30, A, B, eid30, ticks=(2,))
            vs = [c["verdict"] for c in pr["tick2"]["pos"]]
            strat[nm] = {"pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed")} for arc, b in zip(S.ARCS, pr["tick2"]["pos"])],
                         "reading": family_of(vs)}
            fams[nm]["STRATEGIC"] = strat[nm]["reading"]
            R[nm] = {"fam": fams[nm], "U": u_name(pairs_out[nm]["headway"]["U"])}
            names[nm + "_U"] = "%s-%s-U-%s" % (PRE, nm, R[nm]["U"])
            for f in FAMS:
                names["%s_%s" % (nm, f)] = "%s-%s-%s-%s" % (PRE, nm, f, fams[nm][f])
        cons = consequences(R)
        beside = {"P60_pairs": {}, "AT_FRESH_pairs": {}, "difference_of_differences": conv(G30["dod"]), "rule_under_K_at_age": {}, "offsets_at_t0": {}, "levels_P30": conv(G30["levels"]), "levels_P60": conv(G60["levels"]),
                  "counts": {"P30": G30["counts"], "P60": G60["counts"]}, "spread": {"P30": conv(G30["spread"]["headway"]["U"]), "P60": conv(G60["spread"]["headway"]["U"])}}
        for nm in P60_PAIRS:
            p = G60["pairs"][nm]
            pr = S.pair_read(cells60, p["A"], p["B"], eid60, ticks=(2,))
            beside["P60_pairs"][nm] = {"A": p["A"], "B": p["B"], "headway": ucf(p["headway"]), "cells": cells_of(p["family"]), "strategic_pos_30_50_80": [{"mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"]} for b in pr["tick2"]["pos"]]}
        for nm in FRESH_PAIRS:
            p = G30["pairs"][nm]
            beside["AT_FRESH_pairs"][nm] = {"A": p["A"], "B": p["B"], "headway": ucf(p["headway"]), "cells": cells_of(p["family"])}
        for nm in RULE_AT_AGE:
            p = G30["pairs"][nm]
            beside["rule_under_K_at_age"][nm] = {"A": p["A"], "B": p["B"], "headway": ucf(p["headway"])}
        for run, k in [(r, s) for r in sorted(RUNS) for s in SEEDS]:
            po = npz_load(os.path.join(tick_dir(a.pull, run, k), "preroll_offsets_%s.npz" % key_of(run, k)))
            if po is None:
                continue
            m = np.asarray(po["kind"]).astype(str) == "model"
            beside["offsets_at_t0"][key_of(run, k)] = {"position_m": off_stats(np.asarray(po["dpos"], np.float64)[m]), "heading_abs_deg": off_stats(np.abs(np.asarray(po["dpsi_deg"], np.float64)[m])),
                                                         "speed_abs_mps": off_stats(np.abs(np.asarray(po["dv"], np.float64)[m]))}
        res.update(names=names, pairs=pairs_out, cells=cells_out, families=fams, strategic=strat, beside=beside, reported_no_bar={"consequences_fixed_by_card_section_6": [{"case": c, "sentence": s} for c, s in cons]})
        print("== names")
        for k, v in names.items():
            print("   %-34s %s" % (k, v))
        print("== consequences fixed by section 6 of the card (which sentence applies; the card itself changes nothing)")
        for c, s in cons:
            print("   [%s] %s" % (c, s))
        print("== pairs at P = 30: U C F (pooled, interval, per seed) and the family readings (STRATEGIC: pos at 30, 50, 80 m at the tick t0+20)")
        for nm, p in pairs_out.items():
            print("   %-20s %s minus %s | %s | %s" % (nm, p["A"], p["B"], line(p["headway"]), fams[nm]))
        print("== cells (all ten per pair; the named ones decide the readings)")
        for nm, cs in cells_out.items():
            for c, v in cs.items():
                print("   %-20s %-30s D %+8.4f [%+.4f, %+.4f] %s %s%s" % (nm, c, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"], " *" if any(c in NAMED[f] for f in NAMED) else ""))
        print("== STRATEGIC pos (m) at the tick t0+20: diff [lo, hi] verdict")
        for nm, s_ in strat.items():
            print("   %-20s " % nm + " ; ".join("%dm %+.4f [%+.4f,%+.4f] %s" % (int(b["arc_m"]), b["mean"], b["lo"], b["hi"], b["verdict"]) for b in s_["pos_30_50_80m_tick_t0_plus_20"]))
        print("== beside, naming nothing")
        for k, v in beside.items():
            print("   %s %s" % (k, json.dumps(conv(v))[:2500]))
    else:
        print("== a control did not pass: no name, no level, no difference; the read is reported as written")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    detail = {g: {k: v for k, v in G.items() if k in ("restriction", "counts", "spread", "missing_or_incomplete")} for g, G in (("30", G30), ("60", G60), ("full", GF))}
    res["grid_reads_detail_controls_only"] = detail
    json.dump(conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
