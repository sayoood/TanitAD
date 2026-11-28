#!/usr/bin/env python
"""read_telegap.py -- read of card PREREG_V8F_PROOF_ADD122 (TELEGAP-T0). CPU, numpy only. Tier T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance).

  python read_telegap.py --stage controls --out R/TELEGAP_CONTROLS.json  --dump NAME=PATH ... [inputs, see TELEGAP_READ_DEFS.md]
  python read_telegap.py --stage names --controls R/TELEGAP_CONTROLS.json --out R/TELEGAP_RESULT.json [--log R/read_names.log]
  python read_telegap.py --selftest

Stage controls: runs t0 to t6 of card section 5 and PRINTS only `t<i> PASS|FAIL` lines and the last line `ZZ READ-TELEGAP controls all_pass <True|False>`.
Every value goes to the controls file: under `payload` when all seven pass, under `withheld` when one fails (then no name and no arm value exists).
Stage names: refuses unless t0 to t6 are all present and pass; writes the result file and a names log; derives every name a second time apart.
The reader of card ADD108 (read_lookgap.py, and through it read_lookat.py) is imported unchanged; their sha256 are recorded in the outputs."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD122"
PRE = "TELEGAP-T0"
CONTROLS = tuple("t%d" % i for i in range(7))
N_BOOT, BOOT_SEED = 2000, 122
V8 = "/workspace/v8proof"
EXPECT = {"rows": 3841, "clips": 641}
REF_D = {"D_0": 3.960, "D_1": 3.913, "D_2": 4.112}
BANKED_FIXED = (3.960, 3.913, 4.112, 4.079, 4.056)
RANGE_PAD = 0.15
TOL_E = 1e-3
LAB_TOL = 1e-5
SHORT_M = 5.0
BAR_RATIO = 0.85
FAR_HALF = 0.5
MATCH_MIN = 0.95
BAD_CLIP_MAX = 0.05
T2_MIN_SHARE = 0.9
PERM_MIN = 0.95
SLOPE_WIN = 25
ARMS = {"D": ("D_0", "D_1", "D_2"), "M": ("M_0", "M_1", "M_2"), "F": ("F_0", "F_1", "F_2")}
NINE = ARMS["D"] + ARMS["M"] + ARMS["F"]
NEW_SIX = ARMS["M"] + ARMS["F"]
EXTRA = ("D2_0", "FP_0")
ALL_DUMPS = NINE + EXTRA
GAP_BINS = ((0, 10), (10, 20), (20, 40), (40, 60))
T8 = {"az": 0.0, "el": -2.0, "half": 4.0}
T4 = {"az": 0.0, "el": -1.0, "half": 2.0}
ALLOWED_CHANGES = ("N_WIN", "N_TOK", "ARMS", "TokenStore", "build_samples", "token_store", "sample_builder")   # note N2 of card ADD122: the five module values tele_fit.py sets
OK_STATUS = ("ok", "skipped_complete")
TIER = "T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance)"
FAMILIES = ("LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not read (n = 0 for each; reason: T0 read of a perception head, no trajectory is produced; they are read by the rollout card that follows if the bar is met)")
LOOKGAP_DEFAULT = V8 + "/lookgap/kit_read/read_lookgap.py"
LOOKAT_DEFAULT = V8 + "/lookat/kit/read_lookat.py"


# ============================================================================================ the imported readers
def load_lookgap(path):
    """import read_lookgap.py unchanged from `path` (fallback: the sibling folder on the dev box). -> (module, path used)"""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (path, os.path.join(here, "..", "..", "lookgap", "read", "read_lookgap.py")):
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("read_lookgap", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["read_lookgap"] = mod
            spec.loader.exec_module(mod)
            return mod, os.path.abspath(p)
    raise FileNotFoundError("read_lookgap.py not found at %s" % path)


def _res(bad, **kw):
    d = {"pass": not bad, "failures": bad}
    d.update(kw)
    return d


def fnum(x):
    return None if x is None or not np.isfinite(x) else float(x)


def read_text(p):
    return io.open(p, encoding="utf-8", errors="replace").read() if p and os.path.isfile(p) else None


def try_json(p):
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def wj(path, obj):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, default=lambda o: float(o) if hasattr(o, "item") else str(o))


def wt(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def ids_of(p):
    with io.open(p, encoding="utf-8") as fh:
        return [ln.split()[0] for ln in fh if ln.strip()]


# ============================================================================================ values
def lay_all(LK, RL, J, paths):
    """every dump laid on the join rows: gap, lab_gap, lab_present, cov, tele_dropped (None when the dump has no such key). A dump that cannot be read gives None."""
    L = {}
    for k, p in paths.items():
        try:
            D = LK.load_npz(p)
            d = RL.lay(LK, J, D)
        except (OSError, ValueError, KeyError):
            L[k] = None
            continue
        d["tele_dropped"] = LK.on_rows(J, D, "tele_dropped").astype(bool) if "tele_dropped" in D else None
        for extra in ("const_gap",):
            if extra in D:
                d[extra] = float(np.asarray(D[extra]).reshape(-1)[0])
        L[k] = d
    return L


def lead_rows(L):
    """label lead of D_0 (lab_present and finite lab_gap): the base of the card's banked reference. Not clamped to 60 m."""
    d0 = L["D_0"]
    return d0["lab_present"].astype(bool) & np.isfinite(d0["lab_gap"])


def t3_rows(L):
    """control t3 rows: lead rows with a finite gap in D_0 to D_2, BEFORE the common frame set."""
    m = lead_rows(L)
    for k in ARMS["D"]:
        m = m & np.isfinite(L[k]["gap"])
    return m


def common_rows(L):
    """common frame set: lead within 60 m, a finite gap in all nine dumps, not tele_dropped in any new dump. Raises ValueError when a dump lacks tele_dropped."""
    lab = L["D_0"]["lab_gap"]
    with np.errstate(invalid="ignore"):
        m = lead_rows(L) & (lab >= 0.0) & (lab <= 60.0)
    for k in NINE:
        m = m & np.isfinite(L[k]["gap"])
    for k in NEW_SIX:
        if L[k]["tele_dropped"] is None:
            raise ValueError("dump %s has no tele_dropped array" % k)
        m = m & ~L[k]["tele_dropped"]
    return m


def mean_abs(L, k, rows):
    return fnum(np.abs(L[k]["gap"][rows] - L["D_0"]["lab_gap"][rows]).mean()) if rows.any() else None


def boot_idx(nc, n_boot, seed):
    return np.random.RandomState(seed).randint(0, nc, size=(n_boot, nc))


def boot_ratio(num, den, n_boot, seed):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    idx = boot_idx(len(num), n_boot, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    r = r[np.isfinite(r)]
    if r.size < n_boot // 2:
        return None
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def boot_diff_over(a, b, n, n_boot, seed):
    """paired cluster bootstrap of (sum a - sum b) / sum n over clips, the same draws as boot_ratio; two-sided 95 % percentile."""
    a, b, n = (np.asarray(x, np.float64) for x in (a, b, n))
    idx = boot_idx(len(a), n_boot, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (a[idx].sum(axis=1) - b[idx].sum(axis=1)) / n[idx].sum(axis=1)
    d = d[np.isfinite(d)]
    if d.size < n_boot // 2:
        return None
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def slopes(common, off, n, series):
    """windows of SLOPE_WIN consecutive join rows (greedy, non-overlapping, every row in the common set); least-squares slope per frame of each series. -> {name: array}, count"""
    x = np.arange(SLOPE_WIN, dtype=np.float64)
    xc = x - x.mean()
    den = float((xc ** 2).sum())
    out = {k: [] for k in series}
    for o, nn in zip(off, n):
        c = common[o:o + nn]
        i = 0
        while i + SLOPE_WIN <= nn:
            if c[i:i + SLOPE_WIN].all():
                for k, v in series.items():
                    y = v[o + i:o + i + SLOPE_WIN]
                    out[k].append(float((xc * (y - y.mean())).sum() / den))
                i += SLOPE_WIN
            else:
                i += 1
    return {k: np.asarray(v) for k, v in out.items()}


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return fnum(np.corrcoef(a, b)[0, 1])


def compute_values(LK, RL, J, L, box, n_boot=N_BOOT, seed=BOOT_SEED):
    """J the evaluation join, L {dump name: laid dump}, box {has, az, el} on the join rows. -> payload (all values of the read)."""
    n = np.asarray(J["n"], np.int64)
    off = np.asarray(J["off"], np.int64)
    nc = len(n)
    ci_all = np.repeat(np.arange(nc), n)
    lab_all = L["D_0"]["lab_gap"]
    lead = lead_rows(L)
    t3 = t3_rows(L)
    com = common_rows(L)
    differ = np.zeros(len(com), bool)
    for k in NINE + tuple(x for x in EXTRA if L.get(x) is not None):
        o = L[k]
        differ |= o["cov"] & ((o["lab_present"] != L["D_0"]["lab_present"]) | ~np.isclose(o["lab_gap"], lab_all, rtol=LAB_TOL, atol=LAB_TOL, equal_nan=True))
    dropped_any = np.zeros(len(com), bool)
    for k in NEW_SIX:
        dropped_any |= L[k]["tele_dropped"]
    with np.errstate(invalid="ignore"):
        beyond60 = int((lead & (lab_all > 60.0)).sum())
    P = {"n_clips": int(nc), "n_boot": n_boot, "seed": seed, "tier": TIER, "families": FAMILIES,
         "interval_level": "two-sided 95 % percentile, cluster bootstrap over the evaluation clips, fits held fixed",
         "n_lead_rows": int(lead.sum()), "n_lead_beyond_60": beyond60, "n_t3_rows": int(t3.sum()), "n_tele_dropped_on_t3_rows": int((dropped_any & t3).sum()),
         "n_common": int(com.sum()), "n_label_fields_differ_on_common": int((differ & com).sum())}
    # t3 values: before the common frame set
    P["t3"] = {"n_rows": int(t3.sum()), "E": {k: mean_abs(L, k, t3) for k in ARMS["D"]}}
    # t4 / t5 values on the t3 rows
    P["t4"] = {"n_rows": int(t3.sum()), "n_finite": int((t3 & np.isfinite(L["D2_0"]["gap"])).sum()) if L.get("D2_0") else None,
               "E": mean_abs(L, "D2_0", t3 & np.isfinite(L["D2_0"]["gap"])) if L.get("D2_0") else None}
    if L.get("FP_0") is not None:
        fm = t3 & np.isfinite(L["FP_0"]["gap"])
        const, src = None, None
        if L["FP_0"].get("const_gap") is not None:
            const, src = L["FP_0"]["const_gap"], "dump key const_gap of FP_0"
        else:
            const, src = float(np.median(lab_all[t3])) if t3.any() else None, "median label gap of the evaluation frames (t3 rows)"
        fp_drop = L["FP_0"].get("tele_dropped")                                                     # note N3 of card ADD122: rows the head does not hold
        n_marked = int((t3 & fp_drop).sum()) if fp_drop is not None else 0
        n_unmarked_bad = int((t3 & ~np.isfinite(L["FP_0"]["gap"]) & ~(fp_drop if fp_drop is not None else np.zeros_like(t3))).sum())
        P["t5"] = {"n_rows": int(fm.sum()), "n_t3_rows": int(t3.sum()), "n_marked_dropped": n_marked, "n_unmarked_not_finite": n_unmarked_bad,
                   "E_fp": mean_abs(L, "FP_0", fm), "const": const, "const_source": src,
                   "E_const": fnum(np.abs(lab_all[fm] - const).mean()) if (fm.any() and const is not None) else None}
    else:
        P["t5"] = None
    # common frame set values
    rows = np.flatnonzero(com)
    ci = ci_all[rows]
    lab = lab_all[rows]
    r_cl = np.bincount(ci, minlength=nc).astype(np.float64)
    absk = {k: np.abs(L[k]["gap"][rows] - lab) for k in NINE}
    sgn = {k: L[k]["gap"][rows] - lab for k in NINE}
    sums = {k: np.bincount(ci, weights=absk[k], minlength=nc) for k in NINE}
    P["E"] = {k: fnum(absk[k].mean()) if len(rows) else None for k in NINE}
    P["mean_E"] = {X: fnum(np.mean([P["E"][k] for k in ks])) if len(rows) else None for X, ks in ARMS.items()}
    mnum = {X: sum(sums[k] for k in ks) / 3.0 for X, ks in ARMS.items()}
    far_m = RL.gap_bin_mask(lab, 40, 60)
    far_n = np.bincount(ci[far_m], minlength=nc).astype(np.float64)
    far_s = {k: np.bincount(ci[far_m], weights=sgn[k][far_m], minlength=nc) for k in NINE}
    P["sums"] = {"r": r_cl.tolist(), "abs": {k: sums[k].tolist() for k in NINE}, "far_n": far_n.tolist(), "far_signed": {k: far_s[k].tolist() for k in NINE}}
    P["R"], P["iv"] = {}, {}
    for X in ("F", "M"):
        P["R"][X] = fnum(P["mean_E"][X] / P["mean_E"]["D"]) if len(rows) and P["mean_E"]["D"] and P["mean_E"]["D"] > 0 else None
        P["iv"][X] = boot_ratio(mnum[X], mnum["D"], n_boot, seed) if len(rows) else None
    # error by gap bin, mean over the three fits of each arm
    P["bins"] = {}
    for X, ks in ARMS.items():
        out = []
        for lo, hi in GAP_BINS:
            m = RL.gap_bin_mask(lab, lo, hi)
            if not m.any():
                out.append({"bin": "%d-%d m" % (lo, hi), "n": 0, "E": None, "signed": None, "short5": None})
                continue
            out.append({"bin": "%d-%d m" % (lo, hi), "n": int(m.sum()), "E": fnum(np.mean([absk[k][m].mean() for k in ks])), "signed": fnum(np.mean([sgn[k][m].mean() for k in ks])),
                        "short5": fnum(np.mean([(sgn[k][m] < -SHORT_M).mean() for k in ks]))})
        P["bins"][X] = out
    # far leads
    nfar = float(far_n.sum())
    sd = {X: (fnum(np.mean([far_s[k].sum() / nfar for k in ARMS[X]])) if nfar > 0 else None) for X in ARMS}
    fsum = {X: sum(far_s[k] for k in ARMS[X]) / 3.0 for X in ARMS}
    P["far"] = {"n": int(nfar), "n_clips": int((far_n > 0).sum()), "signed": sd,
                "iv_diff_F_minus_D": boot_diff_over(fsum["F"], fsum["D"], far_n, n_boot, seed) if nfar > 0 else None}
    # printed with no name
    has = box["has"][rows]
    az, el = box["az"][rows], box["el"][rows]
    in8 = has & (np.abs(az - T8["az"]) <= T8["half"]) & (np.abs(el - T8["el"]) <= T8["half"])
    in4 = has & (np.abs(az - T4["az"]) <= T4["half"]) & (np.abs(el - T4["el"]) <= T4["half"])
    ef = np.mean([absk[k] for k in ARMS["F"]], axis=0) if len(rows) else np.zeros(0)
    P["inside"] = {"n_lead": int(len(rows)), "n_with_box": int(has.sum()), "share_T8": fnum(in8.sum() / has.sum()) if has.any() else None, "share_T4": fnum(in4.sum() / has.sum()) if has.any() else None,
                   "F_inside_T8": {"n": int(in8.sum()), "E": fnum(ef[in8].mean()) if in8.any() else None},
                   "F_outside_T8": {"n": int((has & ~in8).sum()), "E": fnum(ef[has & ~in8].mean()) if (has & ~in8).any() else None},
                   "note": "frames with a label box only; box directions are in the wide camera frame (T8 centre (0, -2) half 4 deg, T4 centre (0, -1) half 2 deg)"}
    mean_gap = {X: np.mean([L[k]["gap"] for k in ks], axis=0) for X, ks in ARMS.items()}
    sl = slopes(com, off, n, dict({"label": lab_all}, **mean_gap))
    P["slope"] = {"n_windows": int(len(sl["label"])), "window_frames": SLOPE_WIN,
                  "corr_with_label_closing_speed": {X: corr(sl[X], sl["label"]) for X in ARMS},
                  "note": "closing speed = minus the slope of the label gap; the read closing speed = minus the slope of the read gap (the mean of the three fits); the correlation equals that of the two slopes"}
    return P


# ============================================================================================ names (two derivations)
def name_x(X, iv):
    if iv is None:
        return None
    if iv[1] < 1.0:
        return "%s-%s-CUTS-GAP-ERROR" % (PRE, X)
    if iv[0] > 1.0:
        return "%s-%s-WORSE" % (PRE, X)
    return "%s-%s-NOT-SEPARATED" % (PRE, X)


def name_far(sF, sD, iv):
    if sF is not None and sD is not None and iv is not None and abs(sF) <= FAR_HALF * abs(sD) and (iv[0] > 0.0 or iv[1] < 0.0):
        return PRE + "-FAR-LEADS-READ"
    return PRE + "-FAR-LEADS-STAY-SHORT"


def names_first(P):
    """card section 4, first derivation (values of the payload). -> {F, M, FAR, bar}"""
    out = {X: name_x(X, P["iv"][X]) for X in ("F", "M")}
    out["FAR"] = name_far(P["far"]["signed"]["F"], P["far"]["signed"]["D"], P["far"]["iv_diff_F_minus_D"])
    out["bar"] = bool(P["R"]["F"] is not None and P["iv"]["F"] is not None and P["R"]["F"] <= BAR_RATIO and P["iv"]["F"][1] < 1.0)
    return out


def _pct(v, q):
    pos = (len(v) - 1) * q / 100.0
    a = int(pos)
    b = min(a + 1, len(v) - 1)
    return v[a] + (v[b] - v[a]) * (pos - a)


def names_second(P):
    """second derivation, written apart: per-clip sums only, plain loops, hand-made percentiles. -> ({F, M, FAR, bar}, {F: iv, M: iv, FAR: iv})"""
    S = P["sums"]
    nc = len(S["r"])
    nb = P["n_boot"]
    idx = np.random.RandomState(P["seed"]).randint(0, nc, size=(nb, nc)).tolist()

    def arm(X):
        return [sum(S["abs"]["%s_%d" % (X, i)][j] for i in range(3)) / 3.0 for j in range(nc)]

    def farsum(X):
        return [sum(S["far_signed"]["%s_%d" % (X, i)][j] for i in range(3)) / 3.0 for j in range(nc)]

    def interval(vals):
        vals = sorted(vals)
        if len(vals) < nb // 2:
            return None
        return [_pct(vals, 2.5), _pct(vals, 97.5)]

    den = arm("D")
    ivs, names, ratio = {}, {}, {}
    for X in ("F", "M"):
        num = arm(X)
        v = []
        for row in idx:
            d = sum(den[j] for j in row)
            if d > 0:
                v.append(sum(num[j] for j in row) / d)
        ivs[X] = interval(v)
        names[X] = name_x(X, ivs[X])
        ratio[X] = (sum(num) / sum(den)) if sum(den) > 0 else None
    fn = S["far_n"]
    nf = sum(fn)
    fF, fD = farsum("F"), farsum("D")
    v = []
    for row in idx:
        d = sum(fn[j] for j in row)
        if d > 0:
            v.append((sum(fF[j] for j in row) - sum(fD[j] for j in row)) / d)
    ivs["FAR"] = interval(v)
    sF, sD = (sum(fF) / nf, sum(fD) / nf) if nf > 0 else (None, None)
    names["FAR"] = name_far(sF, sD, ivs["FAR"])
    names["bar"] = bool(ratio["F"] is not None and ivs["F"] is not None and ratio["F"] <= BAR_RATIO and ivs["F"][1] < 1.0)
    return names, ivs


def derive_names(P):
    n1 = names_first(P)
    n2, iv2 = names_second(P)
    first_iv = {"F": P["iv"]["F"], "M": P["iv"]["M"], "FAR": P["far"]["iv_diff_F_minus_D"]}
    agree = n1 == n2
    for k in ("F", "M", "FAR"):
        a, b = first_iv[k], iv2[k]
        agree = agree and ((a is None and b is None) or (a is not None and b is not None and bool(np.allclose(a, b, atol=1e-9, rtol=0))))
    return n1, n2, agree


# ============================================================================================ controls (pure functions of facts)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _entries(rec):
    """sha256 entries of the banked list record: top level and the `files` mapping; key names without a trailing _sha256 / _sha."""
    out = {}
    if not isinstance(rec, dict):
        return out
    for src in (rec, rec.get("files") if isinstance(rec.get("files"), dict) else {}):
        for k, v in src.items():
            if isinstance(v, str) and HEX64.match(v.lower()):
                out[re.sub(r"_sha(256)?$", "", k)] = v.lower()
    return out


def ctl_t0(f):
    """lists: the lists the encode used equal, by sha256, the entries of the banked list record; train_list and eval_list must be among them."""
    bad = []
    ent = _entries(f.get("record"))
    if not ent:
        return _res(["the banked list record is missing or holds no sha256 entry"])
    files = f.get("files") or {}
    for need in ("train_list", "eval_list"):
        if need not in files:
            bad.append("no %s passed to compare" % need)
    for name, sha in files.items():
        if sha is None:
            bad.append("%s: file unreadable" % name)
        elif name not in ent:
            bad.append("%s: no entry in the list record" % name)
        elif ent[name] != sha:
            bad.append("%s: sha256 differs from the list record" % name)
    return _res(bad, compared=sorted(files))


def ctl_t1(f, ex=EXPECT):
    """staging: pilot clips without a tele clip or with matched frames below 0.95 of the pilot frames are at most 5 per cent; every staged clip is 1920 x 1080 and crc_ok."""
    bad = []
    clips = f.get("clips") or []
    if len(clips) != ex["clips"]:
        bad.append("%d pilot clips listed, expected %d" % (len(clips), ex["clips"]))
    rows, man, mask = f.get("stage_rows") or {}, f.get("manifest") or {}, f.get("mask") or {}
    no_tele, low, disagree = [], [], []
    for c in clips:
        r, m, mk = rows.get(c), man.get(c), mask.get(c)
        if not r or r.get("status") not in OK_STATUS or m is None or mk is None:
            no_tele.append(c)
            continue
        nfr, nmt = mk
        if int(m.get("n_frames", -1)) != nfr:
            disagree.append(c)
        if nfr <= 0 or nmt < MATCH_MIN * nfr:
            low.append(c)
    n = max(len(clips), 1)
    nbad = len(set(no_tele) | set(low) | set(disagree))
    if nbad > BAD_CLIP_MAX * n:
        bad.append("%d of %d clips without a tele clip (%d) or with fewer matched frames than %.2f of the pilot frames (%d) or a frame count that disagrees (%d): share %.4f above %.2f"
                   % (nbad, len(clips), len(no_tele), MATCH_MIN, len(low), len(disagree), nbad / n, BAD_CLIP_MAX))
    badstage = [c for c, r in rows.items() if r.get("status") in OK_STATUS and not (r.get("width") == 1920 and r.get("height") == 1080 and r.get("crc_ok") is True)]
    if badstage:
        bad.append("%d staged clips not 1920 x 1080 with crc_ok (first %s)" % (len(badstage), badstage[0][:8]))
    return _res(bad, n_clips=len(clips), n_no_tele=len(no_tele), n_low_match=len(low), n_frames_disagree=len(disagree), share_bad=nbad / n, n_staged_ok=sum(1 for r in rows.values() if r.get("status") in OK_STATUS))


def ctl_t2(rec):
    """geometry: the JSON of the tele kit's ctl_t2.py: pass true, share >= 0.9, n_frames > 0, threshold 0.9."""
    if not isinstance(rec, dict):
        return _res(["the t2 record is missing or unreadable"])
    bad = []
    if rec.get("pass") is not True:
        bad.append("pass is not true")
    sh = rec.get("share")
    if not isinstance(sh, (int, float)) or sh < T2_MIN_SHARE:
        bad.append("share %s below %.2f" % (sh, T2_MIN_SHARE))
    if not isinstance(rec.get("n_frames"), int) or rec.get("n_frames") <= 0:
        bad.append("n_frames %s" % rec.get("n_frames"))
    if not isinstance(rec.get("threshold"), (int, float)) or abs(rec["threshold"] - T2_MIN_SHARE) > 1e-9:
        bad.append("threshold %s is not %.2f" % (rec.get("threshold"), T2_MIN_SHARE))
    return _res(bad, share=sh, n_frames=rec.get("n_frames"))


def ctl_t3(P3, ex=EXPECT, ref=REF_D):
    """reference: E of D_0 to D_2 on the rows before the common frame set within 1e-3 m of the banked values, and the expected number of rows."""
    bad = []
    for k, v in ref.items():
        e = (P3.get("E") or {}).get(k)
        if e is None or abs(e - v) > TOL_E:
            bad.append("E(%s) %s, expected %.3f within %g" % (k, e, v, TOL_E))
    if P3.get("n_rows") != ex["rows"]:
        bad.append("%s rows, expected %d" % (P3.get("n_rows"), ex["rows"]))
    return _res(bad)


def _norm_item(x):
    s = x.get("name") if isinstance(x, dict) else x
    s = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
    return re.sub(r"^the_", "", s)


def ctl_t4(P4, rec, banked=BANKED_FIXED, pad=RANGE_PAD, allowed=ALLOWED_CHANGES):
    """variant: the fit record names only declared changes, two windows, seed 0; the D2_0 gap error on the t3 rows lies in [min banked - 0.15, max banked + 0.15]."""
    bad = []
    lo, hi = min(banked) - pad, max(banked) + pad
    e = (P4 or {}).get("E")
    if e is None or not (lo <= e <= hi):
        bad.append("E(D2_0) %s not in [%.3f, %.3f]" % (e, lo, hi))
    if P4 and P4.get("n_finite") != P4.get("n_rows"):
        bad.append("D2_0 has %s finite rows of the %s rows" % (P4.get("n_finite"), P4.get("n_rows")))
    if not isinstance(rec, dict):
        return _res(bad + ["the fit record of D2_0 is missing or unreadable"])
    if rec.get("seed") != 0:
        bad.append("seed %s, expected 0" % rec.get("seed"))
    if rec.get("n_win") != 2:
        bad.append("n_win %s, expected 2" % rec.get("n_win"))
    if not isinstance(rec.get("n_tok"), int):
        bad.append("n_tok %s" % rec.get("n_tok"))
    if not (isinstance(rec.get("crop_fit_sha256"), str) and rec["crop_fit_sha256"]):
        bad.append("no crop_fit_sha256")
    ch = rec.get("module_values_changed", rec.get("changed_module_values"))                      # the key tele_fit.py writes; the second name is the tests' older form
    if not isinstance(ch, list):
        bad.append("changed_module_values is not a list")
    else:
        ok = {_norm_item(a) for a in allowed}
        undeclared = [str(x) for x in ch if _norm_item(x) not in ok]
        if undeclared:
            bad.append("changed values outside the declared items: %s" % undeclared[:5])
    return _res(bad, E=e, lo=lo, hi=hi)


def ctl_t5(P5):
    """permuted labels: E of FP_0 is at least 0.95 of the error of the constant read, on the same rows."""
    if not P5 or P5.get("E_fp") is None or P5.get("E_const") is None:
        return _res(["no permuted-label dump or no constant read"])
    bad = []
    if P5["E_fp"] < PERM_MIN * P5["E_const"]:
        bad.append("E(FP_0) %.4f below %.2f of the constant read's %.4f" % (P5["E_fp"], PERM_MIN, P5["E_const"]))
    if P5.get("n_unmarked_not_finite", (P5.get("n_t3_rows") or 0) - (P5.get("n_rows") or 0)) != 0:      # note N3: rows marked as dropped for a missing tele frame are not asked for
        bad.append("FP_0 has %s finite rows of the %s rows, %s marked as dropped, %s neither marked nor finite"
                   % (P5.get("n_rows"), P5.get("n_t3_rows"), P5.get("n_marked_dropped"), P5.get("n_unmarked_not_finite")))
    return _res(bad, const_source=P5.get("const_source"))


def ctl_t6(f):
    """tests and counts: each lead test output ends with `<n> passed` (no failed, no error) and is older than its start epoch; token files exist for every staged clip."""
    bad = []
    tests = f.get("tests") or {}
    if len(tests) < 2:
        bad.append("%d test outputs, at least 2 expected" % len(tests))
    for k, t in tests.items():
        if t.get("text") is None:
            bad.append("%s: no test output" % k)
            continue
        ln = [x for x in t["text"].splitlines() if x.strip()]
        last = ln[-1] if ln else ""
        if not re.search(r"\b\d+ passed\b", last) or re.search(r"failed|error", last, re.I):
            bad.append("%s: last line is not a passed line (%s)" % (k, last[:60]))
        if t.get("before") is None or t.get("mtime") is None or not t["mtime"] < t["before"]:
            bad.append("%s: modification time %s is not before the start epoch %s" % (k, t.get("mtime"), t.get("before")))
    ns, nt = f.get("n_staged"), f.get("n_tok_files")
    if not isinstance(ns, int) or ns <= 0 or not isinstance(nt, int) or nt < ns:
        bad.append("token files %s for %s staged clips" % (nt, ns))
    return _res(bad)


# ============================================================================================ gathering facts from disk
def parse_kv(items, cast=str):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        out[k] = cast(v)
    return out


def dump_paths(a):
    p = {"D_0": V8 + "/cropread/run1/dump/D_H5/reader_s0_score.npz", "D_1": V8 + "/cropread/run1/dump/D_H5/reader_s1_score.npz",
         "D_2": V8 + "/visseed10k/fitroot/dump/D_H5/reader_s2_score.npz"}
    p.update(parse_kv(a.dump))
    return p


def sha_of(LK, p):
    try:
        return LK.sha_file(p)
    except (OSError, TypeError):
        return None


def gather_t0(a, LK):
    return {"record": try_json(a.list_record) if a.list_record else None, "files": {k: sha_of(LK, v) for k, v in parse_kv(a.list_file).items()}}


def gather_t1(a):
    rows = {}
    for p in a.stage_record or []:
        rec = try_json(p) or {}
        for r in rec.get("clips") or []:
            c = r.get("clip")
            if c is None:
                continue
            if r.get("status") in OK_STATUS or c not in rows:
                rows[c] = r
    man = (try_json(a.enc_manifest) or {}).get("clips") or {}
    clips = ids_of(a.all_clips) if a.all_clips and os.path.isfile(a.all_clips) else []
    mask = {}
    for c in clips:
        pth = os.path.join(a.match_dir or "", c + ".npz")
        if os.path.isfile(pth):
            try:
                m = np.load(pth)["mask"].astype(bool)
                mask[c] = (int(len(m)), int(m.sum()))
            except (OSError, ValueError, KeyError):
                pass
    return {"clips": clips, "stage_rows": rows, "manifest": man, "mask": mask}


def gather_t6(a, n_staged):
    before = parse_kv(a.test_before, float)
    tests = {}
    for k, p in parse_kv(a.test_out).items():
        tests[k] = {"text": read_text(p), "mtime": os.path.getmtime(p) if os.path.isfile(p) else None, "before": before.get(k)}
    return {"tests": tests, "n_staged": n_staged, "n_tok_files": a.n_tok_files}


def tracks_of(folder, clips):
    out = {}
    for c in clips:
        p = os.path.join(folder, c + ".npz")
        if os.path.isfile(p):
            z = np.load(p)
            has = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"]) & np.isfinite(z["ctr_el"])
            out[c] = (has, np.where(has, z["ctr_az"], 0.0), np.where(has, z["ctr_el"], 0.0))
    return out


def assemble(ctl, P):
    allp = sorted(ctl) == sorted(CONTROLS) and all(v["pass"] for v in ctl.values())
    res = {"card": CARD, "tier": TIER, "controls": ctl, "all_pass": bool(allp)}
    res["payload" if allp else "withheld"] = P
    return res


def run_controls(a, LK, RL):
    ctl, P = {}, None
    f1 = gather_t1(a)
    ctl["t0"] = ctl_t0(gather_t0(a, LK))
    ctl["t1"] = ctl_t1(f1, a.expect)
    ctl["t2"] = ctl_t2(try_json(a.t2_json) if a.t2_json else None)
    try:
        J = LK.load_npz(a.eval_join)
        paths = dump_paths(a)
        L = lay_all(LK, RL, J, paths)
        missing = [k for k in NINE + EXTRA if L.get(k) is None]
        if missing:
            raise ValueError("dump not readable or not given: %s" % missing)
        if a.const is not None:
            L["FP_0"]["const_gap"] = float(a.const)
        clips = [str(c) for c in J["clip"]]
        box = LK.box_on_rows(J, tracks_of(a.box_dir, clips))
        P = compute_values(LK, RL, J, L, box, a.n_boot)
        ctl["t3"] = ctl_t3(P["t3"], a.expect, a.ref_d)
        ctl["t4"] = ctl_t4(P["t4"], try_json(a.d2_fit_record) if a.d2_fit_record else None, a.banked, a.range_pad)
        ctl["t5"] = ctl_t5(P["t5"])
    except (OSError, ValueError, KeyError, TypeError, IndexError) as ex:
        for k in ("t3", "t4", "t5"):
            ctl[k] = _res(["the read could not be computed: %s: %s" % (type(ex).__name__, ex)])
    ctl["t6"] = ctl_t6(gather_t6(a, ctl["t1"].get("n_staged_ok")))
    return assemble(ctl, P)


def input_hashes(a, LK):
    paths = {"join": a.eval_join, "list_record": a.list_record, "enc_manifest": a.enc_manifest, "t2_json": a.t2_json, "d2_fit_record": a.d2_fit_record}
    for k, p in parse_kv(a.list_file).items():
        paths["list_" + k] = p
    for k, p in dump_paths(a).items():
        paths["dump_" + k] = p
    for k, p in parse_kv(a.test_out).items():
        paths["tests_" + k] = p
    return {k: sha_of(LK, p) if p else None for k, p in paths.items()}


# ============================================================================================ output
def fv(x, d=3):
    return "None" if x is None else ("%." + str(d) + "f") % x


def fiv(iv):
    return "[%.4f, %.4f]" % (iv[0], iv[1]) if iv else "None"


def print_sections(P, nm):
    L = ["tier: " + TIER, "families: " + FAMILIES]
    L.append("rows: lead rows of D_0 %d (of them beyond 60 m %d); rows of control t3 (finite gap in D_0 to D_2) %d; common frame set %d (tele_dropped removed %d of the t3 rows); rows where another dump's label fields differ from D_0: %d"
             % (P["n_lead_rows"], P["n_lead_beyond_60"], P["n_t3_rows"], P["n_common"], P["n_tele_dropped_on_t3_rows"], P["n_label_fields_differ_on_common"]))
    L.append("E (mean absolute gap error in m, common frame set, %d evaluation clips): " % P["n_clips"] + "; ".join("%s %s" % (k, fv(P["E"][k])) for k in NINE))
    for X in ("F", "M"):
        L.append("ratio %s / D = mean E(%s) / mean E(D) = %s (%s / %s); interval %s (%d draws, RandomState(%d), %s)"
                 % (X, X, fv(P["R"][X], 4), fv(P["mean_E"][X], 4), fv(P["mean_E"]["D"], 4), fiv(P["iv"][X]), P["n_boot"], P["seed"], P["interval_level"]))
    for X in ("F", "M"):
        L.append("name: %s" % (nm[X] if nm[X] else "none (no interval for %s)" % X))
    fr = P["far"]
    L.append("far leads (label gap 40-60 m), n %d in %d clips; mean signed error (read minus label) of the mean of three fits: D %s, M %s, F %s; paired difference F minus D interval %s"
             % (fr["n"], fr["n_clips"], fv(fr["signed"]["D"]), fv(fr["signed"]["M"]), fv(fr["signed"]["F"]), fiv(fr["iv_diff_F_minus_D"])))
    L.append("name: %s" % nm["FAR"])
    L.append("bar met: %s (arm F ratio %s at most %.2f and its interval %s below 1)" % ("yes" if nm["bar"] else "no", fv(P["R"]["F"], 4), BAR_RATIO, fiv(P["iv"]["F"])))
    for X in ("D", "M", "F"):
        L.append("by label gap, arm %s (mean of three fits): " % X + "; ".join("%s n %d E %s signed %s short>5m %s" % (b["bin"], b["n"], fv(b["E"]), fv(b["signed"]), fv(b["short5"])) for b in P["bins"][X]))
    i = P["inside"]
    L.append("printed with no name: share of lead frames with a box whose label box centre lies inside T8 %s and inside T4 %s (%d of %d lead frames have a box)" % (fv(i["share_T8"], 4), fv(i["share_T4"], 4), i["n_with_box"], i["n_lead"]))
    L.append("arm F error (mean of three fits) inside T8: n %d E %s; outside T8: n %d E %s" % (i["F_inside_T8"]["n"], fv(i["F_inside_T8"]["E"]), i["F_outside_T8"]["n"], fv(i["F_outside_T8"]["E"])))
    s = P["slope"]
    L.append("correlation of read closing speed with label closing speed over %d-frame windows (n windows %d): " % (s["window_frames"], s["n_windows"]) + "; ".join("%s %s" % (X, fv(s["corr_with_label_closing_speed"][X], 4)) for X in ARMS)
             + " (the requirement of r 0.9 of row 147 is not a claim of this card)")
    L.append("interval note: it answers what another draw of evaluation clips would say for these fits; three fit seeds on one set of train clips is a small number, and a separated interval is necessary, not sufficient")
    return L


def run_names(a, LK):
    cj = try_json(a.controls) or {}
    if cj.get("card") != CARD or cj.get("all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any(v.get("pass") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass with t0 to t6 present: no name and no arm value is written")
        print("ZZ READ-TELEGAP names rc 3")
        return 3
    P = cj["payload"]
    n1, n2, agree = derive_names(P)
    if not agree:
        print("STOP: the two derivations of the names disagree: %s | %s" % (n1, n2))
        print("ZZ READ-TELEGAP names rc 4")
        return 4
    logp = a.log or (os.path.splitext(a.out)[0] + "_names.log")
    if os.path.exists(a.out) or os.path.exists(logp):
        print("REFUSED: %s or %s exists" % (a.out, logp))
        print("ZZ READ-TELEGAP names rc 2")
        return 2
    res = {"card": CARD, "tier": TIER, "names": n1, "second_derivation_agrees": True, "controls_file": os.path.abspath(a.controls), "controls_sha256": LK.sha_file(a.controls),
           "reader_sha256": LK.sha_file(os.path.abspath(__file__)), "values": P}
    wj(a.out, res)
    rc = 0 if (n1["F"] and n1["M"]) else 5
    lines = print_sections(P, n1) + ["ZZ READ-TELEGAP names rc %d" % rc]
    wt(logp, "\n".join(lines) + "\n")
    for ln in lines:
        print(ln)
    return rc


# ============================================================================================ selftest world and tree
ERR = {"cuts": (1.5, 1.5, 1.5), "mild": (3.5, 3.5, 3.5), "worse": (6.0, 6.0, 6.0), "sep": (4.0, 3.9, 4.1)}
FAR_BIAS = {"cuts": 1.0, "mild": 5.0, "worse": 6.0, "sep": 6.0}
SYN_N, SYN_NFR = 12, 30
TG_D = ARMS["D"]
CTL_FAULTS = {"t0", "t1", "t2", "t3", "t4", "t5", "t6"}


def syn_world(kf="cuts", km="cuts", seed=5, amp=0.2, zero=False):
    """synthetic evaluation world: join, dumps (names D_0..F_2, D2_0, FP_0), box tracks. `zero`: every error 0 (no interval)."""
    rng = np.random.RandomState(seed)
    nc, nfr = SYN_N, SYN_NFR
    nrow = nc * nfr
    J = {"clip": np.asarray(["c%02d" % i for i in range(nc)]), "n": np.full(nc, nfr), "off": np.arange(nc) * nfr, "lead_gap": np.zeros(nrow), "lead_valid": np.ones(nrow, np.int8)}
    ci = np.repeat(np.arange(nc), nfr)
    fr = np.tile(np.arange(nfr), nc)
    base = np.linspace(14.0, 46.0, nc)[ci]
    slope = np.linspace(-0.3, 0.3, nc)[ci]
    lab_full = base + slope * fr
    present = (fr >= 2) & (ci % 4 != 3)
    lab = np.where(present, lab_full, np.nan)
    sign = np.where(rng.rand(nrow) < 0.5, -1.0, 1.0)
    far = np.nan_to_num(lab, nan=0.0) >= 40.0
    arm_kind = {"D": None, "M": km, "F": kf}
    dumps = {}
    names = list(NINE) + list(EXTRA)
    for k in names:
        X = k[0]
        i = int(k[-1])
        if k.startswith("D2") or k == "FP_0":
            err, fb = (4.0, 6.0) if k.startswith("D2") else (0.0, 0.0)
        elif X == "D":
            err, fb = (4.0, 3.9, 4.1)[i], 6.0
        else:
            err, fb = ERR[arm_kind[X]][i], FAR_BIAS[arm_kind[X]]
        if zero:
            err, fb = 0.0, 0.0
        mult = 1.0 + amp * np.random.RandomState(100 + sum(map(ord, k))).uniform(-1, 1, nc)[ci]
        gap = lab + sign * err * mult - fb * far + (0.0 if zero else 0.01) * rng.randn(nrow)
        if k == "FP_0":
            gap = np.where(present, rng.permutation(np.nan_to_num(lab, nan=30.0)), np.nan)
        gap = np.where(present, gap, rng.uniform(5, 50, nrow))
        d = {"clips": J["clip"], "clip_index": ci.copy(), "frame": fr.copy(), "gap": gap, "valid": np.ones(nrow, bool), "lab_present": present.copy(), "lab_gap": lab.copy()}
        if k not in ARMS["D"]:
            d["tele_dropped"] = np.zeros(nrow, bool)
        dumps[k] = d
    dumps["M_0"]["tele_dropped"][np.flatnonzero(present)[:4]] = True           # dropped rows: out of the common set only
    dumps["F_1"]["gap"][np.flatnonzero(present)[10]] = np.nan                  # a row without a finite gap in one new dump
    az = np.where(rng.rand(nrow) < 0.3, rng.choice([-1, 1], nrow) * rng.uniform(5, 12, nrow), rng.normal(0, 1.5, nrow))
    el = -2 + rng.normal(0, 1.5, nrow)
    return J, dumps, {"az": az, "el": el}


def syn_payload(LK, RL, kf="cuts", km="cuts", amp=0.2, zero=False, n_boot=300):
    J, dumps, bx = syn_world(kf, km, amp=amp, zero=zero)
    L = {k: RL.lay(LK, J, d) for k, d in dumps.items()}
    for k, d in dumps.items():
        L[k]["tele_dropped"] = LK.on_rows(J, d, "tele_dropped").astype(bool) if "tele_dropped" in d else None
    box = {"has": np.ones(len(bx["az"]), bool), "az": bx["az"], "el": bx["el"]}
    return compute_values(LK, RL, J, L, box, n_boot)


def write_tree(LK, RL, root, kf="cuts", km="cuts", fault=None):
    """a tiny pod-like tree under `root` (12 evaluation + 8 train clips); the planted `fault` flips exactly the named control. -> args namespace"""
    J, dumps, bx = syn_world(kf, km)
    root = os.path.abspath(root)
    sub = {k: os.path.join(root, k) for k in ("lists", "match", "box", "dump", "rec", "tests")}
    for p in sub.values():
        os.makedirs(p, exist_ok=True)
    ev = [str(c) for c in J["clip"]]
    tr = ["t%d" % i for i in range(8)]
    wt(os.path.join(sub["lists"], "train.txt"), "".join("%s fit\n" % c for c in tr))
    wt(os.path.join(sub["lists"], "eval.txt"), "".join(c + "\n" for c in ev))
    wt(os.path.join(sub["lists"], "all.txt"), "".join(c + "\n" for c in tr + ev))
    th, eh = LK.sha_file(os.path.join(sub["lists"], "train.txt")), LK.sha_file(os.path.join(sub["lists"], "eval.txt"))
    wj(os.path.join(sub["rec"], "list_record.json"), {"train_list_sha256": th, "eval_list_sha256": eh})
    nfr = SYN_NFR
    rows, man = [], {}
    for c in tr + ev:
        rows.append({"clip": c, "chunk": 1, "bytes": 5, "width": 1920, "height": 1080, "n_frames": 40, "crc_ok": True, "status": "ok"})
        man[c] = {"n_frames": 40, "n_matched": 40, "sha256": "0" * 64, "bytes": 1}
        np.savez(os.path.join(sub["match"], c + ".npz"), mask=np.ones(40, bool))
    wj(os.path.join(sub["rec"], "stage_1.json"), {"clips": rows})
    wj(os.path.join(sub["rec"], "MANIFEST.json"), {"clips": man})
    wj(os.path.join(sub["rec"], "T2.json"), {"pass": True, "share": 0.95, "n_frames": 200, "threshold": 0.9})
    wj(os.path.join(sub["rec"], "d2_fit.json"), {"arm": "D", "seed": 0, "n_win": 2, "n_tok": 130, "crop_fit_sha256": "ab" * 32, "changed_module_values": ["N_WIN", "N_TOK"]})
    now = time.time()
    for k in ("enc", "fit"):
        p = os.path.join(sub["tests"], k + ".out")
        wt(p, "....\n18 passed in 1.0s\n")
        os.utime(p, (now - 500, now - 500))
    for c in ev:
        i = ev.index(c)
        s = slice(i * nfr, (i + 1) * nfr)
        np.savez(os.path.join(sub["box"], c + ".npz"), has_lead=np.ones(nfr, bool), ctr_az=bx["az"][s], ctr_el=bx["el"][s])
    jp = os.path.join(root, "join.npz")
    np.savez(jp, **J)
    paths = {}
    for k, d in dumps.items():
        p = os.path.join(sub["dump"], k + ".npz")
        np.savez(p, **d)
        paths[k] = p
    P = syn_payload(LK, RL, kf, km)
    ns = argparse.Namespace(list_record=os.path.join(sub["rec"], "list_record.json"), list_file=["train_list=" + os.path.join(sub["lists"], "train.txt"), "eval_list=" + os.path.join(sub["lists"], "eval.txt")],
                            stage_record=[os.path.join(sub["rec"], "stage_1.json")], enc_manifest=os.path.join(sub["rec"], "MANIFEST.json"), all_clips=os.path.join(sub["lists"], "all.txt"), match_dir=sub["match"],
                            t2_json=os.path.join(sub["rec"], "T2.json"), d2_fit_record=os.path.join(sub["rec"], "d2_fit.json"), const=None,
                            test_out=["enc=" + os.path.join(sub["tests"], "enc.out"), "fit=" + os.path.join(sub["tests"], "fit.out")], test_before=["enc=%f" % (now - 100), "fit=%f" % (now - 100)], n_tok_files=20,
                            box_dir=sub["box"], eval_join=jp, dump=["%s=%s" % kv for kv in paths.items()], n_boot=300, expect={"rows": P["n_t3_rows"], "clips": 20},
                            ref_d={k: round(P["t3"]["E"][k], 3) for k in ARMS["D"]}, banked=tuple(round(P["t3"]["E"][k], 3) for k in TG_D), range_pad=RANGE_PAD)
    ns.dump_paths = paths
    ns.sub = sub
    if fault:
        plant(LK, ns, fault, dumps)
    return ns


def plant(LK, ns, fault, dumps):
    sub = ns.sub
    if fault == "t0":
        wt(os.path.join(sub["lists"], "eval.txt"), "c00\n")
    elif fault == "t1":
        rows = try_json(os.path.join(sub["rec"], "stage_1.json"))["clips"]
        for r in rows[:3]:
            r["status"] = "fail: x"
        wj(os.path.join(sub["rec"], "stage_1.json"), {"clips": rows})
    elif fault == "t2":
        wj(os.path.join(sub["rec"], "T2.json"), {"pass": False, "share": 0.5, "n_frames": 200, "threshold": 0.9})
    elif fault == "t3":
        d = dict(dumps["D_1"])
        d["gap"] = d["gap"] + 1.0
        np.savez(ns.dump_paths["D_1"], **d)
    elif fault == "t4":
        wj(os.path.join(sub["rec"], "d2_fit.json"), {"arm": "D", "seed": 0, "n_win": 2, "n_tok": 130, "crop_fit_sha256": "ab" * 32, "changed_module_values": ["N_WIN", "lr"]})
    elif fault == "t5":
        d = dict(dumps["FP_0"])
        d["gap"] = np.where(d["lab_present"], d["lab_gap"] + 0.1, d["gap"])
        np.savez(ns.dump_paths["FP_0"], **d)
    elif fault == "t6":
        wt(os.path.join(sub["tests"], "fit.out"), "1 failed, 17 passed in 1s\n")


def selftest(LK, RL):
    ok = True
    for kf, km, want in (("cuts", "worse", (PRE + "-F-CUTS-GAP-ERROR", PRE + "-M-WORSE")), ("worse", "cuts", (PRE + "-F-WORSE", PRE + "-M-CUTS-GAP-ERROR")), ("sep", "sep", (PRE + "-F-NOT-SEPARATED", PRE + "-M-NOT-SEPARATED"))):
        P = syn_payload(LK, RL, kf, km)
        n1, n2, agree = derive_names(P)
        good = agree and (n1["F"], n1["M"]) == want
        print("world F %s M %s -> %s %s (%s)" % (kf, km, n1["F"], n1["M"], "ok" if good else "WRONG"))
        ok &= good
    tmp = tempfile.mkdtemp(prefix="telegap_selftest_", dir=os.getcwd())
    try:
        res = run_controls(write_tree(LK, RL, os.path.join(tmp, "clean")), LK, RL)
        print("clean tree: all_pass %s" % res["all_pass"])
        ok &= res["all_pass"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ZZ READ-TELEGAP SELFTEST %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--lookgap-reader", default=LOOKGAP_DEFAULT)
    ap.add_argument("--lookat-reader", default=LOOKAT_DEFAULT)
    ap.add_argument("--eval-join", default=V8 + "/hierrep/pins/join_eval141.npz")
    ap.add_argument("--box-dir", default=V8 + "/cropread/enc_out/box")
    ap.add_argument("--dump", action="append", default=[], help="NAME=PATH of one dump (D_0..F_2, D2_0, FP_0)")
    ap.add_argument("--list-record", default="")
    ap.add_argument("--list-file", action="append", default=[], help="NAME=PATH; train_list and eval_list are required")
    ap.add_argument("--stage-record", action="append", default=[])
    ap.add_argument("--enc-manifest", default="")
    ap.add_argument("--all-clips", default="", help="the pilot clips the encode used (one per line, first token)")
    ap.add_argument("--match-dir", default="")
    ap.add_argument("--t2-json", default="")
    ap.add_argument("--d2-fit-record", default="")
    ap.add_argument("--const", type=float, default=None, help="the constant read of t5; else dump key const_gap of FP_0, else the median label gap")
    ap.add_argument("--test-out", action="append", default=[], help="NAME=PATH")
    ap.add_argument("--test-before", action="append", default=[], help="NAME=EPOCH")
    ap.add_argument("--n-tok-files", type=int, default=-1)
    ap.add_argument("--controls", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--ref-d", default=",".join("%s=%s" % kv for kv in REF_D.items()))
    ap.add_argument("--banked", default=",".join(str(x) for x in BANKED_FIXED))
    ap.add_argument("--range-pad", type=float, default=RANGE_PAD)
    for k in EXPECT:
        ap.add_argument("--expect-" + k, type=int, default=EXPECT[k])
    a = ap.parse_args(argv)
    a.expect = {k: getattr(a, "expect_" + k) for k in EXPECT}
    a.ref_d = {k: float(v) for k, _, v in (t.partition("=") for t in a.ref_d.split(",") if t)}
    a.banked = tuple(float(x) for x in a.banked.split(",") if x)
    RL, rpath = load_lookgap(a.lookgap_reader)
    LK, lsha, lpath = RL.load_lookat(a.lookat_reader)
    if a.selftest:
        return selftest(LK, RL)
    if not a.stage or not a.out:
        ap.error("--stage and --out are required")
    if a.stage == "controls":
        if os.path.exists(a.out):
            print("REFUSED: %s exists" % a.out)
            print("ZZ READ-TELEGAP controls rc 2")
            return 2
        res = run_controls(a, LK, RL)
        res["inputs"] = input_hashes(a, LK)
        res["inputs"].update({"read_telegap.py": LK.sha_file(os.path.abspath(__file__)), "read_lookgap.py": LK.sha_file(rpath), "read_lookat.py": lsha})
        wj_indent = json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        wt(a.out, wj_indent)
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if (res["controls"].get(k) or {}).get("pass") else "FAIL"))
        print("ZZ READ-TELEGAP controls all_pass %s" % res["all_pass"])
        return 0
    return run_names(a, LK)


if __name__ == "__main__":
    sys.exit(main())
