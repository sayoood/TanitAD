#!/usr/bin/env python
"""read_lookgap.py -- read of card PREREG_V8F_PROOF_ADD108 (LOOKGAP-T0). CPU, numpy only. Tier T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance).

  python read_lookgap.py --stage controls --out R/LOOKGAP_CONTROLS.json [--lookgap-root ROOT --run1 RUN1 --lookat-reader PATH ...]
  python read_lookgap.py --stage names --controls R/LOOKGAP_CONTROLS.json --out R/LOOKGAP_RESULT.json [--log R/read_names.log]
  python read_lookgap.py --selftest

Stage controls: runs m0 to m6 of card section 5 and PRINTS only `m<i> PASS|FAIL` lines and the last line `ZZ READ-LOOKGAP controls all_pass <True|False>`.
Every value goes to the controls file: under `payload` when all seven pass, under `withheld` when one fails (then no name exists).
Stage names: refuses unless m0 to m6 are all present and pass; writes the result file and a names log; derives the name a second time apart.
The reader of card ADD107 (read_lookat.py, sha256 recorded in the outputs) is imported unchanged as a sibling module."""
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

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD108"
PRE = "LOOKGAP-T0"
CONTROLS = tuple("m%d" % i for i in range(7))
N_BOOT, BOOT_SEED = 2000, 108
EXPECT = {"n_train": 500, "n_eval": 141, "rows": 3841, "fit": 450, "sel": 50}
REF_E = {"D0": 3.960, "D1": 3.913, "D2": 4.112, "B0": 2.137, "B1": 1.910}
REF_S = {"S4": 0.8748, "S8_out": 0.5844}
TOL_E, TOL_S = 1e-3, 1e-4
TOOL_SHA16 = "61bf2f4e8e7ec61f"
LAB_TOL = 1e-5
SHORT_M = 5.0
GAP_BINS = ((0, 10), (10, 20), (20, 40), (40, 60))
AZ_BINS = ((0, 4), (4, 8), (8, 16), (16, None))
DIR_KEYS = ("ctr_az", "ctr_el", "has_lead")
D_FITS = ("D0", "D1", "D2", "D3", "D4")
P_FITS = ("P0", "P1", "P2")
B_FITS = ("B0", "B1")
X_FITS = ("X0", "X1")
MAIN_FITS = P_FITS + D_FITS + B_FITS
ALL_FITS = P_FITS + D_FITS + B_FITS + ("PERM",)
ALL_DUMPS = D_FITS + P_FITS + B_FITS + ("PERM",) + X_FITS
LOOKAT_DEFAULT = "/workspace/v8proof/lookat/kit/read_lookat.py"
V8 = "/workspace/v8proof"
TIER = "T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance)"
FAMILIES = ("LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not read (n = 0 for each; reason: T0, no rollout and no plan in this card; the measure is a T0 read of one input of the lead rule, card section 6)")
STAGE_A_HEADING = "stage A scores (frames with a label lead and a full history; not the row set of this card):"


# ============================================================================================ the imported reader
def load_lookat(path):
    """import read_lookat.py unchanged from `path` (fallback: the sibling folder of this kit on the dev box). -> (module, sha256, path used)"""
    cands = [path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lookat", "read_lookat.py")]
    for p in cands:
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("read_lookat", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["read_lookat"] = mod
            spec.loader.exec_module(mod)
            return mod, mod.sha_file(p), os.path.abspath(p)
    raise FileNotFoundError("read_lookat.py not found at %s" % path)


def ids_of(p):
    with io.open(p, encoding="utf-8") as fh:
        return [ln.split()[0] for ln in fh if ln.strip()]


def _res(bad, **kw):
    d = {"pass": not bad, "failures": bad}
    d.update(kw)
    return d


def fnum(x):
    return None if x is None or not np.isfinite(x) else float(x)


# ============================================================================================ rows and values
def lay(LK, J, D):
    """one dump laid on the join rows with read_lookat.on_rows: gap, lab_gap (nan where none), lab_present (bool), cov (rows the dump has)."""
    D2 = dict(D)
    D2["lab_present"] = np.asarray(D["lab_present"]).astype(bool)
    D2["cov"] = np.ones(len(np.asarray(D["frame"])), bool)
    return {"gap": LK.on_rows(J, D2, "gap"), "lab_gap": LK.on_rows(J, D2, "lab_gap"), "lab_present": LK.on_rows(J, D2, "lab_present"), "cov": LK.on_rows(J, D2, "cov")}


def abs_err(g, lab):
    return np.abs(g - lab)


def boot_ratio(num, den, n_boot=N_BOOT, seed=BOOT_SEED):
    """cluster bootstrap of sum(num) / sum(den) over clips: idx = RandomState(seed).randint(0, n, size=(n_boot, n)); two-sided 95 % percentile; fewer than half of the draws finite -> None."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    nc = len(num)
    idx = np.random.RandomState(seed).randint(0, nc, size=(n_boot, nc))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    r = r[np.isfinite(r)]
    if r.size < n_boot // 2:
        return None
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def az_bin_mask(az, lo, hi):
    """bins as read_lookat._stats: 0-4 is |az| <= 4; 4-8 is 4 < |az| <= 8; 8-16; above 16."""
    a = np.abs(az)
    if lo == 0:
        return a <= hi
    return (a > lo) & ((a <= hi) if hi else True)


def gap_bin_mask(g, lo, hi):
    return (g >= lo) & ((g < hi) if hi < 60 else (g <= hi))


def bin_name_az(lo, hi):
    return ("%d-%d deg" % (lo, hi)) if hi else "above %d deg" % lo


def compute_values(LK, J, L, box, dirs, n_boot=N_BOOT, seed=BOOT_SEED):
    """J the evaluation join, L {dump name: laid dump}, box / dirs {has, az, el} on the join rows. -> payload (all values of the read)."""
    n = np.asarray(J["n"], np.int64)
    nc = len(n)
    ci_all = np.repeat(np.arange(nc), n)
    d0 = L["D0"]
    lab_all = d0["lab_gap"]
    base = d0["lab_present"] & np.isfinite(lab_all)
    for k in D_FITS:
        base &= np.isfinite(L[k]["gap"])
    sel = base.copy()
    for k in P_FITS + B_FITS:
        sel &= np.isfinite(L[k]["gap"])
    differ = np.zeros(len(base), bool)
    differ_all = np.zeros(len(base), bool)
    for k in ALL_DUMPS:
        if k == "D0":
            continue
        o = L[k]
        bad = o["cov"] & ((o["lab_present"] != d0["lab_present"]) | ~np.isclose(o["lab_gap"], lab_all, rtol=LAB_TOL, atol=LAB_TOL, equal_nan=True))
        differ_all |= bad
        differ |= bad & base
    P = {"n_clips": int(nc), "n_boot": n_boot, "seed": seed, "tier": TIER, "families": FAMILIES,
         "interval_level": "two-sided 95 % percentile, cluster bootstrap over the evaluation clips, fits held fixed",
         "n_base": int(base.sum()), "n_rows": int(sel.sum()), "n_label_fields_differ": int(differ.sum()), "n_label_fields_differ_all_rows": int(differ_all.sum()),
         "missing_in_base": {k: int((base & ~np.isfinite(L[k]["gap"])).sum()) for k in P_FITS + B_FITS}}
    rows = np.flatnonzero(sel)
    ci = ci_all[rows]
    lab = lab_all[rows]
    r_cl = np.bincount(ci, minlength=nc).astype(np.float64)
    absk = {k: abs_err(L[k]["gap"][rows], lab) for k in MAIN_FITS}
    sums = {k: np.bincount(ci, weights=absk[k], minlength=nc) for k in MAIN_FITS}
    P["sums"] = {"r": r_cl.tolist(), "abs": {k: sums[k].tolist() for k in MAIN_FITS}}
    P["E"] = {k: fnum(absk[k].mean()) if len(rows) else None for k in MAIN_FITS}
    # PERM on the rows of the row set where it is finite
    pm = np.isfinite(L["PERM"]["gap"][rows])
    prow = rows[pm]
    pci = ci_all[prow]
    perm_abs = abs_err(L["PERM"]["gap"][prow], lab_all[prow])
    p0_abs = abs_err(L["P0"]["gap"][prow], lab_all[prow])
    s_perm, s_p0 = np.bincount(pci, weights=perm_abs, minlength=nc), np.bincount(pci, weights=p0_abs, minlength=nc)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(s_perm.sum() / s_p0.sum()) if len(prow) else None
    P["E"]["PERM"] = fnum(perm_abs.mean()) if len(prow) else None
    P["perm"] = {"n_rows": int(len(prow)), "E_perm": P["E"]["PERM"], "E_p0_same_rows": fnum(p0_abs.mean()) if len(prow) else None, "ratio": fnum(ratio) if ratio is not None else None,
                 "iv": boot_ratio(s_perm, s_p0, n_boot, seed) if len(prow) else None, "sums": {"perm": s_perm.tolist(), "p0": s_p0.tolist()}}
    # R and its interval
    num = (sums["P0"] + sums["P1"] + sums["P2"]) / 3.0
    den = (sums["D0"] + sums["D1"] + sums["D2"]) / 3.0
    mp = np.mean([P["E"][k] for k in P_FITS]) if len(rows) else float("nan")
    md = np.mean([P["E"][k] for k in ("D0", "D1", "D2")]) if len(rows) else float("nan")
    P["mean_E_P"], P["mean_E_D012"] = fnum(mp), fnum(md)
    P["R"] = fnum(mp / md) if len(rows) and md > 0 else None
    P["iv"] = boot_ratio(num, den, n_boot, seed) if len(rows) else None
    P["min_E_D"] = fnum(min(P["E"][k] for k in D_FITS)) if len(rows) else None
    mD01, mP01, mB01 = (np.mean([P["E"][k] for k in ks]) if len(rows) else float("nan") for ks in (("D0", "D1"), ("P0", "P1"), ("B0", "B1")))
    P["share_of_bound"] = fnum((mD01 - mP01) / (mD01 - mB01)) if len(rows) and abs(mD01 - mB01) > 0 else None
    P["corr_with_label"] = {}
    for k in MAIN_FITS + ("PERM",):
        g = L[k]["gap"][rows] if k != "PERM" else L[k]["gap"][prow]
        lb = lab if k != "PERM" else lab_all[prow]
        P["corr_with_label"][k] = fnum(np.corrcoef(g, lb)[0, 1]) if len(lb) > 2 and np.std(g) > 0 and np.std(lb) > 0 else None
    # bins
    P["bins_gap"], P["bins_az"] = {}, {}
    az_lab = box["az"][rows]
    for k in ("P0", "D0", "B0"):
        g = L[k]["gap"][rows]
        out = []
        for lo, hi in GAP_BINS:
            m = gap_bin_mask(lab, lo, hi)
            e = g[m] - lab[m]
            out.append({"bin": "%d-%d m" % (lo, hi), "n": int(m.sum()), "E": fnum(np.abs(e).mean()) if m.any() else None, "signed": fnum(e.mean()) if m.any() else None,
                        "short5": fnum((e < -SHORT_M).mean()) if m.any() else None})
        P["bins_gap"][k] = out
        out = []
        for lo, hi in AZ_BINS:
            m = az_bin_mask(az_lab, lo, hi)
            out.append({"bin": bin_name_az(lo, hi), "n": int(m.sum()), "E": fnum(np.abs(g[m] - lab[m]).mean()) if m.any() else None})
        P["bins_az"][k] = out
    # cross read: heads of B0, B1 read on the tokens of arm P
    P["cross"] = {}
    for x, b in zip(X_FITS, B_FITS):
        gx = L[x]["gap"][rows]
        m = np.isfinite(gx)
        P["cross"][x] = {"n": int(m.sum()), "E": fnum(np.abs(gx[m] - lab[m]).mean()) if m.any() else None, "E_" + b + "_same_rows": fnum(absk[b][m].mean()) if m.any() else None, "E_" + b + "_row_set": P["E"][b]}
    # directions on the base rows (card ADD107 measures)
    br = np.flatnonzero(base)
    gap_b, az_b, el_b = lab_all[br], box["az"][br], box["el"][br]
    st, _, _, _ = LK._stats(np.abs(dirs["az"][br] - az_b), np.abs(dirs["el"][br] - el_b), gap_b, az_b)
    P["dirs"] = {"n_base": int(len(br)), "n_no_box": int((~box["has"][br]).sum()), "n_no_dir": int((~dirs["has"][br]).sum()), "S4": st["S4"], "S8": st["S8"], "S8_out": st["S8_out"], "stats": st}
    return P


# ============================================================================================ names (two derivations)
def name_of(P):
    """card section 4, first derivation (from the values of the payload). None when there is no interval."""
    iv = P.get("iv")
    if iv is None:
        return None
    if iv[1] < 1.0:
        if all(P["E"][k] < min(P["E"][d] for d in D_FITS) for k in P_FITS):
            return PRE + "-CUTS"
        return PRE + "-CUTS-NOT-EACH-FIT"
    if iv[0] > 1.0:
        return PRE + "-WORSE"
    return PRE + "-NOT-SEPARATED"


def name_second(P):
    """second derivation, written apart: from the per-clip sums only, plain python loops, hand-made percentiles. -> (name or None, [lo, hi] or None)"""
    S = P["sums"]
    r = S["r"]
    nc = len(r)
    R = float(sum(r))
    if R <= 0:
        return None, None

    def E(k):
        return sum(S["abs"][k]) / R

    num = [(S["abs"]["P0"][j] + S["abs"]["P1"][j] + S["abs"]["P2"][j]) / 3.0 for j in range(nc)]
    den = [(S["abs"]["D0"][j] + S["abs"]["D1"][j] + S["abs"]["D2"][j]) / 3.0 for j in range(nc)]
    idx = np.random.RandomState(P["seed"]).randint(0, nc, size=(P["n_boot"], nc)).tolist()
    v = []
    for row in idx:
        d = sum(den[j] for j in row)
        if d > 0:
            v.append(sum(num[j] for j in row) / d)
    if len(v) < P["n_boot"] // 2:
        return None, None
    v.sort()

    def pct(q):
        pos = (len(v) - 1) * q / 100.0
        a = int(pos)
        b = min(a + 1, len(v) - 1)
        return v[a] + (v[b] - v[a]) * (pos - a)
    lo, hi = pct(2.5), pct(97.5)
    mind = min(E(d) for d in D_FITS)
    if hi < 1.0:
        return PRE + ("-CUTS" if all(E(k) < mind for k in P_FITS) else "-CUTS-NOT-EACH-FIT"), [lo, hi]
    if lo > 1.0:
        return PRE + "-WORSE", [lo, hi]
    return PRE + "-NOT-SEPARATED", [lo, hi]


def derive_name(P):
    n1 = name_of(P)
    n2, iv2 = name_second(P)
    agree = n1 == n2 and (n1 is None or (P["iv"] is not None and iv2 is not None and np.allclose(P["iv"], iv2, atol=1e-9, rtol=0)))
    return n1, n2, agree


# ============================================================================================ controls (pure functions of facts)
def ctl_m0(f, ex=EXPECT):
    """directions: CHECK_DIRS all_pass; a direction file for each train and evaluation clip with the frame count of its arm P token file; keys exactly the three; has_lead all true."""
    bad = []
    if f.get("check_dirs_all_pass") is not True:
        bad.append("CHECK_DIRS.json has no all_pass true")
    if f.get("n_train") != ex["n_train"] or f.get("n_eval") != ex["n_eval"]:
        bad.append("%s train and %s evaluation clips listed, expected %d and %d" % (f.get("n_train"), f.get("n_eval"), ex["n_train"], ex["n_eval"]))
    if f.get("n_checked") != ex["n_train"] + ex["n_eval"]:
        bad.append("%s clips checked, expected %d" % (f.get("n_checked"), ex["n_train"] + ex["n_eval"]))
    for key, what in (("missing", "no direction file"), ("keys_bad", "keys other than the three (or unreadable)"), ("frames_bad", "another frame count than the arm P token file"), ("lead_bad", "a frame with has_lead false")):
        if f.get(key):
            bad.append("%d clips with %s (first %s)" % (len(f[key]), what, f[key][0]))
    return _res(bad)


def ctl_m1(enc):
    """encode: ENC_M1.json pass_m1 true and every clause pass true."""
    bad = []
    if not isinstance(enc, dict):
        return _res(["ENC_M1.json missing or unreadable"])
    if enc.get("pass_m1") is not True:
        bad.append("pass_m1 is not true")
    cl = enc.get("clauses")
    items = list(cl.items()) if isinstance(cl, dict) else list(enumerate(cl or []))
    if not items:
        bad.append("no clauses")
    for k, v in items:
        if not (isinstance(v, dict) and v.get("pass") is True):
            bad.append("clause %s does not pass" % (k,))
    return _res(bad)


def _argv_root(argv):
    for i, t in enumerate(argv or []):
        if t == "--root" and i + 1 < len(argv):
            return argv[i + 1]
        if isinstance(t, str) and t.startswith("--root="):
            return t[len("--root="):]
    return None


def _norm(p):
    return None if p is None else str(p).replace("\\", "/").rstrip("/")


def ctl_m2(f, ex=EXPECT):
    """fits: the four records (seeds 0, 1, 2 and the permuted seed 0) as the card names them; lists, tool, run root and token folder as recorded and as read now."""
    bad = []
    rr = f.get("runroot") or {}
    for key in ("train_list_sha256", "eval_list_sha256", "pilot_train_list_sha256", "pilot_eval_list_sha256", "tok_b_realpath"):
        if key not in rr:
            bad.append("RUNROOT.json has no %s" % key)
    for want, key in ((f.get("train_sha"), "pilot_train_list_sha256"), (f.get("eval_sha"), "pilot_eval_list_sha256")):
        if rr.get(key) is not None and rr.get(key) != want:
            bad.append("RUNROOT.json %s differs from the pilot list now" % key)
    if _norm(rr.get("tok_b_realpath")) != _norm(f.get("tok_p_real")):
        bad.append("RUNROOT.json tok_b_realpath is not the arm P token folder")
    if _norm(f.get("live_tok_b_real")) != _norm(f.get("tok_p_real")):
        bad.append("the live real path of run/tok/B is not the arm P token folder")
    for k, seed, perm in (("s0", 0, False), ("s1", 1, False), ("s2", 2, False), ("perm", 0, True)):
        r = (f.get("recs") or {}).get(k)
        if not r:
            bad.append("%s: no fit record" % k)
            continue
        if r.get("seed") != seed:
            bad.append("%s: seed %s" % (k, r.get("seed")))
        if r.get("head") != "H5":
            bad.append("%s: head %s" % (k, r.get("head")))
        if r.get("arm") != "B":
            bad.append("%s: arm %s" % (k, r.get("arm")))
        if bool(r.get("permute_labels")) != perm:
            bad.append("%s: permute_labels is %s" % (k, r.get("permute_labels")))
        if r.get("hparams_equal_card") is not True:
            bad.append("%s: hparams_equal_card is not true" % k)
        if not str(r.get("tool_sha256") or "").startswith(TOOL_SHA16):
            bad.append("%s: tool_sha256 does not begin %s" % (k, TOOL_SHA16))
        for key, want in (("train_list_sha256", f.get("train_sha")), ("eval_list_sha256", f.get("eval_sha"))):
            if r.get(key) != want or r.get(key) != rr.get(key):
                bad.append("%s: %s differs from the pilot list now or from RUNROOT.json" % (k, key))
        cr = r.get("clips_read") or {}
        if len(cr.get("fit") or []) != ex["fit"] or len(cr.get("sel") or []) != ex["sel"]:
            bad.append("%s: clips_read is not %d fit and %d sel" % (k, ex["fit"], ex["sel"]))
        if _norm(_argv_root(r.get("argv"))) != _norm(f.get("run_root")):
            bad.append("%s: --root in argv is not the run root" % k)
    return _res(bad)


def ctl_m3(P):
    """deliberate regression: the lower end of the interval of E(PERM) / E(P0) is above 1."""
    iv = (P.get("perm") or {}).get("iv")
    return _res([] if (iv is not None and iv[0] > 1.0) else ["the permuted fit does not have more gap error than P0 (interval %s)" % (iv,)])


def ctl_m4(P, ex=EXPECT, ref=REF_E):
    """the reference recomputes: five E values within 1e-3 m, the row set has the expected rows, no label field differs between dumps."""
    bad = []
    for k, v in ref.items():
        e = P["E"].get(k)
        if e is None or abs(e - v) > TOL_E:
            bad.append("E(%s) %s, expected %.3f within %g" % (k, e, v, TOL_E))
    if P["n_rows"] != ex["rows"]:
        bad.append("%d rows in the row set, expected %d" % (P["n_rows"], ex["rows"]))
    if P["n_label_fields_differ"] != 0:
        bad.append("%d base rows where a dump's label fields differ from D0" % P["n_label_fields_differ"])
    return _res(bad)


def ctl_m5(P, ex=EXPECT, ref=REF_S):
    """the directions are those of card ADD107: S4 and S8_out on the base rows within 1e-4."""
    bad = []
    d = P["dirs"]
    if d["n_base"] != ex["rows"]:
        bad.append("%d base rows, expected %d" % (d["n_base"], ex["rows"]))
    for k, v in ref.items():
        s = d.get(k)
        if s is None or abs(s - v) > TOL_S:
            bad.append("%s %s, expected %.4f within %g" % (k, s, v, TOL_S))
    return _res(bad, n_no_box=d["n_no_box"], n_no_dir=d["n_no_dir"])


def ctl_m6(texts):
    """kit tests: for each of the four lead_pytest.out files the last non-empty line holds `passed` and neither `failed` nor `error`. texts {name: text or None}"""
    bad = []
    for k, t in texts.items():
        if t is None:
            bad.append("%s: no lead_pytest.out" % k)
            continue
        ln = [x for x in t.splitlines() if x.strip()]
        last = ln[-1] if ln else ""
        if not re.search(r"\bpassed\b", last) or re.search(r"failed|error", last, re.I):
            bad.append("%s: last line is not a passed line (%s)" % (k, last[:60]))
    if len(texts) != 4:
        bad.append("%d test outputs, expected 4" % len(texts))
    return _res(bad)


# ============================================================================================ disk
def dump_paths(a):
    lg = a.lookgap_root
    p = {"D0": V8 + "/cropread/run1/dump/D_H5/reader_s0_score.npz", "D1": V8 + "/cropread/run1/dump/D_H5/reader_s1_score.npz"}
    for i in (2, 3, 4):
        p["D%d" % i] = V8 + "/visseed10k/fitroot/dump/D_H5/reader_s%d_score.npz" % i
    for i in (0, 1):
        p["B%d" % i] = V8 + "/cropread/run1/dump/B_H5/reader_s%d_score.npz" % i
        p["X%d" % i] = lg + "/xdump/B_on_P/reader_s%d_score.npz" % i
    for i in (0, 1, 2):
        p["P%d" % i] = lg + "/run/dump/B_H5/reader_s%d_score.npz" % i
    p["PERM"] = lg + "/run/dump/B_H5_perm/reader_s0_score.npz"
    for ov in a.dump or []:
        k, _, v = ov.partition("=")
        p[k] = v
    return p


def kit_test_paths(a):
    lg = a.lookgap_root
    return {k: lg + "/" + k + "/lead_pytest.out" for k in ("kit_dirs", "kit_enc", "kit_fit", "kit_read")}


def wj(path, obj):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh)


def wt(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def read_text(p):
    return io.open(p, encoding="utf-8", errors="replace").read() if os.path.isfile(p) else None


def try_json(p):
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tok_n(folder, c):
    p = os.path.join(folder, c + ".npy")
    return int(np.load(p, mmap_mode="r").shape[0]) if os.path.isfile(p) else None


def gather_m0(a):
    tr = [ln.split()[0] for ln in io.open(os.path.join(a.run1, "lists", "train_final.txt"), encoding="utf-8") if ln.strip()]
    ev = ids_of(os.path.join(a.run1, "lists", "eval141.txt"))
    cd = try_json(os.path.join(a.lookgap_root, "records", "CHECK_DIRS.json"))
    f = {"check_dirs_all_pass": (cd or {}).get("all_pass"), "n_train": len(tr), "n_eval": len(ev), "n_checked": 0, "missing": [], "keys_bad": [], "frames_bad": [], "lead_bad": []}
    for c in tr + ev:
        f["n_checked"] += 1
        p = os.path.join(a.dirs_dir, c + ".npz")
        if not os.path.isfile(p):
            f["missing"].append(c)
            continue
        try:
            z = np.load(p)
            if sorted(z.files) != sorted(DIR_KEYS):
                f["keys_bad"].append(c)
                continue
            n, tn = len(z["has_lead"]), tok_n(a.tok_p, c)
            if tn is None or not (n == len(z["ctr_az"]) == len(z["ctr_el"]) == tn):
                f["frames_bad"].append(c)
            if not bool(np.all(np.asarray(z["has_lead"]).astype(bool))):
                f["lead_bad"].append(c)
        except (OSError, ValueError, KeyError):
            f["keys_bad"].append(c)
    return f


def gather_m2(a, LK):
    lg_run = a.run_root
    recs = {}
    for k, nm in (("s0", "B_H5_s0"), ("s1", "B_H5_s1"), ("s2", "B_H5_s2"), ("perm", "B_H5_s0_perm")):
        recs[k] = try_json(os.path.join(lg_run, "fits", nm, "FIT_RECORD.json"))
    return {"recs": recs, "runroot": try_json(os.path.join(lg_run, "RUNROOT.json")), "train_sha": LK.sha_file(os.path.join(a.run1, "lists", "train_final.txt")),
            "eval_sha": LK.sha_file(os.path.join(a.run1, "lists", "eval141.txt")), "tok_p_real": os.path.realpath(a.tok_p), "live_tok_b_real": os.path.realpath(a.run_tok_b), "run_root": lg_run}


def tracks_of(LK, folder, clips):
    out = {}
    for c in clips:
        p = os.path.join(folder, c + ".npz")
        if os.path.isfile(p):
            z = np.load(p)
            has = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"]) & np.isfinite(z["ctr_el"])
            out[c] = (has, np.where(has, z["ctr_az"], 0.0), np.where(has, z["ctr_el"], 0.0))
    return out


def run_controls(a, LK):
    ctl, P = {}, None
    ctl["m0"] = ctl_m0(gather_m0(a), a.expect)
    enc = try_json(os.path.join(a.lookgap_root, "ENC_M1.json"))
    ctl["m1"] = ctl_m1(enc)
    ctl["m2"] = ctl_m2(gather_m2(a, LK), a.expect)
    try:
        J = LK.load_npz(a.eval_join)
        L = {k: lay(LK, J, LK.load_npz(p)) for k, p in dump_paths(a).items()}
        clips = [str(c) for c in J["clip"]]
        box = LK.box_on_rows(J, tracks_of(LK, a.box_dir, clips))
        dirs = LK.box_on_rows(J, tracks_of(LK, a.dirs_dir, clips))
        P = compute_values(LK, J, L, box, dirs, a.n_boot)
        P["xdump"] = try_json(os.path.join(a.lookgap_root, "xdump", "XDUMP_IDENT.json"))
        ctl["m3"], ctl["m4"], ctl["m5"] = ctl_m3(P), ctl_m4(P, a.expect, a.ref_e), ctl_m5(P, a.expect, a.ref_s)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as ex:
        ctl["m3"] = ctl["m4"] = ctl["m5"] = _res(["the read could not be computed: %s: %s" % (type(ex).__name__, ex)])
    ctl["m6"] = ctl_m6({k: read_text(p) for k, p in kit_test_paths(a).items()})
    return assemble(ctl, P)


def input_hashes(a, LK):
    h = {}
    paths = {"join": a.eval_join, "run1_train_list": os.path.join(a.run1, "lists", "train_final.txt"), "run1_eval_list": os.path.join(a.run1, "lists", "eval141.txt"),
             "RUNROOT": os.path.join(a.run_root, "RUNROOT.json"), "ENC_M1": os.path.join(a.lookgap_root, "ENC_M1.json"), "CHECK_DIRS": os.path.join(a.lookgap_root, "records", "CHECK_DIRS.json"),
             "XDUMP_IDENT": os.path.join(a.lookgap_root, "xdump", "XDUMP_IDENT.json")}
    for k, nm in (("s0", "B_H5_s0"), ("s1", "B_H5_s1"), ("s2", "B_H5_s2"), ("perm", "B_H5_s0_perm")):
        paths["fit_record_" + k] = os.path.join(a.run_root, "fits", nm, "FIT_RECORD.json")
    for k, p in dump_paths(a).items():
        paths["dump_" + k] = p
    for k, p in kit_test_paths(a).items():
        paths["tests_" + k] = p
    for k, p in paths.items():
        try:
            h[k] = LK.sha_file(p)
        except OSError:
            h[k] = None
    return h


def assemble(ctl, P):
    allp = sorted(ctl) == sorted(CONTROLS) and all(v["pass"] for v in ctl.values())
    res = {"card": CARD, "tier": TIER, "controls": ctl, "all_pass": bool(allp)}
    res["payload" if allp else "withheld"] = P
    return res


# ============================================================================================ output
def fv(x, d=3):
    return "None" if x is None else ("%." + str(d) + "f") % x


def fiv(iv):
    return "[%.4f, %.4f]" % (iv[0], iv[1]) if iv else "None"


def print_sections(P, name):
    L = ["tier: " + TIER, "families: " + FAMILIES]
    L.append("rows: base rows (label lead and a finite gap of D0 to D4) %d; row set of the card (also a finite gap of P0 to P2, B0, B1) %d; rows where another dump's label fields differ from D0: %d on the base rows, %d on all rows"
            % (P["n_base"], P["n_rows"], P["n_label_fields_differ"], P["n_label_fields_differ_all_rows"]))
    if P["n_rows"] != EXPECT["rows"]:
        L.append("row set is not %d: base rows lacking a finite gap: %s" % (EXPECT["rows"], "; ".join("%s %d" % (k, v) for k, v in P["missing_in_base"].items())))
    L.append("E (mean absolute gap error in m, row set; %d evaluation clips): " % P["n_clips"] + "; ".join("%s %s" % (k, fv(P["E"][k])) for k in ALL_FITS) + " (PERM on %d rows where finite)" % P["perm"]["n_rows"])
    L.append("R = mean E(P0..P2) / mean E(D0..D2) = %s (%s / %s); interval %s (%d draws, RandomState(%d), %s)" % (fv(P["R"], 4), fv(P["mean_E_P"], 4), fv(P["mean_E_D012"], 4), fiv(P["iv"]), P["n_boot"], P["seed"], P["interval_level"]))
    L.append("smallest E of D0..D4: %s" % fv(P["min_E_D"]))
    L.append("name: %s" % (name if name else "none (no interval)"))
    L.append("share of the bound's gain, (mean E(D0,D1) - mean E(P0,P1)) / (mean E(D0,D1) - mean E(B0,B1)): %s" % fv(P["share_of_bound"], 4))
    L.append("correlation of gap with label gap: " + "; ".join("%s %s" % (k, fv(P["corr_with_label"][k], 4)) for k in ALL_FITS))
    for k in ("P0", "D0", "B0"):
        L.append("by label gap %s: " % k + "; ".join("%s n %d E %s signed %s short>5m %s" % (b["bin"], b["n"], fv(b["E"]), fv(b["signed"]), fv(b["short5"])) for b in P["bins_gap"][k]))
    for k in ("P0", "D0", "B0"):
        L.append("by |label azimuth| %s: " % k + "; ".join("%s n %d E %s" % (b["bin"], b["n"], fv(b["E"])) for b in P["bins_az"][k]))
    xd = P.get("xdump") or {}
    L.append("XDUMP_IDENT.json: pass %s, max_abs_diff_m %s (the dump of B0 on the tokens of arm B reproduced; stated as a fact)" % (xd.get("pass"), xd.get("max_abs_diff_m")))
    for x, b in zip(X_FITS, B_FITS):
        c = P["cross"][x]
        L.append("cross read %s (head of %s read on the tokens of arm P): n %d E %s; %s on the same rows %s; %s on the row set %s" % (x, b, c["n"], fv(c["E"]), b, fv(c["E_" + b + "_same_rows"]), b, fv(c["E_" + b + "_row_set"])))
    pm = P["perm"]
    L.append("m3 ratio E(PERM) / E(P0) on %d rows: %s interval %s" % (pm["n_rows"], fv(pm["ratio"], 4), fiv(pm["iv"])))
    d = P["dirs"]
    L.append("directions on the %d base rows (no box %d, no direction %d): S4 %s S8 %s S8_out %s" % (d["n_base"], d["n_no_box"], d["n_no_dir"], fv(d["S4"], 4), fv(d["S8"], 4), fv(d["S8_out"], 4)))
    return L


def run_names(a, LK):
    cj = try_json(a.controls) or {}
    if cj.get("card") != CARD or cj.get("all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any(v.get("pass") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass with m0 to m6 present: no name is written")
        print("ZZ READ-LOOKGAP names rc 3")
        return 3
    P = cj["payload"]
    n1, n2, agree = derive_name(P)
    if not agree:
        print("STOP: the two derivations of the name disagree: %s | %s" % (n1, n2))
        print("ZZ READ-LOOKGAP names rc 4")
        return 4
    logp = a.log or (os.path.splitext(a.out)[0] + "_names.log")
    if os.path.exists(a.out) or os.path.exists(logp):
        print("REFUSED: %s or %s exists" % (a.out, logp))
        print("ZZ READ-LOOKGAP names rc 2")
        return 2
    res = {"card": CARD, "tier": TIER, "name": n1, "second_derivation_agrees": True, "controls_file": os.path.abspath(a.controls), "controls_sha256": LK.sha_file(a.controls),
           "reader_sha256": LK.sha_file(os.path.abspath(__file__)), "values": P}
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    rc = 0 if n1 else 5
    lines = print_sections(P, n1) + ["ZZ READ-LOOKGAP names rc %d" % rc]
    with io.open(logp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    for ln in lines:
        print(ln)
    return rc


# ============================================================================================ selftest world
SYN_ERR = {"cuts": {"P0": 1.5, "P1": 1.6, "P2": 1.4}, "notfit": {"P0": 1.0, "P1": 1.0, "P2": 5.0}, "worse": {"P0": 6.0, "P1": 6.0, "P2": 6.0}, "sep": {"P0": 4.0, "P1": 4.0, "P2": 4.0}}
SYN_BASE = {"D0": 4.0, "D1": 3.9, "D2": 4.1, "D3": 4.2, "D4": 4.0, "B0": 2.0, "B1": 1.9, "X0": 2.1, "X1": 2.0}
SYN_N, SYN_NFR = 12, 30


def syn_world(kind, seed=3):
    """synthetic evaluation world: join, dumps (dict of arrays as the card's dump files), label box and direction tracks. kind in cuts | notfit | worse | sep | nointerval."""
    rng = np.random.RandomState(seed)
    nc, nfr = SYN_N, SYN_NFR
    nrow = nc * nfr
    J = {"clip": np.asarray(["c%02d" % i for i in range(nc)]), "n": np.full(nc, nfr), "off": np.arange(nc) * nfr, "lead_gap": rng.uniform(0, 60, nrow), "lead_valid": np.ones(nrow, np.int8)}
    ci = np.repeat(np.arange(nc), nfr)
    fr = np.tile(np.arange(nfr), nc)
    present = rng.rand(nrow) < 0.85
    lab = np.where(present, rng.uniform(3, 58, nrow), np.nan)
    errs = dict(SYN_BASE, **SYN_ERR.get(kind, {"P0": 0.0, "P1": 0.0, "P2": 0.0}))
    errs["PERM"] = 9.0 if kind == "worse" else 6.0
    if kind == "nointerval":
        errs = {k: 0.0 for k in errs}
    sign = np.where(rng.rand(nrow) < 0.5, -1.0, 1.0)
    dumps = {}
    for k in ALL_DUMPS:
        mult = 1.0 + 0.2 * np.random.RandomState(100 + sum(map(ord, k))).uniform(-1, 1, nc)[ci]
        gap = lab + sign * errs[k] * mult + (0.0 if kind == "nointerval" else 0.01) * rng.randn(nrow)
        dumps[k] = {"clips": J["clip"], "clip_index": ci.copy(), "frame": fr.copy(), "gap": gap, "valid": np.ones(nrow, bool), "lab_present": present.copy(), "lab_gap": lab.copy()}
    az = np.where(rng.rand(nrow) < 0.2, rng.choice([-1, 1], nrow) * rng.uniform(10, 25, nrow), rng.normal(0, 3, nrow))
    el = -2 + rng.normal(0, 2.5, nrow)
    return J, dumps, {"az": az, "el": el}, {"az": az + rng.normal(0, 2, nrow), "el": el + rng.normal(0, 2, nrow)}


def syn_payload(LK, kind, n_boot=300):
    J, dumps, bx, dr = syn_world(kind)
    nrow = len(bx["az"])
    box = {"has": np.ones(nrow, bool), "az": bx["az"], "el": bx["el"]}
    dirs = {"has": np.ones(nrow, bool), "az": dr["az"], "el": dr["el"]}
    L = {k: lay(LK, J, d) for k, d in dumps.items()}
    return compute_values(LK, J, L, box, dirs, n_boot)


def write_tree(LK, root, kind="cuts", fault=None):
    """a tiny pod-like tree under `root` (6 fit + 2 sel train clips, 12 evaluation clips); the planted `fault` flips exactly the named control. -> (args namespace, ref dicts)"""
    J, dumps, bx, dr = syn_world(kind)
    root = os.path.abspath(root)
    lg, run1 = os.path.join(root, "lookgap"), os.path.join(root, "run1")
    run, dd, tokp, boxd = os.path.join(lg, "run"), os.path.join(lg, "dirs"), os.path.join(lg, "tok", "P"), os.path.join(root, "box")
    for p in (lg, run1, os.path.join(run1, "lists"), run, dd, tokp, boxd, os.path.join(lg, "records"), os.path.join(lg, "xdump"), os.path.join(lg, "dump")):
        os.makedirs(p, exist_ok=True)
    ev = [str(c) for c in J["clip"]]
    tr = ["t%d" % i for i in range(8)]
    with io.open(os.path.join(run1, "lists", "train_final.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join("%s %s\n" % (c, "fit" if i < 6 else "sel") for i, c in enumerate(tr)))
    with io.open(os.path.join(run1, "lists", "eval141.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(c + "\n" for c in ev))
    nfr = SYN_NFR
    nrow = len(bx["az"])
    for c in tr + ev:
        np.save(os.path.join(tokp, c + ".npy"), np.zeros((nfr, 2), np.float32))
        i = ev.index(c) if c in ev else None
        s = slice(i * nfr, (i + 1) * nfr) if i is not None else None
        az = dr["az"][s] if s else np.zeros(nfr)
        el = dr["el"][s] if s else np.zeros(nfr)
        np.savez(os.path.join(dd, c + ".npz"), has_lead=np.ones(nfr, bool), ctr_az=az, ctr_el=el)
        if s:
            np.savez(os.path.join(boxd, c + ".npz"), has_lead=np.ones(nfr, bool), ctr_az=bx["az"][s], ctr_el=bx["el"][s])
    jp = os.path.join(root, "join.npz")
    np.savez(jp, **J)
    paths = {}
    for k, d in dumps.items():
        p = os.path.join(lg, "dump", k + ".npz")
        np.savez(p, **d)
        paths[k] = p
    tokreal = os.path.realpath(tokp)
    sha = lambda p: LK.sha_file(p)                    # noqa: E731
    tsha, esha = sha(os.path.join(run1, "lists", "train_final.txt")), sha(os.path.join(run1, "lists", "eval141.txt"))
    wj(os.path.join(run, "RUNROOT.json"), {"tok_b_realpath": tokreal, "train_list_sha256": tsha, "eval_list_sha256": esha, "pilot_train_list_sha256": tsha, "pilot_eval_list_sha256": esha})
    for nm, seed, perm in (("B_H5_s0", 0, False), ("B_H5_s1", 1, False), ("B_H5_s2", 2, False), ("B_H5_s0_perm", 0, True)):
        os.makedirs(os.path.join(run, "fits", nm), exist_ok=True)
        rec = {"arm": "B", "head": "H5", "seed": seed, "permute_labels": perm, "argv": ["fit.py", "--root", run.replace("\\", "/"), "--seed", str(seed)], "tool_sha256": TOOL_SHA16 + "0" * 48,
               "train_list_sha256": tsha, "eval_list_sha256": esha, "clips_read": {"fit": ["x"] * 6, "sel": ["y"] * 2}, "hparams_equal_card": True}
        wj(os.path.join(run, "fits", nm, "FIT_RECORD.json"), rec)
    wj(os.path.join(lg, "records", "CHECK_DIRS.json"), {"all_pass": True})
    wj(os.path.join(lg, "ENC_M1.json"), {"pass_m1": True, "clauses": {"c1": {"pass": True}, "c2": {"pass": True}}})
    wj(os.path.join(lg, "xdump", "XDUMP_IDENT.json"), {"pass": True, "max_abs_diff_m": 0.0})
    for k in ("kit_dirs", "kit_enc", "kit_fit", "kit_read"):
        os.makedirs(os.path.join(lg, k), exist_ok=True)
        wt(os.path.join(lg, k, "lead_pytest.out"), "....\n12 passed in 1.0s\n")
    ns = argparse.Namespace(lookgap_root=lg, run1=run1, run_root=run, dirs_dir=dd, box_dir=boxd, tok_p=tokp, run_tok_b=tokp, eval_join=jp, dump=["%s=%s" % kv for kv in paths.items()], n_boot=300,
                            expect={"n_train": 8, "n_eval": 12, "rows": 0, "fit": 6, "sel": 2}, ref_e={}, ref_s={})
    ns.dump_paths = paths
    # references: what the clean world computes (rounded to the card's decimals)
    P = syn_payload(LK, kind)
    ns.expect["rows"] = P["n_rows"]
    ns.ref_e = {k: round(P["E"][k], 3) for k in REF_E}
    ns.ref_s = {"S4": round(P["dirs"]["S4"], 4), "S8_out": round(P["dirs"]["S8_out"], 4)}
    if fault:
        plant(LK, ns, fault, dumps)
    return ns


def plant(LK, ns, fault, dumps):
    lg = ns.lookgap_root
    if fault == "m0":
        p = os.path.join(ns.dirs_dir, "t3.npz")
        np.savez(p, has_lead=np.r_[np.zeros(1, bool), np.ones(SYN_NFR - 1, bool)], ctr_az=np.zeros(SYN_NFR), ctr_el=np.zeros(SYN_NFR))
    elif fault == "m1":
        wj(os.path.join(lg, "ENC_M1.json"), {"pass_m1": True, "clauses": {"c1": {"pass": True}, "c2": {"pass": False}}})
    elif fault == "m2":
        p = os.path.join(ns.run_root, "fits", "B_H5_s1", "FIT_RECORD.json")
        r = json.load(io.open(p, encoding="utf-8"))
        r["seed"] = 5
        wj(p, r)
    elif fault == "m3":
        d = dict(dumps["P0"])
        np.savez(ns.dump_paths["PERM"], **d)
    elif fault == "m4":
        d = dict(dumps["D3"])
        i = int(np.flatnonzero(d["lab_present"])[0])
        lg2 = d["lab_gap"].copy()
        lg2[i] += 1.0
        d["lab_gap"] = lg2
        np.savez(ns.dump_paths["D3"], **d)
    elif fault == "m5":
        p = os.path.join(ns.dirs_dir, "c03.npz")
        z = np.load(p)
        np.savez(p, has_lead=z["has_lead"], ctr_az=z["ctr_az"] + 20.0, ctr_el=z["ctr_el"])
    elif fault == "m6":
        wt(os.path.join(lg, "kit_fit", "lead_pytest.out"), "1 failed, 11 passed in 1s\n")


def selftest(LK):
    ok = True
    want = {"cuts": PRE + "-CUTS", "notfit": PRE + "-CUTS-NOT-EACH-FIT", "worse": PRE + "-WORSE", "sep": PRE + "-NOT-SEPARATED", "nointerval": None}
    for kind, w in want.items():
        P = syn_payload(LK, kind)
        n1, n2, agree = derive_name(P)
        good = n1 == w and agree
        print("world %s -> %s (%s)" % (kind, n1, "ok" if good else "WRONG, expected %s, second %s" % (w, n2)))
        ok &= good
    tmp = tempfile.mkdtemp(prefix="lookgap_selftest_", dir=os.getcwd())
    try:
        ns = write_tree(LK, os.path.join(tmp, "clean"))
        res = run_controls(ns, LK)
        print("clean tree: all_pass %s %s" % (res["all_pass"], {k: v["pass"] for k, v in res["controls"].items()}))
        ok &= res["all_pass"] is True and "payload" in res
        for k in CONTROLS:
            ns = write_tree(LK, os.path.join(tmp, "f" + k), fault=k)
            res = run_controls(ns, LK)
            flipped = sorted(c for c, v in res["controls"].items() if not v["pass"])
            good = flipped == [k] and "withheld" in res and "payload" not in res
            print("fault %s: flipped %s: %s" % (k, flipped, "ok" if good else "WRONG"))
            ok &= good
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ZZ READ-LOOKGAP SELFTEST %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def parse_kv(s, cast=float):
    return {k: cast(v) for k, _, v in (t.partition("=") for t in s.split(",") if t)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--lookat-reader", default=LOOKAT_DEFAULT)
    ap.add_argument("--lookgap-root", default=V8 + "/lookgap")
    ap.add_argument("--run1", default=V8 + "/cropread/run1")
    ap.add_argument("--run-root", default="")
    ap.add_argument("--dirs-dir", default="")
    ap.add_argument("--tok-p", default="")
    ap.add_argument("--run-tok-b", default="")
    ap.add_argument("--box-dir", default=V8 + "/cropread/enc_out/box")
    ap.add_argument("--eval-join", default=V8 + "/hierrep/pins/join_eval141.npz")
    ap.add_argument("--dump", action="append", default=[], help="NAME=PATH override of one dump")
    ap.add_argument("--controls", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--ref-e", default=",".join("%s=%s" % kv for kv in REF_E.items()))
    ap.add_argument("--ref-s", default=",".join("%s=%s" % kv for kv in REF_S.items()))
    for k in EXPECT:
        ap.add_argument("--expect-" + k.replace("_", "-"), type=int, default=EXPECT[k])
    a = ap.parse_args(argv)
    a.expect = {k: getattr(a, "expect_" + k) for k in EXPECT}
    a.ref_e, a.ref_s = parse_kv(a.ref_e), parse_kv(a.ref_s)
    a.run_root = a.run_root or a.lookgap_root + "/run"
    a.dirs_dir = a.dirs_dir or a.lookgap_root + "/dirs"
    a.tok_p = a.tok_p or a.lookgap_root + "/tok/P"
    a.run_tok_b = a.run_tok_b or a.run_root + "/tok/B"
    LK, lsha, lpath = load_lookat(a.lookat_reader)
    if a.selftest:
        return selftest(LK)
    if not a.stage or not a.out:
        ap.error("--stage and --out are required")
    if a.stage == "controls":
        if os.path.exists(a.out):
            print("REFUSED: %s exists" % a.out)
            print("ZZ READ-LOOKGAP controls rc 2")
            return 2
        res = run_controls(a, LK)
        res["inputs"] = input_hashes(a, LK)
        res["inputs"]["read_lookgap.py"] = LK.sha_file(os.path.abspath(__file__))
        res["inputs"]["read_lookat.py"] = lsha
        res["lookat_reader_path"] = lpath
        with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if (res["controls"].get(k) or {}).get("pass") else "FAIL"))
        print("ZZ READ-LOOKGAP controls all_pass %s" % res["all_pass"])
        return 0
    return run_names(a, LK)


if __name__ == "__main__":
    sys.exit(main())
