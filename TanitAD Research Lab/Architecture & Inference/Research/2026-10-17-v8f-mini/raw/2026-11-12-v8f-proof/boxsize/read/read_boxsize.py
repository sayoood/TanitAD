#!/usr/bin/env python
"""read_boxsize.py -- read of card PREREG_V8F_PROOF_ADD124 (BOXSIZE-T0). CPU, numpy only, no import of another kit. Tier T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance).

  python read_boxsize.py --stage controls --out R/BOXSIZE_CONTROLS.json  [inputs, see BOXSIZE_READ_DEFS.md]
  python read_boxsize.py --stage names --controls R/BOXSIZE_CONTROLS.json --out R/BOXSIZE_RESULT.json [--log R/read_names.log]  [the same inputs]
  python read_boxsize.py --selftest

Stage controls: runs s0 to s6 of card section 5 and PRINTS only `s<i> PASS|FAIL` lines (the values of s3, s4 and s5 may be printed) and the last line
`ZZ READ-BOXSIZE controls all_pass <True|False>`. No gap error of arm S and no size error of a fit is computed there: the controls file holds pass flags, the failure texts and the
control values of s3, s4 and s5 only. Exit code 0 when all seven pass, 2 when one fails (or the output file exists).
Stage names: refuses unless the controls file is all-pass with s0 to s6 present and the inputs hash as they did at the controls; then computes every value of card section 3 and the names of
section 4 (derived twice, apart), writes the result file and a names log."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD124"
PRE = "BOXSIZE-T0"
CONTROLS = tuple("s%d" % i for i in range(7))
N_BOOT, BOOT_SEED = 2000, 124
V8 = "/workspace/v8proof"
EXPECT = {"rows": 3841, "clips": 141}
REF_D = {"D_0": 3.960, "D_1": 3.913, "D_2": 4.112}
TOL_E = 1e-3
ID_TOL = 1e-4
RT_TOL = 1e-9
MAP_MAX_E = 2.2
PERM_MIN = 0.95
K_NN = 25
PERM_SEED = 124
SHORT_M = 5.0
BAR_RATIO = 0.85
FAR_HALF = 0.5
U0, ULIM = 1.6, 2.0
ARM_D = ("D_0", "D_1", "D_2")
ARM_S = ("S_0", "S_1", "S_2")
PERM_FIT = "SP_0"
FITS = ARM_S + (PERM_FIT,)
FIT_DIR = {"S_0": "LOC_s0", "S_1": "LOC_s1", "S_2": "LOC_s2", "SP_0": "LOC_s0_perm"}
FIT_SEED = {"S_0": 0, "S_1": 1, "S_2": 2, "SP_0": 0}
GAP_BINS = ((0, 10), (10, 20), (20, 40), (40, 60))
CHANGED_EXPECT = "load_box"
TIER = "T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance)"
FAMILIES = ("LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not read (n = 0 for each; reason: T0 read of a perception head, no trajectory is produced; "
            "a rollout card that reads all four follows only if the bar is met)")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ============================================================================================ small helpers
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_of(p):
    try:
        return sha_file(p)
    except (OSError, TypeError):
        return None


def _res(bad, **kw):
    d = {"pass": not bad, "failures": bad}
    d.update(kw)
    return d


def fnum(x):
    try:
        return None if x is None or not np.isfinite(x) else float(x)
    except (TypeError, ValueError):
        return None


def read_text(p):
    return io.open(p, encoding="utf-8", errors="replace").read() if p and os.path.isfile(p) else None


def try_json(p):
    try:
        with io.open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError):
        return None


def _jdefault(o):
    return float(o) if hasattr(o, "item") else str(o)


def wj(path, obj, indent=None):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=indent, default=_jdefault)


def wt(path, text):
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def ids_of(p):
    with io.open(p, encoding="utf-8") as fh:
        return [ln.split()[0] for ln in fh if ln.strip()]


def parse_kv(items, cast=str):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        out[k] = cast(v)
    return out


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


# ============================================================================================ size encoding (the same two lines as boxsize_fit.decode)
def encode(w, h):
    """degrees -> (az, el) of the locator tool: az = 16 u_w, el = -2 + 8 u_h with u = ln(angle) - 1.6."""
    w, h = np.asarray(w, np.float64), np.asarray(h, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 16.0 * (np.log(w) - U0), -2.0 + 8.0 * (np.log(h) - U0)


def decode(a, b):
    """(az, el) of the locator tool -> (width, height) in degrees: width = exp(1.6 + a / 16), height = exp(1.6 + (b + 2) / 8)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(U0 + a / 16.0), np.exp(U0 + (b + 2.0) / 8.0)


def outside_limits(x):
    """True where u = ln(x) - 1.6 lies outside [-2, 2] (0.67 to 36.6 degrees); nan where x is not finite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.log(np.asarray(x, np.float64)) - U0
    return np.abs(u) > ULIM


# ============================================================================================ join rows
def row_index(J, names, cidx, frame):
    jn = {str(c): i for i, c in enumerate(J["clip"])}
    of = np.array([jn.get(str(c), -1) for c in names], np.int64)
    ji = of[np.asarray(cidx, np.int64)]
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    jc = np.where(ji >= 0, ji, 0)
    fr = np.asarray(frame, np.int64)
    ok = (ji >= 0) & (fr >= 0) & (fr < n[jc])
    return np.where(ok, off[jc] + fr, -1), ok


def on_rows(J, D, key):
    nrow = int(np.asarray(J["n"], np.int64).sum())
    row, ok = row_index(J, D["clips"], D["clip_index"], D["frame"])
    v = np.asarray(D[key])
    out = np.zeros(nrow, bool) if v.dtype == bool else np.full(nrow, np.nan)
    out[row[ok]] = v[ok]
    return out


def lay_dump(J, D):
    """a banked dump of arm D on the join rows: gap, lab_gap (nan where none), lab_present."""
    return {"gap": on_rows(J, D, "gap"), "lab_gap": on_rows(J, D, "lab_gap"), "lab_present": on_rows(J, dict(D, lab_present=np.asarray(D["lab_present"]).astype(bool)), "lab_present").astype(bool)}


def lay_pred(J, D):
    """a prediction file of the locator tool on the join rows: az_pred, el_pred (nan where the file has no row or no output)."""
    az, el = on_rows(J, D, "az_pred"), on_rows(J, D, "el_pred")
    if "has_out" in D:
        ho = on_rows(J, dict(D, has_out=np.asarray(D["has_out"]).astype(bool)), "has_out").astype(bool)
        az, el = np.where(ho, az, np.nan), np.where(ho, el, np.nan)
    return {"az": az, "el": el}


def box_rows(J, box_dir):
    """label sizes on the join rows from <box_dir>/<clip>.npz: has (a lead and a finite positive width and height), w, h (nan elsewhere)."""
    n = np.asarray(J["n"], np.int64)
    off = np.asarray(J["off"], np.int64)
    nrow = int(n.sum())
    has, w, h = np.zeros(nrow, bool), np.full(nrow, np.nan), np.full(nrow, np.nan)
    found = 0
    for i, c in enumerate(J["clip"]):
        p = os.path.join(box_dir, str(c) + ".npz")
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        found += 1
        m = min(len(z["has_lead"]), int(n[i]))
        s = slice(off[i], off[i] + m)
        aw, ah = np.asarray(z["ang_w"], np.float64)[:m], np.asarray(z["ang_h"], np.float64)[:m]
        with np.errstate(invalid="ignore"):
            ok = z["has_lead"].astype(bool)[:m] & np.isfinite(aw) & np.isfinite(ah) & (aw > 0) & (ah > 0)
        has[s] = ok
        w[s] = np.where(ok, aw, np.nan)
        h[s] = np.where(ok, ah, np.nan)
    return {"has": has, "w": w, "h": h, "n_clips_found": found}


# ============================================================================================ the map G (card section 2)
class MapG:
    """25 nearest train frames by Euclidean distance in (ln h, ln w), each scaled by its mean and standard deviation over the train frames; read = median of their label gaps.
    Ties by the lower train index (argpartition finds the 25th distance, the candidates are re-sorted stably in index order). Deterministic, numpy only, chunked."""

    def __init__(self, lnh, lnw, gap, k=K_NN):
        lnh, lnw, gap = (np.asarray(x, np.float64) for x in (lnh, lnw, gap))
        self.mu = np.array([lnh.mean(), lnw.mean()])
        sd = np.array([lnh.std(), lnw.std()])
        self.sd = np.where(sd > 0, sd, 1.0)
        self.t0 = (lnh - self.mu[0]) / self.sd[0]
        self.t1 = (lnw - self.mu[1]) / self.sd[1]
        self.gap = gap
        self.k = int(min(k, len(gap)))
        self.n = int(len(gap))

    def predict(self, w, h, chunk=64):
        w, h = np.asarray(w, np.float64), np.asarray(h, np.float64)
        out = np.full(len(w), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            q0 = (np.log(h) - self.mu[0]) / self.sd[0]
            q1 = (np.log(w) - self.mu[1]) / self.sd[1]
        ok = np.flatnonzero(np.isfinite(q0) & np.isfinite(q1))
        k = self.k
        if k < 1:
            return out
        for s in range(0, len(ok), chunk):
            ii = ok[s:s + chunk]
            d = (q0[ii, None] - self.t0[None, :]) ** 2 + (q1[ii, None] - self.t1[None, :]) ** 2
            part = np.argpartition(d, k - 1, axis=1)[:, :k]
            kth = np.take_along_axis(d, part, axis=1).max(axis=1)
            for r, i in enumerate(ii):
                idx = np.flatnonzero(d[r] <= kth[r])
                if len(idx) > k:
                    idx = idx[np.argsort(d[r, idx], kind="stable")[:k]]
                out[i] = float(np.median(self.gap[idx]))
        return out


def build_map(box_dir, clips, k=K_NN):
    """G from the label tracks of `clips`: frames with a lead, a finite label gap and a finite positive width and height (no cut on the gap). -> (MapG or None, info)"""
    lnh, lnw, gp = [], [], []
    missing = 0
    for c in clips:
        p = os.path.join(box_dir, c + ".npz")
        if not os.path.isfile(p):
            missing += 1
            continue
        z = np.load(p)
        g, w, h = np.asarray(z["gap_re"], np.float64), np.asarray(z["ang_w"], np.float64), np.asarray(z["ang_h"], np.float64)
        with np.errstate(invalid="ignore"):
            ok = z["has_lead"].astype(bool) & np.isfinite(g) & np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0)
        lnh.append(np.log(h[ok]))
        lnw.append(np.log(w[ok]))
        gp.append(g[ok])
    info = {"n_train_clips": len(clips), "n_train_clips_missing_box": missing, "n_train_frames": int(sum(len(x) for x in gp))}
    if not gp or info["n_train_frames"] == 0:
        return None, info
    return MapG(np.concatenate(lnh), np.concatenate(lnw), np.concatenate(gp), k), info


# ============================================================================================ rows and measures
def gap_bin_mask(g, lo, hi):
    return (g >= lo) & ((g < hi) if hi < 60 else (g <= hi))


def t3_rows(L):
    """rows of control t3 of card ADD122: a label lead in D_0 (lab_present, finite lab_gap) and a finite gap in D_0 to D_2. Not clamped to 60 m (as the reader of that card)."""
    d0 = L["D_0"]
    m = d0["lab_present"].astype(bool) & np.isfinite(d0["lab_gap"])
    for k in ARM_D:
        m = m & np.isfinite(L[k]["gap"])
    return m


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
    a, b, n = (np.asarray(x, np.float64) for x in (a, b, n))
    idx = boot_idx(len(a), n_boot, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (a[idx].sum(axis=1) - b[idx].sum(axis=1)) / n[idx].sum(axis=1)
    d = d[np.isfinite(d)]
    if d.size < n_boot // 2:
        return None
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def mean_abs_err(gap, lab, rows):
    return fnum(np.abs(gap[rows] - lab[rows]).mean()) if rows.any() else None


def decoded_gaps(G, pr):
    """gap of one fit on every join row: G at the decoded width and height of the head's output."""
    w, h = decode(pr["az"], pr["el"])
    return G.predict(w, h)


# ============================================================================================ facts for the controls (pure functions below)
def collect_facts(a):
    """everything the controls need, read from disk; nothing of arm S's gap or size error is computed here. -> dict"""
    f = {"errors": []}
    J = None
    try:
        J = load_npz(a.eval_join)
    except (OSError, ValueError, KeyError) as ex:
        f["errors"].append("join: %s: %s" % (type(ex).__name__, ex))
    f["J"] = J
    lf = parse_kv(a.list_file)
    f["list_files"] = {k: sha_of(v) for k, v in lf.items()}
    f["list_record"] = try_json(a.list_record) if a.list_record else None
    f["fit_records"] = {k: try_json(os.path.join(a.out_root, "fits", FIT_DIR[k], "FIT_RECORD.json")) for k in FITS}
    f["box_records"] = {k: try_json(os.path.join(a.out_root, "fits", FIT_DIR[k], "BOXSIZE_FIT_RECORD.json")) for k in FITS}
    f["identity"] = try_json(os.path.join(a.out_root, "IDENTITY_K5.json"))
    f["identity_box"] = try_json(os.path.join(a.out_root, "BOXSIZE_IDENTITY_RECORD.json"))
    before = parse_kv(a.test_before, float)
    f["tests"] = {}
    for k, p in parse_kv(a.test_out).items():
        f["tests"][k] = {"text": read_text(p), "mtime": os.path.getmtime(p) if os.path.isfile(p) else None, "before": before.get(k)}
    return f


def load_arrays(a, J):
    """the arrays both stages use. -> (L {D_k: laid dump}, preds {fit: {az, el}}, t3 rows, label sizes, train clip ids) ; raises on an unreadable input"""
    L = {k: lay_dump(J, load_npz(p)) for k, p in dump_paths(a).items()}
    preds = {}
    for k in FITS:
        preds[k] = lay_pred(J, load_npz(os.path.join(a.out_root, "pred", FIT_DIR[k] + ".npz")))
    lab = box_rows(J, a.box_dir)
    return L, preds, lab


def dump_paths(a):
    p = {"D_0": V8 + "/cropread/run1/dump/D_H5/reader_s0_score.npz", "D_1": V8 + "/cropread/run1/dump/D_H5/reader_s1_score.npz",
         "D_2": V8 + "/visseed10k/fitroot/dump/D_H5/reader_s2_score.npz"}
    p.update(parse_kv(a.dump))
    return p


# ============================================================================================ controls
def _entries(rec):
    out = {}
    if not isinstance(rec, dict):
        return out
    for src in (rec, rec.get("files") if isinstance(rec.get("files"), dict) else {}):
        for k, v in src.items():
            if isinstance(v, str) and HEX64.match(v.lower()):
                out[re.sub(r"_sha(256)?$", "", k)] = v.lower()
    return out


def token_root_of(rec):
    """the value after --root in the record's argv (also `--root=X`); None when the record names none."""
    if not isinstance(rec, dict) or not isinstance(rec.get("argv"), list):
        return None
    av = [str(x) for x in rec["argv"]]
    for i, x in enumerate(av):
        if x == "--root" and i + 1 < len(av):
            return av[i + 1]
        if x.startswith("--root="):
            return x[len("--root="):]
    return None


def ctl_s0(f):
    """lists and tokens: the lists equal, by sha256, the entries of the banked fit record; every fit's own record carries the same two hashes; every record names one token root."""
    ent = _entries(f.get("list_record"))
    if not ent:
        return _res(["the banked list record is missing or holds no sha256 entry"])
    bad = []
    files = f.get("list_files") or {}
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
    roots = set()
    for k in FITS:
        fr, br = (f.get("fit_records") or {}).get(k), (f.get("box_records") or {}).get(k)
        if not isinstance(fr, dict):
            bad.append("%s: FIT_RECORD.json missing or unreadable" % k)
            continue
        for nm in ("train_list", "eval_list"):
            v = fr.get(nm + "_sha256")
            if not (isinstance(v, str) and ent.get(nm) == v.lower()):
                bad.append("%s: %s_sha256 differs from the list record" % (k, nm))
        for rec in (fr, br):
            r = token_root_of(rec)
            if r is None:
                bad.append("%s: a record names no token root (argv --root)" % k)
            else:
                roots.add(r)
    if len(roots) > 1:
        bad.append("the records name %d different token roots" % len(roots))
    return _res(bad, compared=sorted(files), token_root=sorted(roots)[0] if len(roots) == 1 else None)


def ctl_s1(idrec, boxrec, fit_sha=None):
    """identity: the locator tool's identity run passes, within 1e-4, and its record through the new tool exists."""
    bad = []
    if not isinstance(idrec, dict):
        return _res(["IDENTITY_K5.json is missing or unreadable"])
    d = idrec.get("abs_diff")
    if idrec.get("PASS") is not True:
        bad.append("PASS is not true")
    if not isinstance(d, (int, float)) or not np.isfinite(d) or d > ID_TOL:
        bad.append("abs_diff %s is not within %g" % (d, ID_TOL))
    if idrec.get("n_sel_samples") != idrec.get("n_sel_samples_record"):
        bad.append("sample count %s differs from the record's %s" % (idrec.get("n_sel_samples"), idrec.get("n_sel_samples_record")))
    if not isinstance(boxrec, dict):
        bad.append("BOXSIZE_IDENTITY_RECORD.json is missing or unreadable")
    elif fit_sha and boxrec.get("locator_fit_sha256") is not None and boxrec.get("locator_fit_sha256") != fit_sha:
        bad.append("the identity record names another locator_fit_sha256 than the fits")
    return _res(bad)


def _norm_item(x):
    s = x.get("name") if isinstance(x, dict) else x
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def ctl_s2(box_records, roundtrip):
    """one value changed: every record lists exactly the box loader, restored; one locator_fit_sha256; seeds and the permute flag as the arms; encode then decode returns the label sizes within 1e-9."""
    bad = []
    shas = set()
    for k in FITS:
        r = (box_records or {}).get(k)
        if not isinstance(r, dict):
            bad.append("%s: BOXSIZE_FIT_RECORD.json missing or unreadable" % k)
            continue
        ch = r.get("module_values_changed")
        if not isinstance(ch, list) or len(ch) != 1 or _norm_item(ch[0]) != CHANGED_EXPECT:
            bad.append("%s: module_values_changed is not exactly [load_box]: %s" % (k, str(ch)[:80]))
        if r.get("module_values_restored") is not True:
            bad.append("%s: module_values_restored is not true" % k)
        s = r.get("locator_fit_sha256")
        if not (isinstance(s, str) and HEX64.match(s.lower())):
            bad.append("%s: no locator_fit_sha256" % k)
        else:
            shas.add(s.lower())
        if r.get("seed") != FIT_SEED[k] or bool(r.get("permute_targets")) != (k == PERM_FIT):
            bad.append("%s: seed %s / permute_targets %s do not match the arm" % (k, r.get("seed"), r.get("permute_targets")))
    if len(shas) > 1:
        bad.append("%d different locator_fit_sha256 across the records" % len(shas))
    rt = roundtrip or {}
    md = rt.get("max_abs_diff")
    if md is None or not np.isfinite(md) or md > RT_TOL or not rt.get("n_inside"):
        bad.append("encode then decode: max abs diff %s on %s rows inside the limits (tolerance %g)" % (md, rt.get("n_inside"), RT_TOL))
    return _res(bad, locator_fit_sha256=sorted(shas)[0] if len(shas) == 1 else None)


def ctl_s3(E, n_rows, ex=EXPECT, ref=REF_D):
    bad = []
    for k, v in ref.items():
        e = (E or {}).get(k)
        if e is None or abs(e - v) > TOL_E:
            bad.append("E(%s) %s, expected %.3f within %g" % (k, e, v, TOL_E))
    if n_rows != ex["rows"]:
        bad.append("%s rows, expected %d" % (n_rows, ex["rows"]))
    return _res(bad, E=E, n_rows=n_rows)


def ctl_s4(v, max_e=MAP_MAX_E):
    """the map: G on the label sizes has an error of at most 2.2 m; G on label sizes permuted across the frames at least 0.95 of the constant's."""
    bad = []
    if v.get("n_train_clips_missing_box"):
        bad.append("%d train clips without a box file" % v["n_train_clips_missing_box"])
    if not v.get("n_train_frames"):
        return _res(bad + ["no train frame for the map"], **v)
    el, ep, ec = v.get("E_label"), v.get("E_perm"), v.get("E_const")
    if el is None or el > max_e:
        bad.append("E(G on label sizes) %s above %.2f" % (el, max_e))
    if ep is None or ec is None or ep < PERM_MIN * ec:
        bad.append("E(G on permuted label sizes) %s below %.2f of the constant's %s" % (ep, PERM_MIN, ec))
    return _res(bad, **v)


def ctl_s5(v):
    """permuted targets: the gap through arm SP has an error of at least 0.95 of that of the constant."""
    e, c = v.get("E_sp"), v.get("E_const")
    bad = []
    if e is None or c is None or e < PERM_MIN * c:
        bad.append("E(SP_0) %s below %.2f of the constant's %s" % (e, PERM_MIN, c))
    return _res(bad, **v)


def ctl_s6(f):
    """tests and counts: each test output ends with `<n> passed` and is older than its start epoch; every t3 row has a finite size output of each fit."""
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
    fin = f.get("finite") or {}
    if not f.get("n_rows"):
        bad.append("no t3 rows")
    for k in FITS:
        nb = fin.get(k)
        if nb is None or nb != 0:
            bad.append("%s: %s of the t3 rows without a finite size output" % (k, nb))
    return _res(bad, n_rows=f.get("n_rows"))


def control_values(a, f, L, preds, lab, G, ginfo):
    """the values the controls need, from arrays. Arm S's gap and size errors are NOT computed here. -> (s3 values, s4 values, s5 values, s2 roundtrip, finite counts, t3 rows)"""
    t3 = t3_rows(L)
    labg = L["D_0"]["lab_gap"]
    E = {k: mean_abs_err(L[k]["gap"], labg, t3) for k in ARM_D}
    const = float(np.median(labg[t3])) if t3.any() else None
    s4 = dict(ginfo)
    s4.update({"E_label": None, "E_perm": None, "E_const": None, "const": const, "n_rows_with_label_size": int((t3 & lab["has"]).sum()), "n_t3_rows_without_label_size": int((t3 & ~lab["has"]).sum())})
    if G is not None and (t3 & lab["has"]).any():
        m = t3 & lab["has"]
        gl = G.predict(lab["w"][m], lab["h"][m])
        s4["E_label"] = fnum(np.abs(gl - labg[m]).mean())
        perm = np.random.RandomState(PERM_SEED).permutation(int(m.sum()))
        gp = G.predict(lab["w"][m][perm], lab["h"][m][perm])
        s4["E_perm"] = fnum(np.abs(gp - labg[m]).mean())
        s4["E_const"] = fnum(np.abs(const - labg[m]).mean())
    s5 = {"E_sp": None, "E_const": None, "const": const, "n_rows": 0}
    finite = {}
    for k in FITS:
        pr = preds[k]
        finite[k] = int((t3 & ~(np.isfinite(pr["az"]) & np.isfinite(pr["el"]))).sum())
    if G is not None and t3.any():
        g = decoded_gaps(G, preds[PERM_FIT])
        m = t3 & np.isfinite(g)
        if m.any():
            s5.update({"E_sp": fnum(np.abs(g[m] - labg[m]).mean()), "E_const": fnum(np.abs(const - labg[m]).mean()), "n_rows": int(m.sum())})
    inside = t3 & lab["has"] & ~outside_limits(lab["w"]) & ~outside_limits(lab["h"])
    if inside.any():
        ea, eb = encode(lab["w"][inside], lab["h"][inside])
        w2, h2 = decode(ea, eb)
        rt = {"max_abs_diff": fnum(max(np.abs(w2 - lab["w"][inside]).max(), np.abs(h2 - lab["h"][inside]).max())), "n_inside": int(inside.sum())}
    else:
        rt = {"max_abs_diff": None, "n_inside": 0}
    return E, int(t3.sum()), s4, s5, rt, finite


def run_controls(a):
    f = collect_facts(a)
    ctl = {}
    ctl["s0"] = ctl_s0(f)
    fit_sha = ((f["box_records"].get("S_0") or {}) if isinstance(f["box_records"].get("S_0"), dict) else {}).get("locator_fit_sha256")
    ctl["s1"] = ctl_s1(f["identity"], f["identity_box"], fit_sha)
    J = f["J"]
    G, ginfo, vals = None, {}, None
    try:
        if J is None:
            raise ValueError("no evaluation join")
        L, preds, lab = load_arrays(a, J)
        tclips = ids_of(parse_kv(a.list_file)["train_list"])
        G, ginfo = build_map(a.box_dir, tclips)
        vals = control_values(a, f, L, preds, lab, G, ginfo)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as ex:
        why = "the read could not be computed: %s: %s" % (type(ex).__name__, ex)
        ctl["s2"] = ctl_s2(f["box_records"], None)
        ctl["s2"]["pass"] = False
        ctl["s2"]["failures"].append(why)
        for k in ("s3", "s4", "s5", "s6"):
            ctl[k] = _res([why])
        return assemble(ctl)
    E, n_rows, s4, s5, rt, finite = vals
    ctl["s2"] = ctl_s2(f["box_records"], rt)
    ctl["s3"] = ctl_s3(E, n_rows, a.expect, a.ref_d)
    ctl["s4"] = ctl_s4(s4, a.map_max_e)
    ctl["s5"] = ctl_s5(s5)
    f["finite"], f["n_rows"] = finite, n_rows
    ctl["s6"] = ctl_s6(f)
    return assemble(ctl)


def assemble(ctl):
    allp = sorted(ctl) == sorted(CONTROLS) and all(v["pass"] for v in ctl.values())
    return {"card": CARD, "tier": TIER, "controls": ctl, "all_pass": bool(allp)}


def input_hashes(a):
    paths = {"join": a.eval_join, "list_record": a.list_record}
    for k, p in parse_kv(a.list_file).items():
        paths["list_" + k] = p
    for k, p in dump_paths(a).items():
        paths["dump_" + k] = p
    for k in FITS:
        d = os.path.join(a.out_root, "fits", FIT_DIR[k])
        paths["pred_" + k] = os.path.join(a.out_root, "pred", FIT_DIR[k] + ".npz")
        paths["fit_record_" + k] = os.path.join(d, "FIT_RECORD.json")
        paths["boxsize_record_" + k] = os.path.join(d, "BOXSIZE_FIT_RECORD.json")
    paths["identity"] = os.path.join(a.out_root, "IDENTITY_K5.json")
    paths["identity_box"] = os.path.join(a.out_root, "BOXSIZE_IDENTITY_RECORD.json")
    for k, p in parse_kv(a.test_out).items():
        paths["tests_" + k] = p
    out = {k: sha_of(p) if p else None for k, p in paths.items()}
    try:
        lines = "".join("%s:%d\n" % (n, os.path.getsize(os.path.join(a.box_dir, n))) for n in sorted(os.listdir(a.box_dir)) if n.endswith(".npz"))
        out["box_dir_listing"] = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    except OSError:
        out["box_dir_listing"] = None
    out["read_boxsize.py"] = sha_file(os.path.abspath(__file__))
    return out


# ============================================================================================ values of card section 3 (names stage only)
def compute_values(J, L, preds, lab, G, n_boot=N_BOOT, seed=BOOT_SEED, ref_const=None):
    n = np.asarray(J["n"], np.int64)
    nc = len(n)
    ci_all = np.repeat(np.arange(nc), n)
    labg_all = L["D_0"]["lab_gap"]
    t3 = t3_rows(L)
    rows = np.flatnonzero(t3)
    ci, labg = ci_all[rows], labg_all[rows]
    gaps = {k: L[k]["gap"][rows] for k in ARM_D}
    for k in ARM_S:
        gaps[k] = decoded_gaps(G, preds[k])[rows]
    absk = {k: np.abs(gaps[k] - labg) for k in gaps}
    sgn = {k: gaps[k] - labg for k in gaps}
    sums = {k: np.bincount(ci, weights=absk[k], minlength=nc) for k in gaps}
    P = {"n_clips": int(nc), "n_boot": n_boot, "seed": seed, "tier": TIER, "families": FAMILIES, "n_rows": int(len(rows)),
         "interval_level": "two-sided 95 % percentile, cluster bootstrap over the evaluation clips, fits held fixed (it answers what another draw of clips would say, not what another fit would say)"}
    P["E"] = {k: fnum(absk[k].mean()) for k in ARM_D + ARM_S}
    P["mean_E"] = {"S": fnum(np.mean([P["E"][k] for k in ARM_S])), "D": fnum(np.mean([P["E"][k] for k in ARM_D]))}
    mnum = {"S": sum(sums[k] for k in ARM_S) / 3.0, "D": sum(sums[k] for k in ARM_D) / 3.0}
    far_m = gap_bin_mask(labg, 40, 60)
    far_n = np.bincount(ci[far_m], minlength=nc).astype(np.float64)
    far_s = {k: np.bincount(ci[far_m], weights=sgn[k][far_m], minlength=nc) for k in gaps}
    P["sums"] = {"r": np.bincount(ci, minlength=nc).astype(np.float64).tolist(), "abs": {k: sums[k].tolist() for k in gaps}, "far_n": far_n.tolist(), "far_signed": {k: far_s[k].tolist() for k in gaps}}
    P["R"] = fnum(P["mean_E"]["S"] / P["mean_E"]["D"]) if P["mean_E"]["D"] else None
    P["iv"] = boot_ratio(mnum["S"], mnum["D"], n_boot, seed)
    P["bins"] = {}
    for X, ks in (("D", ARM_D), ("S", ARM_S)):
        out = []
        for lo, hi in GAP_BINS:
            m = gap_bin_mask(labg, lo, hi)
            if not m.any():
                out.append({"bin": "%d-%d m" % (lo, hi), "n": 0, "E": None, "signed": None, "short5": None})
                continue
            out.append({"bin": "%d-%d m" % (lo, hi), "n": int(m.sum()), "E": fnum(np.mean([absk[k][m].mean() for k in ks])), "signed": fnum(np.mean([sgn[k][m].mean() for k in ks])),
                        "short5": fnum(np.mean([(sgn[k][m] < -SHORT_M).mean() for k in ks]))})
        P["bins"][X] = out
    nfar = float(far_n.sum())
    P["far"] = {"n": int(nfar), "n_clips": int((far_n > 0).sum()),
                "signed": {X: (fnum(np.mean([far_s[k].sum() / nfar for k in ks])) if nfar > 0 else None) for X, ks in (("D", ARM_D), ("S", ARM_S))},
                "signed_single": {k: (fnum(far_s[k].sum() / nfar) if nfar > 0 else None) for k in ARM_D + ARM_S},
                "iv_diff_S_minus_D": boot_diff_over(sum(far_s[k] for k in ARM_S) / 3.0, sum(far_s[k] for k in ARM_D) / 3.0, far_n, n_boot, seed) if nfar > 0 else None}
    # printed beside, judged by nothing
    has = lab["has"][rows]
    w_lab, h_lab = lab["w"][rows], lab["h"][rows]
    bs = {"size_error_pct": [], "G_on_label_sizes": []}
    gl = G.predict(w_lab[has], h_lab[has]) if has.any() else np.zeros(0)
    gl_full = np.full(len(rows), np.nan)
    gl_full[has] = gl
    dec = {k: decode(preds[k]["az"][rows], preds[k]["el"][rows]) for k in ARM_S}
    for lo, hi in GAP_BINS:
        m = gap_bin_mask(labg, lo, hi) & has
        name = "%d-%d m" % (lo, hi)
        if not m.any():
            bs["size_error_pct"].append({"bin": name, "n": 0, "w": None, "h": None})
            bs["G_on_label_sizes"].append({"bin": name, "n": 0, "E": None})
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ew = np.mean([np.abs(np.log(dec[k][0][m]) - np.log(w_lab[m])).mean() for k in ARM_S]) * 100.0
            eh = np.mean([np.abs(np.log(dec[k][1][m]) - np.log(h_lab[m])).mean() for k in ARM_S]) * 100.0
        bs["size_error_pct"].append({"bin": name, "n": int(m.sum()), "w": fnum(ew), "h": fnum(eh)})
        bs["G_on_label_sizes"].append({"bin": name, "n": int(m.sum()), "E": fnum(np.abs(gl_full[m] - labg[m]).mean())})
    bs["G_on_label_sizes_all"] = {"n": int(has.sum()), "E": fnum(np.abs(gl_full[has] - labg[has]).mean()) if has.any() else None}
    bs["n_rows_without_label_size"] = int((~has).sum())
    ow, oh = outside_limits(w_lab[has]), outside_limits(h_lab[has])
    bs["outside_limits"] = {"n": int(has.sum()), "width": fnum(ow.mean()) if has.any() else None, "height": fnum(oh.mean()) if has.any() else None, "either": fnum((ow | oh).mean()) if has.any() else None,
                            "limits_deg": [float(np.exp(U0 - ULIM)), float(np.exp(U0 + ULIM))]}
    P["beside"] = bs
    return P


# ============================================================================================ names (two derivations)
def name_ratio(iv):
    if iv is None:
        return None
    if iv[1] < 1.0:
        return PRE + "-CUTS-GAP-ERROR"
    if iv[0] > 1.0:
        return PRE + "-WORSE"
    return PRE + "-NOT-SEPARATED"


def name_far(sS, sD, iv):
    if sS is not None and sD is not None and iv is not None and abs(sS) <= FAR_HALF * abs(sD) and (iv[0] > 0.0 or iv[1] < 0.0):
        return PRE + "-FAR-LEADS-READ"
    return PRE + "-FAR-LEADS-STAY-SHORT"


def names_first(P):
    return {"RATIO": name_ratio(P["iv"]), "FAR": name_far(P["far"]["signed"]["S"], P["far"]["signed"]["D"], P["far"]["iv_diff_S_minus_D"]),
            "bar": bool(P["R"] is not None and P["iv"] is not None and P["R"] <= BAR_RATIO and P["iv"][1] < 1.0)}


def _pct(v, q):
    pos = (len(v) - 1) * q / 100.0
    a = int(pos)
    b = min(a + 1, len(v) - 1)
    return v[a] + (v[b] - v[a]) * (pos - a)


def names_second(P):
    """written apart: per-clip sums only, plain loops, hand-made percentiles."""
    S = P["sums"]
    nc, nb = len(S["r"]), P["n_boot"]
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

    num, den = arm("S"), arm("D")
    v = []
    for row in idx:
        d = sum(den[j] for j in row)
        if d > 0:
            v.append(sum(num[j] for j in row) / d)
    iv_r = interval(v)
    ratio = (sum(num) / sum(den)) if sum(den) > 0 else None
    fn = S["far_n"]
    nf = sum(fn)
    fS, fD = farsum("S"), farsum("D")
    v = []
    for row in idx:
        d = sum(fn[j] for j in row)
        if d > 0:
            v.append((sum(fS[j] for j in row) - sum(fD[j] for j in row)) / d)
    iv_f = interval(v)
    sS, sD = (sum(fS) / nf, sum(fD) / nf) if nf > 0 else (None, None)
    names = {"RATIO": name_ratio(iv_r), "FAR": name_far(sS, sD, iv_f), "bar": bool(ratio is not None and iv_r is not None and ratio <= BAR_RATIO and iv_r[1] < 1.0)}
    return names, {"RATIO": iv_r, "FAR": iv_f}


def derive_names(P):
    n1 = names_first(P)
    n2, iv2 = names_second(P)
    first_iv = {"RATIO": P["iv"], "FAR": P["far"]["iv_diff_S_minus_D"]}
    agree = n1 == n2
    for k in ("RATIO", "FAR"):
        x, y = first_iv[k], iv2[k]
        agree = agree and ((x is None and y is None) or (x is not None and y is not None and bool(np.allclose(x, y, atol=1e-9, rtol=0))))
    return n1, n2, agree


# ============================================================================================ output
def fv(x, d=3):
    return "None" if x is None else ("%." + str(d) + "f") % x


def fiv(iv):
    return "[%.4f, %.4f]" % (iv[0], iv[1]) if iv else "None"


def print_sections(P, nm):
    L = ["tier: " + TIER, "families: " + FAMILIES]
    L.append("rows: %d evaluation frames of control t3 (a label lead and a finite gap in D_0 to D_2), %d evaluation clips" % (P["n_rows"], P["n_clips"]))
    L.append("E (mean absolute gap error in m, each fit singly): " + "; ".join("%s %s" % (k, fv(P["E"][k])) for k in ARM_D + ARM_S))
    L.append("ratio S / D = mean E(S) / mean E(D) = %s (%s / %s); interval %s (%d draws, RandomState(%d), %s)" % (fv(P["R"], 4), fv(P["mean_E"]["S"], 4), fv(P["mean_E"]["D"], 4), fiv(P["iv"]), P["n_boot"], P["seed"], P["interval_level"]))
    L.append("name: %s" % (nm["RATIO"] if nm["RATIO"] else "none (no interval for the ratio)"))
    fr = P["far"]
    L.append("far leads (label gap 40-60 m), n %d in %d clips; mean signed error (read minus label) of the mean of three fits: D %s, S %s; singly: %s; paired difference S minus D interval %s"
             % (fr["n"], fr["n_clips"], fv(fr["signed"]["D"]), fv(fr["signed"]["S"]), "; ".join("%s %s" % (k, fv(fr["signed_single"][k])) for k in ARM_D + ARM_S), fiv(fr["iv_diff_S_minus_D"])))
    L.append("name: %s" % nm["FAR"])
    L.append("bar met: %s (ratio %s at most %.2f and its interval %s below 1)" % ("yes" if nm["bar"] else "no", fv(P["R"], 4), BAR_RATIO, fiv(P["iv"])))
    for X in ("D", "S"):
        L.append("by label gap, arm %s (mean of three fits): " % X + "; ".join("%s n %d E %s signed %s short>5m %s" % (b["bin"], b["n"], fv(b["E"]), fv(b["signed"]), fv(b["short5"])) for b in P["bins"][X]))
    bs = P["beside"]
    L.append("printed beside, judged by nothing: size read error in per cent (mean absolute error of ln width and ln height against the label, mean of the three fits), by label gap: "
             + "; ".join("%s n %d width %s height %s" % (b["bin"], b["n"], fv(b["w"], 1), fv(b["h"], 1)) for b in bs["size_error_pct"]))
    L.append("printed beside: gap error of G on the label sizes (the bound of a perfect size read), by label gap: " + "; ".join("%s n %d E %s" % (b["bin"], b["n"], fv(b["E"])) for b in bs["G_on_label_sizes"])
             + "; all n %d E %s (rows without a label size: %d)" % (bs["G_on_label_sizes_all"]["n"], fv(bs["G_on_label_sizes_all"]["E"]), bs["n_rows_without_label_size"]))
    o = bs["outside_limits"]
    L.append("printed beside: share of frames whose label size lies outside the limits %.2f to %.1f degrees of the encoding: width %s, height %s, either %s (n %d)"
             % (o["limits_deg"][0], o["limits_deg"][1], fv(o["width"], 4), fv(o["height"], 4), fv(o["either"], 4), o["n"]))
    L.append("interval note: it answers what another draw of evaluation clips would say for these fits; three fit seeds on one set of train clips is a small number, and a separated interval is necessary, not sufficient")
    return L


def check_inputs(a, cj):
    """the inputs hash as they did at the controls. -> list of differing keys"""
    now = input_hashes(a)
    was = cj.get("inputs") or {}
    return sorted(k for k in set(now) | set(was) if now.get(k) != was.get(k))


def run_names(a):
    cj = try_json(a.controls) or {}
    if cj.get("card") != CARD or cj.get("all_pass") is not True or sorted(cj.get("controls", {})) != sorted(CONTROLS) or any(v.get("pass") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass with s0 to s6 present: no name and no arm value is written")
        print("ZZ READ-BOXSIZE names rc 3")
        return 3
    diff = check_inputs(a, cj)
    if diff:
        print("REFUSED: inputs differ from those of the controls: %s" % diff[:6])
        print("ZZ READ-BOXSIZE names rc 3")
        return 3
    logp = a.log or (os.path.splitext(a.out)[0] + "_names.log")
    if os.path.exists(a.out) or os.path.exists(logp):
        print("REFUSED: %s or %s exists" % (a.out, logp))
        print("ZZ READ-BOXSIZE names rc 2")
        return 2
    J = load_npz(a.eval_join)
    L, preds, lab = load_arrays(a, J)
    G, _ = build_map(a.box_dir, ids_of(parse_kv(a.list_file)["train_list"]))
    P = compute_values(J, L, preds, lab, G, a.n_boot)
    n1, n2, agree = derive_names(P)
    if not agree:
        print("STOP: the two derivations of the names disagree: %s | %s" % (n1, n2))
        print("ZZ READ-BOXSIZE names rc 4")
        return 4
    res = {"card": CARD, "tier": TIER, "names": n1, "second_derivation_agrees": True, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls),
           "reader_sha256": sha_file(os.path.abspath(__file__)), "values": P}
    wj(a.out, res)
    rc = 0 if (n1["RATIO"] and n1["FAR"]) else 5
    lines = print_sections(P, n1) + ["ZZ READ-BOXSIZE names rc %d" % rc]
    wt(logp, "\n".join(lines) + "\n")
    for ln in lines:
        print(ln)
    return rc


# ============================================================================================ selftest world and tree
SYN_NE, SYN_NT, SYN_NFR = 12, 60, 30
LAW_H, LAW_W = 60.0, 90.0                       # synthetic box law: height = 60 / gap, width = 90 / gap (degrees)
ERR = {"cuts": (1.5, 1.5, 1.5), "mild": (3.5, 3.5, 3.5), "worse": (6.0, 6.0, 6.0), "sep": (4.0, 3.9, 4.1)}
FAR_BIAS = {"cuts": 1.0, "mild": 5.0, "worse": 6.0, "sep": 6.0}
CTL_FAULTS = {"s0", "s1", "s2", "s3", "s4", "s5", "s6"}


def law(g):
    g = np.asarray(g, np.float64)
    return LAW_W / g, LAW_H / g


def syn_world(ks="cuts", seed=5, amp=0.2):
    """synthetic evaluation world: join, banked dumps D_0..D_2, label gap and sizes per eval row, size outputs of the four fits (az, el), train label tracks."""
    rng = np.random.RandomState(seed)
    nc, nfr = SYN_NE, SYN_NFR
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
    dumps = {}
    for i, k in enumerate(ARM_D):
        mult = 1.0 + amp * np.random.RandomState(100 + sum(map(ord, k))).uniform(-1, 1, nc)[ci]
        gap = lab + sign * (4.0, 3.9, 4.1)[i] * mult - 6.0 * far + 0.01 * rng.randn(nrow)
        gap = np.where(present, gap, rng.uniform(5, 50, nrow))
        dumps[k] = {"clips": J["clip"], "clip_index": ci.copy(), "frame": fr.copy(), "gap": gap, "valid": np.ones(nrow, bool), "lab_present": present.copy(), "lab_gap": lab.copy()}
    outs = {}
    for i, k in enumerate(ARM_S):
        mult = 1.0 + amp * np.random.RandomState(200 + i).uniform(-1, 1, nc)[ci]
        gd = np.where(present, lab + sign * ERR[ks][i] * mult - FAR_BIAS[ks] * far, 20.0)
        outs[k] = np.maximum(gd, 3.0)
    outs[PERM_FIT] = np.where(present, np.random.RandomState(77).permutation(np.nan_to_num(lab, nan=30.0)), 20.0)
    preds = {}
    for k, gd in outs.items():
        w, h = law(gd)
        az, el = encode(w, h)
        preds[k] = {"clips": J["clip"], "clip_index": ci.copy(), "frame": fr.copy(), "az_pred": az, "el_pred": el, "has_out": np.ones(nrow, bool)}
    wl, hl = law(np.where(present, lab, 20.0))
    boxes = {"c%02d" % i: {"has_lead": present[i * nfr:(i + 1) * nfr].copy(), "gap_re": np.where(present, lab, np.nan)[i * nfr:(i + 1) * nfr], "ang_w": wl[i * nfr:(i + 1) * nfr],
                           "ang_h": hl[i * nfr:(i + 1) * nfr], "ctr_az": np.zeros(nfr), "ctr_el": np.zeros(nfr)} for i in range(nc)}
    tbox = {}
    for i in range(SYN_NT):
        g = np.linspace(8.0, 58.0, SYN_NT)[i] + np.linspace(-6.0, 6.0, nfr)
        w, h = law(g)
        tbox["t%02d" % i] = {"has_lead": np.ones(nfr, bool), "gap_re": g, "ang_w": w, "ang_h": h, "ctr_az": np.zeros(nfr), "ctr_el": np.zeros(nfr)}
    return J, dumps, preds, boxes, tbox


def write_tree(root, ks="cuts", fault=None):
    """a tiny pod-like tree under `root`; the planted `fault` flips the named control. -> (argv list for main, paths dict)"""
    J, dumps, preds, boxes, tbox = syn_world(ks)
    root = os.path.abspath(root)
    sub = {k: os.path.join(root, k) for k in ("lists", "box", "dump", "rec", "tests", "fit_out")}
    for p in sub.values():
        os.makedirs(p, exist_ok=True)
    for name, d in list(boxes.items()) + list(tbox.items()):
        d = dict(d)
        if fault == "s4" and name in tbox:
            d["gap_re"] = np.asarray(d["gap_re"]) + 5.0
        np.savez(os.path.join(sub["box"], name + ".npz"), **d)
    ev = [str(c) for c in J["clip"]]
    tr = sorted(tbox)
    wt(os.path.join(sub["lists"], "train.txt"), "".join("%s fit\n" % c for c in tr))
    wt(os.path.join(sub["lists"], "eval.txt"), "".join(c + "\n" for c in ev))
    th, eh = sha_file(os.path.join(sub["lists"], "train.txt")), sha_file(os.path.join(sub["lists"], "eval.txt"))
    wj(os.path.join(sub["rec"], "list_record.json"), {"train_list_sha256": th, "eval_list_sha256": eh})
    if fault == "s0":
        wt(os.path.join(sub["lists"], "eval.txt"), "c00\n")
    np.savez(os.path.join(root, "join.npz"), **J)
    paths = {}
    for k, d in dumps.items():
        d = dict(d)
        if fault == "s3" and k == "D_1":
            d["gap"] = d["gap"] + 1.0
        p = os.path.join(sub["dump"], k + ".npz")
        np.savez(p, **d)
        paths[k] = p
    lsha = "cd" * 32
    for k in FITS:
        fdir = os.path.join(sub["fit_out"], "fits", FIT_DIR[k])
        os.makedirs(fdir, exist_ok=True)
        os.makedirs(os.path.join(sub["fit_out"], "pred"), exist_ok=True)
        pr = dict(preds[k])
        if fault == "s5" and k == PERM_FIT:
            pr = dict(preds["S_0"])
        if fault == "s6" and k == "S_1":
            pr["az_pred"] = np.array(pr["az_pred"])
            pr["az_pred"][5] = np.nan
            pr["has_out"] = np.array(pr["has_out"])
        np.savez(os.path.join(sub["fit_out"], "pred", FIT_DIR[k] + ".npz"), **pr)
        argv = ["boxsize_fit.py", "--seed", str(FIT_SEED[k]), "--root", "/tok/root", "--out-root", "/o"] + (["--permute-targets"] if k == PERM_FIT else [])
        wj(os.path.join(fdir, "FIT_RECORD.json"), {"card": "PREREG_V8F_PROOF_ADD107", "seed": FIT_SEED[k], "argv": argv, "train_list_sha256": th, "eval_list_sha256": eh})
        ch = [{"name": "load_box"}]
        restored = True
        if fault == "s2" and k == "S_2":
            ch = [{"name": "load_box"}, {"name": "lr"}]
        wj(os.path.join(fdir, "BOXSIZE_FIT_RECORD.json"), {"card": CARD, "tier": "T0", "module_values_changed": ch, "module_values_restored": restored, "locator_fit_sha256": lsha, "tool_sha256": "ef" * 32,
                                                          "argv": argv, "seed": FIT_SEED[k], "permute_targets": k == PERM_FIT, "finished_utc": "2026-11-24T00:00:00Z"})
    idr = {"card": "PREREG_V8F_PROOF_ADD107", "control": "k5", "abs_diff": 2e-6, "PASS": True, "n_sel_samples": 900, "n_sel_samples_record": 900}
    if fault == "s1":
        idr.update({"abs_diff": 0.02, "PASS": False})
    wj(os.path.join(sub["fit_out"], "IDENTITY_K5.json"), idr)
    wj(os.path.join(sub["fit_out"], "BOXSIZE_IDENTITY_RECORD.json"), {"card": CARD, "tier": "T0", "locator_fit_sha256": lsha, "argv": ["boxsize_fit.py", "--identity"]})
    now = time.time()
    for k in ("fit", "read"):
        p = os.path.join(sub["tests"], k + ".out")
        wt(p, ("1 failed, 17 passed in 1s\n" if (fault == "s6" and k == "fit") else "....\n18 passed in 1.0s\n"))
        os.utime(p, (now - 500, now - 500))
    t3 = t3_rows({k: lay_dump(J, d) for k, d in dumps.items()})
    argv = ["--eval-join", os.path.join(root, "join.npz"), "--box-dir", sub["box"], "--out-root", sub["fit_out"], "--list-record", os.path.join(sub["rec"], "list_record.json"),
            "--list-file", "train_list=" + os.path.join(sub["lists"], "train.txt"), "--list-file", "eval_list=" + os.path.join(sub["lists"], "eval.txt"),
            "--test-out", "fit=" + os.path.join(sub["tests"], "fit.out"), "--test-out", "read=" + os.path.join(sub["tests"], "read.out"),
            "--test-before", "fit=%f" % (now - 100), "--test-before", "read=%f" % (now - 100), "--n-boot", "300", "--expect-rows", str(int(t3.sum())), "--expect-clips", str(SYN_NE)]
    for k, p in paths.items():
        argv += ["--dump", "%s=%s" % (k, p)]
    ref = {k: round(mean_abs_err(lay_dump(J, dumps[k])["gap"], lay_dump(J, dumps[k])["lab_gap"], t3), 3) for k in ARM_D}
    argv += ["--ref-d", ",".join("%s=%s" % kv for kv in ref.items())]
    return argv, {"root": root, "sub": sub, "ref": ref}


def syn_payload(ks="cuts", n_boot=300):
    J, dumps, preds, boxes, tbox = syn_world(ks)
    tmp = tempfile.mkdtemp(prefix="boxsize_syn_", dir=os.getcwd())
    try:
        for name, d in list(boxes.items()) + list(tbox.items()):
            np.savez(os.path.join(tmp, name + ".npz"), **d)
        G, _ = build_map(tmp, sorted(tbox))
        L = {k: lay_dump(J, d) for k, d in dumps.items()}
        pl = {k: lay_pred(J, d) for k, d in preds.items()}
        return compute_values(J, L, pl, box_rows(J, tmp), G, n_boot)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def selftest():
    ok = True
    for ks, want in (("cuts", PRE + "-CUTS-GAP-ERROR"), ("worse", PRE + "-WORSE"), ("sep", PRE + "-NOT-SEPARATED")):
        P = syn_payload(ks)
        n1, n2, agree = derive_names(P)
        good = agree and n1["RATIO"] == want
        print("world S %s -> %s %s (%s)" % (ks, n1["RATIO"], n1["FAR"], "ok" if good else "WRONG"))
        ok &= good
    tmp = tempfile.mkdtemp(prefix="boxsize_selftest_", dir=os.getcwd())
    try:
        argv, _ = write_tree(os.path.join(tmp, "clean"))
        res = run_controls(parse_args(argv))
        print("clean tree: all_pass %s" % res["all_pass"])
        ok &= res["all_pass"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ZZ READ-BOXSIZE SELFTEST %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--eval-join", default=V8 + "/hierrep/pins/join_eval141.npz")
    ap.add_argument("--box-dir", default=V8 + "/cropread/enc_out/box")
    ap.add_argument("--out-root", default=V8 + "/boxsize/fit_out", help="<O> of the locator tool: pred/, fits/, IDENTITY_K5.json, BOXSIZE_IDENTITY_RECORD.json")
    ap.add_argument("--dump", action="append", default=[], help="NAME=PATH of one banked dump of arm D (D_0, D_1, D_2); defaults are the banked paths")
    ap.add_argument("--list-record", default="")
    ap.add_argument("--list-file", action="append", default=[], help="NAME=PATH; train_list and eval_list are required")
    ap.add_argument("--test-out", action="append", default=[], help="NAME=PATH")
    ap.add_argument("--test-before", action="append", default=[], help="NAME=EPOCH")
    ap.add_argument("--controls", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--ref-d", default=",".join("%s=%s" % kv for kv in REF_D.items()))
    ap.add_argument("--map-max-e", type=float, default=MAP_MAX_E)
    for k in EXPECT:
        ap.add_argument("--expect-" + k, type=int, default=EXPECT[k])
    return ap


def parse_args(argv):
    ap = build_parser()
    a = ap.parse_args(argv)
    a.expect = {k: getattr(a, "expect_" + k) for k in EXPECT}
    a.ref_d = {k: float(v) for k, _, v in (t.partition("=") for t in a.ref_d.split(",") if t)}
    a._ap = ap
    return a


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.stage or not a.out:
        a._ap.error("--stage and --out are required")
    if a.stage == "controls":
        if os.path.exists(a.out):
            print("REFUSED: %s exists" % a.out)
            print("ZZ READ-BOXSIZE controls rc 2")
            return 2
        res = run_controls(a)
        res["inputs"] = input_hashes(a)
        wt(a.out, json.dumps(res, indent=1, default=_jdefault))
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if (res["controls"].get(k) or {}).get("pass") else "FAIL"))
        for k in ("s3", "s4", "s5"):
            c = res["controls"].get(k) or {}
            if k == "s3":
                print("s3 values: E %s rows %s" % ({q: fnum(v) for q, v in (c.get("E") or {}).items()}, c.get("n_rows")))
            elif k == "s4":
                print("s4 values: E(G on label sizes) %s E(G on permuted label sizes) %s E(const) %s" % (fnum(c.get("E_label")), fnum(c.get("E_perm")), fnum(c.get("E_const"))))
            else:
                print("s5 values: E(SP_0) %s E(const) %s" % (fnum(c.get("E_sp")), fnum(c.get("E_const"))))
        print("ZZ READ-BOXSIZE controls all_pass %s" % res["all_pass"])
        return 0 if res["all_pass"] else 2
    return run_names(a)


if __name__ == "__main__":
    sys.exit(main())
