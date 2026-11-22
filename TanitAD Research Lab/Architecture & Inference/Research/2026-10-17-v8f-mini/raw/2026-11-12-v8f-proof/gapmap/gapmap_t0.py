"""gapmap_t0.py -- GAPMAP-T0 of card PREREG_V8F_PROOF_ADD112, sections 3 to 6.

Tier T0 (recorded evaluation frames; a diagnostic of the gap reader, not driving performance). No map here is admissible at inference.
Usage:
  gapmap_t0.py --kit-gap DIR --join FILE --base FILE --gap-record FILE --tests-out FILE --out JSON [--stage controls|names]
  gapmap_t0.py --selftest --kit-gap DIR
numpy and the standard library only; the gap tool of card ADD111 is imported unchanged from --kit-gap."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

G = None
KIT_FILE = "make_gaperr_gapfiles.py"
KIT_SHA_PREFIX = "08dbe57f2b6c4107"
N_FOLDS = 5
BOOT_DRAWS = 2000
BOOT_SEED = 20261122
BAR = 0.10
RANGES = ("near", "middle", "far", "all")
MAPS = ("identity", "IN", "CF", "LIN")
TOL_C0 = 1e-6
TOL_ISO = 1e-9
STAMP = "TIER T0 (recorded evaluation frames; a diagnostic of the gap reader, not driving performance); no map here is admissible at inference"
NAME_CUTS = "GAPMAP-T0-MAP-CUTS-SLOW-PART"
NAME_NOT = "GAPMAP-T0-MAP-DOES-NOT-CUT-SLOW-PART"
NAME_ND = "GAPMAP-T0-NOT-DECIDED"


def import_kit(kit_dir):
    global G
    kit_dir = os.path.abspath(kit_dir)
    sys.path.insert(0, kit_dir)
    sys.modules.pop("make_gaperr_gapfiles", None)
    import make_gaperr_gapfiles as mod
    G = mod
    return mod


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ===================================================================================== isotonic fit
def pava(x, y):
    """Isotonic (non-decreasing) least-squares fit of y on x. Ties in x are pooled first (one group per distinct x, weight = row count).
    -> bx (block mean of x over rows), by (block value), ux (distinct x), gfit (fitted value of each distinct x), cnt"""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) == 0:
        raise ValueError("no rows to fit")
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o]
    ux, st, cnt = np.unique(xs, return_index=True, return_counts=True)
    sy = np.add.reduceat(ys, st)
    sx = ux * cnt
    blocks = []
    for i in range(len(ux)):
        blocks.append([sy[i], sx[i], float(cnt[i]), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][2] > blocks[-1][0] / blocks[-1][2]:
            b = blocks.pop()
            for k in range(4):
                blocks[-1][k] += b[k]
    bx = np.array([b[1] / b[2] for b in blocks])
    by = np.array([b[0] / b[2] for b in blocks])
    gfit = np.repeat(by, [b[3] for b in blocks])
    return bx, by, ux, gfit, cnt


def maxmin_fit(means, w):
    """second routine: fit_i = max over j<=i of min over k>=i of the weighted mean of groups j..k"""
    means = np.asarray(means, np.float64)
    w = np.asarray(w, np.float64)
    n = len(means)
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cs = np.concatenate([[0.0], np.cumsum(w * means)])
    out = np.empty(n)
    for i in range(n):
        best = -np.inf
        for j in range(i + 1):
            m = min((cs[k + 1] - cs[j]) / (cw[k + 1] - cw[j]) for k in range(i, n))
            best = max(best, m)
        out[i] = best
    return out


def group_means(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ux, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return ux, np.bincount(inv, weights=y) / cnt, cnt.astype(np.float64)


def iso_apply(bx, by, xq):
    return np.interp(xq, bx, by)


def fit_line(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    b = float(((x - xm) * (y - ym)).sum() / sxx) if sxx > 0 else 0.0
    return float(ym - b * xm), b


# ===================================================================================== folds and maps
def make_folds(ci, names, err):
    """clips with at least one error row, sorted by name; fold = position modulo 5 -> (clips_err, fold_of {clip index: fold}, rowfold per row (-1 off error rows))"""
    clips_err = sorted({int(c) for c in np.unique(ci[err])}, key=lambda c: str(names[c]))
    fold_of = {c: i % N_FOLDS for i, c in enumerate(clips_err)}
    lut = np.full(len(names), -1, np.int64)
    for c, k in fold_of.items():
        lut[c] = k
    rowfold = np.where(err, lut[ci], -1)
    return clips_err, fold_of, rowfold


def cross_fit(ci, names, err, x, y, kind, fault=None):
    """cross-fitted map over clips. x (base gap) and y (label gap) full-length. -> mapped (full-length, NaN off error rows), diag"""
    clips_err, fold_of, rowfold = make_folds(ci, names, err)
    mapped = np.full(len(err), np.nan)
    count = np.zeros(len(err), np.int64)
    folds, leaks = [], 0
    for k in range(N_FOLDS):
        test = rowfold == k
        if not test.any():
            continue
        fit = (rowfold >= 0) & (rowfold != k)
        if fault == "leak":
            fit[np.flatnonzero(test)[0]] = True
        leak = int((rowfold[fit] == k).sum())
        leaks += leak
        if not fit.any():
            raise ValueError("a fold has no fit rows")
        if kind == "iso":
            bx, by, _, _, _ = pava(x[fit], y[fit])
            mapped[test] = iso_apply(bx, by, x[test])
        else:
            a, b = fit_line(x[fit], y[fit])
            mapped[test] = a + b * x[test]
        count[test] += 1
        folds.append({"fold": k, "n_clips": int(sum(1 for c in clips_err if fold_of[c] == k)), "n_test_rows": int(test.sum()),
                      "n_fit_rows": int(fit.sum()), "leak_rows": leak})
    diag = {"n_clips": len(clips_err), "n_fold_assignments": len(fold_of), "fold_values": sorted(set(fold_of.values())),
            "folds": folds, "leak_rows_total": leaks, "mapped_once_on_error_rows": bool((count[err] == 1).all()),
            "mapped_zero_off_error_rows": bool((count[~err] == 0).all())}
    return mapped, diag


# ===================================================================================== measures
def range_masks(g, err):
    with np.errstate(invalid="ignore"):
        return {"near": err & (g < G.NEAR_M), "middle": err & (g >= G.NEAR_M) & (g < G.MID_M), "far": err & (g >= G.MID_M), "all": err.copy()}


def analyse(ci, fr, names, lab, bh, g, bg, fault=None):
    err, e, s, f = G.error_parts(ci, fr, lab, bh, g, bg)
    if not err.any():
        raise ValueError("no error rows")
    masks = range_masks(g, err)
    x, y = bg, g
    bx, by, _, _, _ = pava(x[err], y[err])
    gm = {"IN": np.full(len(err), np.nan)}
    gm["IN"][err] = iso_apply(bx, by, x[err])
    gm["CF"], diag_cf = cross_fit(ci, names, err, x, y, "iso", fault)
    gm["LIN"], diag_lin = cross_fit(ci, names, err, x, y, "lin", fault)
    rows = {"identity": (e, s, f)}
    for m in ("IN", "CF", "LIN"):
        bg2 = bg.copy()
        bg2[err] = gm[m][err]
        err2, e2, s2, f2 = G.error_parts(ci, fr, lab, bh, g, bg2)
        if not np.array_equal(err, err2):
            raise ValueError("error rows changed by a map")
        rows[m] = (e2, s2, f2)
    blocks = {}
    for m in MAPS:
        e_, s_, f_ = rows[m]
        blocks[m] = {r: {"n_error_rows": int(mk.sum()), "rms_e": G.rms(e_[mk]), "rms_s": G.rms(s_[mk]), "rms_f": G.rms(f_[mk])} for r, mk in masks.items()}
    cuts = {}
    for m in ("IN", "CF", "LIN"):
        cuts[m] = {"cut_s": 1.0 - blocks[m]["all"]["rms_s"] / blocks["identity"]["all"]["rms_s"],
                   "cut_e": 1.0 - blocks[m]["all"]["rms_e"] / blocks["identity"]["all"]["rms_e"]}
    return {"err": err, "rows": rows, "blocks": blocks, "cuts": cuts, "diag": {"CF": diag_cf, "LIN": diag_lin}}


def bootstrap(ci, err, rows, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """clip bootstrap with the maps held fixed. A draw recomputes the rms over the rows of the drawn clips (a clip drawn twice counts twice)."""
    uc, inv = np.unique(ci[err], return_inverse=True)
    K = len(uc)
    nr = np.bincount(inv, minlength=K).astype(np.float64)
    ss = {}
    for m in MAPS:
        for qi, q in (("e", 0), ("s", 1)):
            v = rows[m][q][err]
            ss[(m, qi)] = np.bincount(inv, weights=v * v, minlength=K)
    rng = np.random.default_rng(seed)
    cnt = np.empty((draws, K))
    for d in range(draws):
        cnt[d] = np.bincount(rng.integers(0, K, size=K), minlength=K)
    n = cnt @ nr
    rm = {k: np.sqrt((cnt @ v) / n) for k, v in ss.items()}
    out = {}
    for m in ("IN", "CF", "LIN"):
        out[m] = {}
        for qi in ("s", "e"):
            c = 1.0 - rm[(m, qi)] / rm[("identity", qi)]
            lo, hi = np.percentile(c, [2.5, 97.5])
            out[m]["cut_" + qi] = [float(lo), float(hi)]
    return out


def name_of(cut_s_cf_interval, cut_s_in_interval):
    cuts = cut_s_cf_interval[0] >= BAR
    notc = cut_s_in_interval[1] < BAR
    if cuts and not notc:
        return NAME_CUTS
    if notc and not cuts:
        return NAME_NOT
    return NAME_ND


def values_of(C, fault=None):
    A = analyse(C["ci"], C["fr"], C["names"], C["lab"], C["base_has"], C["g"], C["base_gap"], fault)
    iv = bootstrap(C["ci"], A["err"], A["rows"])
    cuts = {m: {"cut_s": {"point": A["cuts"][m]["cut_s"], "lo": iv[m]["cut_s"][0], "hi": iv[m]["cut_s"][1]},
                "cut_e": {"point": A["cuts"][m]["cut_e"], "lo": iv[m]["cut_e"][0], "hi": iv[m]["cut_e"][1]}} for m in ("IN", "CF", "LIN")}
    cf = A["diag"]["CF"]
    return A, {"values": A["blocks"], "cuts": cuts, "fold_sizes": cf["folds"], "n_error_clips": cf["n_clips"],
               "bootstrap": {"draws": BOOT_DRAWS, "seed": BOOT_SEED, "percentiles": [2.5, 97.5], "maps_held_fixed": True}}


# ===================================================================================== controls
def planted_world(kind, seed=7, n_clips=30, n_fr=60, off_sd=2.5):
    """planted world: monotone distortion of the gap + noise, or a per-clip offset that does not depend on the gap
    (the offset world has clip means spread widely, so the support of the label gap has no pile-up at its edges that a map could clamp)"""
    rng = np.random.default_rng(seed)
    if kind != "monotone":
        n_clips = 40
    n = n_clips * n_fr
    ci = np.repeat(np.arange(n_clips, dtype=np.int64), n_fr)
    fr = np.tile(np.arange(n_fr, dtype=np.int64), n_clips)
    ph = rng.uniform(0.0, 6.28, n_clips)
    if kind == "monotone":
        g = 30.0 + 25.0 * np.sin(fr / 9.0 + ph[ci])
        bg = 10.0 + g + 0.02 * g ** 2 + rng.normal(0.0, 0.5, n)
    else:
        mu = rng.normal(30.0, 12.0, n_clips)
        g = mu[ci] + 6.0 * np.sin(fr / 9.0 + ph[ci])
        off = rng.normal(0.0, off_sd, n_clips)
        bg = g + off[ci] + rng.normal(0.0, 0.3, n)
    names = ["w%03d" % i for i in range(n_clips)]
    ones = np.ones(n, bool)
    return ci, fr, names, ones, ones.copy(), g, bg


def check_c0(A, record):
    fails = []
    t0 = record.get("t0")
    if not isinstance(t0, dict):
        return ["record has no t0"]
    for r in RANGES:
        mine, rec = A["blocks"]["identity"][r], t0.get(r)
        if rec is None:
            fails.append("%s: missing in record" % r)
            continue
        if mine["n_error_rows"] != rec["n_error_rows"]:
            fails.append("%s: n_error_rows differ" % r)
        for k in ("rms_e", "rms_s", "rms_f"):
            a, b = mine[k], rec[k]
            if (a is None) != (b is None) or (a is not None and abs(a - b) > TOL_C0):
                fails.append("%s: %s differs" % (r, k))
    return fails


def check_c1(fault=None):
    fails = []
    A = analyse(*planted_world("monotone"))
    for m in ("IN", "CF"):
        if not A["cuts"][m]["cut_e"] >= 0.5:
            fails.append("monotone world: cut_e(%s) below 0.5" % m)
    B = analyse(*planted_world("offset"))
    if not B["cuts"]["CF"]["cut_s"] < 0.05:
        fails.append("offset world: cut_s(CF) not below 0.05")
    return fails


def check_c2(n_series=300):
    fails = []
    rng = np.random.default_rng(11)
    for t in range(n_series):
        n = int(rng.integers(2, 61))
        if t % 2 == 0:
            x = rng.integers(0, 12, n).astype(np.float64)
        else:
            x = rng.normal(0.0, 5.0, n)
        y = 0.7 * x + rng.normal(0.0, 3.0, n)
        bx, by, ux, gfit, cnt = pava(x, y)
        if np.any(np.diff(by) < 0) or np.any(np.diff(gfit) < 0):
            fails.append("series %d: not non-decreasing" % t)
        u2, gm, w = group_means(x, y)
        mm = maxmin_fit(gm, w)
        if not np.array_equal(u2, ux) or np.abs(mm - gfit).max() > TOL_ISO:
            fails.append("series %d: differs from the max-min routine" % t)
        if len(fails) > 5:
            break
    return fails


def check_c3(A):
    fails = []
    for m in ("CF", "LIN"):
        d = A["diag"][m]
        if d["n_fold_assignments"] != d["n_clips"] or any(k < 0 or k >= N_FOLDS for k in d["fold_values"]):
            fails.append("%s: a clip is not in exactly one fold" % m)
        if sum(f["n_clips"] for f in d["folds"]) != d["n_clips"]:
            fails.append("%s: fold clip counts do not add up" % m)
        if d["leak_rows_total"] != 0 or any(f["leak_rows"] != 0 for f in d["folds"]):
            fails.append("%s: fit rows hold rows of a clip of the fold" % m)
        if not (d["mapped_once_on_error_rows"] and d["mapped_zero_off_error_rows"]):
            fails.append("%s: an error row is not mapped exactly once" % m)
    return fails


def check_c4(base_path, record, kit_dir, kit_prefix=KIT_SHA_PREFIX):
    fails = []
    want = (record.get("base") or {}).get("sha256")
    got = sha_file(base_path)
    if not want or want != got:
        fails.append("base file sha256 differs from the record")
    kp = os.path.join(kit_dir, KIT_FILE)
    if not os.path.exists(kp) or not sha_file(kp).startswith(kit_prefix):
        fails.append("gap tool sha256 does not begin %s" % kit_prefix)
    return fails


def check_c5(tests_out, t_start):
    if not os.path.exists(tests_out):
        return ["tests output file does not exist"]
    fails = []
    with open(tests_out, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    m = re.search(r"\b(\d+) passed\b", txt)
    if not m or int(m.group(1)) < 1:
        fails.append("no passed count of at least 1")
    if re.search(r"\b\d+ (failed|errors?)\b", txt):
        fails.append("a failed or error count is present")
    if os.path.getmtime(tests_out) > t_start:
        fails.append("tests output is newer than the start of this run")
    return fails


def run_controls(J, B, base_path, record_path, tests_out, kit_dir, t_start, kit_prefix=KIT_SHA_PREFIX, fault=None):
    """-> controls dict, payload (values)"""
    with open(record_path, encoding="utf-8") as fh:
        record = json.load(fh)
    C = G.compose(J, B)
    A, vals = values_of(C, fault)
    res = {"c0": check_c0(A, record), "c1": check_c1(), "c2": check_c2(), "c3": check_c3(A),
           "c4": check_c4(base_path, record, kit_dir, kit_prefix), "c5": check_c5(tests_out, t_start)}
    ctrl = {k: {"pass": not v, "failures": v} for k, v in res.items()}
    return ctrl, vals


# ===================================================================================== stages
def controls_stage(a, t_start):
    if os.path.exists(a.out):
        print("REFUSED: %s exists" % a.out)
        return 2
    J, B = G.load_npz(a.join), G.load_npz(a.base)
    ctrl, vals = run_controls(J, B, a.base, a.gap_record, a.tests_out, a.kit_gap, t_start)
    allp = all(v["pass"] for v in ctrl.values())
    doc = {"controls": ctrl, "all_pass": allp, "tier": STAMP}
    doc["payload" if allp else "withheld"] = vals
    d = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(d, exist_ok=True)
    with open(a.out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
    for k in sorted(ctrl):
        print("%s %s" % (k, "PASS" if ctrl[k]["pass"] else "FAIL"))
    print("ZZ GAPMAP-T0 controls all_pass %s" % allp)
    return 0 if allp else 1


def fmt(v):
    return "None" if v is None else "%.6f" % v


def names_stage(out):
    with open(out, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("all_pass") is not True or "payload" not in doc:
        print("REFUSED: the controls did not all pass; nothing is opened")
        return 2
    p = doc["payload"]
    print(STAMP)
    c = p["cuts"]
    print("NAME %s" % name_of((c["CF"]["cut_s"]["lo"], c["CF"]["cut_s"]["hi"]), (c["IN"]["cut_s"]["lo"], c["IN"]["cut_s"]["hi"])))
    for m in MAPS:
        for r in RANGES:
            v = p["values"][m][r]
            print("%s %s n_error_rows %d rms_e %s rms_s %s rms_f %s" % (m, r, v["n_error_rows"], fmt(v["rms_e"]), fmt(v["rms_s"]), fmt(v["rms_f"])))
    for m in ("IN", "CF", "LIN"):
        for q in ("cut_s", "cut_e"):
            x = c[m][q]
            print("%s(%s) %s interval [%s, %s]" % (q, m, fmt(x["point"]), fmt(x["lo"]), fmt(x["hi"])))
    print("error clips %d; bootstrap draws %d seed %d" % (p["n_error_clips"], p["bootstrap"]["draws"], p["bootstrap"]["seed"]))
    for f in p["fold_sizes"]:
        print("fold %d clips %d test rows %d fit rows %d" % (f["fold"], f["n_clips"], f["n_test_rows"], f["n_fit_rows"]))
    print("ZZ GAPMAP-T0 names done")
    return 0


def selftest(kit_dir):
    t0 = time.time()
    root = tempfile.mkdtemp(prefix="gapmap_selftest_")
    jp, bp, rp, op, lp, lop = G.write_world(os.path.join(root, "world"))
    G.build(jp, bp, rp, lp, os.path.join(root, "rec"))
    rec_path = os.path.join(root, "rec", G.RECORD_NAME)
    tp = os.path.join(root, "lead_tests.out")
    with open(tp, "w", encoding="utf-8") as fh:
        fh.write("3 passed in 0.1s\n")
    os.utime(tp, (t0 - 10, t0 - 10))
    J, B = G.load_npz(jp), G.load_npz(bp)
    bad = []
    ctrl, vals = run_controls(J, B, bp, rec_path, tp, kit_dir, t0)
    for k, v in ctrl.items():
        if not v["pass"]:
            bad.append("clean run: %s FAIL %s" % (k, v["failures"]))
    with open(rec_path, encoding="utf-8") as fh:
        rec = json.load(fh)
    rec["t0"]["all"]["rms_e"] += 1e-3
    shifted = os.path.join(root, "shifted_record.json")
    with open(shifted, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    ctrl2, _ = run_controls(J, B, bp, shifted, tp, kit_dir, t0)
    if ctrl2["c0"]["pass"]:
        bad.append("a shifted record value did not turn c0 red")
    ctrl3, _ = run_controls(J, B, bp, rec_path, tp, kit_dir, t0, fault="leak")
    if ctrl3["c3"]["pass"]:
        bad.append("a leaked clip did not turn c3 red")
    out = os.path.join(root, "read", "GAPMAP_T0.json")
    ns = argparse.Namespace(join=jp, base=bp, gap_record=rec_path, tests_out=tp, out=out, kit_gap=kit_dir)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = controls_stage(ns, t0)
        rc2 = names_stage(out) if rc == 0 else 1
    if rc != 0 or rc2 != 0 or "ZZ GAPMAP-T0 names done" not in buf.getvalue():
        bad.append("stages did not run clean: rc %s %s" % (rc, rc2))
    for b in bad:
        print("SELFTEST FAIL: " + b)
    print("ZZ GAPMAP-T0 SELFTEST " + ("OK" if not bad else "FAIL"))
    return 0 if not bad else 1


def main(argv=None):
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-gap", required=True)
    ap.add_argument("--join")
    ap.add_argument("--base")
    ap.add_argument("--gap-record")
    ap.add_argument("--tests-out")
    ap.add_argument("--out")
    ap.add_argument("--stage", choices=("controls", "names"), default="controls")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    import_kit(a.kit_gap)
    if a.selftest:
        return selftest(a.kit_gap)
    if a.stage == "names":
        if not a.out:
            ap.error("--out is required")
        return names_stage(a.out)
    for k in ("join", "base", "gap_record", "tests_out", "out"):
        if not getattr(a, k):
            ap.error("--%s is required" % k.replace("_", "-"))
    return controls_stage(a, t_start)


if __name__ == "__main__":
    sys.exit(main())
