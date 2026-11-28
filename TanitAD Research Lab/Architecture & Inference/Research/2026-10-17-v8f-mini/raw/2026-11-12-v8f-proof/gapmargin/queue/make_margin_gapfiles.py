#!/usr/bin/env python
"""make_margin_gapfiles.py -- card PREREG_V8F_PROOF_ADD117 (GAPMARGIN10K), control k0. Writes scaled copies of the base camera gap file of card ADD106 (VGAPLE.npz):
gap' = a * gap on every row with an output, every other array unchanged, not a number where the base has no output. CPU only, numpy only; no label is read.

  python make_margin_gapfiles.py --base VGAPLE.npz --out-dir DIR --scales 0.90,0.80      writes VGAPLE_A90.npz, VGAPLE_A80.npz and GAPMARGIN_GAPFILES_RECORD.json
  python make_margin_gapfiles.py --selftest DIR

The record holds pass_k0, the clauses per file, the hashes of the base file and of the scaled files, and per file the count of rows whose base gap lies above 60 m
and at or below 60/a m (a lead for the rule only with the margin). The output folder must not exist."""
import argparse
import datetime
import hashlib
import io
import json
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD117"
KEYS = ("clip_index", "frame", "gap", "has_out", "clips", "seed")
GAP_MAX = 60.0
TOL = 1e-9


def sha_file(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def tag(a):
    return "A%02d" % int(round(a * 100))


def scale(B, a, fault=None):
    """-> a new dict of arrays: gap scaled by a on the rows with an output"""
    out = {k: np.array(v, copy=True) for k, v in B.items()}
    has = np.asarray(B["has_out"], bool)
    g = np.asarray(B["gap"], np.float64).copy()
    g[has] = a * g[has]
    g[~has] = np.nan
    if fault == "shift":
        g[np.flatnonzero(has)[3]] += 0.01
    if fault == "nan_filled":
        k = np.flatnonzero(~has)
        if len(k):
            g[k[0]] = 1.0
    if fault == "other_field":
        out["frame"] = out["frame"].copy()
        out["frame"][0] += 1
    out["gap"] = g
    return out


def clauses_of(B, S, a):
    has = np.asarray(B["has_out"], bool)
    bg, sg = np.asarray(B["gap"], np.float64), np.asarray(S["gap"], np.float64)
    c = {"same_keys": sorted(S) == sorted(B)}
    c["only_gap_differs"] = c["same_keys"] and all(
        np.array_equal(np.asarray(S[k]), np.asarray(B[k])) and np.asarray(S[k]).dtype == np.asarray(B[k]).dtype for k in B if k != "gap")
    c["scaled_on_the_rows_with_an_output"] = bool(sg.shape == bg.shape and np.all(np.abs(sg[has] - a * bg[has]) <= TOL))
    c["not_a_number_where_no_output"] = bool(sg.shape == bg.shape and np.isnan(sg[~has]).all())
    loop = np.full(len(bg), np.nan)                                       # a second routine, row by row
    for i in range(len(bg)):
        if has[i]:
            loop[i] = a * float(bg[i])
    c["second_routine"] = bool(sg.shape == loop.shape and np.array_equal(sg, loop, equal_nan=True))
    return c


def build(base_path, out_dir, scales, fault=None):
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    B = load(base_path)
    miss = [k for k in KEYS if k not in B]
    if miss:
        raise SystemExit("REFUSED: the base file lacks %s" % miss)
    has = np.asarray(B["has_out"], bool)
    bg = np.asarray(B["gap"], np.float64)
    one = scale(B, 1.0)
    ident = all(np.array_equal(np.asarray(one[k]), np.asarray(B[k]), equal_nan=(k == "gap")) for k in B)
    os.makedirs(out_dir)
    rec = {"card": CARD, "base": {"path": base_path, "sha256": sha_file(base_path)}, "scale_one_gives_the_base_arrays": bool(ident), "files": {}, "clauses": {},
           "counts": {"n_rows": int(len(bg)), "n_rows_with_output": int(has.sum()), "n_base_gap_above_60": int((has & (bg > GAP_MAX)).sum())},
           "tol_m": TOL, "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    ok = bool(ident)
    for a in scales:
        nm = "VGAPLE_" + tag(a)
        S = scale(B, a, fault)
        p = os.path.join(out_dir, nm + ".npz")
        np.savez(p, **S)
        back = load(p)
        c = clauses_of(B, back, a)
        rec["clauses"][nm] = c
        ok = ok and all(c.values())
        rec["files"][nm] = {"file": nm + ".npz", "a": float(a), "sha256": sha_file(p),
                            "n_rows_base_gap_above_60_and_scaled_at_most_60": int((has & (bg > GAP_MAX) & (a * bg <= GAP_MAX)).sum())}
    rec["pass_k0"] = bool(ok)
    io.open(os.path.join(out_dir, "GAPMARGIN_GAPFILES_RECORD.json"), "w", encoding="utf-8", newline="\n").write(json.dumps(rec, indent=1))
    return rec


def synthetic_base(n=400, seed=3):
    rng = np.random.default_rng(seed)
    has = rng.random(n) > 0.1
    gap = np.where(has, rng.uniform(2.0, 75.0, n), np.nan)
    return {"clip_index": np.repeat(np.arange(n // 40), 40).astype(np.int64), "frame": np.tile(np.arange(40), n // 40).astype(np.int64), "gap": gap, "has_out": has,
            "clips": np.array(["clip%02d" % i for i in range(n // 40)]), "seed": np.int64(-1)}


def selftest(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    bp = os.path.join(base_dir, "base.npz")
    np.savez(bp, **synthetic_base())
    r = build(bp, os.path.join(base_dir, "clean"), (0.90, 0.80))
    ok = r["pass_k0"] is True and sorted(r["files"]) == ["VGAPLE_A80", "VGAPLE_A90"]
    for fault in ("shift", "nan_filled", "other_field"):
        rf = build(bp, os.path.join(base_dir, "f_" + fault), (0.90, 0.80), fault)
        ok = ok and rf["pass_k0"] is False
    print("ZZ MARGIN-GAPFILES SELFTEST %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--base", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--scales", default="0.90,0.80")
    ap.add_argument("--selftest", default="")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.selftest)
    if not a.base or not a.out_dir:
        print("REFUSED: --base and --out-dir are required")
        return 2
    scales = tuple(float(x) for x in a.scales.split(","))
    if not all(0.5 <= s <= 1.0 for s in scales):
        print("REFUSED: a scale outside 0.5 to 1.0")
        return 2
    rec = build(a.base, a.out_dir, scales)
    print("ZZ MARGIN-GAPFILES pass_k0 %s" % rec["pass_k0"])
    for nm in sorted(rec["clauses"]):
        for k, v in sorted(rec["clauses"][nm].items()):
            print("clause %s %s %s" % (nm, k, v))
    print("counts %s" % json.dumps(rec["counts"], sort_keys=True))
    for nm in sorted(rec["files"]):
        print("file %s rows with a base gap above 60 m and a scaled gap of at most 60 m: %d" % (nm, rec["files"][nm]["n_rows_base_gap_above_60_and_scaled_at_most_60"]))
    return 0 if rec["pass_k0"] else 1


if __name__ == "__main__":
    sys.exit(main())
