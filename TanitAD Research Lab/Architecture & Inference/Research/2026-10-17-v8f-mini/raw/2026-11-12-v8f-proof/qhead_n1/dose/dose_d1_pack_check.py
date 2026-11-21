#!/usr/bin/env python
"""dose_d1_pack_check.py -- control d1 of note PREREG_V8F_PROOF_ADD42_N2 section 3: the label pack built for all training clips equals the banked pack of the probe clips row for row
on manifest positions lo .. hi-1 (default 600 .. 1799). Largest absolute difference 0; any missing or extra row fails.

  python dose_d1_pack_check.py --new-pack train_pack_all.npz --banked-pack train_pack.npz [--lo 600] [--hi 1800] [--out d1.json]

Rows are matched by KEY (clip index, frame), so a row present on one side only is counted (missing = banked rows the new pack lacks, extra = new rows in the range the banked pack lacks).
Compared on matched rows: clip names (new clips[lo:hi] against banked clips), gap (NaN pattern equal, largest absolute difference of finite entries), pres, v (same rule), and the join sha256
of the two meta records. JSON on stdout (and --out). Exit 0 pass, 2 fail or refused. Tier T0 input control.
"""
import argparse
import json
import sys

import numpy as np


def _meta(z):
    try:
        return json.loads(str(z["meta"]))
    except Exception:
        return {}


def _maxdiff(a, b):
    """-> (nan_pattern_equal, largest absolute difference over entries finite on both sides; 0.0 when none)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    both = np.isfinite(a) & np.isfinite(b)
    d = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    inf_ok = bool(np.array_equal(np.where(np.isinf(a), a, 0.0), np.where(np.isinf(b), b, 0.0)))
    return same_nan and inf_ok, d


def compare(new, banked, lo, hi, meta_new=None, meta_banked=None):
    """new, banked: dicts clips, clip_idx, t, gap, pres, v. Returns the result dict (key 'pass')."""
    R = {"lo": int(lo), "hi": int(hi)}
    nb = len(np.asarray(banked["clips"]))
    if hi - lo != nb:
        R.update({"pass": False, "why": "the banked pack has %d clips, the range lo..hi has %d" % (nb, hi - lo)})
        return R
    ci_n = np.asarray(new["clip_idx"]).astype(np.int64)
    sel = (ci_n >= lo) & (ci_n < hi)
    rows_n = np.flatnonzero(sel)
    k_new = (ci_n[rows_n] - lo) * 100000 + np.asarray(new["t"]).astype(np.int64)[rows_n]
    k_ban = np.asarray(banked["clip_idx"]).astype(np.int64) * 100000 + np.asarray(banked["t"]).astype(np.int64)
    R["rows_new_in_range"], R["rows_banked"] = int(len(rows_n)), int(len(k_ban))
    miss = np.setdiff1d(k_ban, k_new)
    extra = np.setdiff1d(k_new, k_ban)
    dup = bool(len(np.unique(k_new)) != len(k_new) or len(np.unique(k_ban)) != len(k_ban))
    R["missing_rows"], R["extra_rows"], R["duplicate_keys"] = int(len(miss)), int(len(extra)), dup
    names_ok = bool(np.array_equal(np.asarray(new["clips"]).astype(str)[lo:hi], np.asarray(banked["clips"]).astype(str)))
    R["clip_names_equal"] = names_ok
    ok = names_ok and len(miss) == 0 and len(extra) == 0 and not dup
    if ok:
        on = rows_n[np.argsort(k_new, kind="stable")]
        ob = np.argsort(k_ban, kind="stable")
        for key in ("gap", "pres", "v"):
            nan_eq, d = _maxdiff(np.asarray(new[key])[on], np.asarray(banked[key])[ob])
            R[key + "_nan_pattern_equal"], R[key + "_max_abs_diff"] = nan_eq, d
            ok = ok and nan_eq and d == 0.0
    else:
        R["note"] = "values not compared: the row sets or names differ"
    jn, jb = (meta_new or {}).get("join_sha256"), (meta_banked or {}).get("join_sha256")
    R["join_sha256_new"], R["join_sha256_banked"] = jn, jb
    R["join_sha256_equal"] = bool(jn is not None and jn == jb)
    ok = ok and R["join_sha256_equal"]
    R["pass"] = bool(ok)
    return R


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--new-pack", required=True)
    ap.add_argument("--banked-pack", required=True)
    ap.add_argument("--lo", type=int, default=600)
    ap.add_argument("--hi", type=int, default=1800)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    try:
        zn = np.load(a.new_pack, allow_pickle=False)
        zb = np.load(a.banked_pack, allow_pickle=False)
        keys = ("clips", "clip_idx", "t", "gap", "pres", "v")
        new = {k: zn[k] for k in keys}
        banked = {k: zb[k] for k in keys}
        R = compare(new, banked, a.lo, a.hi, _meta(zn), _meta(zb))
    except (OSError, KeyError, ValueError) as e:
        R = {"pass": False, "why": "refused: %s: %s" % (type(e).__name__, e)}
    R["control"] = "d1 (note ADD42_N2 section 3): new pack against the banked pack on manifest positions %d..%d" % (a.lo, a.hi - 1)
    s = json.dumps(R, indent=1)
    print(s)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(s + "\n")
    return 0 if R["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
