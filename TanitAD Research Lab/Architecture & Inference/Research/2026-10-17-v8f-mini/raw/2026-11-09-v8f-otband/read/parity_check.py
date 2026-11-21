"""parity_check.py -- control P0 of the OTBAND read: the commit-of-record tree, run on the DAG-OACC s0 checkpoint (a pre-change
checkpoint), must reproduce the banked baseline dump.  Rule, written before the run (README_READ.md):
  PASS-EXACT  every array common to both dumps (per episode file and the ticks file) is identical;
  PASS        identifiers and model-free arrays (eid, clip_index, ws, g, fl, v0) are identical, and every other float array differs by at
              most 1e-4 (metres for plans) -- thread-count float noise, the read stays valid and the gap is reported;
  FAIL        anything else -- the read is INVALID until explained.
Writes PARITY_P0.json next to this file; exit code 0 on PASS-EXACT or PASS, 1 on FAIL.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = "F:/TanitAD-scratch/otband_read/t1"
OLD = "E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1"
EXACT_KEYS = ("eid", "clip_index", "ws", "g", "fl", "v0")
TOL = 1e-4


def cmp_npz(a, b, out, where):
    za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    ka, kb = set(za.files), set(zb.files)
    out["only_new"] |= ka - kb
    out["only_old"] |= kb - ka
    for k in sorted(ka & kb):
        x, y = za[k], zb[k]
        r = out["keys"].setdefault(k, {"n_arrays": 0, "n_identical": 0, "max_abs": 0.0, "shape_mismatch": 0})
        r["n_arrays"] += 1
        if x.shape != y.shape:
            r["shape_mismatch"] += 1
            continue
        if np.array_equal(x, y, equal_nan=(x.dtype.kind == "f")):
            r["n_identical"] += 1
            continue
        if x.dtype.kind in "fc" and y.dtype.kind in "fc":
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            d = d[np.isfinite(d)]
            r["max_abs"] = max(r["max_abs"], float(d.max()) if d.size else 0.0)
        else:
            r["max_abs"] = float("inf")
        r.setdefault("first_diff_at", where)


META = ("tag",)   # the ticks file's run name: differs by construction between two runs; listed, never part of the rule (amendment 1)


def main():
    # default = P0 (new tree vs the banked baseline).  P0b: parity_check.py P0b  (base tree vs new tree, both run here at 2 threads)
    which = sys.argv[1] if len(sys.argv) > 1 else "P0"
    if which == "P0":
        da, ta = f"{NEW}/dump_PARITY_DAGOACC_s0_m10_causal_heads", f"{NEW}/ticks/ticks_PARITY_DAGOACC_s0_m10_causal_heads.npz"
        db, tb = f"{OLD}/dump_DAGOACC_s0_m10_causal_heads", f"{OLD}/ticks/ticks_DAGOACC_s0_m10_causal_heads.npz"
        meta = ()
    else:
        da, ta = f"{NEW}/dump_PARITY_DAGOACC_s0_m10_causal_heads", f"{NEW}/ticks/ticks_PARITY_DAGOACC_s0_m10_causal_heads.npz"
        db, tb = f"{NEW}/dump_PARITY_BASETREE_DAGOACC_s0_m10_causal_heads", f"{NEW}/ticks/ticks_PARITY_BASETREE_DAGOACC_s0_m10_causal_heads.npz"
        meta = META
    out = {"keys": {}, "only_new": set(), "only_old": set()}
    fa = sorted(glob.glob(f"{da}/ep*.npz"))
    fb = sorted(glob.glob(f"{db}/ep*.npz"))
    assert len(fa) == len(fb) == 141, (len(fa), len(fb))
    assert [os.path.basename(x) for x in fa] == [os.path.basename(x) for x in fb]
    for a, b in zip(fa, fb):
        cmp_npz(a, b, out, os.path.basename(a))
    tk = {"keys": {}, "only_new": set(), "only_old": set()}
    cmp_npz(ta, tb, tk, "ticks")
    meta_rows = {k: tk["keys"].pop(k) for k in meta if k in tk["keys"]}
    allk = [("dump", k, r) for k, r in out["keys"].items()] + [("ticks", k, r) for k, r in tk["keys"].items()]
    exact = all(r["n_identical"] == r["n_arrays"] for _, _, r in allk) and not (out["only_new"] or out["only_old"] or tk["only_new"] or tk["only_old"])
    ids_ok = all(out["keys"].get(k, {}).get("n_identical", -1) == out["keys"].get(k, {}).get("n_arrays", -2) for k in EXACT_KEYS)
    near = all(r["shape_mismatch"] == 0 and r["max_abs"] <= TOL for _, _, r in allk)
    label = "PASS-EXACT" if exact else ("PASS" if ids_ok and near else "FAIL")
    rep = {"label": label, "which": which, "a": [da, ta], "b": [db, tb], "rule": __doc__.split("Writes")[0].strip(), "n_episode_files": len(fa),
           "metadata_rows_outside_the_rule": meta_rows,
           "dump": {"keys": out["keys"], "only_new": sorted(out["only_new"]), "only_old": sorted(out["only_old"])},
           "ticks": {"keys": tk["keys"], "only_new": sorted(tk["only_new"]), "only_old": sorted(tk["only_old"])}}
    json.dump(rep, open(os.path.join(HERE, f"PARITY_{which}.json"), "w"), indent=1)
    worst = sorted(((r["max_abs"], w, k) for w, k, r in allk if r["n_identical"] != r["n_arrays"]), reverse=True)[:6]
    print(f"PARITY_{which}", label, "non-identical (max_abs, where, key):", worst)
    sys.exit(0 if label != "FAIL" else 1)


if __name__ == "__main__":
    main()
