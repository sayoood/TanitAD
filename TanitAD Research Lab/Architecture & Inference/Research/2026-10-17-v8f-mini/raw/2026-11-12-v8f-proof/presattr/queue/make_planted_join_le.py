#!/usr/bin/env python
"""make_planted_join_le.py -- card PREREG_V8F_PROOF_ADD114 (PRESATTR10K), control e0. CPU only.

  python make_planted_join_le.py --kit-gap DIR --join FILE --base FILE --out-dir DIR
  python make_planted_join_le.py --selftest --kit-gap DIR

Writes <out-dir>/JLE.npz: the evaluation join with `lead_gap` replaced by the base gap (the mean gap of the three large fits, VGAPLE.npz) on every row where the label gap is
valid and the base file has a finite output; the label gap stays on the label rows with no base output; every other array is written as it was read. The row layout (join rows
of the frames, label validity, base gap laid on the frames) comes from the gap tool of card ADD111 (make_gaperr_gapfiles.py, imported unchanged from --kit-gap). Beside it
JLE_RECORD.json with the clauses of control e0, the counts and the hashes. Prints `ZZ PLANTED-JOIN-LE pass_e0 <True|False>` and one line per clause; no gap value is printed.
A label input in a join is a tool of attribution: a run that reads this join is never a candidate.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import time

import numpy as np

TOL = 1e-4
KIT_SHA16 = "08dbe57f2b6c4107"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_kit(kit_gap):
    p = os.path.join(kit_gap, "make_gaperr_gapfiles.py")
    spec = importlib.util.spec_from_file_location("make_gaperr_gapfiles_for_jle", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p


def plant(G, J, B, fault=None):
    """-> (new lead_gap array of the join, mix mask over the file rows, idx, base gap on the file rows, label mask)"""
    names, n, cum, idx, ci, fr = G.join_frames(J)
    g, lab = G.label_arrays(J, idx)
    bg, bh = G.lay_base(names, n, cum, B)
    mix = lab & bh & np.isfinite(bg)
    if len(np.unique(idx)) != len(idx):
        raise SystemExit("REFUSED: a join row is the row of two frames")
    new = np.array(J["lead_gap"], copy=True)
    val = bg[mix]
    if fault == "scale":
        val = 0.9 * val
    if fault == "shift":
        val = np.roll(val, 1)
    new[idx[mix]] = val.astype(new.dtype)
    if fault == "outside" and (~mix).any():
        k = int(np.flatnonzero(~mix)[0])
        new[idx[k]] = (0.0 if not np.isfinite(new[idx[k]]) else float(new[idx[k]])) + 1.0
    return new, mix, idx, bg, lab


def second_routine(J, B, gap_max):
    """a plain loop, clip by clip and frame by frame, from the two files alone -> expected lead_gap array of the join"""
    d = {}
    bn = [str(c) for c in B["clips"]]
    for c, f, gp, h in zip(B["clip_index"], B["frame"], B["gap"], B["has_out"]):
        if bool(h) and math.isfinite(float(gp)):
            d[(bn[int(c)], int(f))] = float(gp)
    exp = np.array(J["lead_gap"], dtype=np.float64, copy=True)
    nrep = 0
    for i, c in enumerate(J["clip"]):
        for f in range(int(J["n"][i])):
            r = int(J["off"][i]) + f
            g0 = float(J["lead_gap"][r])
            if int(J["lead_valid"][r]) == 1 and math.isfinite(g0) and 0.0 <= g0 <= gap_max and (str(c), f) in d:
                exp[r] = d[(str(c), f)]
                nrep += 1
    return exp, nrep


def same(a, b, tol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    m = ~np.isnan(a)
    return bool(np.all(np.abs(a[m] - b[m]) <= tol))


def clauses_of(G, J, B, saved, mix, idx, bg):
    cl, fails = {}, []
    other = [k for k in J if k != "lead_gap"]
    cl["only_lead_gap_differs"] = sorted(saved.keys()) == sorted(J.keys()) and all(
        saved[k].dtype == J[k].dtype and saved[k].shape == J[k].shape and saved[k].tobytes() == J[k].tobytes() for k in other) and saved["lead_gap"].dtype == J["lead_gap"].dtype
    rows_mix = np.zeros(len(J["lead_gap"]), bool)
    rows_mix[idx[mix]] = True
    a, b = np.asarray(saved["lead_gap"], np.float64), np.asarray(J["lead_gap"], np.float64)
    cl["unchanged_off_the_named_rows"] = bool(np.array_equal(np.isnan(a[~rows_mix]), np.isnan(b[~rows_mix])) and np.array_equal(a[~rows_mix][~np.isnan(a[~rows_mix])], b[~rows_mix][~np.isnan(b[~rows_mix])]))
    cl["equals_base_on_the_named_rows"] = same(a[idx[mix]], bg[mix], TOL)
    exp, nrep = second_routine(J, B, float(G.GAP_MAX_M))
    cl["second_routine"] = same(a, exp, TOL) and nrep == int(mix.sum())
    for k, v in cl.items():
        if not v:
            fails.append(k)
    return cl, fails


def build(G, kit_path, join_path, base_path, out_dir, fault=None):
    J, B = G.load_npz(join_path), G.load_npz(base_path)
    new, mix, idx, bg, lab = plant(G, J, B, fault)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "JLE.npz")
    if os.path.exists(out) or os.path.exists(os.path.join(out_dir, "JLE_RECORD.json")):
        raise SystemExit("REFUSED: output exists in " + out_dir)
    arrs = {k: (new if k == "lead_gap" else J[k]) for k in J}
    if fault == "other_field":
        k2 = [k for k in J if k not in ("lead_gap", "clip", "n", "off")][0]
        x = np.array(J[k2], copy=True)
        x.flat[0] = x.flat[0] + 1
        arrs[k2] = x
    np.savez(out, **arrs)
    saved = G.load_npz(out)
    cl, fails = clauses_of(G, J, B, saved, mix, idx, bg)
    gmax = float(G.GAP_MAX_M)
    with np.errstate(invalid="ignore"):
        counts = {"n_join_rows": int(len(J["lead_gap"])), "n_file_rows": int(len(idx)), "n_label_rows": int(lab.sum()), "n_replaced_rows": int(mix.sum()),
                  "n_label_rows_kept": int((lab & ~mix).sum()), "n_replaced_rows_above_gap_max": int((bg[mix] > gmax).sum()),
                  "n_replaced_rows_below_zero": int((bg[mix] < 0.0).sum()), "gap_max_m": gmax}
    rec = {"card": "PREREG_V8F_PROOF_ADD114", "pass_e0": not fails, "clauses": cl, "failures": fails, "counts": counts,
           "join": {"path": out, "sha256": sha_file(out)}, "base": {"path": base_path, "sha256": sha_file(base_path)}, "eval_join_sha256": sha_file(join_path),
           "gap_tool_sha256": sha_file(kit_path), "tol_m": TOL, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with io.open(os.path.join(out_dir, "JLE_RECORD.json"), "x", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return rec


def selftest(G, kit_path, tmp_root):
    os.makedirs(tmp_root, exist_ok=True)
    J = G.synthetic_join()
    B = G.synthetic_base(J)
    jp, bp = os.path.join(tmp_root, "join.npz"), os.path.join(tmp_root, "base.npz")
    if not os.path.exists(jp):
        np.savez(jp, **J)
        np.savez(bp, **B)
    res = {}
    for fault in (None, "scale", "shift", "outside", "other_field"):
        od = os.path.join(tmp_root, "out_%s" % (fault or "clean"))
        res[fault or "clean"] = build(G, kit_path, jp, bp, od, fault)
    ok = res["clean"]["pass_e0"] and res["clean"]["counts"]["n_replaced_rows"] > 0 and all(not res[k]["pass_e0"] for k in ("scale", "shift", "outside", "other_field"))
    return ok, res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-gap", required=True)
    ap.add_argument("--join", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tmp", default="")
    a = ap.parse_args(argv)
    G, kit_path = load_kit(a.kit_gap)
    if a.selftest:
        import tempfile
        ok, _ = selftest(G, kit_path, a.tmp or tempfile.mkdtemp(prefix="jle_selftest_"))
        print("ZZ PLANTED-JOIN-LE SELFTEST " + ("OK" if ok else "FAILED"))
        return 0 if ok else 1
    if not sha_file(kit_path).startswith(KIT_SHA16):
        raise SystemExit("REFUSED: the gap tool has the sha256 prefix %s, expected %s" % (sha_file(kit_path)[:16], KIT_SHA16))
    rec = build(G, kit_path, a.join, a.base, a.out_dir)
    print("ZZ PLANTED-JOIN-LE pass_e0 %s" % rec["pass_e0"])
    for k in sorted(rec["clauses"]):
        print("clause %s %s" % (k, rec["clauses"][k]))
    print("counts " + json.dumps(rec["counts"], sort_keys=True))
    return 0 if rec["pass_e0"] else 1


if __name__ == "__main__":
    sys.exit(main())
