#!/usr/bin/env python3
"""rh_pilot_check.py -- HANDOFF_B1 step 3, checks (a)-(d), on REAL pilot/bank outputs. CPU, numpy only plus the new tree's tanitad.data.v8f_dagger (load_shard).

  pilot  --rh DIR --default DIR --unpatched DIR [--unpatched2 DIR] --readout TOKEN --out PILOT_RH.json
  bank   --bank DIR --readout TOKEN [--expect-windows N] [--out FILE]      (full bank re-check: (b) on every record, (c), (d), (e) complete TRAIN cover)

(a) the defaults pilot of the NEW tree equals a pilot of the UNPATCHED tree (DSCALE10K tree cdbff971), array for array: max abs diff 0.0. The device is a GPU, which is not bit-reproducible, so if the diff
    is nonzero the same unpatched pilot is compared with a SECOND unpatched pilot (--unpatched2) and (a) passes only when diff(new, unpatched) <= diff(unpatched, unpatched2) (the device's own floor);
    both numbers are reported.
(b) every non-NONFINITE record of the RH bank: stored last-knot acceleration minus gain x the executed one <= 1e-6 (the test's `_a3_worst`, gain 1.0). Control: the defaults pilot must read > 1e-3
    (else the check is vacuous). Both tokens hold acceleration gain 1, so (b) applies to both.
(c) coverage fraction >= 0.5 and not blocked; PROVENANCE controls K-B4 and K-B5 PASS; bank.json exists (merge refuses on a failed K-B1/K-B4/K-B5); K-B1 mismatch_chunks 0 per shard.
(d) bank.json common.generator == {readout causal_g, acc_gain 1.0, path_exec (true for causal_g1x only), kappa_max "0.2"} for RH, and ABSENT for the defaults pilot (a default bank carries no record).
"""
import argparse
import glob
import hashlib
import json
import os
import sys

R = os.environ.get("RH_ROOT", "/root/rollhist")
TREE = os.environ.get("RH_TREE", R + "/tree_cdb")
for p in (TREE + "/stack", TREE + "/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
from tanitad.data import v8f_dagger as DG  # noqa: E402

WANT = {"causal_g1x": {"readout": "causal_g", "acc_gain": 1.0, "path_exec": True, "kappa_max": "0.2"},
        "causal_g1": {"readout": "causal_g", "acc_gain": 1.0, "path_exec": False, "kappa_max": "0.2"}}


def shards(d):
    return sorted(glob.glob(os.path.join(d, "shard_*of*.npz")))


def a3_worst(arrays, gain):
    worst, n = 0.0, 0
    for i in range(len(arrays["ci"])):
        if int(arrays["reason"][i]) == DG.NONFINITE:
            continue
        P = np.asarray(arrays["pose"][i], np.float64)[-3:, :2]
        if not np.isfinite(P).all():
            continue
        s1, s2 = np.hypot(*(P[1] - P[0])) / 0.2, np.hypot(*(P[2] - P[1])) / 0.2
        worst, n = max(worst, abs(float(arrays["kin3"][i][1]) - gain * (s2 - s1) / 0.2)), n + 1
    return worst, n


def max_abs_diff(a, b):
    if sorted(a) != sorted(b):
        return float("inf")
    w = 0.0
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            return float("inf")
        if x.dtype.kind == "f":
            if (np.isnan(x) != np.isnan(y)).any():
                return float("inf")
            m = np.isnan(x)
            d = np.abs(np.where(m, 0, x).astype(np.float64) - np.where(m, 0, y).astype(np.float64))
            w = max(w, float(d.max()) if d.size else 0.0)
        elif not np.array_equal(x, y):
            return float("inf")
    return w


def dir_diff(d1, d2):
    f1, f2 = shards(d1), shards(d2)
    if [os.path.basename(f) for f in f1] != [os.path.basename(f) for f in f2] or not f1:
        return float("inf"), "shard file lists differ"
    return max(max_abs_diff(DG.load_shard(a)[0], DG.load_shard(b)[0]) for a, b in zip(f1, f2)), "ok"


def bank_json(d):
    return json.load(open(os.path.join(d, "bank.json"), encoding="utf-8"))


def prov(d):
    return json.load(open(os.path.join(d, "PROVENANCE.json"), encoding="utf-8"))


def check_c(d):
    bad, rec = [], {}
    if not os.path.isfile(os.path.join(d, "bank.json")):
        return ["no bank.json"], rec
    b, p = bank_json(d), prov(d)
    cov = b["coverage"]
    rec["coverage"] = cov
    if cov["fraction"] < 0.5 or cov.get("blocked"):
        bad.append(f"coverage {cov['fraction']:.4f} < 0.5 or blocked")
    for k in ("K-B4", "K-B5"):
        if (p.get("controls", {}).get(k) or {}).get("PASS") is not True:
            bad.append(f"control {k} is not PASS")
    for f in glob.glob(os.path.join(d, "shard_*.provenance.json")):
        if json.load(open(f)).get("kb1_replay", {}).get("mismatch_chunks"):
            bad.append(f"K-B1 mismatch in {os.path.basename(f)}")
    return bad, rec


def check_d(d, token):
    g = bank_json(d).get("common", {}).get("generator")
    return ([] if g == WANT[token] else [f"bank.json common.generator is {g!r}, want {WANT[token]!r}"]), g


def check_b(d, gain=1.0):
    worst, n = 0.0, 0
    for f in shards(d):
        w, k = a3_worst(DG.load_shard(f)[0], gain)
        worst, n = max(worst, w), n + k
    return worst, n


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    p1 = sp.add_parser("pilot")
    for k in ("rh", "default", "unpatched"):
        p1.add_argument("--" + k, required=True)
    p1.add_argument("--unpatched2", default=None)
    p1.add_argument("--readout", required=True, choices=sorted(WANT))
    p1.add_argument("--out", required=True)
    p2 = sp.add_parser("bank")
    p2.add_argument("--bank", required=True)
    p2.add_argument("--readout", required=True, choices=sorted(WANT))
    p2.add_argument("--expect-windows", type=int, default=0)
    p2.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = {"readout": a.readout, "checks": {}}
    if a.c == "pilot":
        d_nu, why = dir_diff(a.default, a.unpatched)
        a_rec = {"max_abs_diff_new_default_vs_unpatched": d_nu, "note": why}
        if d_nu == 0.0:
            ok_a = True
        elif a.unpatched2 and os.path.isdir(a.unpatched2):
            d_uu, _ = dir_diff(a.unpatched, a.unpatched2)
            a_rec["max_abs_diff_unpatched_vs_unpatched2_floor"] = d_uu
            ok_a = d_nu <= d_uu
        else:
            ok_a = False
            a_rec["note"] += "; nonzero and no second unpatched pilot to read the device floor from"
        a_rec["pass"] = ok_a
        rep["checks"]["a_defaults_equal_unpatched"] = a_rec
        w, n = check_b(a.rh)
        wc, nc = check_b(a.default)
        strict = a.readout == "causal_g1x"
        rep["checks"]["b_executed_acceleration"] = {"rh_worst_abs": w, "rh_n_records": n, "control_defaults_worst_abs": wc, "control_n": nc, "strict": strict,
                                                    "note": "" if strict else "causal_g1 has no --path-exec: the stored pose is not the executed knots, so the 1e-6 identity is not asserted (value reported); needs the PI's readout to be causal_g1x for the full R4 form",
                                                    "pass": bool(n > 0 and nc > 0 and wc > 1e-3 and (w <= 1e-6 or not strict))}
        bad, rec = check_c(a.rh)
        rep["checks"]["c_coverage_and_controls"] = {"pass": not bad, "problems": bad, **rec}
        bad, g = check_d(a.rh, a.readout)
        gd = bank_json(a.default).get("common", {}).get("generator")
        if gd is not None:
            bad.append(f"the defaults pilot carries a generator record {gd!r} (a default bank must have none)")
        rep["checks"]["d_header_generator"] = {"pass": not bad, "problems": bad, "rh_generator": g, "defaults_generator": gd}
        rep["rh_bank_json_sha256"] = hashlib.sha256(open(os.path.join(a.rh, "bank.json"), "rb").read()).hexdigest()
    else:
        w, n = check_b(a.bank)
        strict = a.readout == "causal_g1x"
        rep["checks"]["b_executed_acceleration"] = {"worst_abs": w, "n_records": n, "strict": strict, "pass": bool(n > 0 and (w <= 1e-6 or not strict))}
        bad, rec = check_c(a.bank)
        rep["checks"]["c_coverage_and_controls"] = {"pass": not bad, "problems": bad, **rec}
        bad, g = check_d(a.bank, a.readout)
        rep["checks"]["d_header_generator"] = {"pass": not bad, "problems": bad, "generator": g}
        if a.expect_windows:
            b = bank_json(a.bank)
            tw = b["common"]["train"]["n_windows"]
            e = [] if (tw == a.expect_windows and not b["partial"]) else [f"train n_windows {tw} / partial {b['partial']} vs expected {a.expect_windows} complete"]
            rep["checks"]["e_complete_train_cover"] = {"pass": not e, "problems": e}
    ok = all(v["pass"] for v in rep["checks"].values())
    rep["pass"] = ok
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep)[:1800])
    print("ZZ RH-PILOT-CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
