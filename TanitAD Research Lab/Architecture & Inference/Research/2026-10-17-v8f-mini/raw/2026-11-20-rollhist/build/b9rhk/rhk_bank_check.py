#!/usr/bin/env python3
"""rhk_bank_check.py --bank DIR --expect-windows N --out BANK_CHECK.json   -- the full-bank re-check (b, c, d, e) of rh_pilot_check.py `bank`, for readout K (token causal_g0x). Reuses that file's functions UNCHANGED (imported);
only the readout table and the acceleration gain differ. CPU, numpy + the tree's tanitad.data.v8f_dagger.load_shard.
(b) every non-NONFINITE record: stored last-knot acceleration minus 0.0 x the executed one <= 1e-6 (gain 0: K hands the planner NO own acceleration), AND the vacuity control: the executed one-frame acceleration
    (s2 - s1)/0.2 computed from the stored pose reaches > 1e-3 somewhere (a bank whose stored accelerations were all zero because the motion was, would pass b on nothing).
(c) coverage fraction >= 0.5 and not blocked (else BLOCKED), K-B4 / K-B5 PASS, K-B1 mismatch_chunks 0;  (d) common.generator == {causal_g, acc_gain 0.0, path_exec true, kappa_max "0.2"};  (e) complete TRAIN cover.
Last line `ZZ RH-PILOT-CHECK PASS|FAIL` (same string as the RH bank check)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rh_pilot_check as C  # noqa: E402   (the pod file, unchanged; it puts the tree on sys.path and imports tanitad.data.v8f_dagger as C.DG)

TOKEN = "causal_g0x"
WANT_K = {"readout": "causal_g", "acc_gain": 0.0, "path_exec": True, "kappa_max": "0.2"}
C.WANT[TOKEN] = WANT_K                                    # this process's copy of the module table; the pod file on disk is not touched


def exec_accel_max(arrays):
    np_, DG = C.np, C.DG
    worst = 0.0
    for i in range(len(arrays["ci"])):
        if int(arrays["reason"][i]) == DG.NONFINITE:
            continue
        P = np_.asarray(arrays["pose"][i], np_.float64)[-3:, :2]
        if not np_.isfinite(P).all():
            continue
        s1, s2 = np_.hypot(*(P[1] - P[0])) / 0.2, np_.hypot(*(P[2] - P[1])) / 0.2
        worst = max(worst, abs((s2 - s1) / 0.2))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--expect-windows", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = {"readout": TOKEN, "checks": {}}
    w, n = C.check_b(a.bank, 0.0)
    ex = max((exec_accel_max(C.DG.load_shard(f)[0]) for f in C.shards(a.bank)), default=0.0)
    rep["checks"]["b_no_own_acceleration"] = {"worst_abs": w, "n_records": n, "executed_accel_max": ex, "pass": bool(n > 0 and w <= 1e-6 and ex > 1e-3),
                                              "note": "stored last-knot acceleration == 0.0 x executed; the executed acceleration is non-trivially nonzero in the stored poses (vacuity control)"}
    bad, rec = C.check_c(a.bank)
    rep["checks"]["c_coverage_and_controls"] = {"pass": not bad, "problems": bad, **rec}
    bad, g = C.check_d(a.bank, TOKEN)
    rep["checks"]["d_header_generator"] = {"pass": not bad, "problems": bad, "generator": g}
    if a.expect_windows:
        b = C.bank_json(a.bank)
        tw = b["common"]["train"]["n_windows"]
        e = [] if (tw == a.expect_windows and not b["partial"]) else [f"train n_windows {tw} / partial {b['partial']} vs expected {a.expect_windows} complete"]
        rep["checks"]["e_complete_train_cover"] = {"pass": not e, "problems": e}
    ok = all(v["pass"] for v in rep["checks"].values())
    rep["pass"] = ok
    rep["readout"] = TOKEN
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep)[:1800])
    print("ZZ RH-PILOT-CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
