#!/usr/bin/env python
"""plumb_read4.py -- controls of the source cand-imagkin (note PREREG_V8F_PROOF_ADD40_N4 section 3) from the pulled logs of one checkpoint. Written by the lead.

  k1  on every active tick and candidate: gap_d(h) - gap_0(h) == a_P(h) - a_d(h) to 1e-9 m                       (log of the cand-imagkin run)
  k2  at tick 0 the gap of d = 0 equals the gap of d = 0 of a cand-imag run of the same checkpoint                (max and median absolute difference, stated)
  k3  constant gap 1e9 m: no tick changes the plan;  level shifted by -20 m: more active ticks choose d > 0 than without the shift
  p7  window count, and the active ticks of the two sources are the same set
Beside (names nothing): the second diagnostic of the p4/p5 log (the reader's offsets on the baselines logged, lp, pl, pp, sm): level and response per baseline.
Reads no unsafe rate and no family value.
"""
import argparse
import io
import json
import os

import numpy as np

OFFS = (-2.0, -1.0, 1.0)


def slope0(d, e):
    d, e = np.asarray(d, np.float64).ravel(), np.asarray(e, np.float64).ravel()
    ok = np.isfinite(d) & np.isfinite(e)
    return float((d[ok] * e[ok]).sum() / (e[ok] * e[ok]).sum()) if ok.any() else float("nan")


def med(x):
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def load(root, run):
    p = os.path.join(root, run, "imag_log_%s.npz" % run)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--sfx", default="x")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nm = lambda t: "%s%s_%s" % (t, a.sfx, a.arm)                      # noqa: E731
    K = load(a.pull, nm("IPk"))
    res = {"arm": a.arm, "runs": {t: bool(load(a.pull, nm(t)) is not None) for t in ("IPk", "IPi", "IPkid1", "IPkid3", "IPkn", "IPks")}}
    if K is None:
        raise SystemExit("no log of %s" % nm("IPk"))
    cs = [float(x) for x in K["cand_set"]]
    i0 = cs.index(0.0)
    act = np.asarray(K["active"]).astype(bool)
    gap, ah = np.asarray(K["gap_h"], np.float64)[act], np.asarray(K["a_h"], np.float64)[act]
    k1 = float(np.nanmax(np.abs((gap - gap[:, [i0], :]) - (ah[:, [i0], :] - ah))))
    res["k1"] = {"max_abs_m": k1, "n_active_ticks": int(act.sum()), "largest_known_difference_m": float(np.nanmax(ah[:, [i0], :] - ah)), "PASS": bool(k1 <= 1e-9)}
    res["p7_windows"] = int(len(set(zip(np.asarray(K["ci"]).tolist(), np.asarray(K["t0"]).tolist()))))
    res["source"] = str(K["mode"]) if "mode" in K.files else "?"
    res["shas"] = {k: str(K[k])[:16] for k in ("imag_plan_sha256", "wrapper_sha256") if k in K.files}
    I = load(a.pull, nm("IPi"))
    if I is not None:
        key = lambda Z: {(int(c), int(t), int(k)): i for i, (c, t, k) in enumerate(zip(Z["ci"], Z["t0"], Z["k"]))}      # noqa: E731
        kk, ki = key(K), key(I)
        acti = np.asarray(I["active"]).astype(bool)
        same = set(k for k, i in kk.items() if act[i]) == set(k for k, i in ki.items() if acti[i])
        t0k = [(k, i) for k, i in kk.items() if k[2] == 0 and act[i] and k in ki]
        gk = np.asarray(K["gap_h"], np.float64)[[i for _, i in t0k]][:, i0]
        gi = np.asarray(I["gap_h"], np.float64)[[ki[k] for k, _ in t0k]][:, i0]
        d = np.abs(gk - gi)
        res["k2"] = {"n_tick0_active": int(len(t0k)), "max_abs_m": float(np.nanmax(d)), "median_abs_m": med(d), "PASS": bool(np.nanmax(d) <= 1e-2)}
        res["p7_active_same_set_at_tick0"] = bool(set(k for k, i in kk.items() if act[i] and k[2] == 0) == set(k for k, i in ki.items() if acti[i] and k[2] == 0))
        res["p7_active_same_set_all_ticks"] = bool(same)
    k3 = {}
    G = load(a.pull, nm("IPkid3"))
    if G is not None:
        k3["const_gap_changed_ticks"] = int(np.asarray(G["changed"]).astype(bool).sum())
    S = load(a.pull, nm("IPks"))
    if S is not None:
        sh = lambda Z: float((np.asarray(Z["chosen_c"], np.float64)[(np.asarray(Z["active"]).astype(bool)) & (np.asarray(Z["k"]).astype(int) == 0)] > 0).mean())   # noqa: E731
        k3["share_d_above_0_at_tick0"] = {"unshifted": sh(K), "shifted_minus_20": sh(S)}
    if k3:
        k3["PASS"] = bool(k3.get("const_gap_changed_ticks", 1) == 0 and ("share_d_above_0_at_tick0" not in k3 or k3["share_d_above_0_at_tick0"]["shifted_minus_20"] > k3["share_d_above_0_at_tick0"]["unshifted"])
                          and "const_gap_changed_ticks" in k3 and "share_d_above_0_at_tick0" in k3)
        res["k3"] = k3
    # ---- beside: the second diagnostic
    pp = os.path.join(a.pull, nm("IPk"), "p45_%s.npz" % nm("IPk"))
    if os.path.exists(pp):
        P = np.load(pp, allow_pickle=True)
        k0 = np.asarray(K["k"]).astype(int) == 0
        key0 = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(np.asarray(K["ci"])[k0], np.asarray(K["t0"])[k0]))}
        rows0 = np.array([key0[(int(c), int(t))] for c, t in zip(P["ci"], P["t0"])])
        aw = act[k0][rows0]
        exp = np.stack([np.stack([-0.5 * d * T * T for T in (1.0, 2.0)]) for d in OFFS])           # [3, 2]
        gl, go = np.asarray(P["imag_gap_logged"], np.float64), np.asarray(P["imag_gap_logged_off"], np.float64)
        diag = {"n_windows": int(len(aw)), "n_active_windows": int(aw.sum()), "present_gap_median_m": med(np.asarray(K["g_src"], np.float64)[k0][rows0][aw])}
        bases = {"logged": (gl, go)}
        for b in ("lp", "pl", "pp", "sm"):
            if "imag_gap_hyb_" + b in P.files:
                h = np.asarray(P["imag_gap_hyb_" + b], np.float64)
                bases[b] = (h[:, 0], h[:, 1:])
        for b, (g0, g1) in bases.items():
            d = g1 - g0[:, None, :]
            diag[b] = {"level_median_T1.0_m": med(g0[aw, 0]), "level_median_T2.0_m": med(g0[aw, 1]),
                       "slope_T1.0": slope0(d[aw][:, :, 0], np.broadcast_to(exp[:, 0], d[aw][:, :, 0].shape)),
                       "slope_T2.0": slope0(d[aw][:, :, 1], np.broadcast_to(exp[:, 1], d[aw][:, :, 1].shape)),
                       "median_ratio_T2.0": med(d[aw][:, :, 1] / exp[None, :, 1]),
                       "level_minus_logged_median_abs_T2.0_m": med(np.abs(g0[aw, 1] - gl[aw, 1]))}
        if "act_plan" in P.files:
            A, Lg = np.asarray(P["act_plan"], np.float64)[aw], np.asarray(P["act_cache"], np.float64)[aw]
            ok = np.isfinite(A).all((1, 2)) & np.isfinite(Lg).all((1, 2))
            diag["plan_against_logged_actions"] = {"corr_acc": float(np.corrcoef(A[ok][:, :, 0].ravel(), Lg[ok][:, :, 0].ravel())[0, 1]),
                                                   "corr_yaw": float(np.corrcoef(A[ok][:, :, 1].ravel(), Lg[ok][:, :, 1].ravel())[0, 1]),
                                                   "median_abs_acc_diff": med(np.abs(A[ok][:, :, 0] - Lg[ok][:, :, 0])), "median_abs_yaw_diff": med(np.abs(A[ok][:, :, 1] - Lg[ok][:, :, 1])),
                                                   "std_acc_plan": float(A[ok][:, :, 0].std()), "std_acc_logged": float(Lg[ok][:, :, 0].std()),
                                                   "std_yaw_plan": float(A[ok][:, :, 1].std()), "std_yaw_logged": float(Lg[ok][:, :, 1].std())}
        res["beside_second_diagnostic"] = diag
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)

    def rr(d):
        return {k: (round(v, 4) if isinstance(v, float) else (rr(v) if isinstance(v, dict) else v)) for k, v in d.items()}
    for k in ("runs", "k1", "k2", "k3", "p7_windows", "p7_active_same_set_at_tick0", "p7_active_same_set_all_ticks"):
        if k in res:
            print(k, json.dumps(rr(res[k]) if isinstance(res[k], dict) else res[k]))
    for k, v in res.get("beside_second_diagnostic", {}).items():
        print("  diag", k, json.dumps(rr(v) if isinstance(v, dict) else v))


if __name__ == "__main__":
    main()
