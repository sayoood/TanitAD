"""Label-free analysis of a NAV-GOAL-ONLY trigger for REFe's goal fallback (CPU, numpy only).

Question (PI 2026-10-01): "no lane geometry for model driving; nav commands and nav goals are fine". Amendment 8's trigger
read the ego's distance to the route POLYLINE. Here the trigger reads only the NAV GOAL the planner already receives and
the ego speed: the goal is INVALID when it lies farther than the ego could reach along any path in the goal horizon,
    |p2| > kappa * max(|v0|, 5 m/s) * 12 s            (p2 = the far goal point, ego frame)
because a valid route goal is an arc-length point at exactly that distance, so its Euclidean distance cannot exceed it.

PRE-REGISTERED PROCEDURE (written before the full census was read): kappa is chosen on the 1,123 SELECTION tokens only
(maximum F1 agreement with the old trigger `ego_to_route_m > 20`), then FROZEN and reported on the other 11,023 tokens,
label-free (no PDMS). Second part: which ego-only estimate of the goal best predicts the human's own 4 s future on the
triggered tokens (straight / constant turn rate / + acceleration)."""
import argparse
import gzip
import json
import math
import sys

import os

import numpy as np

# the external drive was re-lettered D: -> E: on 2026-10-03; REFE_DRIVE re-points it without editing paths twice
_DRV = os.environ.get("REFE_DRIVE", "D:")
EXPORT = f"{_DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SEL = f"{_DRV}/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-09-28-m6b-tangent/tokens_1123.json"
HOR, MINV = 12.0, 5.0
KINDS = ("straight", "ctrv", "ca", "ctra")
_DT_INT = 0.01                                  # numeric integration step (s)


def _integrate(v, omega, a, t):
    """heading = omega*t, speed = max(v + a*t, 0) (a vehicle does not reverse into a goal), positions by midpoint
    integration on a 10 ms grid. 2026-10-03: replaces a chord formula that is exact only for CONSTANT speed (an arc of
    constant curvature); with a != 0 and a constant TURN RATE the curvature varies, so the chord form was wrong."""
    tt = np.arange(0.0, float(t.max()) + _DT_INT / 2, _DT_INT)
    tm = tt[:-1] + _DT_INT / 2
    sp = np.maximum(v + a * tm, 0.0)
    x = np.concatenate([[0.0], np.cumsum(sp * np.cos(omega * tm) * _DT_INT)])
    y = np.concatenate([[0.0], np.cumsum(sp * np.sin(omega * tm) * _DT_INT)])
    return np.stack([np.interp(t, tt, x), np.interp(t, tt, y)], 1)


def predict(kind, v, omega, a, t):
    """positions (n,2) in the CURRENT ego frame at times t (s)"""
    if kind == "straight":
        return np.stack([v * t, np.zeros_like(t)], 1)
    if kind == "ctrv":
        if abs(omega) < 1e-4:
            return np.stack([v * t, np.zeros_like(t)], 1)
        return np.stack([v / omega * np.sin(omega * t), v / omega * (1 - np.cos(omega * t))], 1)
    if kind == "ca":                               # straight with the measured acceleration, no turn
        return _integrate(v, 0.0, a, t)
    if kind == "ctra":
        return _integrate(v, omega, a, t)
    raise ValueError(kind)


def selftest():
    """analytic targets, written as literals/closed forms that do NOT call the integrator's own code path"""
    t = np.arange(1, 25) * 0.5
    e1 = np.abs(predict("ctra", 10.0, 0.2, 0.0, t) - np.stack([10 / 0.2 * np.sin(0.2 * t), 10 / 0.2 * (1 - np.cos(0.2 * t))], 1)).max()
    e2 = np.abs(predict("ctra", 4.0, 0.0, 1.5, t)[:, 0] - (4.0 * t + 0.75 * t * t)).max()
    e3 = np.abs(predict("ctra", 4.0, 0.0, -1.0, t)[:, 0] - np.where(t < 4.0, 4.0 * t - 0.5 * t * t, 8.0)).max()  # stops at 4 s, 8 m
    # the case the old chord formula got wrong: turn AND accelerate. Closed form of int (v+a*tau) (cos, sin)(w*tau) dtau
    v, w, a = 6.0, 0.15, 0.8
    cf = np.stack([v * np.sin(w * t) / w + a * (np.cos(w * t) + w * t * np.sin(w * t) - 1) / w ** 2,
                   v * (1 - np.cos(w * t)) / w + a * (np.sin(w * t) - w * t * np.cos(w * t)) / w ** 2], 1)
    e4 = np.abs(predict("ctra", v, w, a, t) - cf).max()
    # MUTANT: the 2026-10-01 chord formula must go RED on that case, or the test cannot see the defect it replaced
    s_ = v * t + 0.5 * a * t * t
    th = w * t
    old = np.stack([s_ * np.sinc(th / np.pi / 2) * np.cos(th / 2), s_ * np.sinc(th / np.pi / 2) * np.sin(th / 2)], 1)
    e_mut = np.abs(old - cf).max()
    ok = e1 < 1e-3 and e2 < 1e-3 and e3 < 1e-3 and e4 < 1e-3 and e_mut > 0.1
    print(f"SELFTEST ctra(a=0)=ctrv {e1:.2e}  ctra(w=0,a>0) {e2:.2e}  ctra(w=0,a<0,stop) {e3:.2e}  "
          f"ctra(w,a) vs closed form {e4:.2e}  OLD chord mutant {e_mut:.2f} m (must be RED >0.1)  -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="goal_census_navtest_full.json")
    ap.add_argument("--out", default="goal_trigger_analysis.json")
    a = ap.parse_args()
    if not selftest():
        return 2
    C = json.load(open(a.census, encoding="utf-8"))["rows"]
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    sel = set(json.load(open(SEL, encoding="utf-8"))["tokens"])
    toks = [t for t in C if "goal" in C[t] and t in E]
    n_err = sum(1 for r in C.values() if "error" in r)
    d_route = np.array([C[t]["ego_to_route_m"] for t in toks])
    g = np.array([C[t]["goal"] for t in toks])                     # (N,4): p1x p1y p2x p2y
    v = np.array([math.hypot(*C[t]["v"]) for t in toks])
    s = np.maximum(v, MINV) * HOR
    ratio = np.hypot(g[:, 2], g[:, 3]) / s
    old = d_route > 20.0
    is_sel = np.array([t in sel for t in toks])
    out = {"n_tokens": len(toks), "n_census_errors": n_err, "n_selection": int(is_sel.sum()), "old_trigger_fires": int(old.sum()),
           "ratio_quantiles_old_fires": {q: float(np.percentile(ratio[old], q)) for q in (0, 5, 25, 50)} if old.any() else {},
           "ratio_quantiles_old_silent": {q: float(np.percentile(ratio[~old], q)) for q in (50, 90, 99, 99.9, 100)}}

    def conf(k, m):
        new = ratio > k
        tp, fp, fn = int((new & old & m).sum()), int((new & ~old & m).sum()), int((~new & old & m).sum())
        return tp, fp, fn, (2 * tp / max(2 * tp + fp + fn, 1))
    ks = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0]
    out["selection_sweep"] = {str(k): dict(zip(("tp", "fp", "fn", "f1"), conf(k, is_sel))) for k in ks}
    kbest = max(ks, key=lambda k: (conf(k, is_sel)[3], -k))
    out["kappa_frozen"] = kbest
    out["held_out_at_kappa"] = dict(zip(("tp", "fp", "fn", "f1"), conf(kbest, ~is_sel)))
    out["all_at_kappa"] = dict(zip(("tp", "fp", "fn", "f1"), conf(kbest, np.ones(len(toks), bool))))
    new = ratio > kbest
    # disagreement tokens, named
    out["new_only"] = [{"token": toks[i], "ego_to_route_m": float(d_route[i]), "ratio": float(ratio[i]), "v": float(v[i])} for i in np.where(new & ~old)[0][:40]]
    out["old_only"] = [{"token": toks[i], "ego_to_route_m": float(d_route[i]), "ratio": float(ratio[i]), "v": float(v[i])} for i in np.where(~new & old)[0][:60]]
    out["old_only_ratio_quantiles"] = {q: float(np.percentile(ratio[~new & old], q)) for q in (0, 25, 50, 75, 100)} if (~new & old).any() else {}
    out["old_only_d_quantiles"] = {q: float(np.percentile(d_route[~new & old], q)) for q in (0, 25, 50, 75, 100)} if (~new & old).any() else {}

    # --- estimator quality on the OLD-trigger tokens, against the human's own 4 s future (8 poses on the 0.5 s grid)
    t8 = np.arange(1, 9) * 0.5
    res = {}
    for name, sel_mask in (("triggered_old", old), ("triggered_old_selection", old & is_sel), ("all_tokens_reference", np.ones(len(toks), bool))):
        rows = []
        for i in np.where(sel_mask)[0]:
            ex = E[toks[i]]
            hum = np.array(ex["human_future_poses"], dtype=np.float64)[:, :2]
            st = ex["ego_statuses"]
            vv = math.hypot(*st[-1]["ego_velocity"])
            h = [s_["ego_pose"][2] for s_ in st[-2:]]
            om = ((h[1] - h[0] + math.pi) % (2 * math.pi) - math.pi) / 0.5
            ac = float(st[-1]["ego_acceleration"][0])
            cmd = int(np.argmax(st[-1]["driving_command"]))
            r = {"cmd": cmd}
            for kind in KINDS:
                p = predict(kind, vv, om, ac, t8)
                e = np.linalg.norm(p - hum, axis=1)
                r[kind] = (float(e.mean()), float(e[-1]))
            rows.append(r)
        if rows:
            res[name] = {"n": len(rows), **{k: {"ADE_mean": float(np.mean([r[k][0] for r in rows])), "ADE_median": float(np.median([r[k][0] for r in rows])),
                                                "FDE4s_mean": float(np.mean([r[k][1] for r in rows])), "FDE4s_median": float(np.median([r[k][1] for r in rows]))}
                                            for k in KINDS}}
            # paired per-token FDE differences vs straight, with a token bootstrap (CPU, report-only)
            rng = np.random.default_rng(20260927)
            fs = np.array([r["straight"][1] for r in rows])
            for k in KINDS[1:]:
                d = np.array([r[k][1] for r in rows]) - fs
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                res[name][f"FDE4s_{k}_minus_straight"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                          "estimator": "token bootstrap, 2000 resamples (NOT log-clustered; report-only)"}
            res[name]["by_command_n"] = {c: int(sum(1 for r in rows if r["cmd"] == c)) for c in range(4)}
    out["estimators_vs_human_4s"] = res
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v_ for k, v_ in out.items() if k not in ("new_only", "old_only")}, indent=1)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
