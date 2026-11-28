"""E2' / E3' exactly as registered in SPEC_ADDENDUM_S2A2.md (sha256 a44bd428…, registered 2026-10-04T12:59Z).

E2': certification C1/C2 FIRST (nothing is read unless both pass); one pre-registered fallback instrument; bar
rho(F0+NAV+RC_noised(v)) - rho(F0+NAV+HEAVY(v)) <= 0.01. E3' on the CLEAN checkpoint per support class.

usage: python s2_e2prime.py <s2_windows_trainE2p.npz> <s2_windows_eval139.npz> <out.json>
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import s2_analyze as A  # noqa: E402  (nav_block, deranged_by_clip — the Stage-2 helpers)

VARS = ("A50", "B", "A30", "A80")
SEED = 20261005
INSTR = {"registered": dict(max_iter=200, max_depth=6, learning_rate=0.05, random_state=0),
         "fallback": dict(max_iter=200, max_depth=6, learning_rate=0.05, random_state=0,
                          min_samples_leaf=200, l2_regularization=1.0)}


def r2_oof(X, Y, g, kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=5).split(X, Y[:, 0], g):
        for j in range(Y.shape[1]):
            pred[te, j] = HistGradientBoostingRegressor(**kw).fit(X[tr], Y[tr, j]).predict(X[te])
    return 1.0 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def rho(r, base):
    return float(np.mean((np.asarray(r) - base) / (1.0 - base)))


def noised(Z, v, rng):
    x, y, psi = Z[f"rc_{v}_x"], Z[f"rc_{v}_y"], np.radians(Z[f"rc_{v}_psi"])
    ea, el = rng.normal(0, 2.0, len(x)), rng.normal(0, 0.75, len(x))
    return np.c_[x + ea * np.cos(psi) - el * np.sin(psi), y + ea * np.sin(psi) + el * np.cos(psi), Z[f"rc_{v}_psi"]]


def e2prime(Z):
    rng = np.random.default_rng(SEED)
    N = {v: noised(Z, v, rng) for v in VARS}                          # noise drawn once per variant, fixed order
    Z = dict(Z)
    Z["rcN_A50_x"], Z["rcN_A50_y"], Z["rcN_A50_psi"] = N["A50"][:, 0], N["A50"][:, 1], N["A50"][:, 2]
    shuf = A.deranged_by_clip(Z, ["rcN_A50_x", "rcN_A50_y", "rcN_A50_psi"], SEED)
    sel = np.isfinite(Z["vfut"][:, 0]) & np.isfinite(Z["o1_ms"])
    for v in VARS:
        sel &= (Z[f"rc_{v}_valid"] == 1) & (Z[f"rcH_{v}_valid"] == 1)
    for k in ("x", "y", "psi"):
        sel &= np.isfinite(shuf[f"rcN_A50_{k}"])
    Y, g, v0 = Z["vfut"][sel], Z["clip_sha12"][sel], Z["v0"][sel][:, None]
    NAV = A.nav_block(Z, sel)
    out = {"n_rows": int(sel.sum()), "n_clips": int(len(np.unique(g))), "instruments": {}}
    for name, kw in INSTR.items():
        t0 = time.time()
        base = r2_oof(v0, Y, g, kw)
        c1 = rho(r2_oof(np.c_[v0, shuf["rcN_A50_x"][sel], shuf["rcN_A50_y"][sel], shuf["rcN_A50_psi"][sel]], Y, g, kw), base)
        c2 = rho([r2_oof(np.c_[v0, Y[:, j]], Y[:, [j]], g, kw)[0] for j in range(6)], base)
        o1 = rho(r2_oof(np.c_[v0, Z["o1_ms"][sel]], Y, g, kw), base)
        cert = abs(c1) <= 0.01 and c2 >= 0.99
        rec = {"C1_deranged_rho": round(c1, 5), "C2_target_rho": round(c2, 5), "O1_oracle_rho": round(o1, 5),
               "O1_ge_0.20": o1 >= 0.20, "certified": bool(cert), "kw": kw}
        print(f"[E2' {name}] C1 {c1:+.4f} C2 {c2:.4f} O1 {o1:+.3f} certified={cert} ({time.time() - t0:.0f}s)", flush=True)
        if cert:
            rn = rho(r2_oof(np.c_[v0, NAV], Y, g, kw), base)
            rec["rho_NAV"] = round(rn, 5)
            rec["variants"] = {}
            for v in VARS:
                H = np.c_[Z[f"rcH_{v}_x"][sel], Z[f"rcH_{v}_y"][sel], Z[f"rcH_{v}_psi"][sel]]
                C = np.c_[Z[f"rc_{v}_x"][sel], Z[f"rc_{v}_y"][sel], Z[f"rc_{v}_psi"][sel]]
                Nv = N[v][sel]
                r_nh = rho(r2_oof(np.c_[v0, NAV, H], Y, g, kw), base)
                r_nn = rho(r2_oof(np.c_[v0, NAV, Nv], Y, g, kw), base)
                r_nc = rho(r2_oof(np.c_[v0, NAV, C], Y, g, kw), base)
                r_n = rho(r2_oof(np.c_[v0, Nv], Y, g, kw), base)
                r_h = rho(r2_oof(np.c_[v0, H], Y, g, kw), base)
                rec["variants"][v] = {"rho_NAV_HEAVY": round(r_nh, 5), "rho_NAV_RCnoised": round(r_nn, 5),
                                      "rho_NAV_RCclean": round(r_nc, 5), "rho_RCnoised": round(r_n, 5), "rho_HEAVY": round(r_h, 5),
                                      "E2prime_delta": round(r_nn - r_nh, 5), "E2prime_pass": bool(r_nn - r_nh <= 0.01),
                                      "clean_minus_heavy": round(r_nc - r_nh, 5)}
                print(f"   {v}: NAV+RCnoised {r_nn:+.4f} vs NAV+HEAVY {r_nh:+.4f} -> delta {r_nn - r_nh:+.4f} "
                      f"{'PASS' if r_nn - r_nh <= 0.01 else 'FAIL'} (clean {r_nc - r_nh:+.4f})", flush=True)
        out["instruments"][name] = rec
        if cert:
            out["instrument_used"] = name
            break
    else:
        out["instrument_used"] = None
        out["E2prime"] = "BLOCKED: no certified nonlinear instrument on this corpus"
    # OLS beside it (decides nothing)
    base = A.r2_oof(v0, Y, g, "ols")
    out["ols_beside"] = {v: {"NAV_RCnoised_minus_NAV_HEAVY": round(
        rho(A.r2_oof(np.c_[v0, NAV, N[v][sel]], Y, g, "ols"), base)
        - rho(A.r2_oof(np.c_[v0, NAV, np.c_[Z[f"rcH_{v}_x"][sel], Z[f"rcH_{v}_y"][sel], Z[f"rcH_{v}_psi"][sel]]], Y, g, "ols"), base), 5)}
        for v in VARS}
    return out


def e3prime(Z):
    exc_lc = (Z["exc_eps_m"] >= 1.5) & (np.abs(Z["exc_net_deg"]) < 15.0)
    no_exc = np.isfinite(Z["exc_eps_m"]) & ~exc_lc
    out = {}
    for v in VARS:
        d = np.abs(Z[f"delta_{v}"])
        ok = np.isfinite(d)
        rng_ = Z[f"support_range_deg_{v}"]
        st, cu = ok & (rng_ <= 10.0), ok & (rng_ > 10.0)
        r = {"n_valid": int(ok.sum()), "straight_n": int(st.sum()),
             "straight_p90": round(float(np.percentile(d[st], 90)), 4),
             "straight_lc_scale_n": int((st & exc_lc).sum()),
             "straight_lc_scale_median": round(float(np.median(d[st & exc_lc])), 4) if (st & exc_lc).any() else None,
             "straight_no_exc_median": round(float(np.median(d[st & no_exc])), 4),
             "curved_n": int(cu.sum()), "curved_p90": round(float(np.percentile(d[cu], 90)), 4), "curved_bins": {}}
        r["bar_straight_p90_le_1m"] = bool(r["straight_p90"] <= 1.0)
        r["bar_lc_scale_median_le_0.5m"] = bool(r["straight_lc_scale_median"] is not None and r["straight_lc_scale_median"] <= 0.5)
        for lo, hi in ((10, 30), (30, 60), (60, 90), (90, 1e9)):
            b = cu & (rng_ > lo) & (rng_ <= hi)
            r["curved_bins"][f"{lo}-{int(hi) if hi < 1e8 else 'inf'}"] = {
                "n": int(b.sum()),
                "lc_scale_median": round(float(np.median(d[b & exc_lc])), 4) if (b & exc_lc).any() else None,
                "no_exc_median": round(float(np.median(d[b & no_exc])), 4) if (b & no_exc).any() else None}
        out[v] = r
    return out


def main(tr_p, ev_p, out_p):
    t0 = time.time()
    ZT = dict(np.load(tr_p, allow_pickle=True))
    ZE = dict(np.load(ev_p, allow_pickle=True))
    for Z in (ZT, ZE):
        Z["clip_sha12"] = np.array([str(s) for s in Z["clip_sha12"]])
    R = {"prereg": "SPEC_ADDENDUM_S2A2.md sha256 a44bd428aa9cc141a2e159848983075fe4033da363336ddd4407836ff2481ad9",
         "meta_train": json.loads(str(ZT["meta"])), "E2prime": e2prime(ZT),
         "E3prime": {"trainE2p": e3prime(ZT), "eval139": e3prime(ZE)}}
    # decision (registered rule)
    used = R["E2prime"].get("instrument_used")
    dec = None
    for v in ("A50", "B"):
        e2ok = bool(used) and R["E2prime"]["instruments"][used]["variants"][v]["E2prime_pass"]
        e3ok = all(R["E3prime"][p][v]["bar_straight_p90_le_1m"] and R["E3prime"][p][v]["bar_lc_scale_median_le_0.5m"]
                   for p in ("trainE2p", "eval139"))
        R.setdefault("decision_trace", {})[v] = {"E2prime": e2ok, "E3prime": e3ok}
        if e2ok and e3ok and dec is None:
            dec = f"RC-{v} noised CONFIRMED"
    R["decision"] = dec or ("BLOCKED: no certified instrument" if not used else "RC returns to the Master Mind")
    R["wall_s"] = round(time.time() - t0, 1)
    json.dump(R, open(out_p, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("[E2'/E3'] decision:", R["decision"], json.dumps(R["decision_trace"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
