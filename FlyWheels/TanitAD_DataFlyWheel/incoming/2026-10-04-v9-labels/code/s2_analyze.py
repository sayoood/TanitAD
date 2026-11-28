"""WP-A Stage 2 analysis: E1 (trivial-planner floor), E2 (speed leak), E3 (+A1, lateral leak), E4 (decision).

Pre-registration: SPEC.md §9.E and SPEC_ADDENDUM_S2A1.md (sha256 in raw/SPEC_SHA256.txt, recorded before this ran).
Route metrics are RE-TYPED from the route package's `route_metrics.py` (not imported). Paired episode-cluster bootstrap,
B = 2000, seed 0, clusters = clips. One fixed rule added before any number was read (stated in the output JSON):
E2's verdict is read on eval139 under BOTH model classes; the trainS replicate is reported, and a variant that passes on
eval139 but fails on trainS is treated as FAILING in the E4 ranking (conservative).

usage: python s2_analyze.py <s2_windows_eval139.npz> <s2_windows_trainS.npz> <out.json> <tables.md>
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------------------------- re-typed metrics #
SLOTS = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
TAU_C = 0.18063741505146028            # 10.35 deg, the run's own compliance tau (route_metrics.py:14)
TURN_DEG, STRAIGHT_DEG, HEAD_AGREE_DEG, MIN_LEN_M, STALL_M = 30.0, 10.0, 15.0, 5.0, 0.05
VARS = ("A30", "A50", "A80", "B")
SEED = 20261004
B_BOOT = 2000


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def terminal_heading(P):
    d = P[..., -1, :] - P[..., -2, :]
    th = np.arctan2(d[..., 1], d[..., 0])
    return np.where(np.hypot(d[..., 0], d[..., 1]) < STALL_M, 0.0, th)


def dir_class(th):
    return np.where(th >= TAU_C, 1, np.where(th <= -TAU_C, -1, 0))


def path_length(P):
    Q = np.concatenate([np.zeros(P.shape[:-2] + (1, 2)), P], axis=-2)
    return np.linalg.norm(np.diff(Q, axis=-2), axis=-1).sum(-1)


def gt_class(gt, gv):
    th = terminal_heading(gt)
    ok = gv[:, -1].astype(bool) & (path_length(gt) >= MIN_LEN_M)
    deg = np.degrees(np.abs(th))
    cls = np.full(len(th), "unclassified", dtype=object)
    cls[ok & (deg >= TURN_DEG) & (th > 0)] = "turnL"
    cls[ok & (deg >= TURN_DEG) & (th < 0)] = "turnR"
    cls[ok & (deg < STRAIGHT_DEG)] = "straight"
    cls[ok & (deg >= STRAIGHT_DEG) & (deg < TURN_DEG)] = "gentle"
    return cls, th


def per_window_metrics(P, gt, gv):
    d = np.linalg.norm(P - gt, axis=-1)
    vv = gv.astype(bool)
    ade = np.where(vv.sum(1) > 0, (d * vv).sum(1) / np.maximum(vv.sum(1), 1), np.nan)
    thp, thg = terminal_heading(P), terminal_heading(gt)
    h15 = (np.degrees(np.abs(wrap(thp - thg))) <= HEAD_AGREE_DEG).astype(float)
    dirok = (dir_class(thp) == np.sign(thg)).astype(float)
    e6 = P[:, -1] - gt[:, -1]
    xtrack6 = np.abs(-np.sin(thg) * e6[:, 0] + np.cos(thg) * e6[:, 1])
    return {"ade": ade, "h15": h15, "dir": dirok, "xtrack6": xtrack6}


# ---------------------------------------------------------------------------------------------- planners #
def tp_arc(xp, yp, s):
    """Positions at arc lengths s [W, 8] along the constant-curvature arc tangent to x through (xp, yp)."""
    r2 = xp ** 2 + yp ** 2
    k = np.where(r2 > 1e-6, 2.0 * yp / np.maximum(r2, 1e-6), 0.0)[:, None]
    small = np.abs(k) < 1e-9
    ks = k * s
    x = np.where(small, s, np.sin(ks) / np.where(small, 1.0, k))
    y = np.where(small, 0.0, (1.0 - np.cos(ks)) / np.where(small, 1.0, k))
    return np.stack([x, y], -1)


def gt_arc_lengths(gt):
    Q = np.concatenate([np.zeros((len(gt), 1, 2)), gt], 1)
    return np.cumsum(np.linalg.norm(np.diff(Q, axis=1), axis=-1), 1)


# ---------------------------------------------------------------------------------------------- bootstrap #
def make_draws(clips, B=B_BOOT, seed=0):
    u, inv = np.unique(clips, return_inverse=True)
    return inv, np.random.default_rng(seed).integers(0, len(u), (B, len(u))), len(u)


def boot_mean(x, mask, inv, draws, E):
    x = np.asarray(x, np.float64)
    m = np.asarray(mask, bool) & np.isfinite(x)
    N = np.bincount(inv, weights=np.where(m, x, 0.0), minlength=E)
    D = np.bincount(inv, weights=m.astype(float), minlength=E)
    cnt = np.stack([np.bincount(d, minlength=E) for d in draws])
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (cnt @ N) / (cnt @ D)
    pt = N.sum() / D.sum() if D.sum() > 0 else np.nan
    bs = bs[np.isfinite(bs)]
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs.size else [None, None]
    return float(pt), ci, int(m.sum())


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def deranged_by_clip(Z, cols, seed):
    """A5 T2c construction: each clip receives a donor clip's value at the same provider index t."""
    clips = Z["clip_sha12"]
    t = Z["t"]
    u = np.unique(clips)
    perm = derangement(len(u), seed)
    out = {c: np.full(len(clips), np.nan) for c in cols}
    by = {c: np.nonzero(clips == c)[0] for c in u}
    for ci, c in enumerate(u):
        rec, don = by[c], by[u[perm[ci]]]
        dt = t[don]
        order = np.argsort(dt)
        j = np.clip(np.searchsorted(dt[order], t[rec]), 0, len(don) - 1)
        src = don[order][j]
        for col in cols:
            out[col][rec] = Z[col][src]
    return out


# ---------------------------------------------------------------------------------------------- E2 models #
def r2_oof(X, Y, groups, model):
    from sklearn.model_selection import GroupKFold
    pred = np.zeros_like(Y)
    for tr_i, te_i in GroupKFold(n_splits=5).split(X, Y[:, 0], groups):
        for j in range(Y.shape[1]):
            if model == "ols":
                A = np.c_[np.ones(len(tr_i)), X[tr_i]]
                w, *_ = np.linalg.lstsq(A, Y[tr_i, j], rcond=None)
                pred[te_i, j] = np.c_[np.ones(len(te_i)), X[te_i]] @ w
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                m = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, random_state=0)
                m.fit(X[tr_i], Y[tr_i, j])
                pred[te_i, j] = m.predict(X[te_i])
    sse = ((Y - pred) ** 2).sum(0)
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - sse / sst


def nav_block(Z, sel):
    tok = Z["nav_token"][sel]
    dn = np.where(np.isfinite(Z["nav_d_next_m"][sel]), np.clip(Z["nav_d_next_m"][sel], 0, 300), 300.0)
    de = np.where(np.isfinite(Z["nav_d_end_m"][sel]), np.clip(Z["nav_d_end_m"][sel], 0, 300), 300.0)
    dy = np.where(np.isfinite(Z["nav_dyaw_next_deg"][sel]), Z["nav_dyaw_next_deg"][sel], 0.0)
    return np.c_[(tok == 1), (tok == 2), dn, de, dy, Z["nav_args_valid"][sel], np.clip(Z["nav_lookahead_m"][sel], 0, 1000)].astype(float)


def e2(Z, name, models=("ols", "hgb")):
    shuf = deranged_by_clip(Z, [f"rc_A50_{k}" for k in ("x", "y", "psi")], SEED)
    sel = np.isfinite(Z["vfut"][:, 0]) & np.isfinite(Z["a0"]) & np.isfinite(Z["o1_ms"])
    for v in VARS:
        sel &= Z[f"rc_{v}_valid"] == 1
    for k in ("x", "y", "psi"):
        sel &= np.isfinite(shuf[f"rc_A50_{k}"])
    Y = Z["vfut"][sel]
    g = Z["clip_sha12"][sel]
    v0 = Z["v0"][sel][:, None]
    NAV = nav_block(Z, sel)
    tt = np.where(np.isfinite(Z["nav_t_next_s"][sel]), np.clip(Z["nav_t_next_s"][sel], -10, 60), 60.0)
    NAVT = np.c_[NAV, tt, Z["nav_token_ttime"][sel] == 1, Z["nav_token_ttime"][sel] == 2].astype(float)
    sets = {"F0": v0, "F0p_v0_a0": np.c_[v0, Z["a0"][sel]], "NAV": np.c_[v0, NAV],
            "DIAG_NAV_TTIME": np.c_[v0, NAVT],
            "CTRL_RC_A50_deranged": np.c_[v0, np.c_[shuf["rc_A50_x"][sel], shuf["rc_A50_y"][sel], shuf["rc_A50_psi"][sel]]],
            "CTRL_O1_oracle": np.c_[v0, Z["o1_ms"][sel]]}
    for v in VARS:
        rc = np.c_[Z[f"rc_{v}_x"][sel], Z[f"rc_{v}_y"][sel], Z[f"rc_{v}_psi"][sel]]
        sets[f"RC_{v}"] = np.c_[v0, rc]
        sets[f"NAV_RC_{v}"] = np.c_[v0, NAV, rc]
    out = {"population": name, "n_rows": int(sel.sum()), "n_clips": int(len(np.unique(g))), "taus_s": [1, 2, 3, 4, 5, 6],
           "models": {}}
    for mdl in models:
        t0 = time.time()
        r2 = {k: r2_oof(X, Y, g, mdl) for k, X in sets.items()}
        # the target-itself control, per tau (feature = v0 + that tau's own target)
        r2["CTRL_target_itself"] = np.array([r2_oof(np.c_[v0, Y[:, j]], Y[:, [j]], g, mdl)[0] for j in range(6)])
        base = r2["F0"]
        rho = {k: float(np.mean((r - base) / (1.0 - base))) for k, r in r2.items()}
        out["models"][mdl] = {"r2": {k: [round(float(a), 5) for a in r] for k, r in r2.items()},
                              "rho_mean": {k: round(v, 5) for k, v in rho.items()},
                              "d": {k: int(X.shape[1]) for k, X in sets.items()},
                              "wall_s": round(time.time() - t0, 1)}
        rn = rho["NAV"]
        out["models"][mdl]["controls"] = {
            "deranged_abs_rho_le_0.01": abs(rho["CTRL_RC_A50_deranged"]) <= 0.01,
            "target_itself_rho_ge_0.99": rho["CTRL_target_itself"] >= 0.99,
            "o1_oracle_rho_ge_0.40": rho["CTRL_O1_oracle"] >= 0.40}
        out["models"][mdl]["bar"] = {v: {"rho_RC": round(rho[f"RC_{v}"], 5), "rho_NAV": round(rn, 5),
                                         "rho_NAV_RC_minus_NAV": round(rho[f"NAV_RC_{v}"] - rn, 5),
                                         "pass": bool(rho[f"RC_{v}"] <= rn + 0.03 and rho[f"NAV_RC_{v}"] - rn <= 0.03)}
                                     for v in VARS}
        print(f"[E2 {name} {mdl}] {out['models'][mdl]['wall_s']}s rho:",
              {k: round(v, 3) for k, v in rho.items()}, flush=True)
    return out


# ---------------------------------------------------------------------------------------------- E3 #
def e3(Z, name):
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
    exc_lc = (Z["exc_eps_m"] >= 1.5) & (np.abs(Z["exc_net_deg"]) < 15.0)
    out = {"population": name, "variants": {}}
    for v in VARS:
        d = Z[f"delta_{v}"]
        ok = np.isfinite(d)
        straight = ok & (Z[f"support_range_deg_{v}"] <= 10.0)
        curved = ok & ~(Z[f"support_range_deg_{v}"] <= 10.0)
        ad = np.abs(d)

        def q(m):
            return {"n": int(m.sum()), "median": round(float(np.median(ad[m])), 4) if m.any() else None,
                    "p90": round(float(np.percentile(ad[m], 90)), 4) if m.any() else None}
        r = {"all": q(ok), "straight_support": q(straight) | {"share_of_valid": round(float(straight.sum() / max(ok.sum(), 1)), 4)},
             "curved_support": q(curved),
             "straight_support_lc_scale_excursion": q(straight & exc_lc),
             "straight_support_no_lc_excursion": q(straight & ~exc_lc & np.isfinite(Z["exc_eps_m"]))}
        r["E3_as_registered_pass"] = bool(r["all"]["p90"] is not None and r["all"]["p90"] <= 1.0)
        r["E3_A1_pass"] = bool(r["straight_support"]["p90"] is not None and r["straight_support"]["p90"] <= 1.0)
        # secondary: within-lane information the input adds over the heavy route (dev6 OOF R^2, trees)
        hv = f"rcH_{v}"
        sel = np.isfinite(Z["dev6"]) & (Z[f"rc_{v}_valid"] == 1) & (Z[f"{hv}_valid"] == 1)
        if sel.sum() > 1000:
            Yd = Z["dev6"][sel][:, None]
            g = Z["clip_sha12"][sel]
            A = np.c_[Z["v0"][sel], Z[f"{hv}_x"][sel], Z[f"{hv}_y"][sel], Z[f"{hv}_psi"][sel]]
            Bm = np.c_[A, Z[f"rc_{v}_x"][sel], Z[f"rc_{v}_y"][sel], Z[f"rc_{v}_psi"][sel]]
            ra, rb = float(r2_oof(A, Yd, g, "hgb")[0]), float(r2_oof(Bm, Yd, g, "hgb")[0])
            r["dev6_oof_r2"] = {"n": int(sel.sum()), "heavy_route_only": round(ra, 4), "heavy_plus_input": round(rb, 4),
                                "delta": round(rb - ra, 4), "dev6_abs_median_m": round(float(np.median(np.abs(Yd))), 4)}
        out["variants"][v] = r
    return out


# ---------------------------------------------------------------------------------------------- E1 #
def plans(Z, rows, gt, v0, rng):
    s_const = v0[:, None] * SLOTS[None, :]
    s_gt = gt_arc_lengths(gt)
    P = {"CV_straight": np.stack([s_const, np.zeros_like(s_const)], -1)}
    shuf = deranged_by_clip(Z, [f"rc_{v}_{k}" for v in VARS for k in ("x", "y", "psi")], SEED)
    for v in VARS:
        x, y, psi = Z[f"rc_{v}_x"][rows], Z[f"rc_{v}_y"][rows], np.radians(Z[f"rc_{v}_psi"][rows])
        P[f"TP_{v}"] = tp_arc(x, y, s_const)
        P[f"TP_{v}_gtspeed"] = tp_arc(x, y, s_gt)
        ea, el = rng.normal(0, 2.0, len(rows)), rng.normal(0, 0.75, len(rows))
        P[f"TP_{v}_noised"] = tp_arc(x + ea * np.cos(psi) - el * np.sin(psi), y + ea * np.sin(psi) + el * np.cos(psi), s_const)
        P[f"CTRL_TP_{v}_deranged"] = tp_arc(shuf[f"rc_{v}_x"][rows], shuf[f"rc_{v}_y"][rows], s_const)
    return P


def summarise(P, gt, gv, cls, clips, ref=None):
    inv, draws, E = make_draws(clips)
    classes = {"turn": np.isin(cls, ["turnL", "turnR"]), "turnL": cls == "turnL", "turnR": cls == "turnR",
               "straight": cls == "straight", "gentle": cls == "gentle", "all_classified": cls != "unclassified"}
    M = {a: per_window_metrics(p, gt, gv) for a, p in P.items()}
    out = {"n": {k: int(m.sum()) for k, m in classes.items()}, "n_clips": int(E), "arms": {}, "paired_vs_ref": {}}
    for a, mm in M.items():
        row = {}
        for cn, cm in classes.items():
            if not cm.any():
                continue
            row[cn] = {"ade": boot_mean(mm["ade"], cm, inv, draws, E)[:2],
                       "h15": boot_mean(mm["h15"], cm, inv, draws, E)[:2],
                       "xtrack6": boot_mean(mm["xtrack6"], cm, inv, draws, E)[:2]}
            if cn.startswith("turn"):
                row[cn]["dir"] = boot_mean(mm["dir"], cm, inv, draws, E)[:2]
        out["arms"][a] = row
    if ref is not None:
        for a, mm in M.items():
            if a == ref:
                continue
            r = {}
            for cn in ("turn", "straight", "all_classified"):
                cm = classes[cn]
                r[cn] = {"d_ade": boot_mean(mm["ade"] - M[ref]["ade"], cm, inv, draws, E)[:2],
                         "d_h15": boot_mean(mm["h15"] - M[ref]["h15"], cm, inv, draws, E)[:2]}
                if cn == "turn":
                    r[cn]["d_dir"] = boot_mean(mm["dir"] - M[ref]["dir"], cm, inv, draws, E)[:2]
            out["paired_vs_ref"][a] = r
    return out


def e1(Z, bank_path):
    res = {}
    rng = np.random.default_rng(SEED)
    # ---- (i) EVAL-DIAG vs refcv7's banked E9 pick ------------------------------------------------------ #
    bk = np.load(bank_path, allow_pickle=True)
    key = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(Z["clip_sha12"], Z["t"]))}
    rows = np.array([key.get((str(s), int(t)), -1) for s, t in zip(bk["win_sha12"], bk["win_t"])])
    res["evaldiag_join"] = {"n_bank": int(len(rows)), "n_joined": int((rows >= 0).sum())}
    rows_ok = rows >= 0
    rows = rows[rows_ok]
    bgt, bgv = bk["gt"][rows_ok].astype(np.float64), bk["gt_valid"][rows_ok].astype(np.int8)
    lg, lgv = Z["gt"][rows], Z["gt_valid"][rows]
    both = (bgv.astype(bool) & (lgv == 1))
    dd = np.linalg.norm(bgt - lg, axis=-1)[both]
    res["gt_crosscheck_log_vs_bank"] = {"n_slots": int(both.sum()), "median_m": round(float(np.median(dd)), 4),
                                        "p95_m": round(float(np.percentile(dd, 95)), 4), "max_m": round(float(dd.max()), 4),
                                        "v0_abs_diff_median": round(float(np.median(np.abs(bk["v0"][rows_ok] - Z["v0"][rows]))), 4)}
    use_bank_gt = res["gt_crosscheck_log_vs_bank"]["median_m"] > 0.10
    res["evaldiag_gt_source"] = "bank gt (addendum rule: median > 0.10 m)" if use_bank_gt else "bank gt (pre-registered for EVAL-DIAG comparisons; log gt agrees)"
    gt, gv = bgt, bgv          # EVAL-DIAG comparisons always on the bank's own gt (the refcv7 pick was scored on it)
    cls, _ = gt_class(gt, gv)
    v0 = bk["v0"][rows_ok].astype(np.float64)
    P = plans(Z, rows, gt, v0, rng)
    P["REFCV7_E9_pick"] = bk["traj"][rows_ok].astype(np.float64)
    res["evaldiag_n_by_class_check"] = {c: int((cls == c).sum()) for c in ("turnL", "turnR", "straight", "gentle", "unclassified")}
    common = np.ones(len(rows), bool)
    for v in VARS:
        common &= Z[f"rc_{v}_valid"][rows] == 1
    res["evaldiag_rc_all_valid"] = int(common.sum())
    sub = {a: p[common] for a, p in P.items()}
    res["evaldiag"] = summarise(sub, gt[common], gv[common], cls[common], Z["clip_sha12"][rows][common], ref="REFCV7_E9_pick")
    # ---- (ii) every eval139 window with a valid log GT and every RC variant valid -------------------------- #
    sel = (Z["gt_valid"][:, -1] == 1)
    for v in VARS:
        sel &= Z[f"rc_{v}_valid"] == 1
    r2 = np.nonzero(sel)[0]
    gt2, gv2 = Z["gt"][r2], Z["gt_valid"][r2]
    cls2, _ = gt_class(gt2, gv2)
    P2 = plans(Z, r2, gt2, Z["v0"][r2], np.random.default_rng(SEED))
    res["all_windows"] = summarise(P2, gt2, gv2, cls2, Z["clip_sha12"][r2], ref="CV_straight")
    res["all_windows"]["population"] = f"eval139 windows with log GT at 6 s and all RC variants valid: {len(r2)} of {len(Z['t'])}"
    return res


def main(ev_p, tr_p, out_p, md_p):
    t0 = time.time()
    ZE = dict(np.load(ev_p, allow_pickle=True))
    ZT = dict(np.load(tr_p, allow_pickle=True))
    for Z in (ZE, ZT):
        Z["clip_sha12"] = np.array([str(s) for s in Z["clip_sha12"]])
    R = {"prereg": "SPEC.md §9.E + SPEC_ADDENDUM_S2A1.md (sha256 in raw/SPEC_SHA256.txt)",
         "rule_added_before_numbers": "E2 verdict on eval139 under both models; trainS replicate reported; pass-on-eval but fail-on-trainS is treated as FAIL in E4",
         "meta_eval139": json.loads(str(ZE["meta"])), "meta_trainS": json.loads(str(ZT["meta"]))}
    R["E1"] = e1(ZE, "D:/refcv7_route_bin/2026-10-04/eval_s0g.npz")
    print(f"[E1] done {time.time() - t0:.0f}s", flush=True)
    R["E3"] = {"eval139": e3(ZE, "eval139"), "trainS": e3(ZT, "trainS")}
    print(f"[E3] done {time.time() - t0:.0f}s", flush=True)
    R["E2"] = {"eval139": e2(ZE, "eval139"), "trainS": e2(ZT, "trainS")}
    print(f"[E2] done {time.time() - t0:.0f}s", flush=True)
    # ---- E4 ---------------------------------------------------------------------------------------- #
    e4 = {}
    for v in VARS:
        e2_ev = all(R["E2"]["eval139"]["models"][m]["bar"][v]["pass"] for m in ("ols", "hgb"))
        e2_tr = all(R["E2"]["trainS"]["models"][m]["bar"][v]["pass"] for m in ("ols", "hgb"))
        e3r = R["E3"]["eval139"]["variants"][v]
        h15 = R["E1"]["all_windows"]["arms"][f"TP_{v}"]["turn"]["h15"][0]
        e4[v] = {"E2_eval139": e2_ev, "E2_trainS": e2_tr, "E2_used": bool(e2_ev and e2_tr),
                 "E3_as_registered": e3r["E3_as_registered_pass"], "E3_A1": e3r["E3_A1_pass"],
                 "E3_failure_localised_on_curved_support": bool((not e3r["E3_as_registered_pass"]) and e3r["E3_A1_pass"]),
                 "E3_used": bool(e3r["E3_as_registered_pass"] or e3r["E3_A1_pass"]),
                 "TP_turn_h15_allwindows": h15}
        e4[v]["eligible"] = bool(e4[v]["E2_used"] and e4[v]["E3_used"])
    elig = [v for v in VARS if e4[v]["eligible"]]
    if elig:
        Ls = {"A30": 30, "A50": 50, "A80": 80, "B": 80}
        best = max(elig, key=lambda v: (round(e4[v]["TP_turn_h15_allwindows"], 6), Ls[v]))
    else:
        best = None
    R["E4"] = {"per_variant": e4, "eligible": elig, "recommended": best,
               "rule": "among variants passing E2 (eval139, both models; trainS replicate) and E3 (as registered, or E3-A1 when the failure is localised on curved-support windows), the highest TP heading-within-15 deg on GT-turn windows (all eval139 windows); ties -> larger L"}
    R["wall_s"] = round(time.time() - t0, 1)
    json.dump(R, open(out_p, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[s2_analyze] done {R['wall_s']}s; recommended={best}; eligible={elig}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
