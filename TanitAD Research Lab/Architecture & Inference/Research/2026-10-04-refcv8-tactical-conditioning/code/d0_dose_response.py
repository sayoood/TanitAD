"""SPEC_D0_DOSE_RESPONSE.md -- how good must the tactical layer be before conditioning SELECTION on it pays?

Zero GPU. Re-scores refcv7-r101-s0 @ 50,400's BANKED fans (route package capture) with SIMULATED tactical posteriors of
known quality. Not a lever, not a training result. Refuses to run if the SPEC on disk differs from its registered hash.

Run (CPU):
  PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe code/d0_dose_response.py
Writes raw/d0_dose_response.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
ROUTE = Path(r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
             r"2026-10-04-refcv7-route-following/code")
sys.path.insert(0, str(ROUTE))
import route_metrics as rm  # noqa: E402  (the route package's own definitions, imported -- never re-typed)

BIN = Path("D:/refcv7_route_bin/2026-10-04")
MD5 = {"eval_s0g": "1c48a53e84e3a6296004efc0a81a0a64", "eval_s1": "5f783c842497b9dff0d878c98b23f883",
       "train_s0": "53e06ec57663474905cef487f2ebc4a9"}
B, SEED, R = 2000, 0, 5
Q_GRID = (1.0 / 3.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
BETA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, "HARD")
SIG_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
LAM_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
LOGP_FLOOR = math.log(1e-4)
SLOT_T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
LON5 = ("HOLD", "CREEP", "BRAKE", "ACCEL", "CRUISE")
#: refcv7's v7 longitudinal vocabulary -> the 5 kinematic classes (stated; FOLLOW has no lead input here)
V7LON_TO_5 = {"FOLLOW": "CRUISE", "CRUISE": "CRUISE", "YIELD_MERGE": "BRAKE", "BRAKE_TO": "BRAKE",
              "CREEP": "CREEP", "HOLD": "HOLD", "ADAPT_SPEED_FOR_CURVE": "BRAKE", "ACCELERATE": "ACCEL"}
V7LON = ('FOLLOW', 'CRUISE', 'YIELD_MERGE', 'BRAKE_TO', 'CREEP', 'HOLD', 'ADAPT_SPEED_FOR_CURVE', 'ACCELERATE')


def spec_check() -> str:
    reg = (PKG / "raw" / "SPEC_SHA256.txt").read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256((PKG / "SPEC_D0_DOSE_RESPONSE.md").read_bytes()).hexdigest()
    if got != reg:
        raise SystemExit(f"[d0] SPEC on disk {got[:12]} != registered {reg[:12]} -- refusing")
    return got


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(tag: str) -> dict:
    p = BIN / f"{tag}.npz"
    m = md5(p)
    if m != MD5[tag]:
        raise SystemExit(f"[d0] {p} md5 {m} != {MD5[tag]}")
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def masked_argmax(score, keep, fallback):
    r = np.where(keep, score, -np.inf)
    empty = ~keep.any(1)
    r = np.where(empty[:, None], np.where(fallback, score, -np.inf), r)
    return r.argmax(1)


def derive(z: dict) -> dict:
    d = {"W": len(z["sel_idx"]), "ep": np.asarray(z["win_sha12"]).astype(str)}
    cls, th_gt = rm.gt_class(z["gt"], z["gt_valid"])
    d["cls"], d["th_gt"] = cls, th_gt
    clsd = np.isin(cls, ["turnL", "turnR", "straight", "gentle"])
    d["target"] = np.where(clsd, rm.dir_class(th_gt), 9)
    d["th_c"] = rm.terminal_heading(z["fan"])
    d["dir_c"] = rm.dir_class(d["th_c"])
    d["ade_c"], _, _ = rm.ade_fde(z["fan"], z["gt"], z["gt_valid"])
    d["reach"] = z["reach"].astype(bool)
    d["P_c"] = rm.path_length(z["fan"])                      # [W, N] 6-s arc
    d["P_g"] = rm.path_length(z["gt"])                       # [W]
    d["pg_ok"] = z["gt_valid"][:, -1].astype(bool) & (d["P_g"] > 0.5)
    d["C_c"] = np.linalg.norm(z["fan"][:, :, -1, :].astype(np.float64), axis=-1)   # 6-s chord
    d["masks"] = {"all": d["target"] != 9, "turn": np.isin(cls, ["turnL", "turnR"]),
                  "straight": cls == "straight"}
    return d


def per_window(pick, z, d) -> dict:
    w = np.arange(d["W"])
    ade = d["ade_c"][w, pick]
    dir_ok = (d["dir_c"][w, pick] == d["target"]).astype(float)
    head = (np.abs(rm.wrap(d["th_c"][w, pick] - d["th_gt"])) <= math.radians(rm.HEAD_AGREE_DEG)).astype(float)
    prog = np.where(d["pg_ok"], np.abs(d["P_c"][w, pick] / np.maximum(d["P_g"], 1e-6) - 1.0), np.nan)
    return {"ade": ade, "dir": dir_ok, "head": head, "prog": prog}


# --------------------------------------------------------------------------------------------- #
# simulated posteriors / constraints (the SAME draws are used for eval_s0g and eval_s1)          #
# --------------------------------------------------------------------------------------------- #
def lat_logp(d, q, r, tag_seed):
    """[W, 3] log posterior over (R, K, L) indexed by class+1, calibrated accuracy q; uniform where unclassified."""
    W = d["W"]
    rng = np.random.default_rng(10_000 * tag_seed + 100 * r + int(round(q * 1000)))
    y = d["target"]
    ok = y != 9
    flip = rng.random(W) >= q
    alt = rng.integers(0, 2, W)                               # which of the two other classes
    yy = np.where(ok, y, 0)
    others = np.stack([np.array([c for c in (-1, 0, 1) if c != v]) for v in yy])   # [W, 2]
    yhat = np.where(flip, others[np.arange(W), alt], yy)
    p = np.full((W, 3), (1.0 - q) / 2.0)
    p[np.arange(W), yhat + 1] = q
    p[~ok] = 1.0 / 3.0
    return np.maximum(np.log(np.maximum(p, 1e-300)), LOGP_FLOOR), yhat


def s_lat(z, d, logp3, yhat, beta):
    lp = np.take_along_axis(logp3, (d["dir_c"] + 1).astype(np.int64), axis=1)    # [W, N]
    if beta == "HARD":
        ok = (d["target"] != 9)[:, None]
        keep = d["reach"] & ((d["dir_c"] == yhat[:, None]) | ~ok)
        return masked_argmax(z["s_e9"], keep, d["reach"])
    return masked_argmax(z["s_e9"] + float(beta) * lp, d["reach"], d["reach"])


def lon_term(Pc, Phat, valid):
    t = np.abs(np.log((Pc + 1.0) / (Phat[:, None] + 1.0)))
    return np.where(valid[:, None], t, 0.0)


def phat_noisy(d, sigma, r, tag_seed):
    rng = np.random.default_rng(20_000 * tag_seed + 100 * r + int(round(sigma * 1000)))
    return d["P_g"] * np.exp(sigma * rng.standard_normal(d["W"]))


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def mean_ade(pick, d, mask):
    a = d["ade_c"][np.arange(d["W"]), pick]
    m = mask & np.isfinite(a)
    return float(a[m].mean())


# --------------------------------------------------------------------------------------------- #
def boot_delta(x_arm, x_ref, d, mask, draws):
    """paired: mean over masked windows of (arm - ref), episode-cluster bootstrap."""
    x = np.asarray(x_arm, np.float64) - np.asarray(x_ref, np.float64)
    pt, bs, _ = rm.boot_mean(x, d["ep"], mask=mask, draws=draws)
    return {"d": None if not np.isfinite(pt) else round(float(pt), 4), "ci": [None if c is None else round(c, 4)
                                                                             for c in rm.ci95(bs)]}


def summarise(pw_arm_draws, pw_v0, d, draws):
    """pw_arm_draws: list over noise draws of per-window dicts -> averaged per window, then bootstrapped."""
    avg = {k: np.nanmean(np.stack([p[k] for p in pw_arm_draws]), axis=0) for k in pw_arm_draws[0]}
    out = {}
    for cname in ("all", "turn", "straight"):
        m = d["masks"][cname]
        out[f"dADE_{cname}"] = boot_delta(avg["ade"], pw_v0["ade"], d, m, draws)
    out["d_turn_dir"] = boot_delta(avg["dir"], pw_v0["dir"], d, d["masks"]["turn"], draws)
    out["turn_dir"] = round(float(avg["dir"][d["masks"]["turn"]].mean()), 4)
    out["turn_head15"] = round(float(avg["head"][d["masks"]["turn"]].mean()), 4)
    pr = avg["prog"][d["masks"]["all"]]
    out["prog_err_median_all"] = round(float(np.nanmedian(pr)), 4)
    # across-draw SD of the all-window and turn mean dADE (noise variance of the simulation itself)
    for cname in ("all", "turn"):
        m = d["masks"][cname]
        per = [np.nanmean((p["ade"] - pw_v0["ade"])[m]) for p in pw_arm_draws]
        out[f"draw_sd_dADE_{cname}"] = round(float(np.std(per)), 4)
    return out


def clears_r1(s0, s1):
    t = s0["dADE_turn"]
    ok = (t["ci"][1] is not None and t["ci"][1] < 0 and s0["d_turn_dir"]["ci"][0] is not None
          and s0["d_turn_dir"]["ci"][0] > 0 and s0["dADE_straight"]["d"] <= 0.05 and s0["dADE_all"]["d"] <= 0
          and s1["dADE_turn"]["d"] < 0 and s1["d_turn_dir"]["d"] > 0)
    return bool(ok)


def clears_r2(s0, s1):
    a = s0["dADE_all"]
    return bool(a["d"] <= -0.303 and a["ci"][1] is not None and a["ci"][1] < 0 and s1["dADE_all"]["d"] < 0)


def main():
    t0 = time.time()
    spec = spec_check()
    Z = {k: load(k) for k in ("train_s0", "eval_s0g", "eval_s1")}
    D = {k: derive(v) for k, v in Z.items()}
    ze, zs1, zt = Z["eval_s0g"], Z["eval_s1"], Z["train_s0"]
    de, ds1, dt = D["eval_s0g"], D["eval_s1"], D["train_s0"]
    res = {"spec_sha256": spec, "inputs_md5": MD5, "controls": {}, "tier": "OPEN-LOOP, banked fans, SIMULATED posteriors"}

    # ---- instrument controls --------------------------------------------------------------- #
    c = res["controls"]
    for tag in ("eval_s0g", "eval_s1", "train_s0"):
        z, d = Z[tag], D[tag]
        a = masked_argmax(z["s_e9"], d["reach"], d["reach"])
        c[f"C0_{tag}_e9_argmax_mismatch"] = int((a != z["sel_idx"]).sum())
    c["C0_same_windows_s0g_s1"] = bool(np.array_equal(ze["win_sha12"], zs1["win_sha12"])
                                       and np.array_equal(ze["win_t"], zs1["win_t"]))
    c["C0_n_classified_turn_straight_eval"] = [int(de["masks"][k].sum()) for k in ("all", "turn", "straight")]
    if any(c[f"C0_{t}_e9_argmax_mismatch"] for t in ("eval_s0g", "eval_s1", "train_s0")) or not c["C0_same_windows_s0g_s1"]:
        raise SystemExit(f"[d0] instrument control failed: {c}")
    draws = rm.make_draws(de["ep"], B, SEED)
    v0_e, v0_s1 = per_window(ze["sel_idx"], ze, de), per_window(zs1["sel_idx"], zs1, ds1)

    # added control (not in SPEC section 1; written before any number): S-LAT(q=1, HARD) must reproduce A4's B1,
    # and a hard 10 % progress filter must reproduce A4's B3 (RESULT.md sec. 2.2: -0.104 / -0.566 and -0.606).
    lp1, yh1 = lat_logp(de, 1.0, 0, 1)
    pB1 = s_lat(ze, de, lp1, yh1, "HARD")
    keep_b3 = de["reach"] & ((np.abs(de["P_c"] / np.maximum(de["P_g"], 1e-6)[:, None] - 1.0) <= 0.10)
                             | ~de["pg_ok"][:, None])
    pB3 = masked_argmax(ze["s_e9"], keep_b3, de["reach"])
    sB1 = summarise([per_window(pB1, ze, de)], v0_e, de, draws)
    sB3 = summarise([per_window(pB3, ze, de)], v0_e, de, draws)
    c["C1_B1_reproduction"] = {"all": sB1["dADE_all"]["d"], "turn": sB1["dADE_turn"]["d"],
                               "expected": [-0.104, -0.566]}
    c["C1_B3_reproduction"] = {"all": sB3["dADE_all"]["d"], "turn": sB3["dADE_turn"]["d"],
                               "expected": [-0.606, -0.892]}
    if abs(sB1["dADE_all"]["d"] + 0.104) > 2e-3 or abs(sB1["dADE_turn"]["d"] + 0.566) > 2e-3 \
            or abs(sB3["dADE_all"]["d"] + 0.606) > 2e-3:
        raise SystemExit(f"[d0] C1 failed -- the instrument does not reproduce A4's bounds: {c}")

    # ---- S-LAT(q) ---------------------------------------------------------------------------- #
    lat_rows = {}
    for q in Q_GRID:
        # fit beta on TRAIN (mean ADE over classified windows, averaged over R draws)
        fit = {}
        for beta in BETA_GRID:
            vals = []
            for r in range(R):
                lp, yh = lat_logp(dt, q, r, 3)
                vals.append(mean_ade(s_lat(zt, dt, lp, yh, beta), dt, dt["masks"]["all"]))
            fit[str(beta)] = float(np.mean(vals))
        best = min(fit, key=lambda k: (fit[k], 0 if k == "HARD" else float(k)))
        beta = "HARD" if best == "HARD" else float(best)
        pw0, pw1 = [], []
        for r in range(R):
            lp, yh = lat_logp(de, q, r, 1)          # the SAME draw for both sampler seeds (same windows)
            pw0.append(per_window(s_lat(ze, de, lp, yh, beta), ze, de))
            pw1.append(per_window(s_lat(zs1, ds1, lp, yh, beta), zs1, ds1))
        s0, s1 = summarise(pw0, v0_e, de, draws), summarise(pw1, v0_s1, ds1, draws)
        lat_rows[f"{q:.3f}"] = {"q": q, "beta_fit": beta, "train_fit_mean_ade": fit, "eval_s0": s0, "eval_s1": s1,
                                "clears_R1": clears_r1(s0, s1)}
    res["S_LAT"] = lat_rows
    qs = [lat_rows[k]["q"] for k in lat_rows if lat_rows[k]["clears_R1"] and lat_rows[k]["q"] > 0.34]
    res["R1_q_star"] = min(qs) if qs else None
    res["R3_C_U_all"] = lat_rows[f"{1/3:.3f}"]["eval_s0"]["dADE_all"]

    # ---- S-LON(sigma) ------------------------------------------------------------------------ #
    def lon_pick(z, d, Phat, valid, lam):
        return masked_argmax(z["s_e9"] - lam * lon_term(d["P_c"], Phat, valid), d["reach"], d["reach"])

    lon_rows = {}
    for sig in SIG_GRID:
        fit = {}
        for lam in LAM_GRID:
            vals = [mean_ade(lon_pick(zt, dt, phat_noisy(dt, sig, r, 3), dt["pg_ok"], lam), dt, dt["masks"]["all"])
                    for r in range(R)]
            fit[str(lam)] = float(np.mean(vals))
        lam = float(min(fit, key=lambda k: (fit[k], float(k))))
        pw0, pw1 = [], []
        for r in range(R):
            ph = phat_noisy(de, sig, r, 1)
            pw0.append(per_window(lon_pick(ze, de, ph, de["pg_ok"], lam), ze, de))
            pw1.append(per_window(lon_pick(zs1, ds1, ph, ds1["pg_ok"], lam), zs1, ds1))
        s0, s1 = summarise(pw0, v0_e, de, draws), summarise(pw1, v0_s1, ds1, draws)
        lon_rows[f"{sig:.2f}"] = {"sigma": sig, "lambda_fit": lam, "train_fit_mean_ade": fit, "eval_s0": s0,
                                  "eval_s1": s1, "clears_R2": clears_r2(s0, s1)}
    res["S_LON"] = lon_rows
    sg = [lon_rows[k]["sigma"] for k in lon_rows if lon_rows[k]["clears_R2"]]
    res["R2_sigma_star"] = max(sg) if sg else None

    # controls: C-SH (shuffled P-hat, sigma 0) and C-CV (6 s x v0)
    def fit_lam(fn_train):
        f = {str(l): mean_ade(fn_train(l), dt, dt["masks"]["all"]) for l in LAM_GRID}
        return float(min(f, key=lambda k: (f[k], float(k)))), f

    for name, mk in (("C_SH", lambda d, tag: (d["P_g"][derangement(d["W"], 7 + tag)],
                                              d["pg_ok"] & d["pg_ok"][derangement(d["W"], 7 + tag)])),
                     ("C_CV", lambda d, tag: (6.0 * np.asarray(Z_of[id(d)]["v0"], np.float64),
                                              np.ones(d["W"], bool)))):
        Z_of = {id(dt): zt, id(de): ze, id(ds1): zs1}
        pht, vt = mk(dt, 3)
        lam, f = fit_lam(lambda l: lon_pick(zt, dt, pht, vt, l))
        phe, ve = mk(de, 1)
        p0 = per_window(lon_pick(ze, de, phe, ve, lam), ze, de)
        p1 = per_window(lon_pick(zs1, ds1, phe, ve, lam), zs1, ds1)
        res[name] = {"lambda_fit": lam, "train_fit_mean_ade": f, "eval_s0": summarise([p0], v0_e, de, draws),
                     "eval_s1": summarise([p1], v0_s1, ds1, draws)}
    g0 = lon_rows["0.00"]["eval_s0"]["dADE_all"]["d"]
    gcv = res["C_CV"]["eval_s0"]["dADE_all"]["d"]
    res["R4_cv_share_of_oracle_gain"] = round(gcv / g0, 4) if (g0 and g0 < 0) else None

    # ---- S-JOINT(q, sigma) --------------------------------------------------------------- #
    joint = {}
    for q in (0.6, 0.7, 0.8, 0.9, 1.0):
        for sig in (0.0, 0.1, 0.2, 0.3):
            fit = {}
            for beta in BETA_GRID[:-1]:
                for lam in LAM_GRID:
                    vals = []
                    for r in range(R):
                        lp, _ = lat_logp(dt, q, r, 3)
                        lpk = np.take_along_axis(lp, (dt["dir_c"] + 1).astype(np.int64), axis=1)
                        s = zt["s_e9"] + beta * lpk - lam * lon_term(dt["P_c"], phat_noisy(dt, sig, r, 3), dt["pg_ok"])
                        vals.append(mean_ade(masked_argmax(s, dt["reach"], dt["reach"]), dt, dt["masks"]["all"]))
                    fit[(beta, lam)] = float(np.mean(vals))
            beta, lam = min(fit, key=lambda k: (fit[k], k))
            pw0, pw1 = [], []
            for r in range(R):
                lp, _ = lat_logp(de, q, r, 1)
                ph = phat_noisy(de, sig, r, 1)
                for z, d, acc in ((ze, de, pw0), (zs1, ds1, pw1)):
                    lpk = np.take_along_axis(lp, (d["dir_c"] + 1).astype(np.int64), axis=1)
                    s = z["s_e9"] + beta * lpk - lam * lon_term(d["P_c"], ph, d["pg_ok"])
                    acc.append(per_window(masked_argmax(s, d["reach"], d["reach"]), z, d))
            joint[f"q{q:.2f}_s{sig:.2f}"] = {"beta": beta, "lambda": lam, "eval_s0": summarise(pw0, v0_e, de, draws),
                                            "eval_s1": summarise(pw1, v0_s1, ds1, draws)}
    res["S_JOINT"] = joint

    # ---- real-model rows ------------------------------------------------------------------- #
    def p3_of(z):
        pl = z["p_lat"].astype(np.float64)
        p3 = np.stack([pl[:, rm.LAT_SIDE == s].sum(1) for s in (-1, 0, 1)], axis=1)
        return p3 / np.maximum(p3.sum(1, keepdims=True), 1e-12)

    hl = {}
    for beta in BETA_GRID[:-1]:
        lpt = np.maximum(np.log(np.maximum(p3_of(zt), 1e-300)), LOGP_FLOOR)
        hl[str(beta)] = mean_ade(s_lat(zt, dt, lpt, p3_of(zt).argmax(1) - 1, beta), dt, dt["masks"]["all"])
    bH = float(min(hl, key=lambda k: (hl[k], float(k))))
    out_h = {}
    for tag, z, d, v0 in (("eval_s0", ze, de, v0_e), ("eval_s1", zs1, ds1, v0_s1)):
        p3 = p3_of(z)
        lp = np.maximum(np.log(np.maximum(p3, 1e-300)), LOGP_FLOOR)
        out_h[tag] = summarise([per_window(s_lat(z, d, lp, p3.argmax(1) - 1, bH), z, d)], v0, d, draws)
    p3e = p3_of(ze)
    yhat = p3e.argmax(1) - 1
    m_all, m_turn = de["masks"]["all"], de["masks"]["turn"]
    res["H_LAT"] = {"beta_fit": bH, "train_fit_mean_ade": hl, **out_h,
                    "acc3_all_classified": round(float((yhat == de["target"])[m_all].mean()), 4),
                    "acc3_turn": round(float((yhat == de["target"])[m_turn].mean()), 4),
                    "mean_p_true_turn": round(float(p3e[np.arange(de["W"]), np.clip(de["target"], -1, 1) + 1][m_turn]
                                                    .mean()), 4)}

    def chord_term(d, z, lam):
        g6 = np.linalg.norm(z["g_tac"][:, 2, :2].astype(np.float64), axis=-1)
        ok = np.isfinite(g6)
        t = np.abs(np.log((d["C_c"] + 1.0) / (np.where(ok, g6, 0.0)[:, None] + 1.0)))
        return np.where(ok[:, None], t, 0.0), g6

    if "g_tac" in zt and np.isfinite(zt["g_tac"]).all():
        f = {}
        for lam in LAM_GRID:
            t, _ = chord_term(dt, zt, lam)
            f[str(lam)] = mean_ade(masked_argmax(zt["s_e9"] - lam * t, dt["reach"], dt["reach"]), dt,
                                   dt["masks"]["all"])
        lamE = float(min(f, key=lambda k: (f[k], float(k))))
        oe = {}
        for tag, z, d, v0 in (("eval_s0", ze, de, v0_e), ("eval_s1", zs1, ds1, v0_s1)):
            if "g_tac" not in z or not np.isfinite(z["g_tac"]).all():
                oe[tag] = "g_tac not captured on this pass"
                continue
            t, g6 = chord_term(d, z, lamE)
            oe[tag] = summarise([per_window(masked_argmax(z["s_e9"] - lamE * t, d["reach"], d["reach"]), z, d)],
                                v0, d, draws)
        _, g6e = chord_term(de, ze, lamE)
        cg = np.linalg.norm(ze["gt"][:, -1, :].astype(np.float64), axis=-1)
        okc = de["pg_ok"] & np.isfinite(g6e)
        res["H_E8"] = {"lambda_fit": lamE, "train_fit_mean_ade": f, **oe,
                       "rel_chord_err_median": round(float(np.median(np.abs(g6e[okc] / np.maximum(cg[okc], 1e-6) - 1))),
                                                     4),
                       "n": int(okc.sum())}
    else:
        res["H_E8"] = "g_tac not captured on train_s0"

    # ---- section 4: refcv7 baselines for R8-4 (i) and (iii) ------------------------------- #
    res["baseline_R8_4"] = baselines(ze, de)
    res["wall_s"] = round(time.time() - t0, 1)
    out = PKG / "raw" / "d0_dose_response.json"
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"[d0] wrote {out} in {res['wall_s']} s; R1 q* = {res['R1_q_star']}; R2 sigma* = {res['R2_sigma_star']}")


def lon_class_from_path(P, v0):
    """[W, 8, 2] slot positions (+ origin) and v0 -> 5-way kinematic class index (LON5), D4's literals, first event."""
    W = P.shape[0]
    Q = np.concatenate([np.zeros((W, 1, 2)), P.astype(np.float64)], axis=1)
    dtv = np.diff(np.concatenate([[0.0], SLOT_T]))
    v = np.linalg.norm(np.diff(Q, axis=1), axis=-1) / dtv           # [W, 8] segment speeds
    v0 = np.asarray(v0, np.float64)
    vmax, vmin = v.max(1), v.min(1)
    out = np.full(W, LON5.index("CRUISE"))
    br = v <= (v0[:, None] - 1.5)
    ac = v >= (v0[:, None] + 1.5)
    fb = np.where(br.any(1), br.argmax(1), 99)
    fa = np.where(ac.any(1), ac.argmax(1), 99)
    out = np.where((fb < 99) & (fb <= fa), LON5.index("BRAKE"), out)
    out = np.where((fa < 99) & (fa < fb), LON5.index("ACCEL"), out)
    out = np.where(vmax <= 2.0, LON5.index("CREEP"), out)
    out = np.where((v0 <= 0.5) & (vmax <= 0.5), LON5.index("HOLD"), out)
    return out


def f1_report(pred, true, n_cls, names):
    rep, f1s = {}, []
    for k in range(n_cls):
        tp = int(((pred == k) & (true == k)).sum())
        fp = int(((pred == k) & (true != k)).sum())
        fn = int(((pred != k) & (true == k)).sum())
        n = tp + fn
        p = tp / (tp + fp) if tp + fp else None
        r = tp / n if n else None
        f1 = (2 * p * r / (p + r)) if (p and r) else (0.0 if n else None)
        if f1 is not None:
            f1s.append(f1)
        rep[names[k]] = {"n": n, "recall": None if r is None else round(r, 4),
                         "precision": None if p is None else round(p, 4), "f1": None if f1 is None else round(f1, 4)}
    rep["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else None
    rep["accuracy"] = round(float((pred == true).mean()), 4)
    maj = np.bincount(true, minlength=n_cls).argmax()
    rep["majority_control_accuracy"] = round(float((true == maj).mean()), 4)
    rep["majority_class"] = names[maj]
    return rep


def baselines(z, d):
    out = {}
    m = d["masks"]["all"]
    p3 = np.stack([z["p_lat"][:, rm.LAT_SIDE == s].sum(1) for s in (-1, 0, 1)], axis=1)
    yhat = p3.argmax(1)                                        # 0=R 1=K 2=L
    ytrue = d["target"] + 1
    out["i_lat3_all_classified"] = f1_report(yhat[m], ytrue[m].astype(int), 3, ("R", "K", "L"))
    mt = d["masks"]["turn"]
    out["i_lat3_turn_side_correct"] = round(float((yhat[mt] == ytrue[mt]).mean()), 4)
    okl = z["gt_valid"][:, -1].astype(bool)
    lon_true = lon_class_from_path(z["gt"], z["v0"])
    lon_pred = np.array([LON5.index(V7LON_TO_5[V7LON[i]]) for i in z["p_lon"].argmax(1)])
    out["i_lon5_valid_6s"] = f1_report(lon_pred[okl], lon_true[okl], 5, LON5)
    out["i_lon_mapping"] = V7LON_TO_5
    # (iii) consistency with the tactical head's 3-way argmax (side): fan / top-8 / pick
    side = yhat - 1
    dc = d["dir_c"]
    W = d["W"]
    reach = d["reach"]
    cons_fan = ((dc == side[:, None]) & reach).sum(1) / np.maximum(reach.sum(1), 1)
    s = np.where(reach, z["s_e9"], -np.inf)
    top8 = np.argsort(-s, axis=1, kind="mergesort")[:, :8]
    cons_top8 = (np.take_along_axis(dc, top8, axis=1) == side[:, None]).mean(1)
    cons_pick = (dc[np.arange(W), z["sel_idx"]] == side).astype(float)
    pick_gt = (dc[np.arange(W), z["sel_idx"]] == d["target"]).astype(float)
    for nm, mk in (("all_classified", m), ("turn", mt)):
        out[f"iii_{nm}"] = {"fan_share_consistent_with_tac_argmax": round(float(cons_fan[mk].mean()), 4),
                            "top8_share_consistent": round(float(cons_top8[mk].mean()), 4),
                            "pick_consistent_with_tac_argmax": round(float(cons_pick[mk].mean()), 4),
                            "pick_dir_correct_vs_gt": round(float(pick_gt[mk].mean()), 4),
                            "n": int(mk.sum())}
    return out


if __name__ == "__main__":
    main()
