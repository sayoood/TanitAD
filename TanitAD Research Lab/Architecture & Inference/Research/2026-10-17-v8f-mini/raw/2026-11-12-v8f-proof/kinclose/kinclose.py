#!/usr/bin/env python
"""kinclose.py -- tool kit of the pre-registered card PREREG_V8F_PROOF_ADD88 (KINCLOSE), tier T0 (reads on logged frames; no rollout).

A closing read from the reader's gap over a window together with the measured ego speed, with no fit. Definitions: KINCLOSE_DEFS.md.
Subcommands: ego (clip files -> ego_v_eval.npz; torch only here), read (controls k0..k4, then the arms and the names; numpy only), show (prints names and a table
after checking that the controls passed and that the result is bound to them), and --selftest. Exit codes of read: 0 all controls pass, 1 a control failed, 2 refused."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CARD = "PREREG_V8F_PROOF_ADD88"
WS = (5, 10, 25, 50)
H_SCORE = 25
BOOT_B = 2000
BOOT_SEED = 88
BAR_R_SEED = 0.30
REQ_R = 0.9
ROW147_SD = 1.24
K0_TOL = 1e-6
K3_TOL = 1e-6
K4_MARGIN = 0.05
K4_BAND = 0.10
SHIFT = 10
V_MEAN_LO, V_MEAN_HI = 1.0, 40.0
RUN_FIVE = 5
T_FRAMES = 101
FAULTS = set()  # in-process regression switches of the self-test; the command line never sets them
CF = None
CL = None


# ============================================================================================ small helpers
def sha256_file(p):
    return hashlib.sha256(io.open(p, "rb").read()).hexdigest()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def write_json(path, obj):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(clean(obj), f, indent=1)


def read_json(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def refuse(msg):
    print("REFUSED: " + msg)
    raise SystemExit(2)


def load_tools(path=None):
    """import closeread_fit and closest (numpy only) from the closeread tools folder."""
    global CF, CL
    cands = [p for p in (path, os.environ.get("KINCLOSE_CRTOOLS"), os.path.join(HERE, "..", "closeread"), os.path.join(HERE, "..", "closeread", "tools")) if p]
    for p in cands:
        if os.path.isfile(os.path.join(p, "closeread_fit.py")) and os.path.isfile(os.path.join(p, "closest.py")):
            if p not in sys.path:
                sys.path.insert(0, p)
            import closeread_fit as cf
            import closest as cl
            CF, CL = cf, cl
            return
    refuse("closeread tools (closeread_fit.py, closest.py) not found in: " + ", ".join(cands))


# ============================================================================================ slope and the arms
def slope_rows(x, m, ci, fr, W, dt):
    """least-squares slope against time of x over frames fr-W+1 .. fr of clip ci, on the frames where m is true (and the frame is >= 0); -> slope [n] (NaN below 2 points), count [n]."""
    off = np.arange(W - 1, -1, -1)
    idx = fr[:, None] - off[None, :]
    inb = idx >= 0
    idc = np.where(inb, idx, 0)
    M = m[ci[:, None], idc] & inb
    X = np.where(M, x[ci[:, None], idc], 0.0)
    t = idc.astype(np.float64) * dt
    n = M.sum(1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        tb = (M * t).sum(1) / n
        xb = X.sum(1) / n
        dtm = np.where(M, t - tb[:, None], 0.0)
        den = (dtm ** 2).sum(1)
        num = (dtm * np.where(M, X - xb[:, None], 0.0)).sum(1)
        sl = num / den
    sl = np.where((n >= 2) & (den > 0), sl, np.nan)
    return sl, M.sum(1)


def ego_path(V, dt, pathmod=None):
    """s(j): cumulative sum of dt * (v(j-1) + v(j)) / 2 from frame 0; pathmod: planted faults of control k1 ('flip', 'shift+', 'shift-')."""
    s = np.zeros_like(V)
    s[:, 1:] = np.cumsum(dt * (V[:, :-1] + V[:, 1:]) / 2.0, axis=1)
    if "ego_sign" in FAULTS:
        s = -s
    T = V.shape[1]
    if "ego_shift" in FAULTS:
        s = s[:, np.clip(np.arange(T) - SHIFT, 0, T - 1)]
    if pathmod == "flip":
        s = -s
    elif pathmod == "shift+":
        s = s[:, np.clip(np.arange(T) - SHIFT, 0, T - 1)]
    elif pathmod == "shift-":
        last = np.maximum(np.isfinite(V).sum(1), 1) - 1          # last frame of each clip (clips have different lengths; V is NaN after the end)
        s = np.take_along_axis(s, np.minimum(np.arange(T)[None, :] + SHIFT, last[:, None]), axis=1)
    return s


def make_ctx(D, ego, ep_map, H=H_SCORE):
    """dense per-clip arrays of one reader seed and the scored rows (rows of closeread_fit.build_rows at H 25)."""
    cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
    nc, T = len(D["clips"]), int(fr.max()) + 1
    f3 = np.stack([np.asarray(D["gap"], np.float64), np.asarray(D["p"], np.float64), np.asarray(D["clos"], np.float64)], 1)
    ok = np.asarray(D["valid"], bool) & np.isfinite(f3).all(1)
    G = np.full((nc, T), np.nan)
    G[cid[ok], fr[ok]] = f3[ok, 0]
    RM = np.zeros((nc, T), bool)
    RM[cid[ok], fr[ok]] = True
    lg = np.asarray(D["lab_gap"], np.float64)
    lp = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
    LG = np.full((nc, T), np.nan)
    LG[cid[lp], fr[lp]] = lg[lp]
    LM = np.zeros((nc, T), bool)
    LM[cid[lp], fr[lp]] = True
    LR = np.zeros((nc, T), np.int64)
    LR[cid, fr] = np.asarray(D["lab_run"], np.int64)
    V = np.full((nc, T), np.nan)                                 # NaN after the end of a clip: the clips of the cache have 100, 101 or 104 frames (note N1 of the card)
    nfr = np.zeros(nc, np.int64)
    np.maximum.at(nfr, cid, fr + 1)
    for i, c in enumerate(D["clips"]):
        v = ego[str(c)]
        if len(v) != int(nfr[i]):
            refuse("ego speed of clip %s has %d frames, the dump has %d" % (c, len(v), int(nfr[i])))
        V[i, :len(v)] = v
    rows_cell = CF.build_rows(D, H_SCORE, ep_map)
    rows = CF.build_rows(D, 10, ep_map) if "wrong_frames" in FAULTS else rows_cell
    name2i = {str(c): i for i, c in enumerate(D["clips"])}
    ci = np.asarray([name2i[str(c)] for c in rows["clip"]], np.int64)
    return {"G": G, "RM": RM, "LG": LG, "LM": LM, "LR": LR, "V": V, "rows": rows, "rows_cell": rows_cell, "ci": ci, "fr": np.asarray(rows["frame"], np.int64), "dt": float(D["dt_s"]), "T": T}


def arm(ctx, kind, W, pathmod=None):
    """estimate of one arm at every scored row of the context (NaN where the arm has no value)."""
    dt = ctx["dt"] * (0.5 if "dt_wrong" in FAULTS else 1.0)
    s = ego_path(ctx["V"], dt, pathmod)
    ci, fr = ctx["ci"], ctx["fr"]
    allm = np.ones(s.shape, bool)
    s5 = 0.0 if "no_slope5" in FAULTS else slope_rows(s, allm, ci, fr, 5, dt)[0]
    if kind == "K" and "k_is_s" in FAULTS:
        kind = "S"
    if kind == "K":
        return slope_rows(ctx["G"] + s, ctx["RM"], ci, fr, W, dt)[0] - s5
    if kind == "S":
        return slope_rows(ctx["G"], ctx["RM"], ci, fr, W, dt)[0]
    if kind == "E":
        return slope_rows(s, ctx["RM"], ci, fr, W, dt)[0] - s5
    LG = ctx["LG"]
    if kind == "B" and "b5_offset" in FAULTS:
        LG = np.concatenate([LG[:, :1], LG[:, :-1]], axis=1)
    if kind == "B":
        sl, cnt = slope_rows(LG + s, ctx["LM"], ci, fr, W, dt)
        return np.where(cnt == W, sl - s5, np.nan)
    if kind == "BS":
        sl, cnt = slope_rows(LG, ctx["LM"], ci, fr, W, dt)
        return np.where(cnt == W, sl, np.nan)
    raise ValueError(kind)


def cell_est(ctxs, est_paths):
    out = []
    for c, p in zip(ctxs, est_paths):
        if "wrong_frames" in FAULTS:
            out.append(np.full(len(c["fr"]), np.nan))
        else:
            out.append(CL.ClosingEstimator.load(p).apply(c["rows_cell"]["X"]))
    return out


# ============================================================================================ summaries (closeread_fit routines by import)
def sub_rows(rows, m):
    return {k: np.asarray(rows[k])[m] for k in ("ep", "y", "frame", "clip")}


def lag1(e, rows):
    if len(e) < 3:
        return float("nan")
    same = (rows["clip"][1:] == rows["clip"][:-1]) & (np.asarray(rows["frame"][1:]) - np.asarray(rows["frame"][:-1]) == 1)
    if same.sum() < 3:
        return float("nan")
    return float(CF.pearson(e[:-1][same], e[1:][same]))


def summarize(ests, rows_list, masks, B, seed):
    sr = [sub_rows(r, m) for r, m in zip(rows_list, masks)]
    es = [np.asarray(e)[m] for e, m in zip(ests, masks)]
    n = sum(len(x["y"]) for x in sr)
    if n < 3:
        return None
    c = CF.cell_summary(es, sr, B, seed)
    err = [e - r["y"] for e, r in zip(es, sr)]
    return {"r_pooled": c["r_pooled"], "r_ci": c["r_ci"], "r_per_seed": c["r_per_seed"], "mae": c["mae_est"], "err_sd": float(np.concatenate(err).std()),
            "err_sd_per_seed": [float(x.std()) if len(x) else float("nan") for x in err], "err_lag1": [lag1(x, r) for x, r in zip(err, sr)],
            "n_frames": int(n), "n_per_seed": [int(len(x["y"])) for x in sr]}


def paired(ea, eb, rows_list, masks, B, seed):
    """paired difference of the absolute error, arm A minus arm B, on the same frames: point and percentile 2.5 / 97.5 of the two-level bootstrap."""
    sr = [sub_rows(r, m) for r, m in zip(rows_list, masks)]
    n = sum(len(x["y"]) for x in sr)
    if n < 3:
        return None
    allep = sorted(set(np.concatenate([x["ep"] for x in sr]).tolist()))
    emap = {e: i for i, e in enumerate(allep)}
    stats = []
    for a, b, r in zip(ea, eb, sr):
        epi = np.asarray([emap[e] for e in r["ep"]], np.int64)
        sa = CF.episode_sums(np.asarray(a)[masks[len(stats)]], r["y"], epi, len(allep))
        sb = CF.episode_sums(np.asarray(b)[masks[len(stats)]], r["y"], epi, len(allep))
        sa["say"] = sb["sae"]
        stats.append(sa)
    d, lo, hi = CF.pair_summary(stats, B, seed)
    return {"diff_abs_err": d, "ci": [lo, hi], "n_frames": int(n)}


def fin(*arrs):
    m = np.isfinite(arrs[0])
    for a in arrs[1:]:
        m = m & np.isfinite(a)
    return m


# ============================================================================================ controls
def mae_of(est, rows_list, masks):
    tot = sum(float(np.abs(e[m] - r["y"][m]).sum()) for e, r, m in zip(est, rows_list, masks))
    n = sum(int(m.sum()) for m in masks)
    return tot / n if n else float("nan")


def control_k0(ctxs):
    worst, n = 0.0, 0
    for c in ctxs:
        b5 = arm(c, "B", 5)
        sel = (c["LR"][c["ci"], c["fr"]] == RUN_FIVE) & np.isfinite(b5)
        n += int(sel.sum())
        if sel.any():
            worst = max(worst, float(np.abs(b5[sel] - c["rows"]["y"][sel]).max()))
    return {"max_abs_diff_ms": worst, "n_frames": n, "tol": K0_TOL, "pass": bool(n > 0 and np.isfinite(worst) and worst <= K0_TOL)}


def control_k1(ctxs, B):
    rl = [c["rows"] for c in ctxs]
    base = [arm(c, "B", 25) for c in ctxs]
    bs = [arm(c, "BS", 25) for c in ctxs]
    mods = {m: [arm(c, "B", 25, m) for c in ctxs] for m in ("shift+", "shift-", "flip")}
    masks = [fin(base[i], bs[i], mods["shift+"][i], mods["shift-"][i], mods["flip"][i]) for i in range(len(ctxs))]
    n = sum(int(m.sum()) for m in masks)
    out = {"n_frames": n}
    if n < 3:
        out.update({"pass": False, "reason": "fewer than 3 frames with a bound value"})
        return out
    pr = paired(base, bs, rl, masks, B, BOOT_SEED)
    out["B25_lower_than_BS25_separated"] = bool(pr["diff_abs_err"] < 0 and pr["ci"][1] < 0)
    mb = mae_of(base, rl, masks)
    for m, e in mods.items():
        out["B25_lower_than_" + m] = bool(mb < mae_of(e, rl, masks))
    out["pass"] = bool(all(v for k, v in out.items() if k.startswith("B25_")))
    return out


def control_k2(ctxs):
    ok_all, vs = True, []
    for c in ctxs:
        mx = {}
        for i, f in zip(c["ci"], c["fr"]):
            mx[int(i)] = max(mx.get(int(i), 0), int(f))
        for i, f in mx.items():
            v = c["V"][i, :f + 1]
            ok_all = ok_all and bool(np.isfinite(v).all() and (v >= 0).all())
        vs.append(c["V"][c["ci"], c["fr"]])
    mean_v = float(np.mean(np.concatenate(vs))) if len(vs) else float("nan")
    lens = np.isfinite(ctxs[0]["V"]).sum(1) if len(ctxs) else np.zeros(0, np.int64)
    return {"frames_per_clip": {str(int(n)): int((lens == n).sum()) for n in sorted(set(lens.tolist()))}, "finite_nonneg_on_used_frames": bool(ok_all), "mean_v_ms": mean_v, "band": [V_MEAN_LO, V_MEAN_HI], "pass": bool(ok_all and np.isfinite(mean_v) and V_MEAN_LO <= mean_v <= V_MEAN_HI)}


def control_k3(ctxs, ks, est_paths, add81_res, add81_ctl):
    out = {"per_seed": {}, "pass": False}
    try:
        a = add81_res["arms"]["T25"]
        fr = add81_ctl["controls"]["t0"]["frames"]
        ests = cell_est(ctxs, est_paths)
        sr = [sub_rows(c["rows"], np.ones(len(c["fr"]), bool)) for c in ctxs]
        good = True
        for i, k in enumerate(ks):
            want = fr["H25_seed%d" % k]
            n = int(len(ctxs[i]["fr"]))
            ok = bool(n == int(want["n_tool"]) == int(want["n_dump"]))
            out["per_seed"]["seed%d" % k] = {"n_frames": n, "n_add81": int(want["n_tool"]), "pass": ok}
            good = good and ok
        c = CF.cell_summary([e for e in ests], sr, 50, BOOT_SEED)
        good = good and bool(c["n_frames"] == int(a["n_frames"]))
        good = good and bool(abs(c["r_pooled"] - float(a["cell_r_pooled"])) <= K3_TOL and abs(c["mae_est"] - float(a["cell_mae"])) <= K3_TOL)
        rs = [float(x) for x in a["cell_r_per_seed"]]
        good = good and len(rs) == len(ks) and all(abs(x - y) <= K3_TOL for x, y in zip(c["r_per_seed"], rs))
        out.update({"n_frames_pooled": c["n_frames"], "n_frames_add81": int(a["n_frames"]), "r_pooled_equal": bool(abs(c["r_pooled"] - float(a["cell_r_pooled"])) <= K3_TOL),
                    "mae_equal": bool(abs(c["mae_est"] - float(a["cell_mae"])) <= K3_TOL), "pass": bool(good and np.isfinite(c["r_pooled"]))})
    except (KeyError, ValueError, TypeError, IndexError) as e:
        out.update({"pass": False, "reason": "%s: %s" % (type(e).__name__, e)})
    return out


def control_k4():
    """planted world of the self-test: a gap read that carries the lead reads better with the ego term; a gap read that is noise alone does not."""
    res = {"pass": True}
    for mode in ("signal", "noise"):
        dumps, ego, epm = gen_world(7, mode)
        ctxs = [make_ctx(dumps[k], ego, epm) for k in sorted(dumps)]
        y = np.concatenate([c["rows"]["y"] for c in ctxs])
        est = {n: np.concatenate([arm(c, n[0], 25) for c in ctxs]) for n in ("K25", "S25", "E25")}
        r = {n: float(CF.pearson(e, y)) for n, e in est.items()}
        mae = {n: float(np.abs(e - y).mean()) for n, e in est.items()}
        if mode == "signal":
            ok = bool(r["K25"] > r["S25"] + K4_MARGIN and mae["K25"] < mae["S25"])
        else:
            ok = bool(abs(r["K25"] - r["E25"]) <= K4_BAND)
        res[mode] = {"r": r, "mae": mae, "pass": ok}
        res["pass"] = bool(res["pass"] and ok)
    return res


# ============================================================================================ names
def names_from(arms, paired_):
    names = []
    kb = paired_["K25-L25"]
    r_ok = all(np.isfinite(x) and x >= BAR_R_SEED for x in arms["K25"]["r_per_seed"])
    names.append("KINCLOSE-K25-BETTER-THAN-OUTPUTS" if (kb and kb["ci"][1] < 0 and r_ok) else "KINCLOSE-K25-NOT-BETTER")
    ks = paired_["K25-S25"]
    if ks and ks["ci"][1] < 0:
        names.append("KINCLOSE-EGO-TERM-HELPS")
    elif ks and ks["ci"][0] > 0:
        names.append("KINCLOSE-EGO-TERM-HURTS")
    else:
        names.append("KINCLOSE-EGO-TERM-NOT-SEPARATED")
    k = arms["K25"]
    if k and k["r_ci"][0] >= REQ_R:
        names.append("KINCLOSE-REACHES-THE-REQUIREMENT")
    elif all(arms["K%d" % w] and arms["K%d" % w]["r_ci"][1] < REQ_R for w in (5, 10, 25, 50)):
        names.append("KINCLOSE-BELOW-THE-REQUIREMENT")
    else:
        names.append("KINCLOSE-REQUIREMENT-NOT-DECIDED")
    b, kon = arms.get("B25"), arms.get("K25_onB")
    if b and kon and b["r_ci"][0] >= REQ_R and kon["r_ci"][1] < REQ_R:
        names.append("KINCLOSE-GAP-READ-IS-THE-LIMIT")
    elif b and b["r_ci"][1] < REQ_R:
        names.append("KINCLOSE-WINDOW-IS-THE-LIMIT")
    return names


def all_arms(ctxs, est_paths, B):
    rl = [c["rows"] for c in ctxs]
    full = [np.ones(len(c["fr"]), bool) for c in ctxs]
    A, est = {}, {}
    for W in WS:
        for kind in ("K", "S", "E", "B", "BS"):
            est["%s%d" % (kind, W)] = [arm(c, kind, W) for c in ctxs]
    est["L25"] = cell_est(ctxs, est_paths)
    for name, e in est.items():
        A[name] = summarize(e, rl, [fin(x) for x in e], B, BOOT_SEED)
    for W in WS:
        mb = [fin(est["B%d" % W][i], est["K%d" % W][i]) for i in range(len(ctxs))]
        A["K%d_onB" % W] = summarize(est["K%d" % W], rl, mb, B, BOOT_SEED)
        A["B%d_onB" % W] = summarize(est["B%d" % W], rl, mb, B, BOOT_SEED)
    P = {}
    pairs = [("K25-L25", "K25", "L25"), ("K25-S25", "K25", "S25"), ("B25-BS25", "B25", "BS25")] + [("K%d-L25" % w, "K%d" % w, "L25") for w in (5, 10, 50)]
    for nm, a, b in pairs:
        masks = [fin(est[a][i], est[b][i]) for i in range(len(ctxs))]
        P[nm] = paired(est[a], est[b], rl, masks, B, BOOT_SEED)
    del full
    return A, P


# ============================================================================================ the read
def input_paths(a, ks):
    ps = {"dump": {k: os.path.join(a.dump_dir, "reader_s%d_score.npz" % k) for k in ks}, "est": [os.path.join(a.est_dir, "closest_L25_s%d.npz" % k) for k in ks]}
    need = list(ps["dump"].values()) + ps["est"] + [a.ego, a.episode_map, a.add81_result, a.add81_controls]
    miss = [p for p in need if not os.path.isfile(p)]
    if miss:
        refuse("missing input: " + ", ".join(os.path.basename(p) for p in miss[:5]))
    return ps


def load_ego(path, clips):
    z = np.load(path, allow_pickle=False)
    if "clips" not in z.files or "v" not in z.files:
        refuse("ego file lacks clips or v")
    if "n" not in z.files:
        refuse("ego file lacks n (frames per clip)")
    names, v, n = [str(c) for c in z["clips"]], np.asarray(z["v"], np.float64), np.asarray(z["n"], np.int64)
    if v.ndim != 2 or v.shape[0] != len(names) or n.shape != (len(names),) or (n < 1).any() or (n > v.shape[1]).any():
        refuse("ego v has shape %s and n has shape %s for %d clips" % (v.shape, n.shape, len(names)))
    d = {c: v[i, :int(n[i])] for i, c in enumerate(names)}
    miss = [c for c in clips if c not in d]
    if miss:
        refuse("ego file lacks %d clip(s), first: %s" % (len(miss), miss[0]))
    return d


def run_read(a, log=print):
    ks = [int(x) for x in a.seeds.split(",")]
    cpath, rpath = os.path.join(a.out_dir, "KINCLOSE_CONTROLS.json"), os.path.join(a.out_dir, "KINCLOSE_RESULT.json")
    if os.path.exists(cpath) or os.path.exists(rpath):
        refuse("an output file exists in " + a.out_dir)
    ps = input_paths(a, ks)
    ep_map = read_json(a.episode_map)
    add81_res, add81_ctl = read_json(a.add81_result), read_json(a.add81_controls)
    dumps = {k: CF.load_dump(ps["dump"][k]) for k in ks}
    ego = load_ego(a.ego, [str(c) for c in dumps[ks[0]]["clips"]])
    ctxs = [make_ctx(dumps[k], ego, ep_map) for k in ks]
    os.makedirs(a.out_dir, exist_ok=True)
    ctl = {"k0": control_k0(ctxs), "k1": control_k1(ctxs, a.boot), "k2": control_k2(ctxs), "k3": control_k3(ctxs, ks, ps["est"], add81_res, add81_ctl), "k4": control_k4()}
    npass = sum(1 for v in ctl.values() if v["pass"])
    allp = bool(npass == len(ctl))
    write_json(cpath, {"card": CARD, "reader_seeds": ks, "controls": ctl, "n_passed": npass, "n_controls": len(ctl), "controls_all_pass": allp})
    for nm, v in ctl.items():
        log("CONTROL %s %s" % (nm, "PASS" if v["pass"] else "FAIL"))
    A, P = all_arms(ctxs, ps["est"], a.boot)
    res = {"card": CARD, "tier": "T0", "reader_seeds": ks, "bootstrap": {"B": a.boot, "seed": BOOT_SEED, "levels": "reader seeds, then evaluation episodes (episode draw shared by the seeds)", "interval": "percentile 2.5 / 97.5"},
           "episode_source": "map file", "controls_sha256": sha256_file(cpath), "controls_all_pass": allp, "row147_error_sd_ms": ROW147_SD}
    if allp or "names_despite_fail" in FAULTS:
        res.update({"arms": A, "paired": P, "names": names_from(A, P)})
        if not allp:
            res["controls_all_pass"] = False
    else:
        res.update({"names": [], "withheld": {"arms": A, "paired": P}})
    write_json(rpath, res)
    log("ZZ KINCLOSE-CONTROLS %d/%d all_pass %s" % (npass, len(ctl), allp))
    return 0 if allp else 1


# ============================================================================================ ego, show
def run_ego(a):
    if os.path.exists(a.out):
        refuse("output exists: " + a.out)
    z = np.load(os.path.join(a.dump_dir, "reader_s0_score.npz"), allow_pickle=False)
    clips = [str(c) for c in z["clips"]]
    import torch  # only here
    vs = []
    for c in clips:
        p = os.path.join(a.eval_cache, "eval", "clips", c + ".pt")
        if not os.path.isfile(p):
            refuse("missing clip file " + p)
        try:
            d = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        except Exception:
            d = torch.load(p, map_location="cpu", weights_only=False)
        if "v" not in d:
            refuse("clip file lacks key v: " + c)
        v = np.asarray(d["v"].detach().cpu().numpy() if hasattr(d["v"], "detach") else d["v"], np.float64)
        if v.ndim != 1 or len(v) < 1:
            refuse("v of %s has shape %s, expected one value per frame" % (c, v.shape))
        vs.append(v.copy())
        del d
    n = np.asarray([len(v) for v in vs], np.int64)
    pad = np.full((len(vs), int(n.max())), np.nan)
    for i, v in enumerate(vs):
        pad[i, :len(v)] = v
    np.savez(a.out, clips=np.asarray(clips), v=pad, n=n)
    print("ego speed of %d clips written: %s" % (len(clips), a.out))
    return 0


def run_show(a):
    for p in (a.result, a.controls):
        if not os.path.isfile(p):
            refuse("missing " + p)
    C, R = read_json(a.controls), read_json(a.result)
    if not C.get("controls_all_pass") or C.get("n_passed") != C.get("n_controls"):
        refuse("the controls did not all pass; no name and no value is shown")
    if R.get("controls_sha256") != sha256_file(a.controls):
        refuse("the result is not bound to this control file (sha256 differs)")
    if "withheld" in R or not R.get("names") or "arms" not in R:
        refuse("the result carries no names or arms")
    print("KINCLOSE (card ADD88, tier T0)")
    for n in R["names"]:
        print("NAME " + n)
    f = lambda v: "n/a" if v is None else "%+.3f" % v
    print("%-9s %8s %17s %6s %7s %6s %6s %6s" % ("arm", "r", "r 95%", "mae", "err_sd", "lag1", "n", "sd/1.24"))
    for nm, v in R["arms"].items():
        if not v:
            print("%-9s (no frames)" % nm)
            continue
        lo, hi = v["r_ci"]
        print("%-9s %8s [%s,%s] %6.3f %7.3f %6s %6d %6.2f" % (nm, f(v["r_pooled"]), f(lo), f(hi), v["mae"], v["err_sd"], f(np.nanmean([x if x is not None else np.nan for x in v["err_lag1"]])), v["n_frames"], v["err_sd"] / R["row147_error_sd_ms"]))
    for nm, v in R["paired"].items():
        print("paired %-9s %s" % (nm, "no frames" if not v else "diff %+.3f [%+.3f, %+.3f] n %d" % (v["diff_abs_err"], v["ci"][0], v["ci"][1], v["n_frames"])))
    return 0


# ============================================================================================ synthetic world (self-test and test file)
def gen_world(seed, mode="signal", n_clips=40, T=T_FRAMES, seeds=(0, 1), dt=0.2):
    """-> dumps {reader seed: dump dict}, ego {clip: v[T]}, episode map. Lead of piecewise constant speed, ego speed that changes, gap read = gap + noise ('signal') or noise alone ('noise')."""
    rng = np.random.default_rng(seed)
    clips = ["synthclip-%04d" % i for i in range(n_clips)]
    ego, lab = {}, {}
    T0 = T
    for i, c in enumerate(clips):
        T = T0 - 1 if i % 7 == 1 else (T0 + 3 if i % 11 == 5 else T0)   # clips of three lengths, as in the cache
        t = np.arange(T)
        vl = np.zeros(T)
        j = 0
        while j < T:
            ln = int(rng.integers(25, 45))
            vl[j:j + ln] = rng.uniform(5, 25)
            j += ln
        per, ph = rng.uniform(50, 90), rng.uniform(0, 2 * np.pi)
        ve = 12.0 + 6.0 * np.sin(2 * np.pi * t / per + ph)
        g = np.zeros(T)
        g[0] = 150.0
        for j in range(1, T):
            g[j] = g[j - 1] + dt * ((vl[j - 1] - ve[j - 1]) + (vl[j] - ve[j])) / 2.0
        pres = np.ones(T, bool)
        if i % 3 == 0:
            pres[50:61] = False
        run = np.zeros(T, np.int64)
        clo = np.zeros(T)
        for j in range(T):
            if not pres[j]:
                continue
            run[j] = min(5, (run[j - 1] if j else 0) + 1)
            if run[j] >= 3:
                tt = np.arange(run[j]) * dt
                gg = g[j - run[j] + 1:j + 1]
                clo[j] = float(((tt - tt.mean()) * (gg - gg.mean())).sum() / ((tt - tt.mean()) ** 2).sum())
        ego[c] = ve
        lab[c] = (g, pres, run, clo)
    dumps = {}
    for k in seeds:
        cols = {n: [] for n in ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run")}
        for i, c in enumerate(clips):
            g, pres, run, clo = lab[c]
            T = len(g)
            cols["cid"].append(np.full(T, i))
            cols["clip_index"].append(np.full(T, 1000 + i))
            cols["frame"].append(np.arange(T))
            cols["p"].append(np.clip(0.95 + rng.normal(0, 0.02, T), 0, 1))
            cols["gap"].append(g + rng.normal(0, 0.5, T) if mode == "signal" else 40.0 + rng.normal(0, 1.0, T))
            cols["clos"].append(rng.normal(0, 1.0, T))
            cols["valid"].append(rng.uniform(size=T) > 0.02)
            cols["lab_present"].append(pres)
            cols["lab_gap"].append(np.where(pres, g, np.nan))
            cols["lab_closing"].append(clo)
            cols["lab_run"].append(run)
        D = {n: np.concatenate(v) for n, v in cols.items()}
        D["clips"], D["dt_s"], D["seed"] = np.asarray(clips), np.float64(dt), np.int64(k)
        dumps[k] = D
    return dumps, ego, {c: "ep%03d" % (i // 2) for i, c in enumerate(clips)}


def write_world(root, seed=3, mode="signal", mut=None):
    """files of a synthetic pod folder under root; mut(files dict) may change values before they are written. -> paths dict."""
    os.makedirs(root, exist_ok=True)
    dumps, ego, epm = gen_world(seed, mode)
    ks = sorted(dumps)
    est_dir, dump_dir = os.path.join(root, "est"), os.path.join(root, "dump")
    os.makedirs(est_dir), os.makedirs(dump_dir)
    rng = np.random.default_rng(seed + 100)
    est = {}
    for k in ks:
        n = 3 * H_SCORE
        est[k] = {"mu": np.zeros(n), "sd": np.ones(n), "w": rng.normal(0, 0.05, n), "b": 0.1}
    clips = list(ego)
    ego_n = np.asarray([len(ego[c]) for c in clips], np.int64)
    ego_v = np.full((len(clips), int(ego_n.max())), np.nan)
    for i, c in enumerate(clips):
        ego_v[i, :ego_n[i]] = ego[c]
    st = {"dumps": dumps, "ego_v": ego_v, "ego_n": ego_n, "ego_clips": list(clips), "epm": epm, "est": est, "dt_scale": None}
    cells = []
    for k in ks:
        CL.save_linear(os.path.join(est_dir, "closest_L25_s%d.npz" % k), H_SCORE, est[k]["mu"], est[k]["sd"], est[k]["w"], est[k]["b"], seed=k)
    for k in ks:
        rows = CF.build_rows(dumps[k], H_SCORE, epm)
        cells.append((CL.ClosingEstimator.load(os.path.join(est_dir, "closest_L25_s%d.npz" % k)).apply(rows["X"]), rows))
    cs = CF.cell_summary([c[0] for c in cells], [c[1] for c in cells], 50, BOOT_SEED)
    st["add81_res"] = {"arms": {"T25": {"cell_r_pooled": cs["r_pooled"], "cell_mae": cs["mae_est"], "cell_r_per_seed": cs["r_per_seed"], "n_frames": cs["n_frames"]}}}
    st["add81_ctl"] = {"controls": {"t0": {"frames": {"H25_seed%d" % k: {"n_tool": len(c[1]["y"]), "n_dump": len(c[1]["y"])} for k, c in zip(ks, cells)}}}}
    if mut:
        mut(st)
    for k in ks:
        D = dict(st["dumps"][k])
        np.savez(os.path.join(dump_dir, "reader_s%d_score.npz" % k), **D)
    P = {"dump_dir": dump_dir, "est_dir": est_dir, "ego": os.path.join(root, "ego_v_eval.npz"), "episode_map": os.path.join(root, "episode_map.json"),
         "add81_result": os.path.join(root, "TOKCLOSE_RESULT.json"), "add81_controls": os.path.join(root, "TOKCLOSE_CONTROLS.json")}
    np.savez(P["ego"], clips=np.asarray(st["ego_clips"]), v=np.asarray(st["ego_v"], np.float64), n=np.asarray(st["ego_n"], np.int64))
    write_json(P["episode_map"], st["epm"])
    write_json(P["add81_result"], st["add81_res"])
    write_json(P["add81_controls"], st["add81_ctl"])
    return P


def ns(P, out_dir, boot=300, seeds="0,1"):
    return argparse.Namespace(dump_dir=P["dump_dir"], ego=P["ego"], est_dir=P["est_dir"], episode_map=P["episode_map"], add81_result=P["add81_result"], add81_controls=P["add81_controls"],
                              out_dir=out_dir, boot=boot, seeds=seeds)


# ============================================================================================ self-test
def selftest():
    import tempfile
    tmp = os.path.join(HERE, "tmp_work")
    os.makedirs(tmp, exist_ok=True)
    bad, lines = [], []

    def chk(name, cond):
        lines.append("%s %s" % ("ok  " if cond else "FAIL", name))
        if not cond:
            bad.append(name)

    def quiet(a):
        FAULTS_BEFORE = set(FAULTS)
        try:
            return run_read(a, log=lambda s: None)
        finally:
            FAULTS.clear(), FAULTS.update(FAULTS_BEFORE)

    def ident_boot_width():
        """property of the shared episode draw: two reader seeds with identical rows give exactly the interval of one seed."""
        dumps, ego, epm = gen_world(5, "signal", n_clips=16)
        c = make_ctx(dumps[0], ego, epm)
        e = arm(c, "K", 25)
        m = fin(e)
        one = summarize([e], [c["rows"]], [m], 300, BOOT_SEED)["r_ci"]
        two = summarize([e, e], [c["rows"], c["rows"]], [m, m], 300, BOOT_SEED)["r_ci"]
        return bool(abs(one[0] - two[0]) < 1e-9 and abs(one[1] - two[1]) < 1e-9)

    with tempfile.TemporaryDirectory(dir=tmp) as root:
        load_tools(None) if CF is None else None
        cnt = [0]

        def world(mut=None, mode="signal"):
            cnt[0] += 1
            return write_world(os.path.join(root, "w%d" % cnt[0]), mode=mode, mut=mut)

        def go(P, faults=(), boot=300):
            FAULTS.clear(), FAULTS.update(faults)
            od = os.path.join(root, "o%d" % cnt[0])
            cnt[0] += 1
            try:
                rc = quiet(ns(P, od, boot))
            finally:
                FAULTS.clear()
            return rc, read_json(os.path.join(od, "KINCLOSE_CONTROLS.json")), read_json(os.path.join(od, "KINCLOSE_RESULT.json"))

        Pm = world()
        rc, C, R = go(Pm)
        chk("signal world: rc 0 and all five controls pass (k4 planted included)", rc == 0 and C["controls_all_pass"] and C["n_passed"] == 5)
        chk("signal world: K25 better than the cell and the ego term helps", "KINCLOSE-K25-BETTER-THAN-OUTPUTS" in R["names"] and "KINCLOSE-EGO-TERM-HELPS" in R["names"])
        chk("result is bound to the control file", R["controls_sha256"] == sha256_file(os.path.join(root, "o%d" % (cnt[0] - 1), "KINCLOSE_CONTROLS.json")))
        chk("limit name present on the bound frames (B25 reads the label)", any(n.endswith("-LIMIT") for n in R["names"]) or "B25" in R["arms"])
        chk("k4 planted world: signal reads better with K, noise stays at E", C["controls"]["k4"]["pass"])
        chk("bootstrap property: identical seeds give the interval of one seed (shared episode draw)", ident_boot_width())
        rcn, Cn, Rn = go(world(mode="noise"))
        chk("noise world: the read runs and controls k0 k1 k2 pass", rcn in (0, 1) and all(Cn["controls"][k]["pass"] for k in ("k0", "k1", "k2")))

        # deliberate regressions: each has to turn its control red (or change a name); the healthy tool passes the same check
        for fault, ctl in (("ego_sign", "k1"),("ego_shift", "k1"), ("dt_wrong", "k0"), ("no_slope5", "k0"), ("b5_offset", "k0"), ("wrong_frames", "k3"), ("k_is_s", "k4")):
            rcf, Cf, Rf = go(Pm, [fault])
            chk("regression %s turns %s red (and the read exits 1)" % (fault, ctl), (not Cf["controls"][ctl]["pass"]) and rcf == 1 and Rf["names"] == [] and "withheld" in Rf and "arms" not in Rf)

        def neg_v(st):
            st["ego_v"] = st["ego_v"].copy()
            st["ego_v"][2, 30] = -1.0
        rcv, Cv, _ = go(world(neg_v))
        chk("regression: a negative ego speed on a used frame turns k2 red", (not Cv["controls"]["k2"]["pass"]) and rcv == 1)

        def alter81(st):
            st["add81_res"]["arms"]["T25"]["cell_r_pooled"] += 0.01
        rc8, C8, _ = go(world(alter81))
        chk("regression: a cell r that differs from the result file of card ADD81 turns k3 red", (not C8["controls"]["k3"]["pass"]) and rc8 == 1)

        rcd, Cd, Rd = go(Pm, ["ego_sign", "names_despite_fail"])
        chk("regression: names written although a control failed is caught (the healthy tool writes none)", bool(Rd["names"]) and not Cd["controls_all_pass"])
        FAULTS.add("boot_unshared")
        try:
            def unshared(stats, B, seed):
                S, E = len(stats), len(stats[0]["n"])
                rng = np.random.default_rng(seed)
                point = {k: float(sum(st[k].sum() for st in stats)) for k in stats[0]}
                boot = {k: 0.0 for k in stats[0]}
                for st in stats:
                    W = np.stack([np.bincount(rng.integers(0, E, E), minlength=E) for _ in range(B)])
                    for k in stats[0]:
                        boot[k] = boot[k] + W @ st[k]
                return point, boot
            real = CF.boot_totals
            CF.boot_totals = unshared
            broken = ident_boot_width()
        finally:
            CF.boot_totals = real
            FAULTS.discard("boot_unshared")
        chk("regression: a bootstrap without the shared episode draw breaks the identical-seeds property", not broken)
    for ln in lines:
        print(ln)
    print("ZZ KINCLOSE-SELFTEST rc %d" % (1 if bad else 0))
    return 1 if bad else 0


# ============================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--closeread-tools", default=None)
    sp = ap.add_subparsers(dest="cmd")
    e = sp.add_parser("ego")
    e.add_argument("--eval-cache", required=True)
    e.add_argument("--dump-dir", required=True)
    e.add_argument("--out", required=True)
    r = sp.add_parser("read")
    for n in ("dump-dir", "ego", "est-dir", "episode-map", "add81-result", "add81-controls", "out-dir"):
        r.add_argument("--" + n, required=True)
    r.add_argument("--closeread-tools", dest="crt", default=None)
    r.add_argument("--boot", type=int, default=BOOT_B)
    r.add_argument("--seeds", default="0,1")
    s = sp.add_parser("show")
    s.add_argument("--result", required=True)
    s.add_argument("--controls", required=True)
    a = ap.parse_args(argv)
    if a.selftest:
        load_tools(a.closeread_tools)
        return selftest()
    if a.cmd == "ego":
        return run_ego(a)
    if a.cmd == "show":
        return run_show(a)
    if a.cmd == "read":
        load_tools(a.crt or a.closeread_tools)
        return run_read(a)
    ap.error("a subcommand or --selftest is required")


if __name__ == "__main__":
    sys.exit(main())
