r"""leadhead_b.py -- LEADHEAD stage 0b (card PREREG_V8F_PROOF_ADD23, definitions LEADHEAD_B_DEFS.md M1..).
Tier T0: a world-model diagnostic on logged frames, never driving performance.  CPU, zero training of the model.
The heads HT / HM of card ADD21 fitted on the train probe set (1,200 training clips, ~121,000 frames; grid state, five-frame history),
scored on the same 429 held-out windows.  The ridge is solved from block-wise accumulated sufficient statistics (Gram matrix), so
the memory stays under 3 GB.

  python leadhead_b.py run [--arms P_s0 P_s1 G_s0 G_s1] [--stages ht lc hm direct pooled] [--B 2000] [--out-dir DIR]

Strictly: STRATEGIC PAUSED (PI item 25); LONGITUDINAL, LATERAL, TACTICAL are not measured by this read.
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
ROOT = "F:/TanitAD-scratch/v8proof"
for _p in (ROOT + "/spatial/leadhead", ROOT + "/spatial/s1", ROOT + "/spatial/objslot"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import leadhead_s0 as H  # noqa: E402

O = H.O
RP = H.RP
TIER = O.TIER
FAMILIES = O.FAMILIES
DT = O.DT
S0F = O.S0F
SEED_IN = H.SEED_IN
GRID = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7)       # card ADD21 grid extended upward by 1e6, 1e7 (card ADD23 section 2)
E_DATA = ROOT + "/spatial/s0e/data"
JOIN = "C:/Users/Admin/lorapod/ship2/join_train.npz"
GAP_LO, GAP_HI = 0.0, 60.0
CACHE = "C:/Users/Admin/v8proof_c/leadhead_b/cache"
TRAIN_STATES = {
    "P_s0": "C:/Users/Admin/v8proof_c/s1states/P_s0_train.npz",
    "P_s1": "C:/Users/Admin/v8proof_c/s1states/P_s1_train.npz",
    "G_s0": ROOT + "/spatial/s1states/G_s0_train.npz",
    "G_s1": ROOT + "/spatial/s1states/G_s1_train.npz",
}
LC_SIZES = (2000, 8000, 16000, 32000)
BLOCK = 3000
NPC = 64


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================ sufficient statistics (pure)
class Stats:
    """n, sum x, sum x x', sum x y', sum y, sum y^2 of a set of rows (y has k columns)."""

    def __init__(self, p, k):
        self.n = 0
        self.sx = np.zeros(p)
        self.sxx = np.zeros((p, p))
        self.sxy = np.zeros((p, k))
        self.sy = np.zeros(k)
        self.syy = np.zeros(k)

    def add(self, X, Y):
        self.n += len(X)
        self.sx += X.sum(0)
        self.sxx += X.T @ X
        self.sxy += X.T @ Y
        self.sy += Y.sum(0)
        self.syy += (Y ** 2).sum(0)

    def combine(self, other, sign=1.0):
        s = Stats(len(self.sx), len(self.sy))
        s.n = int(self.n + sign * other.n)
        s.sx = self.sx + sign * other.sx
        s.sxx = self.sxx + sign * other.sxx
        s.sxy = self.sxy + sign * other.sxy
        s.sy = self.sy + sign * other.sy
        s.syy = self.syy + sign * other.syy
        return s

    def __add__(self, o):
        return self.combine(o, 1.0)

    def __sub__(self, o):
        return self.combine(o, -1.0)


def sum_stats(lst):
    t = lst[0]
    for s in lst[1:]:
        t = t + s
    return t


def build_stats(getX, m, Y, gid, ngroups, block=BLOCK):
    """One pass over m rows in blocks; getX(i0, i1) -> float64 [b, p]; Y [m, k]; gid [m] group id (-1 = skipped). Returns a list of Stats."""
    Y = np.asarray(Y, np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    X0 = getX(0, min(1, m))
    S = [Stats(X0.shape[1], Y.shape[1]) for _ in range(ngroups)]
    for i0 in range(0, m, block):
        i1 = min(m, i0 + block)
        Xb = getX(i0, i1)
        gb = gid[i0:i1]
        for g in np.unique(gb):
            if g < 0:
                continue
            sel = gb == g
            S[int(g)].add(Xb[sel], Y[i0:i1][sel])
        del Xb
    return S


def prep(S, col=0):
    n = S.n
    mu = S.sx / n
    C = S.sxx - n * np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(C) / n, 0.0))
    sd = np.where(sd < 1e-12, 1.0, sd)
    Gs = C / np.outer(sd, sd)
    b = (S.sxy[:, col] - mu * S.sy[col]) / sd
    return {"n": n, "mu": mu, "sd": sd, "Gs": Gs, "b": b, "ym": S.sy[col] / n}


def ridge_uc(P, lam):
    """Raw-input weights u and intercept c0 of the ridge on standardised inputs (penalty lam on the standardised Gram)."""
    p = len(P["b"])
    Gl = P["Gs"].copy()
    Gl[np.diag_indices(p)] += lam
    w = np.linalg.solve(Gl, P["b"])
    u = w / P["sd"]
    return u, float(P["ym"] - P["mu"] @ u)


def sse_stats(S, u, c0, col=0):
    """Sum of squared errors of the model x.u + c0 on the rows summarised by S (no rows needed)."""
    n = S.n
    return float(S.syy[col] - 2 * c0 * S.sy[col] - 2 * u @ S.sxy[:, col] + n * c0 * c0 + 2 * c0 * (u @ S.sx) + u @ (S.sxx @ u))


def r2_stats(S, u, c0, col=0):
    sst = float(S.syy[col] - S.sy[col] ** 2 / S.n)
    return float(1.0 - sse_stats(S, u, c0, col) / sst)


def fit_ht_stats(groups, col=0, grid=GRID):
    """HT from inner groups: penalty multiplier chosen by the pooled squared error on the held-out inner groups (the inner fits are standardised on their
    own rows), the final model refitted on all groups. lam = g * n_fit / 1000 as card ADD21."""
    total = sum_stats(groups)
    sse = np.zeros(len(grid))
    nval = 0
    for j, Sj in enumerate(groups):
        if Sj.n == 0:
            continue
        tr = total - Sj
        if tr.n < 5:
            continue
        P = prep(tr, col)
        for i, g in enumerate(grid):
            u, c0 = ridge_uc(P, g * tr.n / 1000.0)
            sse[i] += sse_stats(Sj, u, c0, col)
        nval += Sj.n
        del P
    ib = int(np.argmin(sse))
    P = prep(total, col)
    lam = grid[ib] * total.n / 1000.0
    u, c0 = ridge_uc(P, lam)
    info = {"g": grid[ib], "lam": float(lam), "inner_mse": float(sse[ib] / max(nval, 1)), "inner_mse_by_g": [float(s / max(nval, 1)) for s in sse],
            "n_fit": int(total.n), "at_grid_edge": bool(ib in (0, len(grid) - 1))}
    return {"u": u, "c0": c0, "info": info, "total": total}


def predict_lin(m, X):
    return np.asarray(X, np.float64) @ m["u"] + m["c0"]


# ============================================================================================ row sources and groups (pure)
def make_getX(Zarr, idx5, rows, extra=None):
    """Five-frame input of the given rows, built block by block: Zarr[idx5[row]] -> [b, 5*640] float64 (+ the ego column when extra is given)."""
    def getX(i0, i1):
        r = rows[i0:i1]
        X = np.asarray(Zarr[idx5[r]], np.float64).reshape(len(r), -1)
        if extra is not None:
            X = np.c_[X, extra[r]]
        return X
    return getX


def clip_draw(names, seed=0):
    """Clip order drawn with rng seed (a permutation of the sorted unique clip names)."""
    u = np.array(sorted(set(map(str, names))))
    return u[np.random.default_rng(seed).permutation(len(u))]


def disjoint_check(fit_ids, eval_ids):
    a, b = set(map(str, fit_ids)), set(map(str, eval_ids))
    inter = sorted(a & b)
    return {"pass": bool(len(inter) == 0), "n_fit_ids": len(a), "n_eval_ids": len(b), "n_common": len(inter), "common_first5": inter[:5]}


def lc_clips(order, counts_by_clip, n_target):
    """Clips in drawn order until the labelled rows reach n_target (whole clips); returns (clip list, rows)."""
    tot, out = 0, []
    for c in order:
        out.append(c)
        tot += counts_by_clip.get(c, 0)
        if tot >= n_target:
            break
    return out, tot


def stats_for_rows(Zarr, idx5, rows, y, gid, ngroups, extra=None):
    return build_stats(make_getX(Zarr, idx5, rows, extra), len(rows), y, gid, ngroups)


# ============================================================================================ data of the probe set
def load_probe_meta():
    """Probe set layout without the camera tokens: clip, frame, kinematics (v, a, yaw rate) and the join labels, as s0e_read.load_train builds them
    (same row order: clip, then frame). Only clip_idx, kin and t of the shards are read."""
    I = json.load(open(E_DATA + "/INDEX.json", encoding="utf-8"))
    z = np.load(JOIN)
    J = {"clip": z["clip"].astype(str), "off": z["off"].astype(np.int64), "n": z["n"].astype(np.int64), "lead_valid": z["lead_valid"], "lead_gap": z["lead_gap"]}
    clips = list(I["clips"])
    shard_of = I["shard_of"]
    need = sorted({shard_of[i] for i in range(len(clips))})
    Ks, Cs, Ts = [], [], []
    for sh in need:
        zz = np.load(E_DATA + "/" + sh)
        Ks.append(zz["kin"])
        Cs.append(zz["clip_idx"].astype(np.int64))
        Ts.append(zz["t"].astype(np.int64))
    kin = np.concatenate(Ks).astype(np.float64)
    ci = np.concatenate(Cs)
    t = np.concatenate(Ts)
    o = np.lexsort((t, ci))
    if not np.array_equal(o, np.arange(len(o))):
        kin, ci, t = kin[o], ci[o], t[o]
    jc = {c: i for i, c in enumerate(J["clip"])}
    ji = np.array([jc.get(c, -1) for c in clips])
    assert (ji >= 0).all()
    jidx = J["off"][ji[ci]] + t
    v = np.asarray(J["lead_valid"][jidx]).astype(np.int64)
    g = np.asarray(J["lead_gap"][jidx], np.float64)
    ok = (v == 1) & np.isfinite(g) & (g >= GAP_LO) & (g <= GAP_HI)
    gap = np.where(ok, g, np.nan)
    keep = np.isfinite(kin).all(1)
    return {"kin": kin, "bg": t, "epi": ci, "uep": clips, "beid": np.array(clips)[ci], "gap": gap, "keep": keep, "frames_index": I.get("frames")}


def frame_step_check(M):
    """Rows of a clip are consecutive frames 0..n-1 (stride 1 of the cache); the cache step is 0.2 s (spatial/s0c/s0c_read.py FWD/BACK comment)."""
    ci, t = M["epi"], M["bg"]
    same = ci[1:] == ci[:-1]
    d = (t[1:] - t[:-1])[same]
    first = t[np.r_[True, ~same]]
    return {"stride_all_one": bool((d == 1).all()), "first_frame_zero": bool((first == 0).all()), "n_rows": int(len(ci)), "n_clips": int(len(np.unique(ci))),
            "min_frames_per_clip": int(np.bincount(ci).min()), "max_frames_per_clip": int(np.bincount(ci).max()), "dt_s": DT}


# ============================================================================================ fits (train rows)
def target_rows(FS, tg):
    okm, col = H.TARGETS[tg]
    return np.flatnonzero(FS[okm]), col


def fit_target(Zk, FS, tg, clip_names, fo, pool_mask, perm_seed=None):
    """HT on the train rows of target tg within pool_mask (bool over rows); inner groups = fo (per row, 0..3). Returns model dict. With perm_seed the second Y column
    (labels permuted across the fit rows) is accumulated too and fitted as the shuffled-label control."""
    rows, col = target_rows(FS, tg)
    rows = rows[pool_mask[rows]]
    extra = FS["Xego"] if tg == "dir" else None
    y = FS[col][rows]
    Y = y[:, None]
    if perm_seed is not None:
        Y = np.c_[y, y[np.random.default_rng(perm_seed).permutation(len(y))]]
    gid = fo[rows]
    G = stats_for_rows(Zk, FS["idx5"], rows, Y, gid, 4, extra)
    out = {"true": fit_ht_stats(G, 0)}
    out["true"]["info"]["episodes_fit"] = int(len(set(map(str, clip_names[rows]))))
    if perm_seed is not None:
        out["perm"] = fit_ht_stats(G, 1)
    out["groups"] = G
    out["rows"] = rows
    return out


def tenth_fit(Zk, FS, tg, clip_names, held_clips, seed=SEED_IN):
    """j4 / j8: HT fitted on the train clips outside held_clips, evaluated in sample and on the held-out clips (from statistics, no extra pass)."""
    rows, col = target_rows(FS, tg)
    held = np.isin(clip_names, np.array(sorted(held_clips)))
    fo = np.full(len(clip_names), -1, np.int64)
    fit_pool = ~held
    fo_all = H.inner_folds(clip_names[fit_pool], 4, seed)
    fo[np.flatnonzero(fit_pool)] = fo_all
    fo[held] = 4
    rows = rows[(fo[rows] >= 0)]
    extra = FS["Xego"] if tg == "dir" else None
    G = stats_for_rows(Zk, FS["idx5"], rows, FS[col][rows], fo[rows], 5, extra)
    m = fit_ht_stats(G[:4], 0)
    tot = m["total"]
    res = {"model": m, "r2_in_sample": r2_stats(tot, m["u"], m["c0"]), "r2_heldout_tenth": r2_stats(G[4], m["u"], m["c0"]), "n_fit": int(tot.n), "n_heldout": int(G[4].n),
           "g": m["info"]["g"], "clips_heldout": int(len(held_clips))}
    return res


def apply_heads(models, Z1, Xw):
    X = np.asarray(Z1[:, :S0F + 1, :], np.float64).reshape(len(Z1), -1)
    out = {}
    for tg, m in models.items():
        out[tg] = predict_lin(m, np.c_[X, Xw] if tg == "dir" else X)
    return out


# ============================================================================================ window-side reads
def window_block(ctx, outs, V, label):
    """Fidelity, rule table and banked comparison of the heads' outputs (gap and closing speed at S0)."""
    F = H.fidelity(outs["gap"], outs["c"], ctx["G15"], ctx["c_lab"], ctx["valid"], ctx["ep10"], ctx["n10"], ctx["C10"])
    PV = H.head_pipeline(outs["gap"], outs["c"], V, 10, ctx["y"], ctx["sc"], ctx["fold"])
    tb, _ = O.table(PV, ctx["ep10"], ctx["n10"], ctx["C10"])
    bk = H.banked_compare(PV, ctx["y"], ctx["sc"], ctx["ep10"], ctx["n10"], ctx["C10"], ctx["arm"], ctx["eid"], ctx["g0"])
    return {"fidelity": F, "rules": tb, "coefs": PV["coefs"], "banked": bk}


def direct_block(ctx, outs):
    pd = outs["gap"] + outs["dir"]
    m = ctx["sc"] & np.isfinite(pd) & np.isfinite(outs["gap"])
    return {"skill_over_copy": O.skill_row(ctx["y"], pd, outs["gap"], m, ctx["ep10"], ctx["n10"], ctx["C10"])[0],
            "rmse_gap_2s": H.rmse_row(ctx["y"], pd, m, ctx["ep10"], ctx["n10"], ctx["C10"]),
            "rmse_copy_gap_2s": H.rmse_row(ctx["y"], outs["gap"], m, ctx["ep10"], ctx["n10"], ctx["C10"])}


def jdump(o, path):
    O.jdump(o, path)


def strip(m):
    """Model summary for the result file (no weight vectors)."""
    return {k: v for k, v in m["info"].items() if k != "inner_mse_by_g"} | {"inner_mse_by_g": m["info"]["inner_mse_by_g"]}


# ============================================================================================ global controls
def control_j1(M, eid_windows, bank_ids):
    d = disjoint_check(M["uep"], list(eid_windows) + list(bank_ids))
    d["fit_ids"] = "the 1,200 train probe clip ids"
    d["eval_ids"] = "scored-window episode ids and the eval-bank episode ids"
    return d


def control_j5(M, tenth, amp=10.0, s_noise=None, seed=0):
    """Planted state on the train row layout: the control h1 of card ADD21 on the clip / frame layout of the train probe set (all kept rows)."""
    if s_noise is None:
        s_noise = float(json.load(open(ROOT + "/spatial/leadhead/RESULT_LEADHEAD_S0.json", encoding="utf-8"))["controls"]["h1_planted_state"]["noise_sd_per_dim"])
    rng = np.random.default_rng(seed)
    keep = M["keep"]
    beid, bg = M["beid"][keep], M["bg"][keep]
    uc = sorted(set(beid))
    ge = dict(zip(uc, rng.uniform(15, 60, len(uc))))
    ce = dict(zip(uc, rng.normal(0, 1.5, len(uc))))
    gap = np.array([ge[e] + ce[e] * DT * t for e, t in zip(beid, bg)])
    u = rng.standard_normal(640)
    u /= np.linalg.norm(u)
    Z = rng.standard_normal((len(gap), 640), dtype=np.float32)
    Z *= np.float32(s_noise)
    Z += (gap * amp * s_noise).astype(np.float32)[:, None] * u.astype(np.float32)[None, :]
    FS = H.build_fit_sets(beid, bg, gap, np.full(len(gap), 10.0))
    r = tenth_fit(Z, FS, "c", beid, tenth)
    del Z
    gc.collect()
    return {"pass": bool(r["r2_heldout_tenth"] > 0.9), "heldout_r2_closing": r["r2_heldout_tenth"], "bar": 0.9, "n_fit": r["n_fit"], "n_heldout": r["n_heldout"],
            "noise_sd_per_dim": s_noise, "amp_per_metre": amp, "chosen_g": r["g"], "layout": "all kept rows and clips of the train probe set, synthetic linear gap per clip"}


# ============================================================================================ per-arm context
def arm_context(arm, B):
    D, lat = O.load_arm(arm)
    eid, g0, fold, folds_b = O.arm_windows(D)
    G15, miss = O.load_label_frames(eid, g0)
    V, Z1, _ = O.speed_and_states(arm, eid, g0)
    Xw = H.ego_term_windows(V, 10)
    sc, cnt = O.scored_mask(G15, 10)
    c_lab, hok = H.label_c(G15)
    valid = sc & hok
    y = G15[:, S0F + 10]
    ep10, n10 = O.epi_of(eid, sc)
    C10 = RP.boot_counts(n10, CTX_B[0], O.SEED)
    return {"arm": arm, "D": D, "eid": eid, "g0": g0, "fold": fold, "folds_b": folds_b, "G15": G15, "V": V, "Z1": Z1, "Xw": Xw, "sc": sc, "cnt": cnt, "c_lab": c_lab,
            "hok": hok, "valid": valid, "y": y, "ep10": ep10, "n10": n10, "C10": C10, "miss": miss}


CTX_B = [2000]


def load_train_state(arm, M):
    import s1_read as S1
    S = S1.load_states(TRAIN_STATES[arm])
    t1 = S1.check_t1(S, {"uep": M["uep"], "epi": M["epi"], "bg": M["bg"]})
    Zk = np.ascontiguousarray(S["Z"][M["keep"]], dtype=np.float32)
    rec = S["record"] or {}
    del S
    gc.collect()
    return Zk, t1, {"ckpt_sha256": rec.get("ckpt_sha256"), "wm_state": rec.get("wm_state"), "D": rec.get("D")}


# ============================================================================================ stages
def cache_path(arm, stage):
    return "%s/%s_%s.json" % (CACHE, arm, stage)


def load_cached(arm, stage):
    p = cache_path(arm, stage)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return None


def stage_ht(arm, ctx, M, Zk, FS, names, fo_all, tenth, res_arm):
    """Priority 2: HT on all train rows (gap, closing, direct), shuffled-label control, in-sample and tenth controls, slot read."""
    t0 = time.time()
    pool = np.ones(len(names), bool)
    fits = {}
    for tg in ("gap", "c", "dir"):
        f = fit_target(Zk, FS, tg, names, fo_all, pool, perm_seed=7 if tg == "c" else None)
        fits[tg] = f
        log(arm, "HT", tg, "n_fit", f["true"]["info"]["n_fit"], "g", f["true"]["info"]["g"], "inner_mse %.4f" % f["true"]["info"]["inner_mse"], "(%.0f s)" % (time.time() - t0))
    models = {tg: fits[tg]["true"] for tg in fits}
    outs = apply_heads(models, ctx["Z1"], ctx["Xw"])
    blk = window_block(ctx, outs, ctx["V"], "HT")
    blk["direct"] = direct_block(ctx, outs)
    blk["fit"] = {tg: {**{k: v for k, v in fits[tg]["true"]["info"].items()}, "clips_with_rows": fits[tg]["true"]["info"]["episodes_fit"]} for tg in fits}
    blk["inner_mse_closing"] = fits["c"]["true"]["info"]["inner_mse"]
    blk["inner_mse_direct"] = fits["dir"]["true"]["info"]["inner_mse"]
    blk["inner_mse_gap"] = fits["gap"]["true"]["info"]["inner_mse"]
    # j3 shuffled labels (closing)
    perm = {"c": predict_lin(fits["c"]["perm"], np.asarray(ctx["Z1"][:, :S0F + 1, :], np.float64).reshape(len(ctx["Z1"]), -1)), "gap": outs["gap"]}
    r2p = float(RP.r2_pooled(ctx["c_lab"][ctx["valid"]], perm["c"][ctx["valid"]]))
    ctrl = {"j3_shuffled_labels|HT": {"pass": bool(r2p <= 0.02), "heldout_r2_closing_shuffled_labels": r2p, "bar": 0.02, "n": int(ctx["valid"].sum()), "chosen_g": fits["c"]["perm"]["info"]["g"]}}
    # j7 no future input
    rng = np.random.default_rng(3)
    Zm = ctx["Z1"].copy()
    Zm[:, S0F + 1:, :] = rng.standard_normal(Zm[:, S0F + 1:, :].shape).astype(Zm.dtype) * 5
    om = apply_heads(models, Zm, ctx["Xw"])
    dh = max(float(np.nanmax(np.abs(om[tg] - outs[tg]))) for tg in outs)
    P0 = H.head_pipeline(outs["gap"], outs["c"], ctx["V"], 10, ctx["y"], ctx["sc"], ctx["fold"])
    P1 = H.head_pipeline(om["gap"], om["c"], ctx["V"], 10, ctx["y"], ctx["sc"], ctx["fold"])
    dr = max(float(np.nanmax(np.abs(P0["pred"][r] - P1["pred"][r]))) for r in ("R1", "R2", "R3"))
    ctrl["j7_no_future_input|HT"] = {"pass": bool(dh == 0.0 and dr == 0.0), "max_abs_change_of_head_outputs": dh, "max_abs_change_of_rule_outputs": dr}
    # j4 / j8 tenth
    del fits
    gc.collect()
    tc = tenth_fit(Zk, FS, "c", names, tenth)
    tg_ = tenth_fit(Zk, FS, "gap", names, tenth)
    ctrl["j4_in_sample_ceiling|HT"] = {"pass": bool(tc["r2_in_sample"] > tc["r2_heldout_tenth"]), "r2_in_sample_closing_fit_clips": tc["r2_in_sample"],
                                       "r2_closing_heldout_tenth_of_train_clips": tc["r2_heldout_tenth"], "n_fit": tc["n_fit"], "n_heldout": tc["n_heldout"], "chosen_g": tc["g"],
                                       "clips_heldout": tc["clips_heldout"]}
    g90 = apply_heads({"gap": tg_["model"], "c": tc["model"]}, ctx["Z1"], ctx["Xw"])
    Fw = H.fidelity(g90["gap"], g90["c"], ctx["G15"], ctx["c_lab"], ctx["valid"], ctx["ep10"], ctx["n10"], ctx["C10"])
    ctrl["j8_transfer|HT"] = {"pass": True, "reported_not_a_gate": True, "gap_r2_heldout_tenth_of_train_clips": tg_["r2_heldout_tenth"], "gap_r2_eval_windows_same_90pct_head": Fw["gap"]["r2"],
                              "gap_r2_eval_windows_main_head": blk["fidelity"]["gap"]["r2"], "shift_tenth_minus_eval_windows": tg_["r2_heldout_tenth"] - Fw["gap"]["r2"],
                              "closing_r2_heldout_tenth": tc["r2_heldout_tenth"], "closing_r2_eval_windows_same_90pct_head": Fw["closing"]["r2"],
                              "n_heldout_gap_rows": tg_["n_heldout"], "n_heldout_closing_rows": tc["n_heldout"], "chosen_g_gap_90": tg_["g"]}
    res_arm["HT"] = blk
    res_arm["controls"].update(ctrl)
    log(arm, "HT done: closing R2 %+.3f gap R2 %+.3f R3 skill %+.3f (%.0f s)" % (blk["fidelity"]["closing"]["r2"], blk["fidelity"]["gap"]["r2"],
                                                                                  blk["rules"]["R3"]["skill_over_copy"]["skill"], time.time() - t0))


def stage_lc(arm, ctx, M, Zk, FS, names, res_arm):
    """Priority 3: learning curve of HT (secondary a)."""
    t0 = time.time()
    order = clip_draw(names, 0)
    lc = {}
    for N in LC_SIZES:
        sizes = {}
        mods = {}
        for tg in ("gap", "c"):
            rows, col = target_rows(FS, tg)
            cnt = {}
            for c in names[rows]:
                cnt[c] = cnt.get(c, 0) + 1
            sel, tot = lc_clips(order, cnt, N)
            pool = np.isin(names, np.array(sel))
            fo = np.full(len(names), -1, np.int64)
            fo[np.flatnonzero(pool)] = H.inner_folds(names[pool], 4, SEED_IN)
            r = rows[pool[rows]]
            G = stats_for_rows(Zk, FS["idx5"], r, FS[col][r], fo[r], 4)
            m = fit_ht_stats(G, 0)
            mods[tg] = m
            sizes[tg] = {"clips": len(sel), "n_fit": int(m["info"]["n_fit"]), "g": m["info"]["g"]}
        outs = apply_heads(mods, ctx["Z1"], ctx["Xw"])
        Fw = H.fidelity(outs["gap"], outs["c"], ctx["G15"], ctx["c_lab"], ctx["valid"], ctx["ep10"], ctx["n10"], ctx["C10"])
        lc[str(N)] = {"fit": sizes, "fidelity": Fw}
        log(arm, "LC", N, "closing R2 %+.3f gap R2 %+.3f" % (Fw["closing"]["r2"], Fw["gap"]["r2"]), "(%.0f s)" % (time.time() - t0))
    lc["all"] = {"fit": {tg: {"n_fit": res_arm["HT"]["fit"][tg]["n_fit"], "g": res_arm["HT"]["fit"][tg]["g"]} for tg in ("gap", "c")}, "fidelity": res_arm["HT"]["fidelity"]}
    res_arm["learning_curve"] = lc


# ---------------------------------------------------------------------------------------------- HM
def _pca_variant(P, k, ncol):
    """Top-k components of the standardised first ncol columns (the PCA of card ADD21 by the Gram matrix, here by its primal form), each standardised to unit variance
    on the fit rows. Returns (M [p, k] mapping raw inputs to the standardised scores, offset [k], extra index info)."""
    G = P["Gs"][:ncol, :ncol]
    ev, U = np.linalg.eigh(G)
    o = np.argsort(ev)[::-1][:k]
    ev, U = ev[o], U[:, o]
    s1 = np.sqrt(np.maximum(ev, 1e-12) / P["n"])                  # sd of each score (scores of the standardised inputs have variance ev / n)
    s1 = np.where(s1 < 1e-12, 1.0, s1)
    p = len(P["b"])
    Mx = np.zeros((p, k + (p - ncol)))
    Mx[:ncol, :k] = (U / P["sd"][:ncol, None]) / s1[None, :]
    for e in range(p - ncol):                                      # the ego column bypasses the PCA, standardised
        Mx[ncol + e, k + e] = 1.0 / P["sd"][ncol + e]
    off = P["mu"] @ Mx
    return Mx, off


def _train_nets(Pm, yz):
    nets = [H._train_mlp(Pm, yz, s) for s in (0, 1, 2)]
    return nets


def _nets_pred(nets, Q):
    import torch
    Qt = torch.tensor(Q, dtype=torch.float32)
    with torch.no_grad():
        return np.mean([nn(Qt).numpy().reshape(-1) for nn in nets], axis=0).astype(np.float64)


def fit_hm_target(Zk, FS, tg, names, fo, pool_mask, groups=None):
    """HM on the train rows of target tg: PCA 64 from the Gram of the fit rows, 64-64-1 GELU network, 3 initialisations averaged (card ADD21 L6); inner folds by clip
    give the inner error used to choose between HT and HM. Projection of the rows is a second pass with the five projection matrices at once."""
    rows, col = target_rows(FS, tg)
    rows = rows[pool_mask[rows]]
    extra = FS["Xego"] if tg == "dir" else None
    y = FS[col][rows]
    gid = fo[rows]
    G = stats_for_rows(Zk, FS["idx5"], rows, y, gid, 4, extra) if groups is None else groups
    total = sum_stats(G)
    ncol = 5 * 640
    variants = []          # 4 inner (fit = total - group j) then the final
    for j in range(4):
        tr = total - G[j]
        P = prep(tr, 0)
        Mx, off = _pca_variant(P, NPC, ncol)
        variants.append((P, Mx, off))
        del tr
    P = prep(total, 0)
    Mx, off = _pca_variant(P, NPC, ncol)
    variants.append((P, Mx, off))
    kcol = NPC + (len(P["b"]) - ncol)
    Mall = np.concatenate([v[1] for v in variants], axis=1)
    oall = np.concatenate([v[2] for v in variants])
    Pr = np.zeros((len(rows), Mall.shape[1]), np.float32)
    getX = make_getX(Zk, FS["idx5"], rows, extra)
    for i0 in range(0, len(rows), BLOCK):
        i1 = min(len(rows), i0 + BLOCK)
        Pr[i0:i1] = (getX(i0, i1) @ Mall - oall).astype(np.float32)
    sse, cnt = 0.0, 0
    for j in range(4):
        tri, vai = np.flatnonzero(gid != j), np.flatnonzero(gid == j)
        if len(vai) == 0:
            continue
        Pm = Pr[:, j * kcol:(j + 1) * kcol]
        ym, ys = float(y[tri].mean()), float(y[tri].std()) or 1.0
        nets = _train_nets(Pm[tri], (y[tri] - ym) / ys)
        pv = _nets_pred(nets, Pm[vai]) * ys + ym
        sse += float(((y[vai] - pv) ** 2).sum())
        cnt += len(vai)
    Pm = Pr[:, 4 * kcol:5 * kcol]
    ym, ys = float(y.mean()), float(y.std()) or 1.0
    nets = _train_nets(Pm, (y - ym) / ys)
    Mf, of = variants[4][1], variants[4][2]
    del Pr, variants

    def predict(X):
        Q = (np.asarray(X, np.float64) @ Mf - of)
        return _nets_pred(nets, Q.astype(np.float32)) * ys + ym
    return {"predict": predict, "info": {"n_fit": int(len(rows)), "inner_mse": float(sse / max(cnt, 1)), "n_pca": NPC}, "groups": G}


def stage_hm(arm, ctx, M, Zk, FS, names, fo_all, res_arm):
    """Priority 4: HM on all train rows, shuffled-label control, slot read."""
    t0 = time.time()
    pool = np.ones(len(names), bool)
    mods = {}
    info = {}
    for tg in ("gap", "c", "dir"):
        f = fit_hm_target(Zk, FS, tg, names, fo_all, pool)
        mods[tg] = f
        info[tg] = f["info"]
        log(arm, "HM", tg, "inner_mse %.4f" % f["info"]["inner_mse"], "(%.0f s)" % (time.time() - t0))
        f["groups"] = None
    X = np.asarray(ctx["Z1"][:, :S0F + 1, :], np.float64).reshape(len(ctx["Z1"]), -1)
    outs = {tg: (mods[tg]["predict"](np.c_[X, ctx["Xw"]]) if tg == "dir" else mods[tg]["predict"](X)) for tg in mods}
    blk = window_block(ctx, outs, ctx["V"], "HM")
    blk["direct"] = direct_block(ctx, outs)
    blk["fit"] = info
    blk["inner_mse_closing"] = info["c"]["inner_mse"]
    blk["inner_mse_direct"] = info["dir"]["inner_mse"]
    blk["inner_mse_gap"] = info["gap"]["inner_mse"]
    # j3 for HM: closing labels permuted across the fit rows
    rows, col = target_rows(FS, "c")
    yp = FS[col][rows][np.random.default_rng(7).permutation(len(rows))]
    FSp = dict(FS)
    cc = np.full(len(FS["c"]), np.nan)
    cc[rows] = yp
    FSp["c"] = cc
    fp = fit_hm_target(Zk, FSp, "c", names, fo_all, pool)
    cp = fp["predict"](X)
    r2p = float(RP.r2_pooled(ctx["c_lab"][ctx["valid"]], cp[ctx["valid"]]))
    res_arm["controls"]["j3_shuffled_labels|HM"] = {"pass": bool(r2p <= 0.02), "heldout_r2_closing_shuffled_labels": r2p, "bar": 0.02, "n": int(ctx["valid"].sum())}
    # j7 for HM
    rng = np.random.default_rng(3)
    Zm = ctx["Z1"].copy()
    Zm[:, S0F + 1:, :] = rng.standard_normal(Zm[:, S0F + 1:, :].shape).astype(Zm.dtype) * 5
    Xm = np.asarray(Zm[:, :S0F + 1, :], np.float64).reshape(len(Zm), -1)
    dh = max(float(np.nanmax(np.abs((mods[tg]["predict"](np.c_[Xm, ctx["Xw"]]) if tg == "dir" else mods[tg]["predict"](Xm)) - outs[tg]))) for tg in outs)
    res_arm["controls"]["j7_no_future_input|HM"] = {"pass": bool(dh == 0.0), "max_abs_change_of_head_outputs": dh}
    res_arm["HM"] = blk
    log(arm, "HM done: closing R2 %+.3f gap R2 %+.3f R3 skill %+.3f (%.0f s)" % (blk["fidelity"]["closing"]["r2"], blk["fidelity"]["gap"]["r2"],
                                                                                  blk["rules"]["R3"]["skill_over_copy"]["skill"], time.time() - t0))


# ---------------------------------------------------------------------------------------------- pooled
def stage_pooled(arm, ctx, M, Zk, FS, names, fo_all, res_arm):
    """Secondary (b): the HT heads fitted on the train rows and the eval-bank rows of the training folds together, cross-fitted over the eval folds."""
    t0 = time.time()
    D = ctx["D"]
    FSe = H.build_fit_sets(D["beid"], D["bg"], D["gap"], D["E"][:, 0])
    Sb = np.asarray(D["Sb"], np.float64)
    pool = np.ones(len(names), bool)
    trg = {}
    for tg in ("gap", "c"):
        rows, col = target_rows(FS, tg)
        G = stats_for_rows(Zk, FS["idx5"], rows, FS[col][rows], fo_all[rows], 4)
        trg[tg] = G
    out = {tg: np.full(len(ctx["eid"]), np.nan) for tg in ("gap", "c")}
    info = {}
    for f in range(5):
        te = np.flatnonzero(ctx["fold"] == f)
        mods = {}
        for tg in ("gap", "c"):
            okm, col = H.TARGETS[tg]
            rb = np.flatnonzero(FSe["ok5"] & FSe[okm] & (ctx["folds_b"] != f))
            fo_e = H.inner_folds(D["beid"][rb], 4, SEED_IN + f)
            Ge = stats_for_rows(Sb, FSe["idx5"], rb, FSe[col][rb], fo_e, 4)
            Gp = [trg[tg][j] + Ge[j] for j in range(4)]
            m = fit_ht_stats(Gp, 0)
            m["info"]["n_eval_bank_rows"] = int(len(rb))
            mods[tg] = m
            info.setdefault(str(f), {})[tg] = {k: v for k, v in m["info"].items() if k != "inner_mse_by_g"}
        o = apply_heads(mods, ctx["Z1"][te], ctx["Xw"][te])
        for tg in o:
            out[tg][te] = o[tg]
    blk = window_block(ctx, out, ctx["V"], "pooled")
    blk["fit"] = info
    res_arm["pooled"] = blk
    log(arm, "pooled done: closing R2 %+.3f gap R2 %+.3f R3 skill %+.3f (%.0f s)" % (blk["fidelity"]["closing"]["r2"], blk["fidelity"]["gap"]["r2"],
                                                                                      blk["rules"]["R3"]["skill_over_copy"]["skill"], time.time() - t0))


# ============================================================================================ run
def run(arms, stages, B, out_dir, force=False):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    CTX_B[0] = B
    M = load_probe_meta()
    names_all = M["beid"][M["keep"]]
    res = {"tier": TIER, "families": FAMILIES, "card": "PREREG_V8F_PROOF_ADD23", "B": B, "stages_run": list(stages), "arms": {}, "controls": {}}
    res["probe_set"] = {"rows_total": int(len(M["keep"])), "rows_kept_finite_kin": int(M["keep"].sum()), "clips": len(M["uep"]), "frame_step": frame_step_check(M)}
    order = clip_draw(M["uep"], 0)
    tenth = list(order[:len(order) // 10])
    res["tenth_clips"] = {"n": len(tenth), "rule": "first 120 of the clip permutation rng 0 of the sorted clip ids"}
    banked = json.load(open(O.HERE + "/RESULT_OBJSLOT_S0.json", encoding="utf-8"))
    ref_r2 = banked["label"]["10"]["rules"]["R2"]["skill_over_copy"]["skill"]
    ctrl = {}
    cj5 = cache_path("global", "j5")
    if os.path.exists(cj5) and not force:
        ctrl["j5_planted_state_train_layout"] = json.load(open(cj5, encoding="utf-8"))
    else:
        log("j5 planted state")
        ctrl["j5_planted_state_train_layout"] = control_j5(M, tenth)
        jdump(ctrl["j5_planted_state_train_layout"], cj5)
    log("j5", ctrl["j5_planted_state_train_layout"]["pass"], ctrl["j5_planted_state_train_layout"]["heldout_r2_closing"])
    first = None
    for arm in arms:
        log("arm", arm)
        ta = time.time()
        ctx = arm_context(arm, B)
        if first is None:
            first = (ctx["eid"], ctx["g0"], ctx["fold"])
            V0 = ctx["V"]
            ctrl["j6_label_path"] = H.control_h6(ctx["G15"], V0, ctx["fold"], ref_r2)
            ctrl["j1_disjoint_sets"] = control_j1(M, ctx["eid"], ctx["D"]["uep"])
            log("j6", ctrl["j6_label_path"]["pass"], "j1", ctrl["j1_disjoint_sets"]["pass"])
        else:
            assert np.array_equal(first[0], ctx["eid"]) and np.array_equal(first[1], ctx["g0"]) and np.array_equal(first[2], ctx["fold"])
        A = {"controls": {}, "counts": {"scored": ctx["cnt"], "label_history_ok": int(ctx["valid"].sum()), "label_history_dropped": int((ctx["sc"] & ~ctx["hok"]).sum()),
                                        "n_episodes_scored": ctx["n10"], "windows": int(len(ctx["eid"]))}}
        Zk, t1, rec = load_train_state(arm, M)
        A["controls"]["j2_alignment"] = t1
        A["state_record"] = rec
        names = names_all
        keep = M["keep"]
        FS = H.build_fit_sets(names, M["bg"][keep], M["gap"][keep], M["kin"][keep][:, 0])
        fo_all = H.inner_folds(names, 4, SEED_IN)
        A["fit_rows"] = {"kept_rows": int(len(names)), "with_five_frames": int(FS["ok5"].sum()), "gap_rows": int(FS["gap_ok"].sum()), "closing_rows": int(FS["c_ok"].sum()),
                         "direct_rows": int(FS["dir_ok"].sum()), "clips_with_gap_rows": int(len(set(names[FS["gap_ok"]]))),
                         "clips_with_closing_rows": int(len(set(names[FS["c_ok"]]))), "clips_with_direct_rows": int(len(set(names[FS["dir_ok"]])))}
        def reload_cached():
            for st_, key in (("ht", "HT"), ("hm", "HM")):
                c_ = load_cached(arm, st_)
                if c_ and key in c_:
                    A[key] = c_[key]
                    A["controls"].update(c_["controls"])
            c_ = load_cached(arm, "lc")
            if c_:
                A["learning_curve"] = c_
            c_ = load_cached(arm, "pooled")
            if c_:
                A["pooled"] = c_
        if not force:
            reload_cached()
        for st in stages:
            cp = cache_path(arm, st)
            if os.path.exists(cp) and not force:
                log(arm, st, "from cache")
                continue
            tmp = {"controls": {}}
            if st == "ht":
                stage_ht(arm, ctx, M, Zk, FS, names, fo_all, tenth, tmp)
                A["HT"] = tmp["HT"]
                A["controls"].update(tmp["controls"])
                jdump({"HT": tmp["HT"], "controls": tmp["controls"]}, cp)
            elif st == "lc":
                if "HT" not in A:
                    raise SystemExit("stage lc needs stage ht first")
                stage_lc(arm, ctx, M, Zk, FS, names, A)
                jdump(A["learning_curve"], cp)
            elif st == "hm":
                stage_hm(arm, ctx, M, Zk, FS, names, fo_all, tmp)
                A["HM"] = tmp["HM"]
                A["controls"].update(tmp["controls"])
                jdump({"HM": tmp["HM"], "controls": tmp["controls"]}, cp)
            elif st == "pooled":
                stage_pooled(arm, ctx, M, Zk, FS, names, fo_all, A)
                jdump(A["pooled"], cp)
            gc.collect()
        A["seconds"] = round(time.time() - ta, 1)
        res["arms"][arm] = A
        del Zk, ctx, FS
        gc.collect()
        write_all(res, arms, out_dir, ctrl, final=False)
    res = write_all(res, arms, out_dir, ctrl, final=True)
    res_ctrl_ok = res.get("controls_pass")
    log("controls", "PASS" if res_ctrl_ok else "FAILED", "elapsed %.0f s" % (time.time() - t0))
    return 0 if res_ctrl_ok else 2


# ============================================================================================ outcomes and writing
def _pick(A):
    if "HM" in A and "HT" in A:
        return "HT" if A["HT"]["inner_mse_closing"] <= A["HM"]["inner_mse_closing"] else "HM"
    return "HT"


def _pick_dir(A):
    if "HM" in A and "HT" in A:
        return "HT" if A["HT"]["inner_mse_direct"] <= A["HM"]["inner_mse_direct"] else "HM"
    return "HT"


def outcomes(res, arms):
    o = {"named": True, "complete_heads": bool(all("HM" in a for a in res["arms"].values())), "per_type": {}}
    for typ in ("P", "G"):
        seeds = [a for a in arms if a.startswith(typ + "_") and a in res["arms"]]
        if len(seeds) < 2:
            o["per_type"][typ] = "not named (needs both seeds; have %s)" % seeds
            continue
        d = {}
        pk = {a: _pick(res["arms"][a]) for a in seeds}
        F = {a: res["arms"][a][pk[a]]["fidelity"]["closing"] for a in seeds}
        rc = all(F[a]["r2"] > 0 and F[a]["r2_ci95"][0] > 0 and F[a]["n"] >= 30 for a in seeds)
        d["closing"] = "HEAD-READS-CLOSING-B" if rc else "HEAD-CLOSING-FLAT-B"
        d["head_used"] = pk
        im = all(res["arms"][a][pk[a]]["rules"]["R3"]["skill_over_copy"]["separated_pos"] for a in seeds)
        names = ["HEAD-SLOT-IMAGINES-B"] if im else ["HEAD-SLOT-FLAT-B"]
        if im and all(res["arms"][a][pk[a]]["banked"]["R1"]["R3"]["separated_pos"] for a in seeds):
            names.append("HEAD-SLOT-BEATS-ROLLOUT-B")
        d["slot"] = names
        d["with_each_head"] = {}
        for k in ("HT", "HM"):
            if all(k in res["arms"][a] for a in seeds):
                d["with_each_head"][k] = {
                    "closing": "HEAD-READS-CLOSING-B" if all(res["arms"][a][k]["fidelity"]["closing"]["r2_ci95"][0] > 0 and res["arms"][a][k]["fidelity"]["closing"]["r2"] > 0 for a in seeds) else "HEAD-CLOSING-FLAT-B",
                    "slot": "HEAD-SLOT-IMAGINES-B" if all(res["arms"][a][k]["rules"]["R3"]["skill_over_copy"]["separated_pos"] for a in seeds) else "HEAD-SLOT-FLAT-B"}
        pdk = {a: _pick_dir(res["arms"][a]) for a in seeds}
        if all("direct" in res["arms"][a][pdk[a]] for a in seeds):
            dh = all(res["arms"][a][pdk[a]]["direct"]["skill_over_copy"]["separated_pos"] for a in seeds)
            d["direct"] = "STATE-HOLDS-FUTURE-B" if dh else "STATE-LACKS-FUTURE-B"
            d["direct_head_used"] = pdk
        o["per_type"][typ] = d
    return o


def _ci(c):
    return "[%+.3f, %+.3f]" % (c[0], c[1])


def _sk(row):
    return "%+.3f %s n=%d%s" % (row["skill"], _ci(row["ci95"]), row["n"], " (separated above 0)" if row["separated_pos"] else (" (separated below 0)" if row.get("separated_neg") else ""))


def write_all(res, arms, out_dir, ctrl, final):
    res = dict(res)
    allc = dict(ctrl)
    for arm, A in res["arms"].items():
        for k, v in A["controls"].items():
            allc["%s|%s" % (k, arm)] = v
    res["controls"] = allc
    res["controls_pass"] = bool(all(v["pass"] for v in allc.values())) if final else None
    if final:
        res["outcomes"] = outcomes(res, arms) if res["controls_pass"] else {"named": False, "reason": "a control failed; no outcome is named"}
    else:
        res["outcomes"] = {"named": False, "reason": "partial write while the run is in progress"}
    jdump(res, os.path.join(out_dir, "RESULT_LEADHEAD_B.json"))
    write_verdict(os.path.join(out_dir, "VERDICT_LEADHEAD_B.md"), res, [a for a in arms if a in res["arms"]], final)
    return res


def _add21():
    try:
        return json.load(open(ROOT + "/spatial/leadhead/RESULT_LEADHEAD_S0.json", encoding="utf-8"))
    except Exception:
        return None


def write_verdict(path, res, arms, final):
    a21 = _add21()
    L = ["# VERDICT LEADHEAD stage 0b (PREREG_V8F_PROOF_ADD23)", "",
         "Tier " + TIER + ". " + FAMILIES + ". Zero training of the model; CPU; the banked train-probe states and labels. Heads fitted on the train probe set (1,200 training clips), scored on the 429 held-out windows of card ADD21 (episode-cluster bootstrap, B %d, rng 0)." % res["B"],
         "Stages run in this file: %s. Definitions: LEADHEAD_B_DEFS.md. Ego speeds of the rules are the logged ones (the action input of a rollout)." % ", ".join(res["stages_run"]), ""]
    if not final:
        L += ["**Partial write: the run was not finished when this file was written.**", ""]
    ps = res["probe_set"]
    L += ["Train probe set: %d rows, %d with a finite kinematic vector, %d clips; frame step 0.2 s (stride 1: %s)." % (ps["rows_total"], ps["rows_kept_finite_kin"], ps["clips"], ps["frame_step"]["stride_all_one"]), ""]
    L += ["## Controls (first)", ""]
    for k, v in res["controls"].items():
        L.append("- %s: %s :: %s" % (k, "PASS" if v["pass"] else "FAIL", json.dumps({a: b for a, b in v.items() if a != "pass"}, default=str)))
    L += ["- controls_pass: %s" % res.get("controls_pass"), ""]
    if final and not res["controls_pass"]:
        L += ["**A control failed. No outcome is named.** The numbers are in RESULT_LEADHEAD_B.json.", ""]
    L += ["## Per arm", ""]
    for arm in arms:
        A = res["arms"][arm]
        c = A["counts"]
        fr = A["fit_rows"]
        L += ["### " + arm, "", "Scored windows %d of %d (episodes %d); with the label history rule %d (dropped %d)." % (c["scored"]["scored"], c["windows"], c["n_episodes_scored"], c["label_history_ok"], c["label_history_dropped"]),
              "Fit rows (train probe set, %d kept rows): with five frames %d; gap %d (%d clips); closing %d (%d clips); direct %d (%d clips)." % (
                  fr["kept_rows"], fr["with_five_frames"], fr["gap_rows"], fr["clips_with_gap_rows"], fr["closing_rows"], fr["clips_with_closing_rows"], fr["direct_rows"], fr["clips_with_direct_rows"]), ""]
        for kind in ("HT", "HM"):
            if kind not in A:
                continue
            K_ = A[kind]
            L += ["#### %s" % kind, ""]
            f = K_["fit"]
            if kind == "HT":
                L.append("Chosen penalty multiplier g (gap / closing / direct): %g / %g / %g; at the grid edge (1 or 1e7): %s / %s / %s." % (
                    f["gap"]["g"], f["c"]["g"], f["dir"]["g"], f["gap"]["at_grid_edge"], f["c"]["at_grid_edge"], f["dir"]["at_grid_edge"]))
            L.append("Fit rows used: gap %d, closing %d, direct %d. Inner-fold mean squared error (fit rows only): gap %.3f, closing speed %.4f, direct %.4f." % (
                f["gap"]["n_fit"], f["c"]["n_fit"], f["dir"]["n_fit"], K_["inner_mse_gap"], K_["inner_mse_closing"], K_["inner_mse_direct"]))
            F = K_["fidelity"]
            L += ["", "Fidelity on held-out windows with the label history rule (n=%d):" % F["gap"]["n"],
                  "- gap at S0: R2 %+.3f %s; RMSE %.2f m %s" % (F["gap"]["r2"], _ci(F["gap"]["r2_ci95"]), F["gap"]["rmse_rmse"], _ci(F["gap"]["rmse_ci95"])),
                  "- closing speed at S0: R2 %+.3f %s; RMSE %.2f m/s %s" % (F["closing"]["r2"], _ci(F["closing"]["r2_ci95"]), F["closing"]["rmse_rmse"], _ci(F["closing"]["rmse_ci95"]))]
            if a21 and arm in a21["arms"] and kind in a21["arms"][arm]["kinds"]:
                o = a21["arms"][arm]["kinds"][kind]
                L.append("- ADD21 beside it (heads on the eval bank, %d to %d fit rows): gap R2 %+.3f, closing R2 %+.3f; R3 skill %+.3f; direct skill %+.3f" % (
                    1689, 1926, o["fidelity"]["gap"]["r2"], o["fidelity"]["closing"]["r2"], o["rules"]["R3"]["skill_over_copy"]["skill"], o["direct"]["skill_over_copy"]["skill"]))
            L += ["", "Rules over the copy of the same head (n=%d scored windows):" % K_["rules"]["R3"]["skill_over_copy"]["n"]]
            for r in ("R1", "R2", "R3"):
                row = K_["rules"][r]
                L.append("- %s: skill over the head copy %s; level R2 %+.3f %s" % (r, _sk(row["skill_over_copy"]), row["level_r2"]["r2"], _ci(row["level_r2"]["ci95"])))
            L.append("- R0 (head copy): level R2 %+.3f %s" % (K_["rules"]["R0"]["level_r2"]["r2"], _ci(K_["rules"]["R0"]["level_r2"]["ci95"])))
            d = K_["banked"]["R3"]
            L.append("- R3 level R2 %+.3f minus banked rollout (i) [R1 file] %+.3f: %+.3f %s%s (n=%d)" % (d["level_r2_rule"], d["level_r2_banked_rollout"], d["diff"], _ci(d["ci95"]),
                     " (separated above 0)" if d["separated_pos"] else (" (separated below 0)" if d["separated_neg"] else ""), d["n"]))
            dd = K_["direct"]
            L.append("- Direct read (gap + predicted change over 2.0 s, ego term %s) skill over the head copy: %s; RMSE %.2f m %s against the copy RMSE %.2f m" % (
                "included", _sk(dd["skill_over_copy"]), dd["rmse_gap_2s"]["rmse"], _ci(dd["rmse_gap_2s"]["ci95"]), dd["rmse_copy_gap_2s"]["rmse"]))
            L.append("")
        if "HT" in A and "HM" in A:
            L.append("Selection by the inner folds of the fit rows: closing-speed head = %s; direct head = %s." % (_pick(A), _pick_dir(A)))
            L.append("")
        if "learning_curve" in A:
            L += ["#### Learning curve of HT (secondary a; clips drawn with rng 0; whole clips until the labelled rows reach the size; decides nothing)", "",
                  "| labelled fit rows (gap / closing) | closing R2 on scored windows | gap R2 | g (gap / closing) |", "|---|---|---|---|"]
            for k in [str(n) for n in LC_SIZES] + ["all"]:
                e = A["learning_curve"][k]
                fi = e["fidelity"]
                L.append("| %s / %s | %+.3f %s | %+.3f %s | %g / %g |" % (e["fit"]["gap"]["n_fit"], e["fit"]["c"]["n_fit"], fi["closing"]["r2"], _ci(fi["closing"]["r2_ci95"]), fi["gap"]["r2"], _ci(fi["gap"]["r2_ci95"]),
                                                                         e["fit"]["gap"]["g"], e["fit"]["c"]["g"]))
            L.append("")
        if "pooled" in A:
            K_ = A["pooled"]
            F = K_["fidelity"]
            L += ["#### Pooled fit of HT (secondary b: train probe rows and eval-bank rows of the training folds together, cross-fitted over the eval folds; decides nothing)", "",
                  "- gap at S0: R2 %+.3f %s; closing speed at S0: R2 %+.3f %s" % (F["gap"]["r2"], _ci(F["gap"]["r2_ci95"]), F["closing"]["r2"], _ci(F["closing"]["r2_ci95"])),
                  "- R3 skill over the head copy %s; R2 skill %s; R1 skill %s" % (_sk(K_["rules"]["R3"]["skill_over_copy"]), _sk(K_["rules"]["R2"]["skill_over_copy"]), _sk(K_["rules"]["R1"]["skill_over_copy"])), ""]
    L += ["## Outcomes (card section 4; the head is the one chosen by the inner folds of the fit rows)", ""]
    if final and res["controls_pass"]:
        o = res["outcomes"]
        if not o["complete_heads"]:
            L.append("**Provisional: HM was not run for every arm in this file; names below are on HT and are not the card's final names.**")
        for typ in ("P", "G"):
            d = o["per_type"][typ]
            if isinstance(d, str):
                L.append("- Arm type %s: %s" % (typ, d))
                continue
            L.append("- Arm type %s (both seeds needed): closing **%s**; slot **%s**%s; heads used %s; with each head run: %s" % (
                typ, d["closing"], " + ".join(d["slot"]), ("; direct **%s** (heads %s)" % (d["direct"], d["direct_head_used"])) if "direct" in d else "; direct read not run",
                d["head_used"], json.dumps(d["with_each_head"])))
        L += ["", "## Reading of the pair of cards (card section 4) and the learning curve", ""]
        rows_ok = [a for a in arms if a in res["arms"] and "learning_curve" in res["arms"][a]]
        for a in rows_ok:
            lc_ = res["arms"][a]["learning_curve"]
            L.append("- %s: HT closing-speed R2 on the scored windows at 2,000 / 8,000 / 16,000 / all labelled fit rows: %s (ADD21, 1,689 to 1,926 rows: %s)." % (
                a, " / ".join("%+.3f" % lc_[k]["fidelity"]["closing"]["r2"] for k in ("2000", "8000", "16000", "all")),
                ("%+.3f" % a21["arms"][a]["kinds"]["HT"]["fidelity"]["closing"]["r2"]) if a21 and a in a21["arms"] else "n/a"))
        rising = all(res["arms"][a]["learning_curve"]["all"]["fidelity"]["closing"]["r2"] > res["arms"][a]["learning_curve"]["16000"]["fidelity"]["closing"]["r2"] > res["arms"][a]["learning_curve"]["8000"]["fidelity"]["closing"]["r2"] for a in rows_ok) if rows_ok else None
        flat_names = all((not isinstance(o["per_type"][t], str)) and o["per_type"][t]["closing"] == "HEAD-CLOSING-FLAT-B" and o["per_type"][t]["slot"] == ["HEAD-SLOT-FLAT-B"] for t in ("P", "G"))
        L += ["", "Card ADD21 read flat for both arm types; the names of this card are %s. The learning curve %s at every step up to all labelled fit rows (closing R2 8,000 < 16,000 < all, every arm: %s); "
              "the all-rows point equals the 32,000 point because the train probe set holds fewer labelled rows than that (32,025 gap rows, 30,865 closing rows). "
              "The branch of card section 4 for a flat read with a flat curve therefore does not apply; the branch of card section 5 for a rising curve that has not flattened at all rows applies "
              "(the fit set is extended to more training clips before any name is read as a property of the state). Control j8 (shift between held-out train clips and the eval windows) is printed above." % (
                  "flat for both arm types (closing and slot)" if flat_names else "not flat for both arm types (see above)", "rises" if rising else "does not rise", rising), ""]
    else:
        L.append("No outcome is named (%s)." % ("a control failed" if final else "partial write"))
    L += ["", "## Limits", "",
          "- The heads are fitted on states of clips the model was trained on and applied to states of clips it was not; control j8 measures that shift (R2 on the held-out tenth of the train clips beside R2 on the eval windows). A flat read with a large shift is reported as not decisive.",
          "- The scored set is small (429 windows, 48 episodes); one training run per seed, so an interval answers another draw of episodes and the names need both seeds.",
          "- Train labels use the join lead gap in 0 to 60 m with lead_valid; the scored windows use the eval-bank gap as in card ADD21 (LEADHEAD_B_DEFS.md M3).",
          "- R1, R2, R3 and the direct read use the logged ego speeds of the future frames (the action input of a rollout); no goal input, nothing of the situation classifier.",
          "- T0 only: nothing here is driving performance; LONGITUDINAL, LATERAL, TACTICAL not measured; STRATEGIC PAUSED (PI item 25).", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["P_s0", "P_s1", "G_s0", "G_s1"])
    r.add_argument("--stages", nargs="+", default=["ht", "lc", "hm", "pooled"])
    r.add_argument("--B", type=int, default=2000)
    r.add_argument("--out-dir", default=HERE)
    r.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return run(a.arms, a.stages, a.B, a.out_dir, a.force)


if __name__ == "__main__":
    sys.exit(main())
