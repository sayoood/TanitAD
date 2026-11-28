r"""objslot_s0.py -- OBJSLOT stage 0 (card PREREG_V8F_PROOF_ADD20, definitions OBJSLOT_DEFS.md D1..D20).
Tier T0: a world-model diagnostic on logged frames, never driving performance.  CPU, zero training.
Can an object-level slot (gap, closing speed) with the kinematic transition  gap(t+T) = gap(t) + T*(vL) - ego travel  read the lead gap
2.0 s ahead better than a copy?  Rules R0 copy, R1 constant closing speed, R2 constant lead speed with the known ego motion, R3 = R2's two
terms with three fitted coefficients (training folds only).  Sources: LABEL (logged gaps) and VISION (the level probe of the r-b read on the grid state).

  python objslot_s0.py run [--arms P_s0 P_s1 G_s0 G_s1] [--B 2000] [--out-dir DIR]

Strictly: STRATEGIC PAUSED (PI item 25); LONGITUDINAL, LATERAL, TACTICAL are not measured by this read.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

ROOT = "F:/TanitAD-scratch/v8proof"
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
for _p in (ROOT + "/res", ROOT + "/spatial/s0", ROOT + "/spatial/q2"):
    if _p not in sys.path:
        sys.path.append(_p)
import resprobe as RP  # noqa: E402

TIER = "T0 (a world-model diagnostic on logged frames, never driving performance)"
FAMILIES = "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)"
DT = 0.2
S0F = 4
STEP_MAX_M = 3.0
H_MIN = 3
SEED = 0
RULES = ("R0", "R1", "R2", "R3")
V_ST = "C:/Users/Admin/v8proof_c/latvis_t/SPEED_WINDOWS.npz"
BANKP = ROOT + "/res/cache/frames.npz"
CT = "C:/Users/Admin/v8proof_c/ctxroll"
ARMDEF = {
    "P_s0": dict(states="C:/Users/Admin/v8proof_c/s1states/P_s0_eval.npz", pt="C:/Users/Admin/v8proof_c/latvis_p/latents/P_s0.pt"),
    "P_s1": dict(states="C:/Users/Admin/v8proof_c/s1states/P_s1_eval.npz", pt="C:/Users/Admin/v8proof_c/latvis_p/latents/P_s1.pt"),
    "G_s0": dict(states=ROOT + "/spatial/s1states/G_s0_eval.npz", pt="C:/Users/Admin/v8proof_c/latvis_gw/latents/G_s0.pt"),
    "G_s1": dict(states=ROOT + "/spatial/s1states/G_s1_eval.npz", pt="C:/Users/Admin/v8proof_c/latvis_gw/latents/G_s1.pt"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================ pure label / rule machinery
def hist_run(G, step_max=STEP_MAX_M):
    """G [n, m]: the gap on the m history frames (last column = the start frame). Returns bool mask [n, m] of the trailing run of finite
    labels whose consecutive 0.2 s steps are within step_max (the same lead, D2/D5)."""
    n, m = G.shape
    mask = np.zeros((n, m), bool)
    for i in range(n):
        if not np.isfinite(G[i, m - 1]):
            continue
        mask[i, m - 1] = True
        j = m - 2
        while j >= 0 and np.isfinite(G[i, j]) and abs(G[i, j + 1] - G[i, j]) <= step_max:
            mask[i, j] = True
            j -= 1
    return mask


def ls_slope(G, mask):
    """Least-squares slope (m/s) of the gap over the masked frames (0.2 s apart); nan below 2 frames."""
    n, m = G.shape
    t = np.arange(m, dtype=np.float64) * DT
    w = mask.astype(np.float64)
    cnt = w.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tb = (w * t).sum(1) / cnt
        Gz = np.where(mask, G, 0.0)
        gb = Gz.sum(1) / cnt
        num = (w * (t[None, :] - tb[:, None]) * (Gz - gb[:, None])).sum(1)
        den = (w * (t[None, :] - tb[:, None]) ** 2).sum(1)
        c = num / den
    c[cnt < 2] = np.nan
    return c


def scored_mask(G15, K, step_max=STEP_MAX_M):
    """Windows with a finite gap on every frame S0 .. S0+K and no step above step_max (note N6). Returns (mask, counts)."""
    seg = G15[:, S0F:S0F + K + 1]
    finite_all = np.isfinite(seg).all(1)
    with np.errstate(invalid="ignore"):
        jump = np.nanmax(np.abs(np.diff(np.where(np.isfinite(seg), seg, np.nan), axis=1)), axis=1) > step_max
    jump = np.where(finite_all, jump, False)
    ends = np.isfinite(seg[:, 0]) & np.isfinite(seg[:, -1])
    ok = finite_all & ~jump
    cnt = {"windows": int(len(G15)), "lead_at_S0_and_target": int(ends.sum()), "scored": int(ok.sum()),
           "dropped_lead_changed_or_gap_missing": int((ends & ~ok).sum()), "of_which_jump_above_step_max": int((ends & finite_all & jump).sum()),
           "of_which_gap_missing_in_between": int((ends & ~finite_all).sum())}
    return ok, cnt


def ego_terms(V, K):
    """V [n, >=5+K] logged speeds from frame 0. Returns (V4, ego_sum) with ego_sum = sum_k DT * mean(V[4+k], V[5+k])."""
    ego_sum = np.zeros(len(V))
    for k in range(K):
        ego_sum += DT * 0.5 * (V[:, S0F + k] + V[:, S0F + k + 1])
    return V[:, S0F].astype(np.float64), ego_sum


def rule_inputs(g0, c, V, K):
    """The inputs of the rules: gap(S0), closing speed term T*c, ego term X (= T*V4 - ego travel), so that R2 = g0 + T*c + X."""
    T = DT * K
    V4, ego_sum = ego_terms(V, K)
    return {"g0": g0, "Tc": T * c, "X": T * V4 - ego_sum, "T": T}


def rule_preds(inp):
    g0, Tc, X = inp["g0"], inp["Tc"], inp["X"]
    return {"R0": g0.copy(), "R1": g0 + Tc, "R2": g0 + Tc + X}


def fit_r3(inp, y, valid, fold, nfold=5):
    """R3: g0 + a*Tc + b*X + d, (a, b, d) by least squares on the scored windows of the training folds, applied to the held-out fold."""
    g0, Tc, X = inp["g0"], inp["Tc"], inp["X"]
    n = len(g0)
    pred = np.full(n, np.nan)
    coefs = {}
    fin = np.isfinite(g0) & np.isfinite(Tc) & np.isfinite(X)
    for f in range(nfold):
        tr = valid & fin & np.isfinite(y) & (fold != f)
        te = fin & (fold == f)
        if tr.sum() < 10 or not te.any():
            continue
        A = np.c_[Tc[tr], X[tr], np.ones(int(tr.sum()))]
        coef = np.linalg.lstsq(A, (y - g0)[tr], rcond=None)[0]
        pred[te] = g0[te] + np.c_[Tc[te], X[te], np.ones(int(te.sum()))] @ coef
        coefs[f] = [float(v) for v in coef]
    return pred, coefs


def label_pipeline(G15, V, K, fold, y=None, hmin=H_MIN):
    """The LABEL path: history run -> closing speed -> rules R0..R3. Returns dict with the rule predictions, the scored mask, the rule mask."""
    sc, cnt = scored_mask(G15, K)
    Gh = G15[:, :S0F + 1]
    m = hist_run(Gh)
    c = ls_slope(Gh, m)
    hok = m.sum(1) >= hmin
    c = np.where(hok, c, np.nan)
    inp = rule_inputs(G15[:, S0F], c, V, K)
    pr = rule_preds(inp)
    y = G15[:, S0F + K] if y is None else y
    valid = sc & hok
    pr["R3"], coefs = fit_r3(inp, y, valid, fold)
    return {"pred": pr, "scored": sc, "valid": valid, "y": y, "inp": inp, "counts": cnt, "coefs": coefs,
            "history_ok": int((sc & hok).sum()), "history_dropped": int((sc & ~hok).sum())}


def vision_pipeline(VG, V, K, y, scored, fold):
    """The VISION path. VG [n, 5] = the probe values at frames 0..4 (nothing later enters). y = logged target gap (fit target only for R3)."""
    c = ls_slope(VG, np.ones(VG.shape, bool))
    inp = rule_inputs(VG[:, S0F], c, V, K)
    pr = rule_preds(inp)
    pr["R3"], coefs = fit_r3(inp, y, scored, fold)
    return {"pred": pr, "scored": scored, "valid": scored, "y": y, "inp": inp, "coefs": coefs}


# ============================================================================================ bootstrap and skills
def epi_of(eids, mask):
    ue = sorted(set(eids[mask]))
    idx = {e: i for i, e in enumerate(ue)}
    ep = np.array([idx.get(e, -1) for e in eids])
    return ep, len(ue)


def skill_point(y, p, p0, valid):
    m = valid & np.isfinite(y) & np.isfinite(p) & np.isfinite(p0)
    sse, sse0 = ((y - p)[m] ** 2).sum(), ((y - p0)[m] ** 2).sum()
    return float(1.0 - sse / sse0) if sse0 > 0 else float("nan")


def skill_boot(y, p, p0, valid, ep, n_ep, C):
    m = valid & np.isfinite(y) & np.isfinite(p) & np.isfinite(p0) & (ep >= 0)
    Sr = np.zeros(n_ep)
    S0 = np.zeros(n_ep)
    np.add.at(Sr, ep[m], (y - p)[m] ** 2)
    np.add.at(S0, ep[m], (y - p0)[m] ** 2)
    return 1.0 - (C @ Sr) / (C @ S0)


def skill_row(y, p, p0, valid, ep, n_ep, C):
    pt = skill_point(y, p, p0, valid)
    b = skill_boot(y, p, p0, valid, ep, n_ep, C)
    c = RP.ci(b)
    n = int((valid & np.isfinite(y) & np.isfinite(p) & np.isfinite(p0)).sum())
    return {"skill": pt, "ci95": c, "n": n, "separated_pos": bool(pt > 0 and c[0] > 0), "separated_neg": bool(c[1] < 0)}, b


def r2_level(y, p, valid):
    m = valid & np.isfinite(y) & np.isfinite(p)
    return float(RP.r2_pooled(y[m], p[m])) if m.sum() > 2 else float("nan")


def level_row(y, p, valid, ep, n_ep, C):
    m = valid & np.isfinite(y) & np.isfinite(p) & (ep >= 0)
    S = RP.ep_stats_r2(ep[m], y[m], p[m], n_ep)
    b = RP.boot_r2(S, C)
    return {"r2": r2_level(y, p, valid), "ci95": RP.ci(b), "n": int(m.sum())}, b


def table(pipe, ep, n_ep, C, rules=RULES):
    """Skill over R0 of the same source (on the windows of that rule) and the level R2, for every rule; returns (rows, boots)."""
    out, boots = {}, {}
    y, v = pipe["y"], pipe["valid"]
    for r in rules[1:]:
        row, b = skill_row(y, pipe["pred"][r], pipe["pred"]["R0"], v & np.isfinite(pipe["pred"][r]), ep, n_ep, C)
        lv, lb = level_row(y, pipe["pred"][r], v & np.isfinite(pipe["pred"][r]), ep, n_ep, C)
        out[r] = {"skill_over_copy": row, "level_r2": lv}
        boots[r] = b
    lv0, _ = level_row(y, pipe["pred"]["R0"], v, ep, n_ep, C)
    out["R0"] = {"level_r2": lv0, "skill_over_copy": {"skill": 0.0, "n": int(v.sum())}}
    return out, boots


# ============================================================================================ controls (label path, synthetic)
def planted_series(V, base, vL, K=10, linear_history=True):
    """G15 [n,15]: frames 0..4 linear with slope vL - V4 (c exact) when linear_history, else strictly constant lead speed (history by the
    same identity run backward); frames 4..14 by the identity gap_{f+1} = gap_f + DT*(vL - mean(V_f, V_{f+1}))."""
    n = len(V)
    G = np.full((n, 15), np.nan)
    G[:, S0F] = base
    for f in range(S0F, 14):
        G[:, f + 1] = G[:, f] + DT * (vL - 0.5 * (V[:, f] + V[:, f + 1]))
    if linear_history:
        for f in range(S0F):
            G[:, f] = base + (vL - V[:, S0F]) * DT * (f - S0F)
    else:
        for f in range(S0F - 1, -1, -1):
            G[:, f] = G[:, f + 1] - DT * (vL - 0.5 * (V[:, f] + V[:, f + 1]))
    return G


def control_o1(V, fold, K=10, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(V)
    base = rng.uniform(15, 60, n)
    vL = V[:, S0F] + rng.uniform(-3, 3, n)
    G = planted_series(V, base, vL, K, True)
    P = label_pipeline(G, V, K, fold)
    y = G[:, S0F + K]
    v = P["valid"]
    s2 = skill_point(y, P["pred"]["R2"], P["pred"]["R0"], v)
    s1 = skill_point(y, P["pred"]["R1"], P["pred"]["R0"], v)
    Gc = planted_series(V, base, vL, K, False)
    Pc = label_pipeline(Gc, V, K, fold)
    yc = Gc[:, S0F + K]
    s2c = skill_point(yc, Pc["pred"]["R2"], Pc["pred"]["R0"], Pc["valid"])
    ev = float(np.std(V[:, S0F:S0F + K + 1], axis=1).mean())
    ok = bool(abs(s2 - 1.0) <= 1e-9 and s1 < 1.0 - 1e-6 and ev > 0)
    return {"pass": ok, "r2_skill": s2, "r1_skill": s1, "n": int(v.sum()), "n_planted": int(n), "mean_ego_speed_std_over_window": ev,
            "tolerance": 1e-9, "report_only_strict_constant_lead_speed_history_r2_skill": s2c, "n_strict": int(Pc["valid"].sum())}


def control_o3(pipe):
    y, v = pipe["y"], pipe["valid"]
    copy = skill_point(y, pipe["pred"]["R0"], pipe["pred"]["R0"], v)
    true = skill_point(y, y.copy(), pipe["pred"]["R0"], v)
    return {"pass": bool(copy == 0.0 and true == 1.0), "copy_skill": copy, "true_as_rule_skill": true}


def control_o2(inp_pipe_fn, y, valid, ep, n_ep, C, seed=SEED):
    """inp_pipe_fn(y_perm) -> pipe built with the permuted future gap; every rule at most +0.02."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(valid)
    yp = y.copy()
    yp[idx] = y[rng.permutation(idx)]
    pipe = inp_pipe_fn(yp)
    vals = {}
    for r in RULES[1:]:
        v = pipe["valid"] & np.isfinite(pipe["pred"][r])
        vals[r] = skill_point(yp, pipe["pred"][r], pipe["pred"]["R0"], v)
    return {"pass": bool(all(x <= 0.02 for x in vals.values())), "bar": 0.02, "skill_by_rule": vals}


def control_o5(G15, V, K, fold):
    """R1 with the history moved one frame into the future (frames 1..5, base gap = frame 5, same T) against R1, both over the S0 copy."""
    P = label_pipeline(G15, V, K, fold)
    y = P["y"]
    G5 = G15[:, 1:6]
    m = hist_run(G5)
    c = ls_slope(G5, m)
    c = np.where(m.sum(1) >= H_MIN, c, np.nan)
    T = DT * K
    p_shift = G15[:, 5] + T * c
    v = P["valid"] & np.isfinite(p_shift) & np.isfinite(P["pred"]["R1"])
    s_shift = skill_point(y, p_shift, P["pred"]["R0"], v)
    s_r1 = skill_point(y, P["pred"]["R1"], P["pred"]["R0"], v)
    return {"pass": bool(s_shift > s_r1), "r1_skill": s_r1, "r1_shifted_history_skill": s_shift, "n": int(v.sum())}


def control_o6_label_inputs():
    """The rule functions take only history gaps, ego speeds and gap(S0): asserted by signature."""
    import inspect
    names = {"rule_inputs": ["g0", "c", "V", "K"], "rule_preds": ["inp"], "ls_slope": ["G", "mask"]}
    got = {k: list(inspect.signature(globals()[k]).parameters) for k in names}
    return {"pass": got == names, "signatures": got}


# ============================================================================================ data (real)
def load_label_frames(eid, g0):
    Z = np.load(BANKP)
    beid, bg, gap = Z["beid"].astype(str), Z["bg"].astype(np.int64), Z["lead_gap"].astype(np.float64)
    pos = {(e, int(g)): i for i, (e, g) in enumerate(zip(beid, bg))}
    G = np.full((len(eid), 15), np.nan)
    miss = 0
    for j, (e, g) in enumerate(zip(eid, g0)):
        for f in range(15):
            i = pos.get((e, int(g) - S0F + f), -1)
            if i >= 0:
                G[j, f] = gap[i]
            else:
                miss += 1
    return G, miss


def load_arm(arm):
    import rb_trained as RB
    cfg = ARMDEF[arm]
    lat = "%s/LATENTS_CTX_%s_R1.npz" % (CT, arm)
    D = RB.load(cfg["states"], lat, 640)
    return D, lat


def arm_windows(D):
    """Window arrays of one arm: eid, g0 (S0 step), t = g0 + 11, fold (fold of the episode in RP.make_folds of the bank rows)."""
    folds_b = RP.make_folds(D["beid"])
    fold_of = {e: int(f) for e, f in zip(D["beid"], folds_b)}
    fw = np.array([fold_of[e] for e in D["w_eid"]])
    if D["w_fold_file"] is not None:
        assert bool(np.array_equal(fw, np.array([int(x) for x in D["w_fold_file"]]))), "fold map differs from the latents file"
    return D["w_eid"].astype(str), D["w_g0"].astype(np.int64), fw, folds_b


def speed_and_states(arm, eid, g0):
    import torch
    S = np.load(V_ST, allow_pickle=True)
    key = {(str(e), int(t)): i for i, (e, t) in enumerate(zip(S["eid"].astype(str), S["t"]))}
    idx = np.array([key.get((str(e), int(g) + 11), -1) for e, g in zip(eid, g0)])
    assert (idx >= 0).all(), "windows without a speed row"
    V = np.asarray(S["V"], np.float64)[idx]
    assert np.isfinite(V).all(), "non-finite logged speed"
    d = torch.load(ARMDEF[arm]["pt"], map_location="cpu", weights_only=False)
    Z1 = d["Z1"].float().numpy()[idx]
    assert bool(np.asarray(d["filled"])[idx].all()), "unfilled window"
    return V, Z1, idx


def vision_gaps(fits, Z1, fold, frames=S0F + 1):
    """Probe values at frames 0..frames-1 of every window from the fit of ITS fold. Only those frames of Z1 are read."""
    n = len(Z1)
    VG = np.full((n, frames), np.nan)
    for f, fit in fits.items():
        te = np.flatnonzero(fold == f)
        if len(te) == 0:
            continue
        X = Z1[te][:, :frames, :].astype(np.float64)
        VG[te] = fit(X.reshape(-1, X.shape[-1])).reshape(len(te), frames)
    return VG


def fit_vision_probes(D, folds_b):
    import s0_read as S
    fits = {}
    gap = D["gap"]
    for f in range(5):
        tr = np.flatnonzero(folds_b != f)
        vtr = tr[np.isfinite(gap[tr])]
        fits[f] = S.fit_probe(D["Sb"][vtr], gap[vtr], D["epi"][vtr], S.SEED + f)
        log("probe fold", f, "fitted on", len(vtr), "bank rows")
    return fits


def jdump(o, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


# ============================================================================================ main run
HORIZONS = (2, 4, 6, 8, 10)


def run(arms, B, out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    res = {"tier": TIER, "families": FAMILIES, "card": "PREREG_V8F_PROOF_ADD20", "B": B, "arms": {}, "controls": {}, "counts": {}}
    base = None
    cache = {}
    for arm in arms:
        D, lat = load_arm(arm)
        eid, g0, fold, folds_b = arm_windows(D)
        if base is None:
            base = (eid, g0, fold)
        else:
            assert np.array_equal(base[0], eid) and np.array_equal(base[1], g0) and np.array_equal(base[2], fold), "arms list different windows"
        cache[arm] = (D, lat)
        log(arm, "windows", len(eid), "episodes", len(D["uep"]))
    eid, g0, fold = base
    n = len(eid)
    G15, miss = load_label_frames(eid, g0)
    res["counts"].update({"windows_in_S0_file": int(n), "label_frames_without_bank_row": int(miss), "frame0_read_windows_banked": 1809})
    # ------------------------------------------------------------------ LABEL source
    V, _, _ = speed_and_states(arms[0], eid, g0)
    lab = {}
    ctrl = {}
    for K in HORIZONS:
        P = label_pipeline(G15, V, K, fold)
        ep, n_ep = epi_of(eid, P["scored"])
        C = RP.boot_counts(n_ep, B, SEED)
        tb, _ = table(P, ep, n_ep, C)
        lab[K] = {"counts": P["counts"], "history_ok": P["history_ok"], "history_dropped": P["history_dropped"], "n_episodes": n_ep, "rules": tb,
                  "coefs": P["coefs"]}
        if K == 10:
            P10, ep10, n10, C10 = P, ep, n_ep, C
    res["label"] = lab
    log("label done", "scored", lab[10]["counts"]["scored"])
    ctrl["o1_planted_series"] = control_o1(V, fold)
    ctrl["o3_identity_label"] = control_o3(P10)
    ctrl["o5_leak_detector"] = control_o5(G15, V, 10, fold)
    ctrl["o6_rule_inputs_label"] = control_o6_label_inputs()
    ctrl["o2_shuffled_target_label"] = control_o2(lambda yp: label_pipeline(G15, V, 10, fold, y=yp), P10["y"], P10["valid"], ep10, n10, C10)
    # ------------------------------------------------------------------ VISION source
    outc = {}
    for arm in arms:
        D, lat = cache[arm]
        folds_b = RP.make_folds(D["beid"])
        _, Z1, _ = speed_and_states(arm, eid, g0)
        vgp = "%s/VG_%s.npz" % (out_dir, arm)
        fits = fit_vision_probes(D, folds_b)
        VG = vision_gaps(fits, Z1, fold)
        np.savez(vgp, VG=VG, eid=eid, g0=g0)
        bk = np.load("%s/rb/OOF_RB_%s_R1.npz" % (CT, arm), allow_pickle=True)
        assert np.array_equal(bk["w_eid"].astype(str), eid) and np.array_equal(bk["w_g0"], g0), "banked R1 file lists other windows"
        ykb = bk["yk"]
        # o4 probe fidelity
        okb = np.isfinite(ykb)
        r2_mine = float(RP.r2_pooled(ykb[okb], VG[okb, S0F]))
        r2_bank = float(RP.r2_pooled(ykb[okb], bk["p_ii"][okb]))
        maxdiff = float(np.nanmax(np.abs(VG[okb, S0F] - bk["p_ii"][okb])))
        o4 = {"pass": bool(abs(r2_mine - r2_bank) <= 0.005), "level_r2_vision_R0": r2_mine, "banked_level_r2_ii": r2_bank, "n": int(okb.sum()),
              "tol": 0.005, "max_abs_diff_of_the_two_gap_values": maxdiff}
        # o6 mutation checks
        rng = np.random.default_rng(3)
        Zm = Z1.copy()
        Zm[:, S0F + 1:, :] = rng.standard_normal(Zm[:, S0F + 1:, :].shape).astype(np.float32) * 5
        VGm = vision_gaps(fits, Zm, fold)
        d_state = float(np.abs(VGm - VG).max())
        arm_res = {"controls": {"o4_probe_fidelity": o4}, "horizons": {}, "banked": {}}
        for K in HORIZONS:
            sc, cnt = scored_mask(G15, K)
            ep, n_ep = epi_of(eid, sc)
            C = RP.boot_counts(n_ep, B, SEED)
            y = G15[:, S0F + K]
            PV = vision_pipeline(VG, V, K, y, sc, fold)
            tb, boots = table(PV, ep, n_ep, C)
            arm_res["horizons"][K] = {"counts": cnt, "n_episodes": n_ep, "rules": tb, "coefs": PV["coefs"]}
            if K == 10:
                # paired against the banked rollout reads (i) on the common scored windows
                for nm, fn in (("R1", "OOF_RB_%s_R1.npz" % arm), ("R3h", "OOF_RB_%s_R3h.npz" % arm), ("R3g", "OOF_RB_%s_R3g.npz" % arm)):
                    bz = np.load("%s/rb/%s" % (CT, fn), allow_pickle=True)
                    assert np.array_equal(bz["w_eid"].astype(str), eid) and np.array_equal(bz["w_g0"], g0)
                    pi = bz["p_i"]
                    vv = sc & np.isfinite(PV["pred"]["R3"]) & np.isfinite(pi)
                    m = vv & (ep >= 0)
                    dr = {}
                    for rr in ("R3", "R2", "R1"):
                        pa = PV["pred"][rr]
                        m2 = vv & np.isfinite(pa) & (ep >= 0)
                        Sa = RP.ep_stats_r2(ep[m2], y[m2], pa[m2], n_ep)
                        Sb = RP.ep_stats_r2(ep[m2], y[m2], pi[m2], n_ep)
                        dd = RP.diff_row(RP.boot_r2(Sa, C), RP.boot_r2(Sb, C), RP.r2_pooled(y[m2], pa[m2]), RP.r2_pooled(y[m2], pi[m2]))
                        dd["level_r2_rule"] = float(RP.r2_pooled(y[m2], pa[m2]))
                        dd["level_r2_banked_rollout"] = float(RP.r2_pooled(y[m2], pi[m2]))
                        dd["n"] = int(m2.sum())
                        dr[rr] = dd
                    arm_res["banked"][nm] = dr
                # o6: fold-label leak mutation of R3 and the vision copy of the same arm
                lm = {}
                dmax = 0.0
                for f in range(5):
                    yp = y.copy()
                    te = fold == f
                    yp[te] = rng.standard_normal(int(te.sum())) * 50
                    Pm = vision_pipeline(VG, V, K, yp, sc, fold)
                    dmax = max(dmax, float(np.nanmax(np.abs(Pm["pred"]["R3"][te] - PV["pred"]["R3"][te]))))
                arm_res["controls"]["o6_no_future_input"] = {"pass": bool(d_state == 0.0 and dmax == 0.0), "max_abs_change_when_states_after_frame_4_are_random": d_state,
                                                             "max_abs_change_of_R3_when_heldout_fold_targets_are_random": dmax}
                # o3 / o2 for the vision source
                arm_res["controls"]["o3_identity"] = control_o3(PV)
                arm_res["controls"]["o2_shuffled_target"] = control_o2(lambda yp, _VG=VG: vision_pipeline(_VG, V, 10, yp, sc, fold), y, sc, ep, n_ep, C)
                arm_res["R3_n_windows"] = int(sc.sum())
        res["arms"][arm] = arm_res
        log(arm, "vision done", "R3 skill %+.3f" % arm_res["horizons"][10]["rules"]["R3"]["skill_over_copy"]["skill"], "(%.0f s)" % (time.time() - t0))
    for arm, a in res["arms"].items():
        for k, v in a["controls"].items():
            ctrl["%s|%s" % (k, arm)] = v
    res["controls"] = ctrl
    res["controls_pass"] = bool(all(v["pass"] for v in ctrl.values()))
    res["outcomes"] = outcomes(res, arms) if res["controls_pass"] else {"named": False, "reason": "a control failed; no outcome is named"}
    res["elapsed_s"] = round(time.time() - t0, 1)
    jdump(res, os.path.join(out_dir, "RESULT_OBJSLOT_S0.json"))
    write_verdict(os.path.join(out_dir, "VERDICT_OBJSLOT_S0.md"), res, arms)
    log("controls", "PASS" if res["controls_pass"] else "FAILED", "elapsed %.0f s" % res["elapsed_s"])
    return 0 if res["controls_pass"] else 2


def outcomes(res, arms):
    o = {"named": True}
    r2 = res["label"][10]["rules"]["R2"]["skill_over_copy"]
    o["label"] = "KIN-CEILING-HOLDS" if r2["separated_pos"] else "KIN-CEILING-FLAT"
    for typ in ("P", "G"):
        seeds = [a for a in arms if a.startswith(typ + "_")]
        if len(seeds) < 2:
            o[typ] = "not named (needs both seeds; have %s)" % seeds
            continue
        im = all(res["arms"][a]["horizons"][10]["rules"]["R3"]["skill_over_copy"]["separated_pos"] for a in seeds)
        names = []
        if im:
            names.append("SLOT-IMAGINES")
            bt = all(res["arms"][a]["banked"]["R1"]["R3"]["separated_pos"] for a in seeds)
            if bt:
                names.append("SLOT-BEATS-ROLLOUT")
            bt_other = {nm: all(res["arms"][a]["banked"][nm]["R3"]["separated_pos"] for a in seeds) for nm in ("R3h", "R3g")}
            o[typ + "_beats_rollout_variants_context3"] = bt_other
        else:
            names.append("SLOT-FLAT")
        o[typ] = names
    ceil = res["label"][10]["rules"]["R2"]["skill_over_copy"]["skill"]
    o["share_of_label_ceiling"] = {a: (res["arms"][a]["horizons"][10]["rules"]["R3"]["skill_over_copy"]["skill"] / ceil if ceil > 0 else None) for a in arms}
    return o


def _ci(c):
    return "[%+.3f, %+.3f]" % (c[0], c[1])


def _sk(row):
    return "%+.3f %s n=%d%s" % (row["skill"], _ci(row["ci95"]), row["n"], " (separated above 0)" if row["separated_pos"] else (" (separated below 0)" if row.get("separated_neg") else ""))


def write_verdict(path, res, arms):
    L = ["# VERDICT OBJSLOT stage 0 (PREREG_V8F_PROOF_ADD20)", "", "Tier " + TIER + ". " + FAMILIES + ". Zero training; CPU; the banked states and labels.",
         "Ego speeds of the rollout are the logged ones (the action input of a rollout, teacher-forced as in every T0 read). Definitions: OBJSLOT_DEFS.md.", "",
         "## Controls (first)", ""]
    for k, v in res["controls"].items():
        L.append("- %s: %s :: %s" % (k, "PASS" if v["pass"] else "FAIL", json.dumps({a: b for a, b in v.items() if a != "pass"}, default=str)))
    L += ["- controls_pass: %s" % res["controls_pass"], ""]
    if not res["controls_pass"]:
        L += ["**A control failed. No outcome is named.** The numbers are in RESULT_OBJSLOT_S0.json.", ""]
    lab = res["label"][10]
    c = lab["counts"]
    L += ["## LABEL source (privileged ceiling; one result)", "",
          "Windows %d; lead at S0 and at frame 14: %d; dropped by the continuity rule: %d (jump above 3.0 m: %d; gap missing in between: %d); scored: %d; with the history rule (>= 3 labelled frames): %d (dropped %d); episodes %d."
          % (c["windows"], c["lead_at_S0_and_target"], c["dropped_lead_changed_or_gap_missing"], c["of_which_jump_above_step_max"], c["of_which_gap_missing_in_between"], c["scored"],
             lab["history_ok"], lab["history_dropped"], lab["n_episodes"]), ""]
    for r in ("R1", "R2", "R3"):
        row = lab["rules"][r]
        L.append("- %s: skill over the label copy %s; level R2 %+.3f %s" % (r, _sk(row["skill_over_copy"]), row["level_r2"]["r2"], _ci(row["level_r2"]["ci95"])))
    L.append("- R0 (copy): skill 0 by definition; level R2 %+.3f %s" % (lab["rules"]["R0"]["level_r2"]["r2"], _ci(lab["rules"]["R0"]["level_r2"]["ci95"])))
    L += ["", "## VISION source (grid state, level probe of the r-b read; frames 0 to 4 only)", ""]
    for arm in arms:
        a = res["arms"][arm]
        h = a["horizons"][10]
        L += ["### " + arm, "", "scored windows %d of %d; episodes %d" % (h["counts"]["scored"], h["counts"]["windows"], h["n_episodes"]), ""]
        for r in ("R1", "R2", "R3"):
            row = h["rules"][r]
            L.append("- %s: skill over the vision copy %s; level R2 %+.3f %s" % (r, _sk(row["skill_over_copy"]), row["level_r2"]["r2"], _ci(row["level_r2"]["ci95"])))
        L.append("- R0 (vision copy): level R2 %+.3f %s" % (h["rules"]["R0"]["level_r2"]["r2"], _ci(h["rules"]["R0"]["level_r2"]["ci95"])))
        for nm, dr in a["banked"].items():
            d = dr["R3"]
            L.append("- R3 level R2 %+.3f minus banked rollout (i) [%s] %+.3f: %+.3f %s%s (n=%d)" % (d["level_r2_rule"], nm, d["level_r2_banked_rollout"], d["diff"], _ci(d["ci95"]),
                     " (separated above 0)" if d["separated_pos"] else (" (separated below 0)" if d["separated_neg"] else ""), d["n"]))
        L.append("")
    L += ["## Outcomes (card section 4)", ""]
    if res["controls_pass"]:
        o = res["outcomes"]
        L.append("- Label: **%s**" % o["label"])
        for typ in ("P", "G"):
            L.append("- Vision, arm type %s (both seeds needed): **%s**" % (typ, o[typ] if isinstance(o[typ], str) else " + ".join(o[typ])))
        L.append("- Share of the label ceiling (vision R3 skill / label R2 skill; different window sets, decides nothing): " + json.dumps(o["share_of_label_ceiling"]))
    else:
        L.append("No outcome is named (a control failed).")
    L += ["", "## Horizon profile (decides nothing; skill over the copy of the same source)", "", "| horizon s | source | R1 | R2 | R3 |", "|---|---|---|---|---|"]
    for K in HORIZONS:
        T = DT * K
        row = res["label"][K]["rules"]
        L.append("| %.1f | LABEL (n=%d) | %s | %s | %s |" % (T, row["R2"]["skill_over_copy"]["n"], *[("%+.3f %s" % (row[r]["skill_over_copy"]["skill"], _ci(row[r]["skill_over_copy"]["ci95"]))) for r in ("R1", "R2", "R3")]))
        for arm in arms:
            row = res["arms"][arm]["horizons"][K]["rules"]
            L.append("| %.1f | %s (n=%d) | %s | %s | %s |" % (T, arm, row["R3"]["skill_over_copy"]["n"], *[("%+.3f %s" % (row[r]["skill_over_copy"]["skill"], _ci(row[r]["skill_over_copy"]["ci95"]))) for r in ("R1", "R2", "R3")]))
    L += ["", "## Limits", "",
          "- The scored set is small and one training run per seed; a separated interval answers another draw of episodes only (the vision outcome needs both seeds).",
          "- The probe is the level probe of the r-b read, fitted on levels and not on changes; vision history uses all five frames, presence is not a vision output.",
          "- R2 and R3 use the logged ego speeds of the future frames (the action input of a rollout); that is not a goal input and carries nothing of the situation classifier.",
          "- Label and vision rules are compared with their own copy on their own windows; the share of the ceiling compares two different window sets.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["P_s0", "P_s1", "G_s0", "G_s1"])
    r.add_argument("--B", type=int, default=2000)
    r.add_argument("--out-dir", default=HERE)
    a = ap.parse_args(argv)
    return run(a.arms, a.B, a.out_dir)


if __name__ == "__main__":
    sys.exit(main())
