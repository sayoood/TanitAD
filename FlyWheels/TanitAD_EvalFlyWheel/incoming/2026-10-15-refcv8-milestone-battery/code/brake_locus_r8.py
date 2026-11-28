"""brake_locus_r8.py -- WHERE is refcv8's cruise false-brake (H-R8-S6-LON-1) DECIDED?  HEAD / SELECTION / CANDIDATE.
CPU only, banked dumps only (numpy; no torch, no model forward, no GPU).

    cd <PKG>/code && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
      PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 brake_locus_r8.py            # writes <PKG>/raw/refcv9_headroom/BRAKE_LOCUS.{json,md}

Pre-registration: raw/refcv9_headroom/PREREG_BRAKE_LOCUS.md (+ .sha256).  This tool REFUSES to run if the sealed sha256 does not match the file's bytes.
What it can answer from the banked dumps: the HEAD test (share of r8 false-brake windows whose tactical lon head says decel; false-brake rate on head-says-cruise
windows vs refcv7) plus descriptive proxies.  What it CANNOT answer: (ii) / (iii) -- the REFINED FAN is not banked (inventory in the prereg), so the 4-way label is
UNDETERMINED here; `fan_cells()` below is the pre-registered analysis for the fan dump a re-roll would produce, self-tested on a synthetic fan (control C6).
Definitions (plan class, GT-cruise, false-brake) are the instrument of `cruise_brake_retime_r8.py` / `s6_lon_decomp.py`, IMPORTED, not re-implemented; the only local
re-implementation is the threshold-parametrised twin (`CB.plan_class_thr`) used by the mutations, asserted equal to the original at 1.5 m/s.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "refcv9_headroom"
DEP = PKG / "raw" / "retime_deploy"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
TREE = "step30000ema_unc"
R7ROOT = "D:/refcv7_eval_kit/battery/step50400_a6text"
SEEDS = (0, 1)
DT = 1.0
S6_SLOTS = [1, 3, 4, 5, 6, 7]
N_BASE = 117                       # manifest model.n_anchors (checkpoint anchor buffer); fan = 117 base + 32 appended = 149
LON = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD", "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")
D_SET = (2, 3, 4, 5, 6)            # decel-class set D (PREREG section 2)
D_PLUS = (0, 2, 3, 4, 5, 6)        # sensitivity: FOLLOW added
THR_I, E_SEL, E_CAND = 0.6, 0.5, 0.3
X_FDE6_M, X_ADE_M = 2.91, 1.21     # PREREG section 4 (median refcv7 FDE@6 s / ADE on GT-cruise compliant windows, both seeds pooled)
N_PERM = 1000
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
ESTIMATOR = ("episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed pair_read_r8.SEED: answers 'another draw of EPISODES' only; "
             "NOT another training run (single training seed per arm, H-ESTIM-SEED-1), NOT another inference run (seeds 0 and 1 shown separately, never pooled)")

# ---- LITERALS for control C1, typed from raw/refcv9_headroom/L2_HEADROOM.json (compliant surface); never derived from the code under test
LIT = {"n_compliant": 3251, "ep_compliant": 138, "n_cruise": 1856, "ep_cruise": 135,
       "fd_r8": {0: 0.265625, 1: 0.25754310344827586}, "fd_r7": {0: 0.09105603448275862, 1: 0.09375}}


# ------------------------------------------------------------------------------------------ helpers
def sha256_bytes(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fl(x):
    return None if x is None else float(x)


# ------------------------------------------------------------------------------------------ loading (numpy only)
DEC_KEYS = ("sel_idx", "sel_idx_base", "a_star", "sel_agrees_oracle", "lon_pred_nav_true", "lon_label")


def load_tree(root: str, seed: int, want_r7_embedded: bool) -> dict:
    """S6 panel + decisions + extras join for one tree/seed, in the panel's own window order (sorted ep files)."""
    pdir = f"{root}/s6_s{seed}"
    man = jload(f"{pdir}/manifest.json")
    epi = {int(e["file_index"]): e for e in man["episodes"]}
    cols: dict = {}
    for f in sorted(glob.glob(pdir + "/ep*.npz")):
        fi = int(os.path.basename(f)[2:5])
        z, d = np.load(f), np.load(f"{pdir}/decisions/{os.path.basename(f)}")
        if not np.array_equal(z["ws"], d["ws"]):
            raise SystemExit(f"[brake_locus] {f}: panel ws != decisions ws")
        n = len(z["ws"])
        rec = {"os": z["os"], "g": z["g"], "v0": z["v0"].astype(np.float64), "ws": z["ws"].astype(np.int64),
               "ep_index": np.full(n, int(epi[fi]["episode_index"])), "fi": np.full(n, fi), "clip": np.array([str(epi[fi]["clip_id"])] * n, dtype=object)}
        for k in DEC_KEYS:
            if k in d.files:
                rec[k] = d[k]
        if want_r7_embedded:
            rec["b_r7"] = z[f"b_refcv7_50400_s{seed}"]
        for k, v in rec.items():
            cols.setdefault(k, []).append(v)
    T = {k: np.concatenate(v) for k, v in cols.items()}
    T["slots"] = list(man["grid"]["slots"])
    T["dt"] = float(man["grid"]["dt_s"])
    return T


def join_extras(root: str, seed: int, T: dict, by: str) -> dict:
    """rows of `dump_s{seed}/refcv6_extras.npz` for T's windows: `by='episode'` keys (episode_index, ws) (refcv8: same file indexing as the S6 panel),
    `by='clip'` keys (clip_id, ws) through the tree's own dump manifest (refcv7)."""
    ex = np.load(f"{root}/dump_s{seed}/refcv6_extras.npz")
    if by == "episode":
        key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
        rows = np.array([key[(int(e), int(w))] for e, w in zip(T["ep_index"], T["ws"])])
    else:
        man = jload(f"{root}/dump_s{seed}/manifest.json")
        cid = {int(e["episode_index"]): str(e["clip_id"]) for e in man["episodes"]}
        key = {(cid[int(c)], int(w)): i for i, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
        rows = np.array([key[(str(c), int(w))] for c, w in zip(T["clip"], T["ws"])])
    X = {"rows": rows, "path": f"{root}/dump_s{seed}/refcv6_extras.npz"}
    need = ["nav_true.tacv6_lon_logits", "nav_true.lon_logits_tac", "nav_true.traj", "nav_true.sel_idx"]
    for k in need + [k for k in ex.files if k.startswith(("filter_off.", "vmax_zero."))]:
        X[k] = ex[k][rows]
    return X



def load_sel_bank(root: str, seed: int, T: dict, n_base: int = N_BASE) -> tuple:
    """AMENDMENT 1 A1: the selected anchor's UNREFINED 8-slot bank path per S6 window, NaN where the selected candidate is an appended one (sel_idx >= n_base).
    `sel_bank_nav_true` in the S2 decisions has one row per window with sel_idx < n_base, in window order; S6 windows are mapped through S2 by `ws`."""
    ddir = f"{root}/dump_s{seed}/decisions"
    B = np.full((len(T["ws"]), 8, 2), np.nan)
    ident = {"files": 0, "rows_equal_count_sel_lt_nbase": 0, "s6_sel_idx_equals_s2_on_mapped": 0, "windows_mapped": 0, "windows_in_bank": 0}
    for fi in np.unique(T["fi"]):
        d = np.load(f"{ddir}/ep{int(fi):03d}.npz")
        si, sb = d["sel_idx_nav_true"], d["sel_bank_nav_true"]
        ok = si < n_base
        ident["files"] += 1
        ident["rows_equal_count_sel_lt_nbase"] += int(sb.shape[0] == int(ok.sum()))
        s2i = {int(w): i for i, w in enumerate(d["ws"])}
        row_of, k = {}, 0
        for w, o in zip(d["ws"], ok):
            if o:
                row_of[int(w)] = k
                k += 1
        ident_ok = True
        for j in np.where(T["fi"] == fi)[0]:
            w = int(T["ws"][j])
            ident["windows_mapped"] += 1
            ident_ok &= bool(int(si[s2i[w]]) == int(T["sel_idx"][j]))
            r = row_of.get(w)
            if r is not None:
                B[j] = sb[r]
                ident["windows_in_bank"] += 1
        ident["s6_sel_idx_equals_s2_on_mapped"] += int(ident_ok)
    return B, ident


# ------------------------------------------------------------------------------------------ AMENDMENT 1 A3: along-track statistics (pre-registered, runnable only on a fan dump)
def arc_len(P) -> np.ndarray:
    """[W,K,2] -> [W,K] cumulative chord length of the polyline through (0,0) and the K points."""
    P = np.asarray(P, np.float64)
    Z = np.concatenate([np.zeros_like(P[:, :1]), P], axis=1)
    return np.cumsum(np.linalg.norm(np.diff(Z, axis=1), axis=-1), axis=1)


def _ep_sums(d, h, eid):
    uniq, inv = np.unique(np.asarray([str(x) for x in eid]), return_inverse=True)
    return (np.bincount(inv, (d * h).sum(1), len(uniq)), np.bincount(inv, (d * d).sum(1), len(uniq)), np.bincount(inv, (h * h).sum(1), len(uniq)), inv, len(uniq))


def along_stats(d, h, eid) -> dict:
    """d = s_X - s_P, h = s_H - s_P, both [W,K]; pooled over windows x instants.  g* = <d,h>/<d,d>, r = <d,h>/sqrt(<d,d><h,h>) (UNCENTRED, as along_track_localise.py), A, rho;
    r_centred = corr(d,h) (sensitivity)."""
    dh, dd, hh, _inv, _n = _ep_sums(d, h, eid)
    DH, DD, HH = dh.sum(), dd.sum(), hh.sum()
    dc, hc = (d - d.mean()).ravel(), (h - h.mean()).ravel()
    return {"g_star": float(DH / DD), "r": float(DH / np.sqrt(DD * HH)), "A": float(DH / HH), "rho": float(np.sqrt(DD / HH)),
            "r_centred": float((dc * hc).sum() / np.sqrt((dc * dc).sum() * (hc * hc).sum()))}


def along_r_diff_boot(d1, d2, h, eid, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired episode-cluster bootstrap of r(d1) - r(d2): ONE shared resample matrix (episode multiplicities from S6.boot_counts == taniteval.ci._draws), per-episode sums."""
    import s6_lon_decomp as S6
    C, _ei, n_ep = S6.boot_counts([str(x) for x in eid], n_boot, seed)
    a = _ep_sums(d1, h, eid)
    b = _ep_sums(d2, h, eid)
    ra = (C @ a[0]) / np.sqrt((C @ a[1]) * (C @ a[2]))
    rb = (C @ b[0]) / np.sqrt((C @ b[1]) * (C @ b[2]))
    pt = a[0].sum() / np.sqrt(a[1].sum() * a[2].sum()) - b[0].sum() / np.sqrt(b[1].sum() * b[2].sum())
    lo, hi = np.percentile(ra - rb, [2.5, 97.5])
    return {"delta_r": float(pt), "lo": float(lo), "hi": float(hi), "separated_above": bool(lo > 0), "n_windows": int(len(h)), "n_episodes": int(n_ep)}


def along_verdict(per_seed_diff: dict) -> str:
    """AMENDMENT 1 A3 reading rule on dR = r_os - r_RAND, per seed."""
    ups = [bool(v["lo"] > 0) for v in per_seed_diff.values()]
    return "SELECTOR-ADDS-VALUE" if all(ups) else ("SELECTION-SUPPORTED" if not any(ups) else "SEED-SPLIT")


def selftest_along() -> dict:
    rng = np.random.default_rng(7)
    W, K = 400, 4
    h = rng.normal(size=(W, K))
    eid = np.repeat(np.arange(20), W // 20)
    e = np.stack([-h[:, 1], h[:, 0], -h[:, 3], h[:, 2]], axis=1)             # orthogonal to h row-wise, |e| = |h|
    res = {}
    for name, d, want in (("K1_d_eq_h", h, {"g_star": 1.0, "r": 1.0, "A": 1.0, "rho": 1.0}), ("K2_d_eq_2h", 2 * h, {"g_star": 0.5, "r": 1.0, "A": 2.0, "rho": 2.0}),
                          ("K3_d_eq_h_plus_orthogonal_e", h + e, {"g_star": 0.5, "r": 2 ** -0.5, "A": 1.0, "rho": 2 ** 0.5})):
        g = along_stats(d, h, eid)
        res[name] = {"want": want, "got": {k: g[k] for k in want}, "pass": bool(all(abs(g[k] - want[k]) < 1e-9 for k in want))}
    # mutation: g* computed with <h,h> in the denominator (= A) must NOT equal 0.5 on K2
    dh, dd, hh, _i, _n = _ep_sums(2 * h, h, eid)
    res["mutation_gstar_denominator_hh"] = {"value": float(dh.sum() / hh.sum()), "went_RED": bool(abs(dh.sum() / hh.sum() - 0.5) > 1e-6)}
    # iid residual (a random candidate unrelated to the human) reads r ~ 0; a selector that picks the human-exact candidate reads r = 1
    big = rng.normal(size=(4000, K))
    hb = rng.normal(size=(4000, K))
    eb = np.repeat(np.arange(40), 100)
    res["iid_residual_r_near_0"] = {"r": along_stats(big, hb, eb)["r"], "pass": bool(abs(along_stats(big, hb, eb)["r"]) < 0.05)}
    bd = along_r_diff_boot(hb, big, hb, eb, n_boot=500)
    res["human_exact_minus_iid_separated_above"] = {"delta_r": bd["delta_r"], "lo": bd["lo"], "pass": bool(bd["separated_above"])}
    bz = along_r_diff_boot(big, big, hb, eb, n_boot=500)
    res["same_candidate_minus_itself_is_zero_width"] = {"delta_r": bz["delta_r"], "lo": bz["lo"], "hi": bz["hi"], "pass": bool(bz["delta_r"] == 0.0 and bz["lo"] == 0.0 and bz["hi"] == 0.0)}
    res["verdict_logic"] = {"pass": bool(along_verdict({0: {"lo": 0.1}, 1: {"lo": 0.2}}) == "SELECTOR-ADDS-VALUE" and along_verdict({0: {"lo": -0.1}, 1: {"lo": 0.0}}) == "SELECTION-SUPPORTED"
                                       and along_verdict({0: {"lo": 0.1}, 1: {"lo": -0.2}}) == "SEED-SPLIT")}
    res["pass"] = bool(all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v) and res["mutation_gstar_denominator_hh"]["went_RED"])
    return res


# ------------------------------------------------------------------------------------------ cells
def slim_cell(c: dict) -> dict:
    import cruise_brake_retime_r8 as CB
    return CB.slim(c)


def share_cell(PR, ind, eid) -> dict:
    """One-sample share of an indicator (NaN outside the population) with its episode-cluster CI: paired against an all-zero reference on the same windows."""
    ref = np.where(np.isfinite(ind), 0.0, np.nan)
    return slim_cell(PR.paired_cell(ind, ref, eid))


def ind_where(mask, val):
    return np.where(mask, np.asarray(val, dtype=np.float64), np.nan)


# ------------------------------------------------------------------------------------------ the pure rule (also driven by the decision-function arms)
def head_test(i_by_seed: dict, d_by_seed: dict, base_rate_ok: dict) -> dict:
    """PREREG section 3.  i_by_seed[s] = point (i); d_by_seed[s] = paired point delta r8 - r7 on head-says-cruise windows; base_rate_ok[s] = (i) exceeds the permutation null."""
    seeds = sorted(i_by_seed)
    c1 = {s: bool(i_by_seed[s] is not None and i_by_seed[s] >= THR_I) for s in seeds}
    c2 = {s: bool(d_by_seed[s] is not None and d_by_seed[s] <= 0.0) for s in seeds}
    p1 = {s: bool(base_rate_ok[s]) for s in seeds}
    passed = all(c1.values()) and all(c2.values()) and all(p1.values())
    return {"clause1_i_ge_0p6": c1, "clause2_r8_le_r7_on_head_says_cruise": c2, "P1_i_exceeds_permutation_null": p1,
            "HEAD_TEST": "PASS" if passed else ("BASE-RATE (void)" if not all(p1.values()) else "FAIL")}


def final_label(t_head, e) -> dict:
    """PREREG section 5.  e = E from the fan stage or None (not measurable)."""
    if e is None:
        return {"label": "UNDETERMINED", "T_HEAD": bool(t_head), "T_SEL": None, "T_CAND": None,
                "note": "SELECTION vs CANDIDATE not measurable without the refined fan (re-roll design in BRAKE_LOCUS_REROLL_DESIGN.md)"}
    t_sel, t_cand = e >= E_SEL, e < E_CAND
    fired = [n for n, v in (("HEAD", t_head), ("SELECTION", t_sel), ("CANDIDATE", t_cand)) if v]
    return {"label": fired[0] if len(fired) == 1 else "MIXED", "T_HEAD": bool(t_head), "T_SEL": bool(t_sel), "T_CAND": bool(t_cand), "fired": fired}


# ------------------------------------------------------------------------------------------ the fan stage (PREREG section 4) -- pre-registered, not runnable on banked data
def fan_cells(F, G, v0, fb, keep=None, x_m=X_FDE6_M, thr=1.5, dt=DT, use_ade=False) -> dict:
    """F [W,N,6,2] refined fan on the S6 grid, G [W,6,2], v0 [W], fb [W] bool (false-brake windows by the re-rolled arm's own class), keep [W,N] bool (reach survivors).
    E(w) = any candidate that is NON-BRAKING (plan class not decel, same rule as the instrument) AND within x_m of the human at 6 s (FDE@6 s; or ADE <= x_m if use_ade)."""
    import cruise_brake_retime_r8 as CB
    W, N = F.shape[0], F.shape[1]
    Fv = F.astype(np.float64)
    cls = np.stack([CB.plan_class_thr(Fv[:, c], v0, dt, thr) for c in range(N)], axis=1)           # [W,N]
    nonbrake = cls != "decel"
    d = (np.linalg.norm(Fv - G[:, None].astype(np.float64), axis=-1).mean(-1) if use_ade
         else np.linalg.norm(Fv[:, :, -1] - G[:, None, -1].astype(np.float64), axis=-1))
    good = nonbrake & (d <= x_m)
    if keep is not None:
        good = good & keep
    e_w = good.any(axis=1)
    return {"E": fl(e_w[fb].mean()) if fb.any() else None, "n_fb": int(fb.sum()), "E_w": e_w, "n_good": good.sum(1)}


def selftest_fan(x_m=X_FDE6_M) -> dict:
    """Synthetic fan with a KNOWN answer.  v0 = 10 m/s, GT = straight line at 10 m/s.  Window 0: one candidate exactly on the human -> E 1.  Window 1: only braking candidates -> 0.
    Window 2: only a far non-braking candidate (offset 6 m) -> 0.  Window 3: a good candidate that the reach mask removed -> 0 with the mask, 1 without."""
    t = np.arange(1, 7, dtype=np.float64)[:, None] * np.array([[10.0, 0.0]])          # [6,2] constant 10 m/s
    G = np.repeat(t[None], 4, 0)
    v0 = np.full(4, 10.0)
    brake = np.cumsum(np.maximum(10.0 - 2.0 * np.arange(1, 7), 0.0))[:, None] * np.array([[1.0, 0.0]])    # decelerates 2 m/s per second
    far = t + np.array([[0.0, 6.0]])
    F = np.zeros((4, 3, 6, 2))
    F[0] = np.stack([t, brake, far]); F[1] = np.stack([brake, brake * 0.9, brake * 0.8]); F[2] = np.stack([far, brake, brake]); F[3] = np.stack([t, brake, brake])
    keep = np.ones((4, 3), bool)
    keep[3, 0] = False
    fb = np.ones(4, bool)
    r_mask = fan_cells(F, G, v0, fb, keep, x_m)
    r_all = fan_cells(F, G, v0, fb, None, x_m)
    r_ade = fan_cells(F, G, v0, fb, keep, X_ADE_M, use_ade=True)
    r_mut = fan_cells(F, G, v0, fb, keep, 0.0 - 1.0)                                    # mutation: X negative -> no candidate can be good
    r_mut2 = fan_cells(F, G, v0, fb, keep, 1e6)                                         # mutation: X huge -> the far candidate (window 2) becomes good
    expect = {"E_with_reach_mask": 0.25, "E_without_mask": 0.5, "E_ade": 0.25}
    got = {"E_with_reach_mask": r_mask["E"], "E_without_mask": r_all["E"], "E_ade": r_ade["E"]}
    ok = all(abs(got[k] - expect[k]) < 1e-12 for k in expect)
    return {"expected_literal": expect, "got": got, "pass": bool(ok),
            "mutation_X_negative": {"E": r_mut["E"], "went_RED": bool(r_mut["E"] != expect["E_with_reach_mask"])},
            "mutation_X_huge": {"E": r_mut2["E"], "went_RED": bool(r_mut2["E"] != expect["E_with_reach_mask"])}}


# ------------------------------------------------------------------------------------------ the measurement for one seed (parametrised so the mutations can break it)
def measure(PR, CB, S6, D: dict, seed: int, thr: float = 1.5, head_key: str = "nav_true.tacv6_lon_logits", head_override=None,
            join_shift: int = 0, swap_cols=None, with_ci: bool = True, d_set=D_SET) -> dict:
    T8, T7, X8, X7, z = D["T8"][seed], D["T7"][seed], D["X8"][seed], D["X7"][seed], D["z"][seed]
    v0, G, eid = T8["v0"], T8["g"], list(D["eid"][seed])
    comp = np.asarray(z["S6_compliant"], bool)
    gtc = CB.plan_class_thr(G, v0, DT, thr)
    c8, c7 = CB.plan_class_thr(T8["os"], v0, DT, thr), CB.plan_class_thr(T8["b_r7"], v0, DT, thr)
    m = comp & (gtc == "cruise")
    fb8, fb7 = m & (c8 == "decel"), m & (c7 == "decel")

    def head_of(X, key, shift=0, swap=None):
        lg = np.array(X[key], dtype=np.float64)
        if swap is not None:
            lg[:, [swap[0], swap[1]]] = lg[:, [swap[1], swap[0]]]
        h = np.argmax(lg, axis=1)
        return np.roll(h, shift) if shift else h
    h8 = head_of(X8, head_key, join_shift, swap_cols) if head_override is None else np.asarray(head_override)
    h7 = head_of(X7, head_key, 0, swap_cols)
    inD8, inD7 = np.isin(h8, d_set), np.isin(h7, d_set)
    cell = (lambda ind: share_cell(PR, ind, eid)) if with_ci else (lambda ind: {"mean_new": float(np.nanmean(ind)), "n_windows": int(np.isfinite(ind).sum())})
    out = {"seed": seed, "thr": thr, "n_compliant": int(comp.sum()), "ep_compliant": len(set(np.asarray(eid)[comp].tolist())),
           "n_cruise": int(m.sum()), "ep_cruise": len(set(np.asarray(eid)[m].tolist())),
           "fd_r8": float((c8[m] == "decel").mean()), "fd_r7": float((c7[m] == "decel").mean()), "n_fb8": int(fb8.sum()), "n_fb7": int(fb7.sum())}
    # ---- (i)
    out["i"] = cell(ind_where(fb8, inD8))
    out["head_class_counts_on_r8_fb"] = {LON[k]: int(((h8 == k) & fb8).sum()) for k in range(8) if ((h8 == k) & fb8).any()}
    out["head_class_counts_on_gt_cruise"] = {LON[k]: int(((h8 == k) & m).sum()) for k in range(8) if ((h8 == k) & m).any()}
    out["r8_head_argmaxes_masked_class_2_or_6"] = int(np.isin(h8[m], (2, 6)).sum())
    # ---- clause 2 (paired, same windows) -- and the sensitivities
    M2 = m & ~inD8
    dec8, dec7 = (c8 == "decel").astype(np.float64), (c7 == "decel").astype(np.float64)
    pc = PR.paired_cell(ind_where(M2, dec8), ind_where(M2, dec7), eid) if with_ci else {"delta": float((dec8[M2] - dec7[M2]).mean()), "mean_new": float(dec8[M2].mean()), "mean_ref": float(dec7[M2].mean()), "n_windows": int(M2.sum())}
    out["clause2_paired"] = slim_cell(pc) if with_ci else pc
    out["clause2_literal_point_delta_le_0"] = bool(out["clause2_paired"]["delta"] <= 0.0)
    out["clause2_not_separated_worse_lo_le_0"] = bool(out["clause2_paired"].get("lo", 0.0) <= 0.0) if with_ci else None
    M2u7 = m & ~inD7
    out["clause2_unpaired_each_arm_own_head"] = {"r8_rate": float(dec8[M2].mean()), "n8": int(M2.sum()), "r7_rate": float(dec7[M2u7].mean()), "n7": int(M2u7.sum())}
    h_plus = np.isin(h8, D_PLUS)
    M2p = m & ~h_plus
    out["sens_Dplus_FOLLOW"] = {"i": float(h_plus[fb8].mean()), "clause2_n": int(M2p.sum()), "r8_rate": float(dec8[M2p].mean()), "r7_rate": float(dec7[M2p].mean()),
                                "delta": float((dec8[M2p] - dec7[M2p]).mean())}
    Mc = m & (h8 == 1)
    out["sens_head_is_CRUISE_only"] = {"n": int(Mc.sum()), "r8_rate": float(dec8[Mc].mean()), "r7_rate": float(dec7[Mc].mean()), "delta": float((dec8[Mc] - dec7[Mc]).mean())}
    # ---- P1: permutation null for (i)
    rng = np.random.default_rng(0)
    pool = inD8[m]
    perm = np.empty(N_PERM)
    fbm = fb8[m]
    for k in range(N_PERM):
        perm[k] = rng.permutation(pool)[fbm].mean()
    out["P1"] = {"i_observed": float(inD8[fb8].mean()), "perm_mean": float(perm.mean()), "perm_p97_5": float(np.percentile(perm, 97.5)),
                 "exceeds_null": bool(inD8[fb8].mean() > np.percentile(perm, 97.5)), "n_perm": N_PERM}
    # ---- S1, S2
    out["S1_head_decel_share_on_gt_cruise"] = {"r8": cell(ind_where(m, inD8)), "r7": cell(ind_where(m, inD7))}
    out["S2_coupling_P_plan_brakes_given_own_head_in_D"] = {"r8": cell(ind_where(m & inD8, dec8)), "r7": cell(ind_where(m & inD7, dec7)),
                                                              "r8_given_head_not_D": cell(ind_where(m & ~inD8, dec8)), "r7_given_head_not_D": cell(ind_where(m & ~inD7, dec7))}
    # ---- S3: strata decomposition of the paired false-brake excess (identity-checked)
    Nm = float(m.sum())
    d = dec8 - dec7
    tot = float(d[m].sum() / Nm)
    A_, B_ = m & inD8, m & ~inD8
    sA, sB = float(d[A_].sum() / Nm), float(d[B_].sum() / Nm)
    out["S3_excess_decomposition"] = {"total_excess_r8_minus_r7": tot, "stratum_r8_head_in_D": {"n": int(A_.sum()), "contribution": sA, "share": sA / tot if abs(tot) > 1e-12 else None,
                                                                                               "r8_rate": float(dec8[A_].mean()), "r7_rate": float(dec7[A_].mean())},
                                      "stratum_r8_head_not_D": {"n": int(B_.sum()), "contribution": sB, "share": sB / tot if abs(tot) > 1e-12 else None,
                                                                "r8_rate": float(dec8[B_].mean()), "r7_rate": float(dec7[B_].mean())},
                                      "identity_sum_equals_total": bool(abs(sA + sB - tot) < 1e-12)}
    # ---- S4: selection provenance
    si, sb, a_star = T8["sel_idx"], T8["sel_idx_base"], T8["a_star"]
    nfb8 = m & ~fb8
    prov = lambda msk: {"n": int(msk.sum()), "sel_in_allocated_group_ge117": float((si[msk] >= N_BASE).mean()), "sel_eq_a_star": float((si[msk] == a_star[msk]).mean()),
                        "sel_ne_sel_idx_base": float((si[msk] != sb[msk]).mean())}
    out["S4_selection_provenance"] = {"r8_false_brake": prov(fb8), "r8_gt_cruise_not_false_brake": prov(nfb8),
                                      "r8_false_brake_head_in_D": prov(fb8 & inD8), "r8_false_brake_head_not_D": prov(fb8 & ~inD8),
                                      "r7_sel_eq_a_star_false_brake": float((T7["sel_idx"][fb7] == T7["a_star"][fb7]).mean()),
                                      "r7_sel_eq_a_star_gt_cruise_not_false_brake": float((T7["sel_idx"][m & ~fb7] == T7["a_star"][m & ~fb7]).mean()),
                                      "r8_sel_idx_max": int(si.max()), "r7_sel_idx_max": int(T7["sel_idx"].max())}
    # ---- S5: banked ablation arms (seed 0 only; seed 1 extras carry nav_true and obey30 only)
    if "filter_off.traj" in X8:
        cf = CB.plan_class_thr(X8["filter_off.traj"][:, S6_SLOTS], v0, DT, thr)
        cv = CB.plan_class_thr(X8["vmax_zero.traj"][:, S6_SLOTS], v0, DT, thr)
        hz = np.argmax(X8["vmax_zero.tacv6_lon_logits"], axis=1)
        out["S5_ablation_arms_false_decel"] = {"native": float((c8[m] == "decel").mean()), "filter_off": float((cf[m] == "decel").mean()), "vmax_zero": float((cv[m] == "decel").mean()),
                                               "ceiling_filter_changes_selection_on_r8_false_brake": float((X8["nav_true.sel_idx"][fb8] != X8["filter_off.sel_idx"][fb8]).mean())
                                               if "filter_off.sel_idx" in X8 else None,
                                               "head_in_D_share_native_vs_vmax_zero_on_gt_cruise": [float(inD8[m].mean()), float(np.isin(hz[m], d_set).mean())]}
    # ---- S7 (AMENDMENT 1 A2): is the braking already in the SELECTED UNREFINED anchor?  (in-bank windows only: sel_idx < 117)
    B8, B7 = D["B8"][seed], D["B7"][seed]
    ok8, ok7 = np.isfinite(B8[:, 0, 0]), np.isfinite(B7[:, 0, 0])
    cb8 = np.full(len(v0), "n/a", dtype=object)
    cb7 = np.full(len(v0), "n/a", dtype=object)
    cb8[ok8] = CB.plan_class_thr(B8[ok8][:, S6_SLOTS], v0[ok8], DT, thr)
    cb7[ok7] = CB.plan_class_thr(B7[ok7][:, S6_SLOTS], v0[ok7], DT, thr)
    dm = lambda msk: float((cb8[msk] == "decel").mean()) if msk.any() else None
    fbb = fb8 & ok8
    out["S7_unrefined_selected_anchor"] = {
        "coverage_in_bank_share_of_r8_false_brake": float(ok8[fb8].mean()), "coverage_in_bank_share_of_gt_cruise": float(ok8[m].mean()),
        "r8_false_brake_in_bank": {"n": int(fbb.sum()), "bank_path_already_decel": dm(fbb), "bank_not_decel_but_refined_decel": 1.0 - dm(fbb) if fbb.any() else None},
        "r8_false_brake_in_bank_head_in_D": {"n": int((fbb & inD8).sum()), "bank_path_already_decel": dm(fbb & inD8)},
        "r8_false_brake_in_bank_head_not_D": {"n": int((fbb & ~inD8).sum()), "bank_path_already_decel": dm(fbb & ~inD8)},
        "r8_gt_cruise_not_false_brake_in_bank": {"n": int((m & ~fb8 & ok8).sum()), "bank_path_decel": dm(m & ~fb8 & ok8)},
        "crosstab_bank_x_refined_r8_gt_cruise_in_bank": {f"bank_{b}__refined_{r}": int(((m & ok8) & (cb8 == b) & (c8 == r)).sum()) for b in ("decel", "cruise", "accel") for r in ("decel", "cruise", "accel")},
        "r7_bank_path_decel_on_gt_cruise_in_bank": float((cb7[m & ok7] == "decel").mean()),
        "r7_false_brake_in_bank": {"n": int((fb7 & ok7).sum()), "bank_path_already_decel": float((cb7[fb7 & ok7] == "decel").mean()) if (fb7 & ok7).any() else None}}
    both = m & ok8 & ok7
    out["S7_unrefined_selected_anchor"]["paired_bank_decel_r8_minus_r7_gt_cruise_both_in_bank"] = (
        slim_cell(PR.paired_cell(ind_where(both, cb8 == "decel"), ind_where(both, cb7 == "decel"), eid)) if with_ci else None)
    # ---- POST-HOC P-2 (AMENDMENT 2, written after run 1; descriptive, not pre-registered): is the GT-nearest BASE anchor itself braking where it was selected?
    ph = {}
    for nm, fbm_, okm_, cbm_, sel_, ast_ in (("r8", fb8, ok8, cb8, T8["sel_idx"], T8["a_star"]), ("r7", fb7, ok7, cb7, T7["sel_idx"], T7["a_star"])):
        for pop, msk in (("false_brake", fbm_), ("gt_cruise_not_false_brake", m & ~fbm_)):
            q = msk & okm_ & (sel_ == ast_)
            ph[f"{nm}_{pop}_sel_eq_a_star_in_bank"] = {"n": int(q.sum()), "n_episodes": len(set(np.asarray(eid)[q].tolist())),
                                                       "a_star_bank_path_decel": float((cbm_[q] == "decel").mean()) if q.any() else None}
    qq = fb8 & ok8 & (T8["sel_idx"] == T8["a_star"])
    if qq.any() and with_ci:
        ph["r8_false_brake_sel_eq_a_star_a_star_bank_decel_cell"] = slim_cell(share_cell(PR, ind_where(qq, cb8 == "decel"), eid))
    out["POSTHOC_P2_gt_nearest_base_anchor_class"] = ph
    # ---- S6: legacy head sensitivity
    if head_key == "nav_true.tacv6_lon_logits":
        hl = head_of(X8, "nav_true.lon_logits_tac")
        inL = np.isin(hl, d_set)
        ML = m & ~inL
        out["sens_legacy_head_lon_logits_tac"] = {"i": float(inL[fb8].mean()), "clause2_n": int(ML.sum()), "r8_rate": float(dec8[ML].mean()), "r7_rate": float(dec7[ML].mean()),
                                                   "delta": float((dec8[ML] - dec7[ML]).mean()),
                                                   "argmax_equals_decisions_lon_pred_nav_true": float((hl == T8["lon_pred_nav_true"]).mean()),
                                                   "argmax_disagrees_with_tacv6_frac_all_windows": float((hl != head_of(X8, "nav_true.tacv6_lon_logits")).mean())}
    out["_arrays"] = {"m": m, "fb8": fb8, "h8": h8}
    return out


# ------------------------------------------------------------------------------------------ controls
def controls(PR, CB, S6, D: dict, res: dict) -> dict:
    ctl: dict = {"C1": {}, "C2": {}, "C3": {}, "C4": {}, "C5": {}, "C6": {}, "C7": {}}
    for s in SEEDS:
        r = res[s]
        ctl["C1"][f"s{s}_n_compliant"] = bool(r["n_compliant"] == LIT["n_compliant"] and r["ep_compliant"] == LIT["ep_compliant"])
        ctl["C1"][f"s{s}_n_cruise"] = bool(r["n_cruise"] == LIT["n_cruise"] and r["ep_cruise"] == LIT["ep_cruise"])
        ctl["C1"][f"s{s}_fd_r8"] = bool(abs(r["fd_r8"] - LIT["fd_r8"][s]) < 5e-9)
        ctl["C1"][f"s{s}_fd_r7"] = bool(abs(r["fd_r7"] - LIT["fd_r7"][s]) < 5e-9)
    ctl["C1_pass"] = bool(all(ctl["C1"].values()))
    # C2 alignment + head join
    for s in SEEDS:
        T8, T7, X8, X7, z = D["T8"][s], D["T7"][s], D["X8"][s], D["X7"][s], D["z"][s]
        ctl["C2"][f"s{s}_ws_equal_deploy8"] = bool(np.array_equal(z["S6_ws"], T8["ws"]))
        ctl["C2"][f"s{s}_os_bit_equal_deploy8"] = bool(np.array_equal(z["S6_os"], T8["os"]))
        ctl["C2"][f"s{s}_r7_embedded_bit_equal_a6text_s6_os"] = bool(np.array_equal(T8["b_r7"], T7["os"]) and np.array_equal(T8["ws"], T7["ws"]))
        ctl["C2"][f"s{s}_r8_head_join_traj_bit_equal_os"] = bool(np.array_equal(X8["nav_true.traj"][:, T8["slots"]], T8["os"]))
        ctl["C2"][f"s{s}_r7_head_join_traj_bit_equal_embedded"] = bool(np.array_equal(X7["nav_true.traj"][:, T8["slots"]], T8["b_r7"]))
        ctl["C2"][f"s{s}_extras_sel_idx_equals_decisions"] = bool(np.array_equal(X8["nav_true.sel_idx"].astype(np.int64), T8["sel_idx"]))
        ctl["C2"][f"s{s}_slots_are_S6_grid"] = bool(T8["slots"] == S6_SLOTS)
    ctl["C2"]["compliant_mask_equal_across_seeds"] = bool(np.array_equal(D["z"][0]["S6_compliant"], D["z"][1]["S6_compliant"]))
    ctl["C2"]["twin_plan_class_equals_S6"] = bool(all(np.array_equal(CB.plan_class_thr(P, D["T8"][0]["v0"], DT, 1.5), S6.plan_lon_class(P, D["T8"][0]["v0"], DT))
                                                      for P in (D["T8"][0]["os"], D["T8"][0]["b_r7"], D["T8"][0]["g"])))
    ctl["C2_pass"] = bool(all(ctl["C2"].values()))
    # C3 identities
    for s in SEEDS:
        sl = res[s].get("sens_legacy_head_lon_logits_tac")
        ctl["C3"][f"s{s}_legacy_head_argmax_equals_decisions_lon_pred"] = bool(sl is not None and sl["argmax_equals_decisions_lon_pred_nav_true"] == 1.0)
        ctl["C3"][f"s{s}_S3_strata_sum_equals_total"] = bool(res[s]["S3_excess_decomposition"]["identity_sum_equals_total"])
    for s_ in SEEDS:
        for tag in ("B8", "B7"):
            idn = D["IDN_" + tag][s_]
            ctl["C3"][f"s{s_}_{tag}_sel_bank_rows_equal_count_sel_lt_nbase_every_file"] = bool(idn["rows_equal_count_sel_lt_nbase"] == idn["files"])
            ctl["C3"][f"s{s_}_{tag}_s6_sel_idx_equals_s2_every_file"] = bool(idn["s6_sel_idx_equals_s2_on_mapped"] == idn["files"])
    ctl["C3_pass"] = bool(all(ctl["C3"].values()))
    # C4 decision-function arms
    r0 = res[0]
    fb8, m = r0["_arrays"]["fb8"], r0["_arrays"]["m"]
    oracle = np.where(fb8, 3, 1)
    never = np.full(len(m), 1)
    ro = measure(PR, CB, S6, D, 0, head_override=oracle)
    rn = measure(PR, CB, S6, D, 0, head_override=never)
    ctl["C4"]["oracle_head_i_is_1"] = bool(abs(ro["i"]["mean_new"] - 1.0) < 1e-12)
    ctl["C4"]["never_decel_head_i_is_0"] = bool(abs(rn["i"]["mean_new"]) < 1e-12)
    tbl = [(({0: 0.9, 1: 0.9}, {0: -0.01, 1: 0.0}, {0: True, 1: True}), "PASS"), (({0: 0.9, 1: 0.5}, {0: -0.01, 1: 0.0}, {0: True, 1: True}), "FAIL"),
           (({0: 0.9, 1: 0.9}, {0: -0.01, 1: 0.01}, {0: True, 1: True}), "FAIL"), (({0: 0.9, 1: 0.9}, {0: -0.01, 1: 0.0}, {0: True, 1: False}), "BASE-RATE (void)")]
    ctl["C4"]["rule_logic_table"] = bool(all(head_test(i, d, p)["HEAD_TEST"] == want for (i, d, p), want in tbl))
    ctl["C4_pass"] = bool(all(ctl["C4"].values()))
    # C5 mutations of the instrument: each must go RED
    base_i = r0["i"]["mean_new"]
    mut: dict = {}
    rm1 = measure(PR, CB, S6, D, 0, thr=1.0, with_ci=False)
    mut["m1_decel_threshold_1p0"] = {"went_RED": bool(not (abs(rm1["fd_r8"] - LIT["fd_r8"][0]) < 5e-9 and abs(rm1["fd_r7"] - LIT["fd_r7"][0]) < 5e-9)),
                                     "fd_r8": rm1["fd_r8"], "literal": LIT["fd_r8"][0]}
    T8, X8 = D["T8"][0], D["X8"][0]
    shifted = np.roll(X8["nav_true.traj"][:, T8["slots"]], 1, axis=0)
    mut["m2_head_join_shifted_one_window"] = {"went_RED": bool(not np.array_equal(shifted, T8["os"])), "detail": "C2 head-join traj equality fails under a one-window shift"}
    rm3 = measure(PR, CB, S6, D, 0, swap_cols=(1, 3), with_ci=False)
    mut["m3_CRUISE_BRAKE_TO_columns_swapped"] = {"went_RED": bool(abs(rm3["i"]["mean_new"] - base_i) > 0.2), "i_mutated": rm3["i"]["mean_new"], "i_observed": base_i}
    ctl["C5"] = mut
    ctl["C5_pass"] = bool(all(v["went_RED"] for v in mut.values()))
    # C6 fan function self-test
    st = selftest_fan()
    ctl["C6"] = st
    ctl["C6_pass"] = bool(st["pass"] and st["mutation_X_negative"]["went_RED"] and st["mutation_X_huge"]["went_RED"])
    sa = selftest_along()
    ctl["C7"] = sa
    ctl["C7_pass"] = bool(sa["pass"])
    ctl["ALL_PASS"] = bool(all(ctl[f"C{k}_pass"] for k in range(1, 8)))
    return ctl



def ckpt_scalars() -> int:
    """MEASURED fact for the design: the LEARNED refcv8 ranking-term scalars (`core.decoder.r8_sel.*`, `r8_rc_gate`) in the EMA-30k checkpoint.  torch is imported only here (mmap, 4 scalars read).
    Writes raw/refcv9_headroom/BRAKE_LOCUS_ckpt_scalars.json.  CPU only."""
    import torch
    path = "E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt"
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    sd = ck["model"]
    keys = ["core.decoder.r8_sel.beta_lat", "core.decoder.r8_sel.beta_lon", "core.decoder.r8_sel.gamma_prog", "core.decoder.r8_sel.gamma_head", "core.decoder.r8_rc_gate"]
    vals = {k.split("core.decoder.")[1]: float(sd[k]) for k in keys}
    J = {"tool": "code/brake_locus_r8.py --ckpt-scalars", "ckpt": path, "step": int(ck["step"]), "kind": ck.get("kind"), "values": vals,
         "meaning": "ranking term r8_sel = beta_lat*log p(lat3 of the candidate) + beta_lon*log p(lon6 of the candidate) + gamma_prog*sat_prog + gamma_head*sat_head (refcv8_conditioning.py:472-499); all four are zero-init and LEARNED.  "
                    "beta_lon ~ 0 means the head's lon POSTERIOR hardly enters the ranking; gamma_prog is the weight on satisfying the head's lon CONSTRAINT output (progress), which is NOT banked.",
         "evidence_class": "MEASURED (ours, checkpoint read, mmap, 4 scalars)"}
    with open(OUT / "BRAKE_LOCUS_ckpt_scalars.json", "w", encoding="utf-8") as f:
        json.dump(J, f, indent=1)
    print(json.dumps(vals))
    return 0

# ------------------------------------------------------------------------------------------ report
def fmt_cell(c) -> str:
    if c is None or "mean_new" not in c:
        return "n/a"
    ci = f" [{c['lo']:.3f}, {c['hi']:.3f}]" if "lo" in c and c.get("lo") is not None else ""
    return f"{c['mean_new']:.3f}{ci} (n {c['n_windows']})"


def write_md(J: dict, path: Path) -> None:
    L = []
    a = L.append
    a("# BRAKE LOCUS: where is refcv8's cruise false-brake decided?  (HEAD / SELECTION / CANDIDATE)\n")
    a(f"**LABEL (banked data): {J['label']['label']}**  |  HEAD test (association): **{J['head_test']['HEAD_TEST']}**  |  SELECTION vs CANDIDATE: **NOT MEASURABLE from banked dumps (the refined fan is not banked)**\n")
    a(f"* tier: **{TIER}**; refcv8 EMA-30k (`{TREE}`) vs refcv7-50,400; surface = S6 compliant windows (3,251 / 138 ep), GT-cruise 1,856; evidence class MEASURED (ours), CPU only, no model forward.")
    a(f"* estimator: {ESTIMATOR}.")
    a(f"* pre-registered rule: `raw/refcv9_headroom/PREREG_BRAKE_LOCUS.md` (sha256 `{J['prereg_sha256'][:12]}` sealed before any locus number was computed; Amendment 1 `{J['amendment1_sha256'][:12]}`, also sealed before the first run: it corrects the `sel_bank_*` inventory row, adds S7 and the ALONG-RAND row); script sha256 `{J['script_sha256'][:12]}`; wall {J['wall_s']:.0f} s.")
    a("* **Pre-seal look disclosed in the prereg**: head-class counts were inspected before sealing, so the HEAD clause outcomes were not blind.\n")
    a("## 1. Inventory: what the banked dumps carry\n")
    a("| item | banked? |\n|---|---|")
    for k, v in J["inventory"].items():
        a(f"| {k} | {v} |")
    a("\n## 2. HEAD test (the pre-registered rule, per inference seed)\n")
    a("| seed | (i) share of r8 false-brake windows with head in D [CI] | clause 1 (>= 0.6) | clause 2: r8 - r7 false-brake rate on r8-head-says-cruise windows (paired) [CI] | literal (<= 0) | not-separated-worse (lo <= 0) | P1 permutation null (97.5th pct) |")
    a("|---|---|---|---|---|---|---|")
    for s in SEEDS:
        r = J["per_seed"][str(s)]
        c2 = r["clause2_paired"]
        a(f"| {s} | {fmt_cell(r['i'])} | {J['head_test']['clause1_i_ge_0p6'][str(s)]} | {c2['delta']:+.4f} [{c2['lo']:+.4f}, {c2['hi']:+.4f}]  (r8 {c2['mean_new']:.3f} vs r7 {c2['mean_ref']:.3f}, n {c2['n_windows']}) | "
          f"{J['head_test']['clause2_r8_le_r7_on_head_says_cruise'][str(s)]} | {r['clause2_not_separated_worse_lo_le_0']} | obs {r['P1']['i_observed']:.3f} vs null {r['P1']['perm_p97_5']:.3f} -> {r['P1']['exceeds_null']} |")
    a(f"\n**HEAD test: {J['head_test']['HEAD_TEST']}.**  {J['reading']}\n")
    a("## 3. Descriptive supplements (not part of the rule)\n")
    a("| seed | head-decel share on GT-cruise: r8 / r7 | P(plan brakes \\| own head in D): r8 / r7 | P(plan brakes \\| own head NOT in D): r8 / r7 | excess r8-r7 false-brake rate | of which r8-head-in-D stratum / not-D stratum |")
    a("|---|---|---|---|---|---|")
    for s in SEEDS:
        r = J["per_seed"][str(s)]
        c = r["S2_coupling_P_plan_brakes_given_own_head_in_D"]
        e = r["S3_excess_decomposition"]
        a(f"| {s} | {r['S1_head_decel_share_on_gt_cruise']['r8']['mean_new']:.3f} / {r['S1_head_decel_share_on_gt_cruise']['r7']['mean_new']:.3f} | {fmt_cell(c['r8'])} / {fmt_cell(c['r7'])} | "
          f"{fmt_cell(c['r8_given_head_not_D'])} / {fmt_cell(c['r7_given_head_not_D'])} | {e['total_excess_r8_minus_r7']:+.4f} | "
          f"{e['stratum_r8_head_in_D']['contribution']:+.4f} ({100 * (e['stratum_r8_head_in_D']['share'] or 0):.0f} %) / {e['stratum_r8_head_not_D']['contribution']:+.4f} ({100 * (e['stratum_r8_head_not_D']['share'] or 0):.0f} %) |")
    a("\nSelection provenance (S4, descriptive).  `sel_idx >= 117` = the selected candidate is one of the 32 APPENDED (head-allocated) candidates; `sel == a_star` = the selected candidate is the GT-nearest BASE-bank anchor; `sel != base` = the E9 goal-selection graft moved the pick (CORRECTED by Amendment 2 P-1: the core pick `sel_idx_base` already includes the refcv8 head-coupled ranking term).\n")
    a("| seed | population | n | sel >= 117 | sel == a_star | sel != sel_idx_base (E9 graft moved it) |\n|---|---|---|---|---|---|")
    for s in SEEDS:
        for nm, p in J["per_seed"][str(s)]["S4_selection_provenance"].items():
            if isinstance(p, dict):
                a(f"| {s} | r8 {nm} | {p['n']} | {p['sel_in_allocated_group_ge117']:.3f} | {p['sel_eq_a_star']:.3f} | {p['sel_ne_sel_idx_base']:.3f} |")
        pr = J["per_seed"][str(s)]["S4_selection_provenance"]
        a(f"| {s} | r7 sel == a_star: false-brake / other GT-cruise | - | - | {pr['r7_sel_eq_a_star_false_brake']:.3f} / {pr['r7_sel_eq_a_star_gt_cruise_not_false_brake']:.3f} | - |")
    s5 = J["per_seed"]["0"].get("S5_ablation_arms_false_decel")
    if s5:
        a(f"\nSeed-0 banked ablation arms (S5): false-decel rate native {s5['native']:.3f}, ceiling filter OFF {s5['filter_off']:.3f}, speed label zeroed {s5['vmax_zero']:.3f}; the ceiling filter changes the selected candidate on {100 * (s5['ceiling_filter_changes_selection_on_r8_false_brake'] or 0):.1f} % of r8 false-brake windows; head-in-D share on GT-cruise native vs label-zeroed {s5['head_in_D_share_native_vs_vmax_zero_on_gt_cruise'][0]:.3f} / {s5['head_in_D_share_native_vs_vmax_zero_on_gt_cruise'][1]:.3f}.")
    a("\nSensitivities (S6): " + "; ".join(
        f"seed {s}: D+FOLLOW (i) {J['per_seed'][str(s)]['sens_Dplus_FOLLOW']['i']:.3f}, clause-2 delta {J['per_seed'][str(s)]['sens_Dplus_FOLLOW']['delta']:+.4f}; head==CRUISE only delta {J['per_seed'][str(s)]['sens_head_is_CRUISE_only']['delta']:+.4f}; "
        f"legacy head (i) {J['per_seed'][str(s)]['sens_legacy_head_lon_logits_tac']['i']:.3f}, delta {J['per_seed'][str(s)]['sens_legacy_head_lon_logits_tac']['delta']:+.4f}; "
        f"unpaired (each arm's own head-says-cruise) r8 {J['per_seed'][str(s)]['clause2_unpaired_each_arm_own_head']['r8_rate']:.3f} vs r7 {J['per_seed'][str(s)]['clause2_unpaired_each_arm_own_head']['r7_rate']:.3f}" for s in SEEDS) + ".\n")
    cs = J.get("ckpt_scalars")
    if cs:
        v = cs["values"]
        a("\nLearned ranking-term scalars in the EMA-30k checkpoint (MEASURED, `BRAKE_LOCUS_ckpt_scalars.json`): beta_lat " + f"{v['r8_sel.beta_lat']:+.4f}, beta_lon {v['r8_sel.beta_lon']:+.4f}, gamma_prog {v['r8_sel.gamma_prog']:+.4f}, gamma_head {v['r8_sel.gamma_head']:+.4f}.  "
          "beta_lon is ~0: the head's lon POSTERIOR hardly enters the ranking, so the argmax-based HEAD test is a proxy; the head can reach the pick through its lon CONSTRAINT output (gamma_prog, NOT banked), through candidate allocation (appended candidates are chosen on ~10 % of false-brake windows) or through shared scene features (common cause).\n")
    a("\n### S7 (Amendment 1 A2): is the braking already in the SELECTED UNREFINED anchor?  (windows with sel_idx < 117 only)\n")
    a("| seed | r8 false-brake windows in-bank (share of all r8 FB) | unrefined selected anchor ALREADY decel | unrefined cruise/accel but REFINED decel | ... when head in D / not in D (bank already decel) | r8 GT-cruise non-FB: bank decel | r7 GT-cruise: bank decel | r8 - r7 bank-decel, paired [CI] |")
    a("|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        q = J["per_seed"][str(s)]["S7_unrefined_selected_anchor"]
        f = q["r8_false_brake_in_bank"]
        pr = q["paired_bank_decel_r8_minus_r7_gt_cruise_both_in_bank"]
        a(f"| {s} | {f['n']} ({q['coverage_in_bank_share_of_r8_false_brake']:.3f}) | {f['bank_path_already_decel']:.3f} | {f['bank_not_decel_but_refined_decel']:.3f} | "
          f"{q['r8_false_brake_in_bank_head_in_D']['bank_path_already_decel']:.3f} / {q['r8_false_brake_in_bank_head_not_D']['bank_path_already_decel']:.3f} | {q['r8_gt_cruise_not_false_brake_in_bank']['bank_path_decel']:.3f} | "
          f"{q['r7_bank_path_decel_on_gt_cruise_in_bank']:.3f} | {pr['delta']:+.3f} [{pr['lo']:+.3f}, {pr['hi']:+.3f}] |")
    a("\n### POST-HOC (Amendment 2, written after run 1; exploratory): is the GT-nearest BASE anchor itself braking where the model selected it?\n")
    a("| seed | r8 false-brake, sel == a_star, in-bank: n / a_star bank path decel [CI] | r8 GT-cruise non-FB, sel == a_star: n / decel | r7 false-brake, sel == a_star: n / decel | r7 non-FB: n / decel |")
    a("|---|---|---|---|---|")
    for s in SEEDS:
        ph = J["per_seed"][str(s)]["POSTHOC_P2_gt_nearest_base_anchor_class"]
        f8, g8, f7, g7 = ph["r8_false_brake_sel_eq_a_star_in_bank"], ph["r8_gt_cruise_not_false_brake_sel_eq_a_star_in_bank"], ph["r7_false_brake_sel_eq_a_star_in_bank"], ph["r7_gt_cruise_not_false_brake_sel_eq_a_star_in_bank"]
        ci8 = ph.get("r8_false_brake_sel_eq_a_star_a_star_bank_decel_cell")
        ci_txt = f" [{ci8['lo']:.3f}, {ci8['hi']:.3f}]" if ci8 and ci8.get("lo") is not None else ""
        fm = lambda q: f"{q['n']} / {q['a_star_bank_path_decel']:.3f}" if q["a_star_bank_path_decel"] is not None else f"{q['n']} / n/a"
        a(f"| {s} | {f8['n']} / {f8['a_star_bank_path_decel']:.3f}{ci_txt} | {fm(g8)} | {fm(f7)} | {fm(g7)} |")
    a("\n### Row ALONG-RAND (coordinator add-on; sibling `refcv9/L1_DESIGN_ANNEX.md` 1.2 'SELECTION' verdict): NOT MEASURABLE -- the refined fan is not banked.  "
      "Random-candidate (`RAND`) and median-progress (`MED`) alignment r / g* on the same S2 compliant windows, both seeds, need a re-roll; definitions, reading rule (SELECTOR-ADDS-VALUE iff `r_os - r_RAND` separated above 0 at both seeds, SELECTION-SUPPORTED iff not) "
      "and controls are pre-registered in `PREREG_BRAKE_LOCUS_AMENDMENT1.md`, the statistics are implemented and self-tested (control C7), the design is `BRAKE_LOCUS_REROLL_DESIGN.md`.\n")
    a("## 4. Controls\n")
    c = J["controls"]
    a(f"* **ALL_PASS: {c['ALL_PASS']}** (C1 {c['C1_pass']}, C2 {c['C2_pass']}, C3 {c['C3_pass']}, C4 {c['C4_pass']}, C5 {c['C5_pass']}, C6 {c['C6_pass']}, C7 {c['C7_pass']}).")
    a(f"* C1 literals (from `L2_HEADROOM.json`): false-decel r8 0.265625 / 0.257543, r7 0.091056 / 0.093750, n 3,251 / 138 and 1,856 / 135: {c['C1']}.")
    a("* C2 alignment + head join (extras `nav_true.traj` bit-equal `os`, so a mis-join cannot pass), C3 identities, C4 decision-function arms (oracle head, never-decel head, rule-logic table), see JSON.")
    a("* C5 mutations: " + "; ".join(f"{k}: {'RED' if v['went_RED'] else 'NOT RED'}" for k, v in c["C5"].items()) + ".")
    a(f"* C7 along-track statistics self-test (K1 / K2 / K3 known values, g* mutation RED, iid residual r ~ 0, verdict logic): pass {c['C7']['pass']}.")
    a(f"* C6 fan function self-test (synthetic fan, known E 0.25 / 0.5 / 0.25): pass {c['C6']['pass']}; mutations X<0 RED {c['C6']['mutation_X_negative']['went_RED']}, X huge RED {c['C6']['mutation_X_huge']['went_RED']}.\n")
    a("## 5. Caveats\n")
    for x in J["caveats"]:
        a(f"* {x}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[brake_locus] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    if "--ckpt-scalars" in sys.argv:
        return ckpt_scalars()
    t0 = time.time()
    pre_md, pre_sha = OUT / "PREREG_BRAKE_LOCUS.md", OUT / "PREREG_BRAKE_LOCUS.sha256"
    sealed, actual = pre_sha.read_text(encoding="utf-8").strip(), sha256_bytes(pre_md)
    if sealed != actual:
        raise SystemExit(f"[brake_locus] REFUSED: PREREG sha256 sealed {sealed} != file {actual}")
    am_md, am_sha = OUT / "PREREG_BRAKE_LOCUS_AMENDMENT1.md", OUT / "PREREG_BRAKE_LOCUS_AMENDMENT1.sha256"
    am_sealed, am_actual = am_sha.read_text(encoding="utf-8").strip(), sha256_bytes(am_md)
    if am_sealed != am_actual:
        raise SystemExit(f"[brake_locus] REFUSED: AMENDMENT1 sha256 sealed {am_sealed} != file {am_actual}")
    am2_md, am2_sha = OUT / "PREREG_BRAKE_LOCUS_AMENDMENT2.md", OUT / "PREREG_BRAKE_LOCUS_AMENDMENT2.sha256"
    am2_sealed, am2_actual = am2_sha.read_text(encoding="utf-8").strip(), sha256_bytes(am2_md)
    if am2_sealed != am2_actual:
        raise SystemExit(f"[brake_locus] REFUSED: AMENDMENT2 sha256 sealed {am2_sealed} != file {am2_actual}")
    import pair_read_r8 as PR
    import cruise_brake_retime_r8 as CB
    import s6_lon_decomp as S6

    root8 = f"{ARCHIVE}/{TREE}"
    D: dict = {"T8": {}, "T7": {}, "X8": {}, "X7": {}, "z": {}, "eid": {}, "B8": {}, "B7": {}, "IDN_B8": {}, "IDN_B7": {}}
    inputs = {}
    for s in SEEDS:
        D["T8"][s] = load_tree(root8, s, True)
        D["T7"][s] = load_tree(R7ROOT, s, False)
        D["B8"][s], D["IDN_B8"][s] = load_sel_bank(root8, s, D["T8"][s])
        D["B7"][s], D["IDN_B7"][s] = load_sel_bank(R7ROOT, s, D["T7"][s])
        D["X8"][s] = join_extras(root8, s, D["T8"][s], "episode")
        D["X7"][s] = join_extras(R7ROOT, s, D["T8"][s], "clip")
        zp = DEP / f"DEPLOY8_POS_{TREE}_s{s}.npz"
        D["z"][s] = np.load(zp, allow_pickle=True)
        D["eid"][s] = [str(x) for x in D["z"][s]["S6_eid"]]
        inputs[f"r8_extras_s{s}"] = {"path": D["X8"][s]["path"], "sha256_12": sha256_bytes(D["X8"][s]["path"])[:12]}
        inputs[f"r7_extras_s{s}"] = {"path": D["X7"][s]["path"], "sha256_12": sha256_bytes(D["X7"][s]["path"])[:12]}
        inputs[f"deploy8_npz_s{s}"] = {"path": str(zp), "sha256_12": sha256_bytes(zp)[:12]}
    print(f"[{time.time() - t0:6.1f}s] loaded banked trees", flush=True)
    res = {s: measure(PR, CB, S6, D, s) for s in SEEDS}
    print(f"[{time.time() - t0:6.1f}s] measured", flush=True)
    ctl = controls(PR, CB, S6, D, res)
    print(f"[{time.time() - t0:6.1f}s] controls ALL_PASS={ctl['ALL_PASS']}", flush=True)

    ht = head_test({s: res[s]["i"]["mean_new"] for s in SEEDS}, {s: res[s]["clause2_paired"]["delta"] for s in SEEDS}, {s: res[s]["P1"]["exceeds_null"] for s in SEEDS})
    ht = {k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v) for k, v in ht.items()}
    lab = final_label(ht["HEAD_TEST"] == "PASS", None)
    r0, r1 = res[0], res[1]
    q0, q1 = r0["S7_unrefined_selected_anchor"], r1["S7_unrefined_selected_anchor"]
    c2_0, c2_1 = r0["clause2_paired"], r1["clause2_paired"]
    e0, e1 = r0["S3_excess_decomposition"], r1["S3_excess_decomposition"]
    k0, k1 = r0["S2_coupling_P_plan_brakes_given_own_head_in_D"], r1["S2_coupling_P_plan_brakes_given_own_head_in_D"]
    s10, s11 = r0["S1_head_decel_share_on_gt_cruise"], r1["S1_head_decel_share_on_gt_cruise"]
    reading = (
        f"Clause 1 holds ((i) = {r0['i']['mean_new']:.3f} / {r1['i']['mean_new']:.3f} at seeds 0 / 1, chance {r0['P1']['perm_mean']:.3f}) but clause 2 does not: where r8's own head says cruise, "
        f"r8 brakes {c2_0['mean_new']:.3f} vs refcv7's {c2_0['mean_ref']:.3f} (paired delta {c2_0['delta']:+.3f} [{c2_0['lo']:+.3f}, {c2_0['hi']:+.3f}], seed 1 {c2_1['delta']:+.3f} [{c2_1['lo']:+.3f}, {c2_1['hi']:+.3f}]).  "
        f"The label stays {lab['label']}: SELECTION vs CANDIDATE needs the refined fan.  "
        f"What the banked data DO say (association): {100 * e0['stratum_r8_head_in_D']['share']:.0f} % / {100 * e1['stratum_r8_head_in_D']['share']:.0f} % of the r8-minus-r7 false-brake excess sits in windows where r8's head says decel; "
        f"r8's plan follows its own decel head {k0['r8']['mean_new']:.2f} / {k1['r8']['mean_new']:.2f} of the time against refcv7's {k0['r7']['mean_new']:.2f} / {k1['r7']['mean_new']:.2f}, although refcv7's head says decel MORE often on GT-cruise windows "
        f"({s10['r7']['mean_new']:.2f} vs {s10['r8']['mean_new']:.2f}) -- the difference is the head-to-plan coupling, not the head's decel rate.  "
        f"The braking is already in the SELECTED UNREFINED anchor in {100 * q0['r8_false_brake_in_bank']['bank_path_already_decel']:.0f} % / {100 * q1['r8_false_brake_in_bank']['bank_path_already_decel']:.0f} % of in-bank false-brake windows "
        f"(vs {100 * q0['r8_gt_cruise_not_false_brake_in_bank']['bank_path_decel']:.0f} % on other GT-cruise windows), so it is decided in WHICH ANCHOR is chosen, not in refinement; the ceiling filter, the speed-label input and the E9 graft are not the locus (S5, S4).  "
        f"POST-HOC: where the selector did pick the GT-nearest base anchor on a false-brake window, that anchor itself brakes in {100 * r0['POSTHOC_P2_gt_nearest_base_anchor_class']['r8_false_brake_sel_eq_a_star_in_bank']['a_star_bank_path_decel']:.0f} % / "
        f"{100 * r1['POSTHOC_P2_gt_nearest_base_anchor_class']['r8_false_brake_sel_eq_a_star_in_bank']['a_star_bank_path_decel']:.0f} % of cases (base-bank coverage is thin there, for refcv7 too).  "
        f"NOT separable from banked data: whether the head CAUSES the pick (needs the force_lon F2 / head-neutral-ranking interventions), and whether the refined fan holds a good cruising candidate (needs the fan).")
    J = {"tool": "code/brake_locus_r8.py", "tier": TIER, "estimator": ESTIMATOR, "tree": TREE, "r7_reference": R7ROOT, "prereg_sha256": sealed, "amendment1_sha256": am_sealed, "amendment2_sha256_POSTHOC": am2_sealed,
         "script_sha256": sha256_bytes(__file__), "ckpt_scalars": (jload(OUT / "BRAKE_LOCUS_ckpt_scalars.json") if (OUT / "BRAKE_LOCUS_ckpt_scalars.json").exists() else None), "inputs": inputs, "label": lab, "head_test": ht, "reading": reading,
         "fan_stage": {"status": "NOT_MEASURABLE", "reason": "the refined fan (out['anchor_traj'] [W,149,8,2]), per-candidate selection scores and candidate tags are not in the banked dumps; "
                                                              "only fan_term_heading [W,149] (terminal heading, no longitudinal content) and reach_keep [W,149] are",
                       "X_FDE6_m": X_FDE6_M, "X_ADE_m": X_ADE_M, "rule": {"CANDIDATE": f"E < {E_CAND}", "SELECTION": f"E >= {E_SEL}"}},
         "along_track_control_ALONG_RAND": {"status": "NOT_MEASURABLE", "reason": "needs a random / median-progress candidate of the REFINED fan; the fan is not banked (AMENDMENT 1 A3).  Statistics, arms, reading rule and controls are pre-registered and implemented "
                                                                                    "(along_stats, along_r_diff_boot, along_verdict, selftest_along = control C7); the re-roll design is raw/refcv9_headroom/BRAKE_LOCUS_REROLL_DESIGN.md",
                                            "reading_rule": "dR = r_os - r_RAND, paired episode-cluster bootstrap, per seed: SELECTOR-ADDS-VALUE iff lo > 0 at both seeds; SELECTION-SUPPORTED iff lo <= 0 at both seeds; else SEED-SPLIT",
                                            "sibling_values_for_reference_not_ours": {"os_r": 0.402, "orc_r": 0.760, "source": "refcv9/L1_DESIGN_ANNEX.md 1.2 (in-bank compliant S2 windows, EMA-30k)"}},
         "inventory": {"refined fan (all candidates, full trajectories)": "NO (only terminal heading [W,149] + reach mask)", "per-candidate selection scores": "NO (only the max, sel_score_max)",
                       "candidate tags (lon6 / alloc / kind)": "NO", "selected index, pre-graft index, GT-nearest base anchor": "yes (decisions/sel_idx, sel_idx_base, a_star)",
                       "selected plan, 8 slots": "yes (decisions/plan_full_nav_true)", "tactical lon head logits (tacv6 = planner-feeding; lon_logits_tac = legacy)": "yes (refcv6_extras.npz)",
                       "head constraint outputs (cons_lon8)": "NO", "oracle_sel (refined GT-nearest plan)": "S2 grid only (0-2 s)", "sel_bank_* (selected anchor's UNREFINED 8-slot path)": "yes, for windows with sel_idx < 117 only (rows = those windows in order; AMENDMENT 1 A1 corrects the original prereg's inadmissible); S6 files carry the S2 array unsliced",
                       "ablated arms (filter_off, vmax_zero)": "seed 0 only"},
         "per_seed": {str(s): {k: v for k, v in res[s].items() if k != "_arrays"} for s in SEEDS}, "controls": ctl,
         "caveats": ["single TRAINING seed per arm: the intervals are episode draws only and are blind to training variance (H-ESTIM-SEED-1); the two inference seeds are the replicate for sampler noise and are not pooled.",
                     "T1 open loop on logged eval139 frames: a statement about plans scored against the logged future, not about driving.",
                     "ASSOCIATION, not intervention.  In refcv8 the head reaches the planner through candidate allocation, the ranking score (beta_lon * log p) and the constraint terms, so HEAD and SELECTION are coupled by design; only an intervention (force_lon F2, head-neutral ranking S0) separates them.",
                     "the speed-label input differs between refcv7 (per-clip continuous v_max_ms) and refcv8 (4-class road-type label): every refcv8-vs-refcv7 row crosses that input change.",
                     "the head argmax is a hard decision; head confidence, the constraint outputs and the allocation histogram are not used here.",
                     "the two tactical lon heads (tacv6, legacy lon_logits_tac) disagree on 22.6 % of windows; the primary is tacv6 (the planner-feeding head, the one H-R8-S6-LON-1 used); the legacy head is a sensitivity.",
                     "(ii) / (iii) are degenerate as worded (see prereg section 0); the fan stage is defined and self-tested but cannot run on banked data.",
                     "S7 covers only in-bank selections (sel_idx < 117); the share is reported.  The unrefined bank path of the selected anchor is the residual family rolled from the prior (v0-conditioned); its class uses the same +-1.5 m/s rule on the last chord (5 s -> 6 s)."],
         "wall_s": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return o
    J = clean(J)
    with open(OUT / "BRAKE_LOCUS.json", "w", encoding="utf-8") as f:
        json.dump(J, f, indent=1)
    write_md(J, OUT / "BRAKE_LOCUS.md")
    print(f"[{time.time() - t0:6.1f}s] wrote BRAKE_LOCUS.json / .md  label={lab['label']}  HEAD_TEST={ht['HEAD_TEST']}  controls_ALL_PASS={ctl['ALL_PASS']}", flush=True)
    return 0 if ctl["ALL_PASS"] else 2


if __name__ == "__main__":
    sys.exit(main())
