"""DOES A PER-SITUATION (window, lead_s) BEAT THE SINGLE FROZEN SETTING?

Pre-registration: ./PRE_REGISTRATION.md — written and staged BEFORE any number here was read.

THE FINDING THIS ACTS ON (MEASURED by the sibling stream, ../2026-08-03-sitclf-temporal/)
-----------------------------------------------------------------------------------------
`intersection` skill DECAYS monotonically with the anticipation horizon (+0.982 -> +0.378 over
1->5 s; precision-lift 2.713 -> 1.794) while `lane_change` RISES and separates ONLY at 5 s. The
programme forces ONE window and ONE horizon onto two phenomena with opposite timescales.

TWO QUESTIONS, ONE SET OF SCORE COLUMNS (pre-registration Sec 1)
  Q-A TRAIN-HORIZON   L is only the head's TRAINING target; evaluation stays the frozen
                      lead_s = 3.0 label. Touches no frozen constant.
  Q-B DEPLOY-HORIZON  L is the declared warning horizon; evaluated on a horizon-agnostic
                      EVENT-level yardstick (Sec 3) whose denominators are identical for
                      every arm.

⛔ VISION-ONLY at inference (PI ruling 2026-08-03). Every arm reads frozen v1 camera latents and
nothing else. The ego block is used ONLY for the LONGITUDINAL/LATERAL family STRATA, and this run
uses the CAUSAL rebuild from rebuild_causal_ego.py, not the substrate's legacy block.

usage:
  python run_per_situation_horizon.py --substrate <npz> --out results_horizon_ps.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.data.situations import (anticipation_target,            # noqa: E402
                                     detect_intersection,
                                     detect_lane_change,
                                     detect_roundabout, kinematics)
from tanitad.eval.ap_ci import _draws, average_precision             # noqa: E402
from tanitad.eval.sitclf import (causal_window, clip_runs,           # noqa: E402
                                 cluster_folds, ridge_scores)
from tanitad.eval.sitclf_deploy import precision_recall_at_budget    # noqa: E402

SITS = ("lane_change", "roundabout", "intersection")
CACHES = (r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74",
          r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836")

# ---- pre-registered grid and constants (Sec 2.1). NOTHING here moves after a number is read.
WINS = (1, 4, 8, 16, 32)
LEADS = (1.0, 2.0, 3.0, 4.0, 5.0)
WIN_MAX = 32
RANK = 16
FROZEN_CELL = (8, 3.0)
H_MAX_S = 5.0                    # the event yardstick's common look-back, Sec 3
DEPLOY_LEAD_S = 3.0              # the frozen deployed horizon — Q-A's evaluation label
SEL_FRAC = 0.20
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
TOP_FRAC = 0.05
C_POW_BAR = 40
HZ = 10.0
N_BOOT = 2000
SEED = 0
ALPHA = 0.05


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


# --------------------------------------------------------------------------- #
# substrate-side rebuild                                                       #
# --------------------------------------------------------------------------- #
def events_per_clip(limit_frames):
    files = []
    for root in CACHES:
        files += sorted(glob.glob(os.path.join(root, "ep_*.pt")))
    out = []
    for f in files:
        P = np.asarray(torch.load(f, map_location="cpu", weights_only=True,
                                  mmap=True)["poses"]).astype(np.float64)
        K = kinematics(P)
        out.append((int(K["T"]), {
            "lane_change": detect_lane_change(K),
            "roundabout": detect_roundabout(K, bracket=True),
            "intersection": detect_intersection(K, cross=None)[0]}))
    tot = sum(t for t, _ in out)
    if tot != limit_frames:
        raise SystemExit(f"C-FID: rebuilt {tot} frames but substrate has {limit_frames}")
    log(f"events rebuilt for {len(out)} clips, {tot:,} frames — C-FID OK")
    return out


def targets_at(per_clip, lead_s):
    Y, V = [], []
    for T, ev in per_clip:
        y = np.zeros((T, 3), bool)
        v = np.zeros((T, 3), bool)
        for i, s in enumerate(SITS):
            y[:, i], v[:, i] = anticipation_target(T, ev[s], lead_s=lead_s)
        Y.append(y)
        V.append(v)
    return np.concatenate(Y), np.concatenate(V)


def fit_pca(F, rows, r, device="cpu"):
    A = torch.from_numpy(np.asarray(F[rows], dtype=np.float32)).to(device)
    mu = A.mean(0, keepdim=True)
    A = A - mu
    _u, _s, v = torch.svd_lowrank(A, q=r + 16, niter=4)
    del A
    if device != "cpu":
        torch.cuda.empty_cache()
    return mu.cpu().numpy(), v[:, :r].cpu().numpy().astype(np.float32)


def project(F, mu, W):
    """Chunked so the fp16 -> fp32 promotion never materialises a 99k x 2048 float32 copy."""
    out = np.empty((F.shape[0], W.shape[1]), np.float32)
    for b in range(0, F.shape[0], 20000):
        out[b:b + 20000] = (np.asarray(F[b:b + 20000], np.float32) - mu) @ W
    out /= max(float(np.abs(out).mean()), 1e-6)
    return out


def ap_lift_safe(y, s):
    b = y.mean()
    if b <= 0:
        return float("nan")
    return average_precision(y, s) / b


# --------------------------------------------------------------------------- #
# the cluster-draw stream — delegated, so it is the programme's estimator       #
# --------------------------------------------------------------------------- #
def cluster_picks(cluster_ids, n_boot=N_BOOT, seed=SEED):
    """`n_boot` multisets of CLUSTER POSITIONS, bit-identical to the row-level draws.

    ``ap_ci._draws`` yields ``concat(idx_by_ep[p] for p in rng.choice(uniq, n_ep))``. Feeding it a
    map whose value for cluster ``u`` is the single position of ``u`` therefore returns exactly the
    cluster pick that the row-level call would have expanded — same RNG stream, same multiset. That
    is what lets ONE draw set drive the frame-level AP-lift (Q-A) and the event-level yardstick
    (Q-B) without either re-implementing the estimator.
    """
    uniq = np.unique(np.asarray([str(x) for x in cluster_ids]))
    pos = {u: np.array([i], dtype=np.int64) for i, u in enumerate(uniq)}
    return uniq, [sel for sel in _draws(uniq, pos, n_boot, seed)]


def pack_paired(point, d):
    ok = np.isfinite(d)
    v = d[ok]
    lo, hi = ((np.percentile(v, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))
              if v.size else (np.nan, np.nan))
    return {"delta": round(float(point), 5), "lo": round(float(lo), 5),
            "hi": round(float(hi), 5), "ci95": round(float((hi - lo) / 2.0), 5),
            "p_delta_gt0": (round(float((v > 0).mean()), 4) if v.size else float("nan")),
            "separated": bool(lo > 0 or hi < 0), "n_boot": N_BOOT,
            "n_boot_valid": int(ok.sum()),
            "estimator": "paired_ap_episode_cluster_bootstrap (cluster draws, taniteval)"}


def pack_single(point, b):
    ok = np.isfinite(b)
    v = b[ok]
    lo, hi = ((np.percentile(v, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))
              if v.size else (np.nan, np.nan))
    return {"point": round(float(point), 5), "lo": round(float(lo), 5),
            "hi": round(float(hi), 5), "ci95": round(float((hi - lo) / 2.0), 5),
            "n_boot": N_BOOT, "n_boot_valid": int(ok.sum()),
            "estimator": "ap_episode_cluster_bootstrap (cluster draws, taniteval)"}


# --------------------------------------------------------------------------- #
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--substrate",
                     default=r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz")
    ap_.add_argument("--ego-causal",
                     default=r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.ego_causal.npz")
    ap_.add_argument("--parent-horizon",
                     default=str(Path(__file__).resolve().parents[1] /
                                 "2026-08-03-sitclf-temporal" / "results_horizon.json"))
    ap_.add_argument("--out", default="results_horizon_ps.json")
    ap_.add_argument("--scores-out", default="results_horizon_ps.scores.npz")
    ap_.add_argument("--device", default=None)
    a = ap_.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    z = np.load(a.substrate)
    F = z["F"]
    cc = z["clip_cluster"]
    st, en = clip_runs(cc)
    N = F.shape[0]
    folds = cluster_folds(cc, 2, seed=0)
    log(f"substrate {N:,} x {F.shape[1]}  {len(st)} clips  device={device}")

    per_clip = events_per_clip(N)

    # ---- labels at every lead + the ONE evaluation row set (pre-reg Sec 2) -------------
    Y_at, V_at = {}, {}
    for L in LEADS:
        Y_at[L], V_at[L] = targets_at(per_clip, L)
    _, hist32 = causal_window(np.zeros((N, 1), np.float32), st, en, WIN_MAX)
    EVAL = V_at[H_MAX_S] & hist32[:, None]           # valid(5.0) ∧ hist_ok(32)
    Y_EVAL = Y_at[DEPLOY_LEAD_S]                     # the FROZEN deployed label
    log(f"hist_ok(32) keeps {hist32.sum():,}/{N:,} rows "
        f"({100*hist32.mean():.1f}%)")

    # ---- C-POW, counted and written BEFORE any score is read --------------------------
    cpow = {}
    for i, s in enumerate(SITS):
        m = EVAL[:, i]
        yv = Y_EVAL[m, i]
        cl = int(len(np.unique(cc[m][yv])))
        cpow[s] = {"n_scorable": int(m.sum()), "n_pos_lead3": int(yv.sum()),
                   "base_rate_lead3": round(float(yv.mean()), 6),
                   "n_clusters_with_a_positive": cl,
                   "C_POW_pass": bool(cl >= C_POW_BAR), "bar": C_POW_BAR}
        log(f"C-POW {s:>13}: {int(m.sum()):>6,} rows, {int(yv.sum()):>5,} pos, "
            f"{cl:>3} positive clusters -> {'PASS' if cl >= C_POW_BAR else 'UNDERPOWERED'}")
    Path("c_pow_precommit.json").write_text(json.dumps(cpow, indent=1), encoding="utf-8")

    # ---- folds / inner selection split (parent convention, verbatim) ------------------
    splits = []
    for f in (0, 1):
        te = folds == f
        tr = ~te
        tr_cl = np.unique(cc[tr])
        perm = np.random.default_rng(0 + f).permutation(tr_cl)
        sel_cl = set(int(x) for x in perm[:max(1, int(round(SEL_FRAC * len(tr_cl))))])
        is_sel = np.array([int(x) in sel_cl for x in cc])
        splits.append((tr & ~is_sel, tr & is_sel, te))

    # ---- the NEG_FEAT clip permutation (parent run_temporal.py verbatim) --------------
    perm_clip = np.random.default_rng(0).permutation(len(st))
    shuf_rows = np.arange(N)
    for i, (s0, e0) in enumerate(zip(st, en)):
        j = perm_clip[i]
        n = min(e0 - s0, en[j] - st[j])
        shuf_rows[s0:s0 + n] = st[j] + np.arange(n)
        if e0 - s0 > n:
            shuf_rows[s0 + n:e0] = st[j] + n - 1

    # ---- the 25-cell grid, one substrate at a time so the windows never co-reside ------
    # CELL[(shuf, W, L)] -> [N, 3] out-of-fold scores;  SELCRIT[(shuf, W, L, f, i)] -> dict
    CELL, SELCRIT, CHOSEN_LAM, fid_ridge = {}, {}, {}, {}
    for shuf in (False, True):
        src = F[shuf_rows] if shuf else F
        XW = {}
        for f in (0, 1):
            mu, Wp = fit_pca(src, np.flatnonzero(splits[f][0]), RANK, device=device)
            P = project(src, mu, Wp)
            for W in WINS:
                XW[(f, W)] = causal_window(P, st, en, W)[0]
            log(f"  PCA+windows built shuf={shuf} fold={f} ({time.time()-t_start:.0f}s)")
        del src
        if not shuf:
            # C-FID-RIDGE: the multi-lambda path must BE stack's ridge_scores
            for W in WINS:
                X = XW[(0, W)]
                fit = splits[0][0] & hist32
                Yl = Y_at[3.0][fit].astype(np.float32)
                Vl = (V_at[3.0] & hist32[:, None])[fit].astype(np.float32)
                ref = ridge_scores(X[fit], Yl, Vl, X[:5000], lam=100.0)
                got = ridge_multi(X[fit], Yl, Vl, X[:5000], (100.0,))[100.0]
                d = float(np.abs(ref - got).max())
                fid_ridge[f"W{W}"] = d
                if d > 0.0:
                    raise SystemExit(f"C-FID-RIDGE FAILED at W={W}: {d:.3e} vs ridge_scores")
            log(f"C-FID-RIDGE: multi-lambda path bit-identical to stack ridge_scores at "
                f"every W (max|diff| {max(fid_ridge.values()):.1e})")
        for W in WINS:
            for L in LEADS:
                out = np.zeros((N, 3), np.float32)
                for f in (0, 1):
                    X = XW[(f, W)]
                    fit, sel, te = splits[f]
                    fitr = fit & hist32
                    vL = V_at[L] & hist32[:, None]
                    S = ridge_multi(X[fitr], Y_at[L][fitr].astype(np.float32),
                                    vL[fitr].astype(np.float32), X, LAMBDAS)
                    for i in range(3):
                        msel = sel & vL[:, i]
                        best = (-np.inf, LAMBDAS[0])
                        for lam in LAMBDAS:
                            if msel.sum() < 50 or Y_at[L][msel, i].sum() < 5:
                                continue
                            v = ap_lift_safe(Y_at[L][msel, i].astype(np.float64),
                                             S[lam][msel, i].astype(np.float64))
                            if np.isfinite(v) and v > best[0]:
                                best = (v, lam)
                        out[te, i] = S[best[1]][te, i]
                        CHOSEN_LAM[(shuf, W, L, f, i)] = best[1]
                        # the SELECTION criterion, on SEL rows only, never on the test fold
                        ms = sel & EVAL[:, i]
                        qa = (ap_lift_safe(Y_EVAL[ms, i].astype(np.float64),
                                           S[best[1]][ms, i].astype(np.float64))
                              if ms.sum() > 50 and Y_EVAL[ms, i].sum() >= 5 else np.nan)
                        SELCRIT[(shuf, W, L, f, i)] = {"qa_ap_lift": float(qa)}
                CELL[(shuf, W, L)] = out
            log(f"  grid W={W:>2} shuf={shuf} done ({time.time()-t_start:.0f}s)")
        del XW

    # ---- the EVENT universe (Q-B), fixed before any arm is scored --------------------
    HB = int(round(H_MAX_S * HZ))
    onsets, onset_cl, onset_sit = [], [], []
    for ci, (s0, e0) in enumerate(zip(st, en)):
        ev = per_clip[ci][1]
        for i, s in enumerate(SITS):
            for (aa, _bb) in ev[s]:
                g = s0 + aa
                lo = max(s0, g - HB)
                if lo < g and EVAL[lo:g, i].any():
                    onsets.append(g)
                    onset_cl.append(cc[s0])
                    onset_sit.append(i)
    onsets = np.array(onsets, np.int64)
    onset_cl = np.array(onset_cl)
    onset_sit = np.array(onset_sit, np.int64)
    log(f"event universe: {len(onsets)} scorable onsets "
        f"({[int((onset_sit==i).sum()) for i in range(3)]} per situation)")

    # per-row distance to the next onset of that situation, within the clip
    NEXT = np.full((N, 3), np.inf, np.float64)
    for ci, (s0, e0) in enumerate(zip(st, en)):
        for i in range(3):
            g = onsets[(onset_sit == i) & (onsets >= s0) & (onsets < e0)]
            if g.size == 0:
                continue
            t = np.arange(s0, e0)
            k = np.searchsorted(g, t, side="right")
            has = k < g.size
            NEXT[t[has], i] = g[k[has]] - t[has]

    # ---- assemble the ARMS from the grid (pre-reg Sec 4) ------------------------------
    def assemble(shuf, pick):
        """pick(f, i) -> (W, L); returns the [N,3] out-of-fold column."""
        out = np.zeros((N, 3), np.float32)
        chosen = {}
        for f in (0, 1):
            te = splits[f][2]
            for i in range(3):
                W, L = pick(f, i)
                out[te, i] = CELL[(shuf, W, L)][te, i]
                chosen[f"fold{f}.{SITS[i]}"] = {"win": W, "lead_s": L}
        return out, chosen

    def sel_best(shuf, f, i):
        cand = [(SELCRIT[(shuf, W, L, f, i)]["qa_ap_lift"], W, L)
                for W in WINS for L in LEADS]
        cand = [c for c in cand if np.isfinite(c[0])]
        return (max(cand)[1], max(cand)[2]) if cand else FROZEN_CELL

    def global_best(shuf, f):
        best = (-np.inf, *FROZEN_CELL)
        for W in WINS:
            for L in LEADS:
                v = [SELCRIT[(shuf, W, L, f, i)]["qa_ap_lift"] for i in range(3)]
                v = [x for x in v if np.isfinite(x)]
                if v and np.mean(v) > best[0]:
                    best = (float(np.mean(v)), W, L)
        return best[1], best[2]

    arms, arm_cfg = {}, {}
    for shuf, tag in ((False, ""), (True, "NULL_")):
        arms[tag + "FROZEN"], arm_cfg[tag + "FROZEN"] = assemble(
            shuf, lambda f, i: FROZEN_CELL)
        arms[tag + "PS_SEL"], arm_cfg[tag + "PS_SEL"] = assemble(
            shuf, lambda f, i, sh=shuf: sel_best(sh, f, i))
        arms[tag + "C_GLOBAL"], arm_cfg[tag + "C_GLOBAL"] = assemble(
            shuf, lambda f, i, sh=shuf: global_best(sh, f))
    for k, v in arm_cfg.items():
        log(f"  arm {k:>14}: {v}")

    # C-ORACLE-PS — chosen on the TEST fold (cheating, upper bound), Q-A criterion
    def oracle_pick(f, i):
        best = (-np.inf, *FROZEN_CELL)
        m = splits[f][2] & EVAL[:, i]
        if m.sum() < 50 or Y_EVAL[m, i].sum() < 5:
            return FROZEN_CELL
        yv = Y_EVAL[m, i].astype(np.float64)
        for W in WINS:
            for L in LEADS:
                v = ap_lift_safe(yv, CELL[(False, W, L)][m, i].astype(np.float64))
                if np.isfinite(v) and v > best[0]:
                    best = (v, W, L)
        return best[1], best[2]
    arms["C_ORACLE_PS"], arm_cfg["C_ORACLE_PS"] = assemble(False, oracle_pick)
    log(f"  arm    C_ORACLE_PS: {arm_cfg['C_ORACLE_PS']}")

    res = {"_what": "does a PER-SITUATION (window, lead_s) beat the single frozen setting?",
           "_prereg": "./PRE_REGISTRATION.md (staged before any number here was read)",
           "_vision_only": "every arm reads frozen v1 camera latents only (PI ruling 2026-08-03)",
           "substrate": a.substrate, "grid": {"wins": list(WINS), "leads_s": list(LEADS)},
           "frozen_cell": {"win": FROZEN_CELL[0], "lead_s": FROZEN_CELL[1]},
           "eval_rows": "valid(lead=5.0) AND hist_ok(WIN_MAX=32)",
           "eval_label_qa": f"lead_s = {DEPLOY_LEAD_S} (the frozen deployed target)",
           "lambda_selection": ("per situation on the SEL split, by AP-lift against the arm's OWN "
                                "training label — question-agnostic, so lambda is not a moving part"),
           "C_POW": cpow, "C_FID_RIDGE_max_abs_diff": fid_ridge,
           "C_FID_PARENT": (json.loads(Path("cfid_parent.json").read_text(encoding="utf-8"))
                            if Path("cfid_parent.json").exists() else "NOT RUN"),
           "arm_configurations": arm_cfg, "n_boot": N_BOOT,
           "per_situation": {}}

    # ---- ONE cluster-draw stream for both questions ----------------------------------
    uniq_cl, picks = cluster_picks(cc)
    cl_index = {int(u): k for k, u in enumerate(uniq_cl.astype(np.int64))}
    log(f"{len(picks)} cluster draws over {len(uniq_cl)} clusters prepared "
        f"({time.time()-t_start:.0f}s)")

    grid_names = {f"CELL_w{W}_L{L}": (False, W, L) for W in WINS for L in LEADS}
    for i, s in enumerate(SITS):
        t0 = time.time()
        m = EVAL[:, i]
        rows = np.flatnonzero(m)
        yv = Y_EVAL[rows, i].astype(np.float64)
        rcl = np.array([cl_index[int(x)] for x in cc[rows]])
        # rows grouped by cluster, so a draw is one concatenate
        order = np.argsort(rcl, kind="mergesort")
        rows_by_cl = [order[np.searchsorted(rcl[order], k, "left"):
                            np.searchsorted(rcl[order], k, "right")]
                      for k in range(len(uniq_cl))]

        cols = {n: arms[n][rows, i].astype(np.float64) for n in arms}
        for n, (sh, W, L) in grid_names.items():
            cols[n] = CELL[(sh, W, L)][rows, i].astype(np.float64)

        # ---------- Q-A : frame-level AP-lift on the FROZEN label ---------------------
        pts = {n: ap_lift_safe(yv, c) for n, c in cols.items()}
        boots = {n: np.empty(len(picks)) for n in cols}
        for k, pick in enumerate(picks):
            sel = np.concatenate([rows_by_cl[p] for p in pick])
            ys = yv[sel]
            b = ys.mean()
            if b <= 0:
                for n in cols:
                    boots[n][k] = np.nan
                continue
            for n, c in cols.items():
                boots[n][k] = average_precision(ys, c[sel]) / b
        log(f"  {s}: Q-A bootstrapped {len(cols)} columns over {len(rows):,} rows "
            f"({time.time()-t0:.0f}s)")

        qa = {"n_scorable": int(len(rows)), "n_pos": int(yv.sum()),
              "base_rate": round(float(yv.mean()), 6),
              "n_clusters": int(len(np.unique(rcl))),
              "n_clusters_with_a_positive": int(len(np.unique(rcl[yv > 0]))),
              "arms": {}}
        for n in cols:
            r_ = pack_single(pts[n], boots[n])
            r_["ap"] = round(average_precision(yv, cols[n]), 5)
            r_["op_5pct"] = precision_recall_at_budget(yv, cols[n], np.ones(len(rows), bool))
            base = "NULL_FROZEN" if n.startswith("NULL_") else "FROZEN"
            r_["paired_vs_frozen"] = (None if n == base else
                                      pack_paired(pts[n] - pts[base], boots[n] - boots[base]))
            if not n.startswith("NULL_") and n in ("FROZEN", "PS_SEL", "C_GLOBAL"):
                nn = "NULL_" + n
                r_["paired_vs_own_null"] = pack_paired(pts[n] - pts[nn], boots[n] - boots[nn])
            qa["arms"][n] = r_
        qa["PS_SEL_vs_C_GLOBAL"] = pack_paired(pts["PS_SEL"] - pts["C_GLOBAL"],
                                               boots["PS_SEL"] - boots["C_GLOBAL"])
        qa["C_SEL_NULL"] = qa["arms"]["NULL_PS_SEL"]["paired_vs_frozen"]
        hw = [qa["arms"][n]["paired_vs_frozen"]["ci95"] for n in grid_names]
        qa["mde_widest_ci95"] = round(float(np.max(hw)), 5)
        qa["mde_median_ci95"] = round(float(np.median(hw)), 5)
        beats = [n for n in grid_names
                 if qa["arms"][n]["paired_vs_frozen"]["separated"]
                 and qa["arms"][n]["paired_vs_frozen"]["delta"] > 0]
        qa["grid_cells_separating_ABOVE_frozen"] = beats

        # ---------- Q-B : the horizon-agnostic EVENT yardstick ------------------------
        on_m = onset_sit == i
        og = onsets[on_m]
        ocl = np.array([cl_index[int(x)] for x in onset_cl[on_m]])
        nxt = NEXT[rows, i]
        qb = {"n_onsets": int(on_m.sum()),
              "n_onset_bearing_clusters": int(len(np.unique(ocl))),
              "C_POW_pass": bool(len(np.unique(ocl)) >= C_POW_BAR),
              "h_max_s": H_MAX_S, "top_frac": TOP_FRAC, "arms": {}}
        row_of = np.full(N, -1, np.int64)
        row_of[rows] = np.arange(len(rows))
        qb_stat = {}
        for n, c in cols.items():
            k = max(1, int(round(TOP_FRAC * len(rows))))
            al = np.zeros(len(rows), bool)
            al[np.argsort(-c, kind="mergesort")[:k]] = True
            warned = np.zeros(len(og), bool)
            lead = np.full(len(og), np.nan)
            for j, g in enumerate(og):
                idx = row_of[max(0, g - HB):g]
                idx = idx[idx >= 0]
                hit = idx[al[idx]]
                if hit.size:
                    warned[j] = True
                    lead[j] = (g - rows[hit[0]]) / HZ
            qb_stat[n] = {"alarm": al, "warned": warned, "lead": lead,
                          "useful5": nxt[al] <= H_MAX_S * HZ,
                          "useful3": nxt[al] <= DEPLOY_LEAD_S * HZ,
                          "acl": rcl[al]}
        # cluster-resampled event recall / precision / median lead
        def qb_boot(n):
            d = qb_stat[n]
            rec = np.bincount(ocl, weights=d["warned"].astype(float), minlength=len(uniq_cl))
            tot = np.bincount(ocl, minlength=len(uniq_cl)).astype(float)
            p5 = np.bincount(d["acl"], weights=d["useful5"].astype(float), minlength=len(uniq_cl))
            p3 = np.bincount(d["acl"], weights=d["useful3"].astype(float), minlength=len(uniq_cl))
            na = np.bincount(d["acl"], minlength=len(uniq_cl)).astype(float)
            R, P5, P3, LD = (np.empty(len(picks)), np.empty(len(picks)),
                             np.empty(len(picks)), np.empty(len(picks)))
            wl = d["lead"][d["warned"]]
            wcl = ocl[d["warned"]]
            for k, pick in enumerate(picks):
                cnt = np.bincount(pick, minlength=len(uniq_cl)).astype(float)
                T = cnt @ tot
                R[k] = (cnt @ rec) / T if T > 0 else np.nan
                A = cnt @ na
                P5[k] = (cnt @ p5) / A if A > 0 else np.nan
                P3[k] = (cnt @ p3) / A if A > 0 else np.nan
                rep = cnt[wcl].astype(np.int64)
                LD[k] = np.median(np.repeat(wl, rep)) if rep.sum() > 0 else np.nan
            return R, P5, P3, LD
        qb_boots = {n: qb_boot(n) for n in cols}
        for n in cols:
            d = qb_stat[n]
            R, P5, P3, LD = qb_boots[n]
            e = {"n_alarm": int(d["alarm"].sum()),
                 "n_onsets": int(len(og)),
                 "n_onsets_warned": int(d["warned"].sum()),
                 "n_onsets_no_alarm": int((~d["warned"]).sum()),
                 "event_recall": pack_single(float(d["warned"].mean()) if len(og) else np.nan, R),
                 "alarm_precision_5s": pack_single(float(d["useful5"].mean()), P5),
                 "alarm_precision_3s": pack_single(float(d["useful3"].mean()), P3),
                 "median_lead_s": pack_single(
                     float(np.median(d["lead"][d["warned"]])) if d["warned"].any() else np.nan, LD)}
            base = "NULL_FROZEN" if n.startswith("NULL_") else "FROZEN"
            if n != base:
                Rb, P5b, P3b, LDb = qb_boots[base]
                e["paired_vs_frozen"] = {
                    "event_recall": pack_paired(
                        float(d["warned"].mean() - qb_stat[base]["warned"].mean()), R - Rb),
                    "alarm_precision_5s": pack_paired(
                        float(d["useful5"].mean() - qb_stat[base]["useful5"].mean()), P5 - P5b),
                    "alarm_precision_3s": pack_paired(
                        float(d["useful3"].mean() - qb_stat[base]["useful3"].mean()), P3 - P3b),
                    "median_lead_s": pack_paired(
                        float(np.median(d["lead"][d["warned"]]) -
                              np.median(qb_stat[base]["lead"][qb_stat[base]["warned"]]))
                        if d["warned"].any() and qb_stat[base]["warned"].any() else np.nan,
                        LD - LDb)}
            qb["arms"][n] = e
        # Q-B per-situation preference: which grid cell maximises event recall?
        qb["best_grid_cell_by_event_recall"] = max(
            grid_names, key=lambda n: qb["arms"][n]["event_recall"]["point"])

        # ---------- the pre-registered verdict ----------------------------------------
        ps = qa["arms"]["PS_SEL"]["paired_vs_frozen"]
        orc = qa["arms"]["C_ORACLE_PS"]["paired_vs_frozen"]
        gl = qa["PS_SEL_vs_C_GLOBAL"]
        selnull = qa["C_SEL_NULL"]
        powered = cpow[s]["C_POW_pass"]
        verdict = (
            "UNDERPOWERED_C_POW" if not powered else
            "SELECTION_ARTEFACT" if (selnull["separated"] and selnull["delta"] > 0) else
            ("PER_SITUATION_WINS" if (gl["separated"] and gl["delta"] > 0)
             else "GAIN_IS_SELECTION_NOT_PER_SITUATION")
            if (ps["separated"] and ps["delta"] > 0) else
            "NO_EFFECT_ABOVE_MDE" if (orc["separated"] and orc["delta"] > 0) else
            "NO_PER_SITUATION_GAIN_EXISTS")
        res["per_situation"][s] = {"C_POW": cpow[s], "VERDICT": verdict,
                                   "QA_TRAIN_HORIZON": qa, "QB_DEPLOY_HORIZON": qb}
        log(f"  {s}: VERDICT {verdict} | PS_SEL vs FROZEN {ps['delta']:+.4f} "
            f"[{ps['lo']:+.4f},{ps['hi']:+.4f}]{' SEP' if ps['separated'] else ''} | "
            f"ORACLE {orc['delta']:+.4f}{' SEP' if orc['separated'] else ''} | "
            f"cells above frozen: {beats or 'NONE'} | Q-B best cell "
            f"{qb['best_grid_cell_by_event_recall']} ({time.time()-t0:.0f}s)")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---- bank the score columns for 0-GPU re-analysis --------------------------------
    np.savez_compressed(a.scores_out, y_lead3=Y_EVAL.astype(np.uint8),
                        eval_rows=EVAL.astype(np.uint8), clip_cluster=cc,
                        situations=np.array(SITS),
                        onsets=onsets, onset_sit=onset_sit, onset_cluster=onset_cl,
                        **{n: arms[n] for n in arms},
                        **{f"CELL_w{W}_L{L}": CELL[(False, W, L)] for W in WINS for L in LEADS})
    log(f"wrote {a.scores_out}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {a.out} ({time.time()-t_start:.0f}s total)")


def ridge_multi(Xtr, Ytr, Vtr, Xall, lams):
    """`sitclf.ridge_scores` for SEVERAL lambdas, sharing the one Gram matrix.

    ⚠️ This is a SPEED path for an estimator that must not drift, so it is asserted BIT-IDENTICAL
    to `stack`'s `ridge_scores` at one cell per window before it is used (C-FID-RIDGE). The
    standardisation, the +-1 targets, the unpenalised intercept and the per-situation validity mask
    are `ridge_scores`'s, verbatim; the only change is that `Am.T @ Am` — which does not depend on
    lambda — is formed once instead of five times.
    """
    A0 = np.asarray(Xtr, dtype=np.float64)
    B0 = np.asarray(Xall, dtype=np.float64)
    Y = np.asarray(Ytr)
    V = np.asarray(Vtr).astype(bool)
    mu = A0.mean(0, keepdims=True)
    sd = np.maximum(A0.std(0, keepdims=True), 1e-3)
    A = np.concatenate([(A0 - mu) / sd, np.ones((len(A0), 1))], 1)
    B = np.concatenate([(B0 - mu) / sd, np.ones((len(B0), 1))], 1)
    eye = np.eye(A.shape[1])
    eye[-1, -1] = 0.0
    out = {lam: np.zeros((len(B0), Y.shape[1]), np.float64) for lam in lams}
    for i in range(Y.shape[1]):
        m = V[:, i]
        if m.sum() < 50:
            continue
        Am = A[m]
        G = Am.T @ Am
        rhs = Am.T @ np.where(Y[m, i].astype(bool), 1.0, -1.0)
        for lam in lams:
            out[lam][:, i] = B @ np.linalg.solve(G + lam * eye, rhs)
    return {lam: v.astype(np.float32) for lam, v in out.items()}


if __name__ == "__main__":
    main()
