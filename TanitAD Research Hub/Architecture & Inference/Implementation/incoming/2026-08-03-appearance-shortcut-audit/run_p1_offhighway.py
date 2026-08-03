"""D-APPEAR P1 — does the APPEARANCE SHORTCUT survive OFF-HIGHWAY, on PhysicalAI-AV?

Pre-registration: ``Project Steering/PREREG_APPEARANCE_SHORTCUT.md`` (outcomes S / C / P / VOID
and every threshold fixed BEFORE this ran).

THE QUESTION
    On comma2k19 HIGHWAY a single 32x32 grayscale STILL FRAME reads ``speed`` at R2 +0.6642
    against the 18,432-feature 800 ms learned latent's +0.7145 -- 93 % -- while every pure
    temporal-difference basis sits at the null. Highway is the easiest possible case for that:
    near-constant road furniture, a narrow speed distribution, little manoeuvre variety.

    PhysicalAI-AV is not highway-dominated. If the still frame still keeps ~93 % of the read
    there, the shortcut is a property of OUR PIPELINE. If the ratio collapses, it is a property
    of comma2k19 and the programme-scale claim must be withdrawn.

WHAT MAKES THE CONTRAST ADMISSIBLE
    ⭐ ENCODER-MATCHED. Both corpora are probed with the SAME frozen encoder,
    ``v1_speedjerk_ckpt.pt`` step 29999 (comma: ``idm_derived_accel_latents.pt``'s
    ``encoder`` field; PhysicalAI: ``sitclf_b4_substrate.meta.json``'s ``trunk.ckpt``).
    The ridge recipe, the split rule, the alpha path, the skill gate, the tie-break and the
    bootstrap are the comma panel's, unchanged. The corpus is the only thing that moves.

usage:
  OMP_NUM_THREADS=6 python run_p1_offhighway.py --n-boot 2000 --out results_p1_physicalai.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "taniteval"))

SUBSTRATE = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_pai_substrate.pt")
HORIZONS = (5, 10, 15, 20)
SCALARS = ("speed", "yaw_rate", "steer", "long_accel")
SPEED_J, ACCEL_J = 0, 3
SEL_TOL, MIN_SKILL = 0.005, 0.01
MEAN_KEY = (None, float("inf"))
RBF_GAMMA_MULTS = (0.25, 1.0, 4.0)
#: pre-registered stratification (m/s). Chosen to match the corpus stream's tactical-lossy bins.
SPEED_BINS = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0), (15.0, 1e9))
MIN_STRATUM_N, MIN_STRATUM_EP = 100, 5           # below this -> UNPOWERED, not evidence
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# --------------------------------------------------------------------------- #
# arms                                                                         #
# --------------------------------------------------------------------------- #
#: (arm, substrate key, feature basis, kernel, frame-order-shuffle?)
ARMS: tuple[tuple[str, str, str, str, bool], ...] = (
    # ⭐ the two arms the PRIMARY RATIO is built from
    ("v1_window",             "Z",     "window", "linear", False),
    ("pix32_centre_rbf",      "pix32", "centre", "rbf",    False),
    # the latent family
    ("v1_centre",             "Z",     "centre", "linear", False),
    ("v1_tdiff",              "Z",     "tdiff",  "linear", False),
    # the single-instant family (linear + the true one-RGB-frame rung)
    ("pix32_centre",          "pix32", "centre", "linear", False),
    ("stk32_centre",          "stk32", "centre", "linear", False),
    ("stk32_centre_rbf",      "stk32", "centre", "rbf",    False),
    # motion-only arms -- the 1.75x comparison
    ("pix8_tdiff_rbf",        "pix8",  "tdiff",  "rbf",    False),
    ("mot8_window_rbf",       "mot8",  "window", "rbf",    False),
    ("mot16_window_rbf",      "mot16", "window", "rbf",    False),
    ("mot8_centre_rbf",       "mot8",  "centre", "rbf",    False),
    ("pix1_window_rbf",       "pix1",  "window", "rbf",    False),
    # ⭐ P1b -- the FRAME-ORDER-SHUFFLE control. Same frames, same marginal, order destroyed.
    ("v1_window_shufframes",  "Z",     "window", "linear", True),
    ("pix32_window",          "pix32", "window", "linear", False),
    ("pix32_window_shufframes", "pix32", "window", "linear", True),
)


# --------------------------------------------------------------------------- #
def split_episodes(eps, hold_every: int = 3):
    tr = [e for i, e in enumerate(eps) if i % hold_every != 0]
    ho = [e for i, e in enumerate(eps) if i % hold_every == 0]
    return tr, ho


def frame_order_shuffle(X: torch.Tensor, seed: int) -> torch.Tensor:
    """[N, W, D] -> the SAME frames in a random PER-ROW order.

    The control for "does the read use temporal ORDER at all". A per-row permutation is
    the strong version: it destroys order globally AND removes any systematic position
    effect, while leaving the multiset of frames in each window untouched.
    """
    g = np.random.default_rng(seed)
    n, w, _ = X.shape
    perm = np.argsort(g.random((n, w)), axis=1)
    return X[torch.arange(n)[:, None], torch.from_numpy(perm)]


def stack_sub(eps, key, *, shuffle_seed=None, order_seed=None):
    """-> X [N, W, D], Y [N, 4+2H], eid [N], centre-speed [N], manoeuvre [N]."""
    X = torch.cat([e[key] for e in eps]).float()
    if X.ndim == 2:
        X = X[:, None, :]
    S = torch.cat([e["S"] for e in eps]).float()
    Tj = torch.cat([e["T"] for e in eps]).float()
    man = torch.cat([e["man"] for e in eps]).numpy()
    eid = np.concatenate([np.full(e["n"], e["name"]) for e in eps])
    if order_seed is not None:
        X = frame_order_shuffle(X, order_seed)
    if shuffle_seed is not None:                     # the NEGATIVE control
        g = np.random.default_rng(shuffle_seed)
        X = X[torch.from_numpy(g.permutation(len(X)))]
    Y = torch.cat([S, Tj.reshape(len(Tj), -1)], 1)
    return X, Y, eid, S[:, SPEED_J].numpy(), man


def unpack(Y):
    Y = np.asarray(Y, dtype=np.float64)
    return Y[:, :4], Y[:, 4:].reshape(len(Y), len(HORIZONS), 2)


def ridge_arm(AP, Xfit, Yfit, Xsel, Xfull, Yfull, Xho, *, feat, kernel="linear",
              gamma_mults=(None,), device="cpu"):
    Xf = AP.window_features(Xfit, feat)
    Xs = AP.window_features(Xsel, feat)
    XF = AP.window_features(Xfull, feat)
    Xh = AP.window_features(Xho, feat)
    Xf, Xs = AP.standardize(Xf, Xs)
    XF, Xh = AP.standardize(XF, Xh)
    ymu = Yfull.mean(0, keepdim=True)
    ysd = Yfull.std(0, keepdim=True).clamp_min(1e-6)
    kwbase = dict(kernel=kernel, matmul_device=device, matmul_dtype=torch.float32)
    alphas = AP.DualRidge.alpha_grid(2, -4, 10)
    sel_preds, ho_preds, tr_preds, gammas = {}, {}, {}, {}
    for gm in gamma_mults:
        gamma = None
        if kernel == "rbf":
            g = torch.Generator().manual_seed(20260803)
            idx = torch.randperm(len(Xf), generator=g)[:1000]
            d2 = torch.cdist(Xf[idx].double(), Xf[idx].double()).pow(2)
            gamma = gm / max(float(d2[d2 > 0].median()), 1e-9)
        gammas[gm] = gamma
        kw = dict(gamma=gamma, **kwbase)
        inner = AP.DualRidge(Xf, ((Yfit - ymu) / ysd).double(), **kw)
        for al in alphas:
            sel_preds[(gm, al)] = inner.predict(Xs, al)
        del inner
        full = AP.DualRidge(XF, ((Yfull - ymu) / ysd).double(), **kw)
        for al in alphas:
            ho_preds[(gm, al)] = full.predict(Xh, al)
            tr_preds[(gm, al)] = full.predict(XF, al)
        del full
    k0 = next(iter(sel_preds))
    for d in (sel_preds, ho_preds, tr_preds):
        d[MEAN_KEY] = torch.zeros_like(d[k0])
    return sel_preds, ho_preds, tr_preds, list(sel_preds), {
        "feature": feat, "kernel": kernel,
        "gammas": {str(k): v for k, v in gammas.items()},
        "n_features": int(Xf.shape[1]), "ymu": ymu, "ysd": ysd}


def select_hparam(AP, sel_preds, Ysel, keys, meta):
    ymu, ysd = meta["ymu"], meta["ysd"]
    Yt = np.asarray(Ysel, dtype=np.float64)
    k_shrunk = max(keys, key=lambda k: k[1])
    chosen = []
    for j in range(Yt.shape[1]):
        scored = [(k, AP.r2_score((sel_preds[k][:, j] * ysd[0, j] + ymu[0, j]).numpy(),
                                  Yt[:, j])) for k in keys]
        smap = dict(scored)
        best = max(smap.values())
        if best < MIN_SKILL:
            k_sel, rule = k_shrunk, "skill_gate_to_train_mean"
        else:
            ok = [(k, s) for k, s in scored if s >= best - SEL_TOL]
            k_sel, rule = max(ok, key=lambda t: t[0][1])[0], "one_se_tiebreak"
        chosen.append((k_sel, float(smap[k_sel]), float(best), rule))
    return chosen


def assemble(preds, chosen, meta):
    ymu, ysd = meta["ymu"], meta["ysd"]
    return np.stack([(preds[c[0]][:, j] * ysd[0, j] + ymu[0, j]).numpy()
                     for j, c in enumerate(chosen)], 1)


# --------------------------------------------------------------------------- #
def score_arm(AP, TCI, APCI, FF, pred, Yho, eid, n_boot):
    gs, gt = unpack(Yho)
    ps, pt = pred[:, :4], pred[:, 4:].reshape(len(pred), len(HORIZONS), 2)
    out = {"r2": {}, "mae": {}}
    for j, ch in enumerate(SCALARS):
        out["r2"][ch] = APCI.stat_episode_cluster_bootstrap(
            (lambda p, g: (lambda sel: AP.r2_score(p[sel], g[sel])))(ps[:, j], gs[:, j]),
            eid, n_boot=n_boot, name=f"r2_{ch}")
        out["mae"][ch] = round(float(np.abs(ps[:, j] - gs[:, j]).mean()), 5)
    out["ade_2s"] = TCI.episode_cluster_bootstrap(
        np.linalg.norm(pt - gt, axis=-1).mean(1), eid, n_boot=n_boot)
    out["four_families"] = FF.all_families(pt, gt, FF.IDM_DT_S,
                                           pred_scalars=ps, gt_scalars=gs)
    return out


def paired_delta(AP, APCI, pred, ctrl, Yho, eid, n_boot):
    gs, _ = unpack(Yho)
    out = {}
    for j, ch in enumerate(SCALARS):
        a, b, g = pred[:, j], ctrl[:, j], gs[:, j]
        d = APCI.paired_stat_episode_cluster_bootstrap(
            (lambda p, gg: (lambda sel: AP.r2_score(p[sel], gg[sel])))(a, g),
            (lambda p, gg: (lambda sel: AP.r2_score(p[sel], gg[sel])))(b, g),
            eid, n_boot=n_boot, name=f"delta_r2_{ch}")
        if np.allclose(a, b, rtol=0, atol=1e-12):
            d["separated"] = False
            d["degenerate_identical_predictions"] = True
        out[ch] = d
    return out


# --------------------------------------------------------------------------- #
# stratification -- the pre-registered part that a pooled number would hide     #
# --------------------------------------------------------------------------- #
def strata_masks(v_centre, man, man_classes):
    out = {}
    for lo, hi in SPEED_BINS:
        out[f"speed_{lo:g}_{hi:g}" if hi < 1e8 else f"speed_{lo:g}_inf"] = \
            (v_centre >= lo) & (v_centre < hi)
    for i, name in enumerate(man_classes):
        out[f"man_{name}"] = (man == i)
    out["man_invalid"] = (man < 0)
    return out


def strat_scores(AP, APCI, preds, Yho, eid, masks, arms, n_boot):
    gs, _ = unpack(Yho)
    g = gs[:, SPEED_J]
    res = {}
    for sname, m in masks.items():
        n = int(m.sum())
        n_ep = int(len(np.unique(eid[m]))) if n else 0
        rec = {"n": n, "n_episodes": n_ep,
               "powered": bool(n >= MIN_STRATUM_N and n_ep >= MIN_STRATUM_EP),
               "gt_speed_mean": round(float(g[m].mean()), 4) if n else None,
               "gt_speed_std": round(float(g[m].std()), 4) if n else None,
               "arms": {}}
        for a in arms:
            p = preds[a][:, SPEED_J]
            if n < 3:
                rec["arms"][a] = {"r2_speed": None, "corr": None, "mae": None}
                continue
            pm, gm = p[m], g[m]
            corr = (float(np.corrcoef(pm, gm)[0, 1])
                    if pm.std() > 0 and gm.std() > 0 else None)
            rec["arms"][a] = {
                "r2_speed": round(AP.r2_score(pm, gm), 5),
                "corr": None if corr is None else round(corr, 5),
                "mae": round(float(np.abs(pm - gm).mean()), 5)}
        if rec["powered"] and len(arms) >= 2:
            a0, a1 = arms[0], arms[1]
            rec["paired_delta_r2_speed"] = APCI.paired_stat_episode_cluster_bootstrap(
                (lambda p, gg: (lambda sel: AP.r2_score(p[sel], gg[sel])))(
                    preds[a1][m, SPEED_J], g[m]),
                (lambda p, gg: (lambda sel: AP.r2_score(p[sel], gg[sel])))(
                    preds[a0][m, SPEED_J], g[m]),
                eid[m], n_boot=min(n_boot, 1000), name=f"{a1}_minus_{a0}")
        res[sname] = rec
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default=str(SUBSTRATE))
    ap.add_argument("--out", default="results_p1_physicalai.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arms", default="", help="comma-separated subset")
    a = ap.parse_args()

    import tanitad.eval.accel_probe as AP
    import tanitad.eval.ap_ci as APCI
    import tanitad.eval.idm_families as FF
    import taniteval.ci as TCI                 # the DECISION-GRADE interval estimator
    from tanitad.refs.refb import MANEUVER_CLASSES

    sub = torch.load(a.substrate, map_location="cpu", weights_only=False)
    eps = sub["episodes"]
    tr, ho = split_episodes(eps)
    log(f"{len(eps)} episodes -> {len(tr)} train / {len(ho)} held out; "
        f"{sum(e['n'] for e in tr)} / {sum(e['n'] for e in ho)} windows")

    inner_tr = [e for i, e in enumerate(tr) if i % 3 != 0]
    inner_sel = [e for i, e in enumerate(tr) if i % 3 == 0]

    arm_list = [x for x in ARMS if not a.arms or x[0] in a.arms.split(",")]
    results = {"meta": {
        "prereg": "Project Steering/PREREG_APPEARANCE_SHORTCUT.md",
        "corpus": sub["corpus"], "encoder": sub["encoder"],
        "encoder_step": sub["encoder_step"], "k": sub["k"], "stride": sub["stride"],
        "n_episodes": len(eps), "n_train_eps": len(tr), "n_holdout_eps": len(ho),
        "n_train_windows": sum(e["n"] for e in tr),
        "n_holdout_windows": sum(e["n"] for e in ho),
        "alignment_checks": sub["alignment_checks"],
        "rig_counts": {r: sum(1 for e in eps if e["rig"] == r)
                       for r in sorted({e["rig"] for e in eps})},
        "n_boot": a.n_boot, "device": a.device,
        "estimator": "paired episode-cluster bootstrap (tanitad/eval/ap_ci.py); "
                     "NEVER overlapping_holdout_se",
    }, "arms": {}, "strata": {}}

    _, Yho_ref, eid_ho, v_ho, man_ho = stack_sub(ho, "pix1")
    _, Ytr_ref, _, _, _ = stack_sub(tr, "pix1")
    gs_ho, gt_ho = unpack(Yho_ref)

    # the NULL arm -- reproduce the train-mean floor on THIS split
    null_pred = np.repeat(np.asarray(Ytr_ref.mean(0), dtype=np.float64)[None, :],
                          len(Yho_ref), 0)
    results["arms"]["NULL_train_mean"] = {
        "n_features": 0, "kernel": "none", "feature": "train_mean",
        **score_arm(AP, TCI, APCI, FF, null_pred, Yho_ref, eid_ho, a.n_boot)}
    log(f"NULL_train_mean speed R2 "
        f"{results['arms']['NULL_train_mean']['r2']['speed']['point']:+.4f}")

    preds_by_arm = {}
    for name, key, feat, kernel, shuf_order in arm_list:
        t_arm = time.time()
        order_seed = 20260803 if shuf_order else None
        Xfit, Yfit, _, _, _ = stack_sub(inner_tr, key, order_seed=order_seed)
        Xsel, Ysel, _, _, _ = stack_sub(inner_sel, key, order_seed=order_seed)
        Xful, Yful, _, _, _ = stack_sub(tr, key, order_seed=order_seed)
        Xho, Yho, eidh, _, _ = stack_sub(ho, key, order_seed=order_seed)
        assert np.array_equal(eidh, eid_ho)
        gms = RBF_GAMMA_MULTS if kernel == "rbf" else (None,)
        sp, hp, _, keys, meta = ridge_arm(
            AP, Xfit, Yfit, Xsel, Xful, Yful, Xho,
            feat=feat, kernel=kernel, gamma_mults=gms, device=a.device)
        chosen = select_hparam(AP, sp, Ysel, keys, meta)
        pred = assemble(hp, chosen, meta)
        # its OWN shuffled control -- same recipe, substrate<->target link destroyed
        Xfit_s, _, _, _, _ = stack_sub(inner_tr, key, shuffle_seed=11, order_seed=order_seed)
        Xsel_s, _, _, _, _ = stack_sub(inner_sel, key, shuffle_seed=12, order_seed=order_seed)
        Xful_s, _, _, _, _ = stack_sub(tr, key, shuffle_seed=13, order_seed=order_seed)
        Xho_s, _, _, _, _ = stack_sub(ho, key, shuffle_seed=14, order_seed=order_seed)
        sp2, hp2, _, keys2, meta2 = ridge_arm(
            AP, Xfit_s, Yfit, Xsel_s, Xful_s, Yful, Xho_s,
            feat=feat, kernel=kernel, gamma_mults=gms, device=a.device)
        ctrl = assemble(hp2, select_hparam(AP, sp2, Ysel, keys2, meta2), meta2)

        rec = {"n_features": meta["n_features"], "kernel": kernel, "feature": feat,
               "substrate": key, "frame_order_shuffled": shuf_order,
               "selected": [{"channel": (SCALARS + tuple(f"traj{i}" for i in range(8)))[j],
                             "gamma_mult": c[0][0],
                             "alpha": ("inf(exact_train_mean)" if np.isinf(c[0][1])
                                       else c[0][1]),
                             "sel_r2": round(c[1], 5), "rule": c[3]}
                            for j, c in enumerate(chosen[:4])],
               **score_arm(AP, TCI, APCI, FF, pred, Yho, eidh, a.n_boot)}
        rec["delta_vs_shuffled"] = paired_delta(AP, APCI, pred, ctrl, Yho, eidh, a.n_boot)
        rec["shuffled_control_r2_speed"] = round(
            AP.r2_score(ctrl[:, SPEED_J], gs_ho[:, SPEED_J]), 5)
        results["arms"][name] = rec
        preds_by_arm[name] = pred
        d = rec["delta_vs_shuffled"]["speed"]
        log(f"{name:26s} feat={meta['n_features']:6d} speed R2 "
            f"{rec['r2']['speed']['point']:+.4f}  d_vs_shuf {d['delta']:+.4f} "
            f"[{d['lo']:+.4f},{d['hi']:+.4f}] {'SEP' if d['separated'] else '---'}  "
            f"({time.time()-t_arm:.0f}s)")
        Path(a.out).write_text(json.dumps(results, indent=1, default=str))

    # ------------------------------------------------------------------ #
    # THE PRIMARY STATISTIC, exactly as pre-registered                     #
    # ------------------------------------------------------------------ #
    if "v1_window" in results["arms"] and "pix32_centre_rbf" in results["arms"]:
        num = results["arms"]["pix32_centre_rbf"]["r2"]["speed"]["point"]
        den = results["arms"]["v1_window"]["r2"]["speed"]["point"]
        ratio = num / den if den > 0 else float("nan")
        sep_num = results["arms"]["pix32_centre_rbf"]["delta_vs_shuffled"]["speed"]["separated"]
        sep_den = results["arms"]["v1_window"]["delta_vs_shuffled"]["speed"]["separated"]
        if not sep_den:
            outcome = "VOID (v1_window does not separate on speed)"
        elif ratio >= 0.70 and sep_num:
            outcome = "S (SHORTCUT SURVIVES)"
        elif ratio <= 0.40:
            outcome = "C (CORPUS-SPECIFIC -- withdraw the programme-scale claim)"
        else:
            outcome = "P (PARTIAL)"
        # the paired interval on the DIFFERENCE of the two arms (not a ratio interval)
        paired = APCI.paired_stat_episode_cluster_bootstrap(
            (lambda p, gg: (lambda sel: AP.r2_score(p[sel], gg[sel])))(
                preds_by_arm["pix32_centre_rbf"][:, SPEED_J], gs_ho[:, SPEED_J]),
            (lambda p, gg: (lambda sel: AP.r2_score(p[sel], gg[sel])))(
                preds_by_arm["v1_window"][:, SPEED_J], gs_ho[:, SPEED_J]),
            eid_ho, n_boot=a.n_boot, name="stillframe_minus_latentwindow")
        # ⭐ and the RATIO's own episode-cluster interval, resampled jointly
        uniq = np.unique(eid_ho)
        idx_by = {u: np.flatnonzero(eid_ho == u) for u in uniq}
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(a.n_boot):
            draw = np.concatenate([idx_by[u] for u in rng.choice(uniq, len(uniq), True)])
            d_ = AP.r2_score(preds_by_arm["v1_window"][draw, SPEED_J], gs_ho[draw, SPEED_J])
            n_ = AP.r2_score(preds_by_arm["pix32_centre_rbf"][draw, SPEED_J],
                             gs_ho[draw, SPEED_J])
            boots.append(n_ / d_ if d_ > 0 else np.nan)
        boots = np.asarray(boots, float)
        bb = boots[np.isfinite(boots)]
        results["primary"] = {
            "definition": "R2_speed(pix32_centre_rbf) / R2_speed(v1_window), same "
                          "held-out windows, encoder-matched",
            "numerator_still_frame": round(num, 5),
            "denominator_latent_window": round(den, 5),
            "RATIO": round(ratio, 5),
            "ratio_ci95_episode_cluster_bootstrap": [round(float(np.percentile(bb, 2.5)), 5),
                                                     round(float(np.percentile(bb, 97.5)), 5)],
            "ratio_n_finite_draws": int(bb.size),
            "comma2k19_reference_ratio": round(0.6642 / 0.7145, 5),
            "paired_delta_still_minus_latent": paired,
            "still_frame_separates": bool(sep_num),
            "latent_window_separates": bool(sep_den),
            "PREREG_OUTCOME": outcome}
        log(f"⭐ PRIMARY RATIO = {ratio:.4f} (comma reference 0.9296) -> {outcome}")

    # ------------------------------------------------------------------ #
    scored = [x for x in ("v1_window", "pix32_centre_rbf", "mot8_window_rbf")
              if x in preds_by_arm]
    if scored:
        results["strata"] = strat_scores(
            AP, APCI, preds_by_arm, Yho_ref, eid_ho,
            strata_masks(v_ho, man_ho, MANEUVER_CLASSES), scored, a.n_boot)
    results["meta"]["maneuver_classes"] = list(MANEUVER_CLASSES)
    Path(a.out).write_text(json.dumps(results, indent=1, default=str))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
