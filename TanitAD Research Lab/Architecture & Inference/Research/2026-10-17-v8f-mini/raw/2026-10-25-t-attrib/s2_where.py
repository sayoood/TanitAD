"""s2_where.py -- Q1: where does T's constraint error live? (EXPLORATORY; MEASURED on eval-141, T0-class diagnostic of T's own code:
the error is T's std-unit constraint block against the GT block, no manipulation of O; estimator for every share = episode-cluster bootstrap B=2000
over the 141 episodes (it answers 'another draw of EPISODES' only, H-ESTIM-SEED-1)).

Units (verified in s1_build_table.out): SD units of the standardiser mu/sd (cons_stats, v8f_mini.py:773-781), the units of T's head `cons_std`
(v8f_mini.py:573) and of the Huber train loss (v8f_mini.py:675-683). Error = T cons_std - GT std, over PRESENT cells only (finite GT, window has a v9 row).
Absent cells carry no GT and therefore no error (R63: T's absent-slot values are O's input, not error).
"""
import sys

import numpy as np

from tcommon import *

B = 2000
rng = np.random.default_rng(0)


def head(t):
    print("\n" + "=" * 110 + "\n" + t + "\n" + "=" * 110)


def boot_share(num_by_ep, den_by_ep, n_boot=B, seed=0):
    r = np.random.default_rng(seed)
    n = len(num_by_ep)
    out = np.empty(n_boot)
    for b in range(n_boot):
        k = r.integers(0, n, n)
        out[b] = num_by_ep[k].sum() / max(den_by_ep[k].sum(), 1e-12)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def run(T, arm):
    head(f"ARM {arm}")
    E, P = sq(T, arm)
    has = T["has"]
    E[~has], P[~has] = 0.0, False
    eid = T["ci"]
    ue = np.unique(eid)
    # ---- 0. reproduce the banked numbers (verification of units + reduction)
    gr = {g: float(np.sqrt(E[:, i][P[:, i]].sum() / P[:, i].sum())) if False else float(np.sqrt(E[:, i].sum() / P[:, i].sum())) for g, i in GROUPS.items()}
    npc = P.sum(1)
    wmse = np.where(npc > 0, E.sum(1) / np.maximum(npc, 1), np.nan)
    pres_w = np.isfinite(wmse)
    print(f"[verify] slot-pooled RMSE by group (SD): " + ", ".join(f"{g} {v:.4f}" for g, v in gr.items()))
    print(f"[verify] present-slot RMSE, window-mean of per-window mean-sq (WC reader definition): {np.sqrt(np.nanmean(wmse)):.4f} (n_windows {pres_w.sum()})")
    print(f"[verify] cell-pooled RMSE over all present cells: {np.sqrt(E.sum() / P.sum()):.4f} (n_cells {int(P.sum())})")
    TOT = E.sum()
    # ---- 1. per group / per cell
    head(f"{arm} -- 1. error by group and by cell (share of total squared error, SD units)")
    print(f"{'group/cell':26s} {'n_present':>9s} {'win%':>6s} {'MSE':>7s} {'RMSE':>6s} {'SSEshare':>8s} {'[CI95]':>16s} {'R2_vs_mean':>10s}")
    ep_tot = np.array([E[eid == e].sum() for e in ue])
    for g, idx in GROUPS.items():
        eg = np.array([E[eid == e][:, idx].sum() for e in ue])
        lo, hi = boot_share(eg, ep_tot)
        n = P[:, idx].sum()
        print(f"{g:26s} {n:9d} {'':>6s} {E[:, idx].sum() / n:7.3f} {np.sqrt(E[:, idx].sum() / n):6.3f} {E[:, idx].sum() / TOT:8.3f} [{lo:6.3f},{hi:6.3f}]")
        for c in idx:
            m = P[:, c]
            if m.sum() == 0:
                continue
            g0 = T["gt_std"][m, c]
            var0 = np.mean(g0 ** 2)                                     # the constant-mu (train mean) predictor's MSE on the present cells
            ec = np.array([E[eid == e][:, c].sum() for e in ue])
            lo, hi = boot_share(ec, ep_tot)
            print(f"   {CONS_NAMES[c]:23s} {m.sum():9d} {m.mean():6.1%} {E[:, c].sum() / m.sum():7.3f} {np.sqrt(E[:, c].sum() / m.sum()):6.3f} "
                  f"{E[:, c].sum() / TOT:8.3f} [{lo:6.3f},{hi:6.3f}] {1 - (E[:, c].sum() / m.sum()) / var0:10.3f}")

    # ---- 2. strata
    nav = T["nav_tok"]
    strata = {
        "GT lat class (join lat_cls at T tick)": np.array([LAT_NAMES.get(int(x), str(x)) for x in T["lat_cls"]]),
        "GT lon class (join lon_cls at T tick)": np.array([LON_NAMES.get(int(x), str(x)) for x in T["lon_cls"]]),
        "ego speed band at T tick (m/s)": band(T["v_T"], [1, 5, 10, 15], ["<1", "1-5", "5-10", "10-15", ">=15"]),
        "GT path heading change over next 6 s (|deg|)": band(np.abs(T["dpsi6"]), [3, 10, 30], ["<3 straight", "3-10", "10-30", ">=30 turn"]),
        "lead agent at T tick": np.where(T["lead_valid"] > 0, "lead present", "no lead"),
        "nav token at NOW (argmax of nav[:3])": np.array([f"nav{int(x)}" for x in nav]),
        "turn constraints present (cell 1 finite)": np.where(np.isfinite(T["gt_std"][:, 1]), "turn label", "no turn label"),
    }
    ep_g = {g: None for g in GROUPS}
    for sname, lab in strata.items():
        head(f"{arm} -- 2. stratum: {sname}")
        print(f"{'stratum':16s} {'n_win':>6s} {'win%':>6s} {'n_cells':>8s} {'cell%':>6s} {'RMSE':>6s} {'SSEshare':>8s} {'[CI95]':>16s} {'lift':>5s} | "
              f"{'latc':>6s} {'lons':>6s} {'stpl':>6s}  (share of EACH GROUP's own SSE)")
        for v in sorted(set(lab.tolist())):
            m = (lab == v) & has
            if m.sum() == 0:
                continue
            e = E[m]
            nc = P[m].sum()
            es = np.array([E[(eid == k) & m].sum() for k in ue])
            lo, hi = boot_share(es, ep_tot)
            sh = e.sum() / TOT
            gs = [E[m][:, idx].sum() / E[:, idx].sum() for g, idx in GROUPS.items()]
            print(f"{str(v):16s} {m.sum():6d} {m.sum() / has.sum():6.1%} {int(nc):8d} {nc / P.sum():6.1%} "
                  f"{np.sqrt(e.sum() / max(nc, 1)):6.3f} {sh:8.3f} [{lo:6.3f},{hi:6.3f}] {sh / max(nc / P.sum(), 1e-9):5.2f} | "
                  f"{gs[0]:6.3f} {gs[1]:6.3f} {gs[2]:6.3f}")
    # ---- 3. concentration
    head(f"{arm} -- 3. concentration of the squared error")
    wsse = E.sum(1)
    order = np.argsort(-wsse)
    nwin = int(has.sum())
    cum = np.cumsum(wsse[order]) / TOT
    cellcum = np.cumsum(P.sum(1)[order]) / P.sum()
    for q in (0.01, 0.05, 0.10, 0.25, 0.50):
        k = int(round(q * nwin))
        print(f"worst {q:5.0%} of windows by window SSE ({k:4d} windows): carry {cum[k - 1]:.3f} of total SSE, holding {cellcum[k - 1]:.3f} of the present cells")
    # per-cell-normalised: windows by mean per-cell squared error (removes the 'more present cells = more SSE' composition effect)
    ok = (npc > 0) & has
    mord = np.argsort(-np.where(ok, wmse, -1))
    mcum = np.cumsum(wsse[mord]) / TOT
    mcell = np.cumsum(npc[mord]) / P.sum()
    for q in (0.10, 0.25):
        k = int(round(q * ok.sum()))
        print(f"worst {q:5.0%} of windows by window MEAN-per-cell sq. error ({k:4d} windows): carry {mcum[k - 1]:.3f} of total SSE, holding {mcell[k - 1]:.3f} of the cells")
    # episodes
    esse = ep_tot
    eo = np.argsort(-esse)
    ecum = np.cumsum(esse[eo]) / TOT
    ecells = np.array([P[eid == e].sum() for e in ue])
    ecc = np.cumsum(ecells[eo]) / P.sum()
    for q in (0.10, 0.25):
        k = int(round(q * len(ue)))
        print(f"worst {q:5.0%} of episodes by episode SSE ({k:3d} episodes): carry {ecum[k - 1]:.3f} of total SSE, holding {ecc[k - 1]:.3f} of the cells")
    emean = esse / np.maximum(ecells, 1)
    eo2 = np.argsort(-emean)
    ecum2 = np.cumsum(esse[eo2]) / TOT
    ecc2 = np.cumsum(ecells[eo2]) / P.sum()
    for q in (0.10, 0.25):
        k = int(round(q * len(ue)))
        print(f"worst {q:5.0%} of episodes by episode MEAN-per-cell sq. error ({k:3d} episodes): carry {ecum2[k - 1]:.3f} of total SSE, holding {ecc2[k - 1]:.3f} of the cells")
    # per-cell concentration: for the three biggest cells, the worst 10 % of that cell's present windows
    cell_sse = E.sum(0)
    for c in np.argsort(-cell_sse)[:6]:
        m = np.where(P[:, c])[0]
        e = np.sort(E[m, c])[::-1]
        k = max(int(round(0.10 * len(e))), 1)
        print(f"   cell {CONS_NAMES[c]:18s}: share of total SSE {cell_sse[c] / TOT:.3f}; worst 10% of its {len(e)} present windows carry {e[:k].sum() / e.sum():.3f} of the cell's SSE")
    return dict(E=E, P=P, wsse=wsse)


if __name__ == "__main__":
    T = load()
    for arm in ("M2b_s0", "M2b_s1"):
        run(T, arm)
