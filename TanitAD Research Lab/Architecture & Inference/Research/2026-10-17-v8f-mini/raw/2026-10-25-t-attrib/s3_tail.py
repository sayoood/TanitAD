"""s3_tail.py -- Q1 continued: the anatomy of the tail (EXPLORATORY; MEASURED on eval-141, M2b_s0 and M2b_s1; units SD; estimator: episode-cluster
bootstrap B=2000 for the headline shares). Which cells and which label magnitudes carry the squared error, by cell KIND, by |GT| magnitude, by
time-to-event versus geometry; and how heavy-tailed the labels are (a label-side finding if one cell kind is dominated by a few extreme GT values).
"""
import numpy as np

from tcommon import *

KIND = {}
for i, n in enumerate(CONS_NAMES):
    if n in ("turn_t_start_s", "turn_t_end_s", "lc_t_cross_s", "lon_t_reach_s", "lon_t_start_s"):
        KIND[i] = "TIME-TO-EVENT (turn/lc/reach/start)"
    elif n in ("lead_tg_s", "lead_tg_min_s"):
        KIND[i] = "TIME-GAP to lead (gap/closing speed)"
    elif n in ("lead_gap_m", "lead_gap_min_m"):
        KIND[i] = "LEAD GAP distance"
    elif n in ("turn_d_start_m", "turn_len_m", "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "lc_d_cross_m", "lon_d_reach_m",
               "stop_x_m", "stop_y_m"):
        KIND[i] = "GEOMETRY distance/position (m)"
    elif n in ("turn_dyaw_deg", "turn_exit_psi_deg", "lat_theta_deg"):
        KIND[i] = "ANGLE (deg)"
    else:
        KIND[i] = "SPEED (m/s)"


def boot_share(num_by_ep, den_by_ep, n_boot=2000, seed=0):
    r = np.random.default_rng(seed)
    n = len(num_by_ep)
    out = np.empty(n_boot)
    for b in range(n_boot):
        k = r.integers(0, n, n)
        out[b] = num_by_ep[k].sum() / max(den_by_ep[k].sum(), 1e-12)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    T = load()
    eid = T["ci"]
    ue = np.unique(eid)
    has = T["has"]
    G = T["gt_std"]
    for arm in ("M2b_s0", "M2b_s1"):
        E, P = sq(T, arm)
        E[~has], P[~has] = 0.0, False
        R = resid(T, arm)
        TOT = E.sum()
        ep_tot = np.array([E[eid == e].sum() for e in ue])
        print("\n" + "=" * 100 + f"\n{arm}: cell KINDS (share of total squared error; CI = episode-cluster bootstrap)\n" + "=" * 100)
        print(f"{'kind':40s} {'n_cells':>8s} {'cell%':>6s} {'RMSE':>6s} {'SSEshare':>8s} {'[CI95]':>16s} {'lift':>5s}")
        kinds = sorted(set(KIND.values()))
        for k in kinds:
            idx = [i for i in KIND if KIND[i] == k]
            n = P[:, idx].sum()
            es = np.array([E[eid == e][:, idx].sum() for e in ue])
            lo, hi = boot_share(es, ep_tot)
            sh = E[:, idx].sum() / TOT
            print(f"{k:40s} {n:8d} {n / P.sum():6.1%} {np.sqrt(E[:, idx].sum() / n):6.3f} {sh:8.3f} [{lo:6.3f},{hi:6.3f}] {sh / (n / P.sum()):5.2f}")

        print("\n" + "-" * 100 + f"\n{arm}: by |GT std| magnitude of the cell (all present cells)\n" + "-" * 100)
        gm = np.abs(np.where(P, G, 0.0))
        edges = [0.5, 1, 2, 3, 5]
        names = ["<0.5", "0.5-1", "1-2", "2-3", "3-5", ">=5"]
        b = np.digitize(gm, edges)
        print(f"{'|GT| bin':10s} {'n_cells':>8s} {'cell%':>6s} {'RMSE':>6s} {'SSEshare':>8s} {'bias(T-GT)':>10s} {'T/GT slope':>10s}")
        for j, nm in enumerate(names):
            m = P & (b == j)
            if m.sum() == 0:
                continue
            gg, tt_ = G[m], (G + R)[m]
            print(f"{nm:10s} {m.sum():8d} {m.sum() / P.sum():6.1%} {np.sqrt(E[m].mean()):6.3f} {E[m].sum() / TOT:8.3f} {np.mean(tt_ - gg):10.3f} "
                  f"{(np.sum(tt_ * gg) / np.sum(gg * gg)):10.3f}")
        print(f"\ncells with |GT| >= 3 SD: {int((P & (gm >= 3)).sum())} ({(P & (gm >= 3)).sum() / P.sum():.2%} of cells) carry "
              f"{E[P & (gm >= 3)].sum() / TOT:.3f} of the SSE; cells with |GT| >= 2: {(P & (gm >= 2)).sum() / P.sum():.2%} carry {E[P & (gm >= 2)].sum() / TOT:.3f}")

        print("\n" + "-" * 100 + f"\n{arm}: GT label tail per cell (std units): max, p99, share of the cell's SSE from cells with |GT|>=3; and T's range on those\n" + "-" * 100)
        print(f"{'cell':20s} {'n':>5s} {'GTmax':>7s} {'GTp99':>7s} {'GTskew':>7s} {'SSEfrom|GT|>=3':>14s} {'meanT on those':>14s} {'meanGT on those':>15s}")
        for c in np.argsort(-E.sum(0))[:10]:
            m = P[:, c]
            gc = G[m, c]
            big = m & (np.abs(np.where(P, G, 0.0))[:, c] >= 3)
            sk = float(np.mean(((gc - gc.mean()) / gc.std()) ** 3))
            tm = float((G + R)[big, c].mean()) if big.sum() else float("nan")
            gmn = float(G[big, c].mean()) if big.sum() else float("nan")
            print(f"{CONS_NAMES[c]:20s} {m.sum():5d} {gc.max():7.2f} {np.percentile(gc, 99):7.2f} {sk:7.2f} {E[big, c].sum() / E[:, c].sum():14.3f} {tm:14.2f} {gmn:15.2f}")

        print("\n" + "-" * 100 + f"\n{arm}: top-24 windows by SSE (1% of windows) -- which cells and what GT\n" + "-" * 100)
        wsse = E.sum(1)
        top = np.argsort(-wsse)[:24]
        agg = {}
        for w in top:
            c = int(np.argmax(E[w]))
            agg[CONS_NAMES[c]] = agg.get(CONS_NAMES[c], 0) + 1
        print("dominant cell of each of the 24 worst windows:", dict(sorted(agg.items(), key=lambda kv: -kv[1])))
        print("episodes of those 24 windows:", len(set(eid[top].tolist())), "distinct")
        print(f"{'win':>5s} {'ep':>3s} {'t':>3s} {'SSE':>7s} {'topcell':>16s} {'GTraw?':>8s} {'GTstd':>7s} {'Tstd':>7s} {'lead_gap':>8s} {'v_T':>5s} {'lon':>10s} {'lat':>9s}")
        for w in top[:12]:
            c = int(np.argmax(E[w]))
            print(f"{w:5d} {eid[w]:3d} {T['tt'][w]:3d} {wsse[w]:7.2f} {CONS_NAMES[c]:>16s} {'':>8s} {G[w, c]:7.2f} {(G + R)[w, c]:7.2f} {T['lead_gap'][w]:8.1f} "
                  f"{T['v_T'][w]:5.1f} {LON_NAMES.get(int(T['lon_cls'][w]), '?'):>10s} {LAT_NAMES.get(int(T['lat_cls'][w]), '?'):>9s}")

        # robustness: what does a Huber (the TRAIN loss) pooled error say, and how much SSE is above |e|>2
        H = huber(R, 1.0)
        H = np.where(P, H, 0.0)
        big_e = P & (np.abs(np.where(P, R, 0.0)) >= 2)
        print(f"\nHuber(delta=1) pooled over present cells: {H.sum() / P.sum():.4f}; cells with |error| >= 2 SD: {big_e.sum()} ({big_e.sum() / P.sum():.2%}) carry {E[big_e].sum() / TOT:.3f} of the SSE")

        print("\n" + "-" * 100 + f"\n{arm}: lead-present windows -- lead-time-gap cells vs label construction\n" + "-" * 100)
        lead = T["lead_valid"] > 0
        for c in (19, 21):
            gc = G[lead & P[:, c], c]
            print(f"{CONS_NAMES[c]}: GT std percentiles 50/90/99/max = {np.percentile(gc, 50):.2f}/{np.percentile(gc, 90):.2f}/{np.percentile(gc, 99):.2f}/{gc.max():.2f}; "
                  f"raw max {(gc.max() * T['sd'][c] + T['mu'][c]):.1f} s; share of cell SSE from the worst 1% of its windows: "
                  f"{np.sort(E[lead & P[:, c], c])[::-1][:max(int(0.01 * (lead & P[:, c]).sum()), 1)].sum() / E[:, c].sum():.3f}")


if __name__ == "__main__":
    main()
