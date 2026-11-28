"""E-EXP-1 — does HAD's axis-factorised local expansion open the LONGITUDINAL axis?

Pre-registration: ../PREREG_E_EXP1.md  (blob 000ed1cdc45da59c7b4ca406f921ba18c024ce4e)
0 GPU. Reads banked fan dumps only. Estimator: taniteval paired episode-cluster bootstrap.

Operator is HAD's published Structure-Preserved Trajectory Expansion (arXiv 2604.03581):
polar radial scale lambda in {0.92,0.96,1.00,1.04,1.08} x angular offset delta in {-6,-3,0,3,6} deg.
"""
import sys, os, json, argparse
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402

LAMBDAS = np.array([0.92, 0.96, 1.00, 1.04, 1.08], dtype=np.float64)
DELTAS_DEG = np.array([-6.0, -3.0, 0.0, 3.0, 6.0], dtype=np.float64)
N_BOOT, SEED = 2000, 0


def ade(traj, gt):
    """traj (...,T,2), gt (W,T,2) -> mean-over-T L2, broadcast on leading dims."""
    return np.linalg.norm(traj - gt, axis=-1).mean(axis=-1)


def rotate(f, deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    x, y = f[..., 0], f[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


def run(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    fan = d["fan"].double().numpy()            # (W, N, T, 2)
    gt = d["gt"].double().numpy()              # (W, T, 2)
    sel = d["sel"].numpy()                     # (W,)
    eid = list(d["eid"])
    v0 = d["v0"].double().numpy()
    W, N, T, _ = fan.shape
    fails = []

    # ---- INSTRUMENT-FAIL branch (prereg section 7) ---------------------------
    if W != 881 or len(set(eid)) != 40:
        fails.append(f"window/episode count {W}/{len(set(eid))} != 881/40")
    # frame convention: ego-forward must be axis 0, positive, and correlate with v0
    fwd_pos = float((gt[:, -1, 0] > 0).mean())
    corr = float(np.corrcoef(gt[:, -1, 0], v0)[0, 1])
    if fwd_pos < 0.90 or corr < 0.50:
        fails.append(f"frame convention: frac(gt_x>0)={fwd_pos:.4f}, corr(gt_x,v0)={corr:.4f}")
    # identity element must reproduce the fan bit-identically
    ident = rotate(fan * 1.00, 0.0)
    if not np.array_equal(ident, fan):
        fails.append("identity (lambda=1, delta=0) is not bit-identical to the fan")

    ade_cand = ade(fan, gt[:, None])                      # (W, N)
    ade_sel = ade_cand[np.arange(W), sel]                 # (W,)
    ade_oracle_fan = ade_cand.min(axis=1)                 # (W,)

    # ---- arms ---------------------------------------------------------------
    best_L = np.full(W, np.inf)
    best_A = np.full(W, np.inf)
    best_LA = np.full(W, np.inf)
    for lam in LAMBDAS:
        for dg in DELTAS_DEG:
            a = ade(rotate(fan * lam, dg), gt[:, None]).min(axis=1)
            best_LA = np.minimum(best_LA, a)
            if dg == 0.0:
                best_L = np.minimum(best_L, a)
            if lam == 1.00:
                best_A = np.minimum(best_A, a)

    # ---- deployable-shaped ceiling: refine only the SELECTED trajectory ------
    fsel = fan[np.arange(W), sel]                         # (W, T, 2)
    sel_L = np.stack([ade(fsel * lam, gt) for lam in LAMBDAS]).min(axis=0)
    sel_A = np.stack([ade(rotate(fsel, dg), gt) for dg in DELTAS_DEG]).min(axis=0)

    # ---- L-global: ONE lambda, fitted LEAVE-ONE-EPISODE-OUT ------------------
    grid = np.arange(0.80, 1.2001, 0.001)
    err = np.stack([ade(fsel * g, gt) for g in grid])     # (G, W)
    eids = np.array(eid)
    sel_Lglobal = np.empty(W)
    lam_fit = {}
    for ep in sorted(set(eid)):
        m = eids == ep
        g = grid[err[:, ~m].mean(axis=1).argmin()]        # fitted on the OTHER 39
        lam_fit[str(ep)] = float(g)
        sel_Lglobal[m] = ade(fsel[m] * g, gt[m])

    # ---- along/cross decomposition at the 2 s waypoint -----------------------
    def axes(traj):
        e = traj - gt
        return float(np.abs(e[:, -1, 0]).mean()), float(np.abs(e[:, -1, 1]).mean())
    along_sel, cross_sel = axes(fsel)

    B = lambda a, b: paired_episode_cluster_bootstrap(a, b, eid, n_boot=N_BOOT, seed=SEED)
    out = {
        "arm": os.path.basename(path), "ckpt_step": int(d["ckpt_step"]),
        "n_windows": W, "n_anchors": N, "n_episodes": len(set(eid)),
        "denoise_steps": int(d["steps"]), "nav_mode": str(d["nav_mode"]),
        "instrument_fail": fails,
        "frame_check": {"frac_gt_x_positive": fwd_pos, "corr_gt_x_v0": corr},
        "point": {
            "ade_sel": float(ade_sel.mean()),
            "ade_oracle_fan": float(ade_oracle_fan.mean()),
            "ade_oracle_L": float(best_L.mean()),
            "ade_oracle_A": float(best_A.mean()),
            "ade_oracle_LA": float(best_LA.mean()),
            "ade_sel_plus_L": float(sel_L.mean()),
            "ade_sel_plus_A": float(sel_A.mean()),
            "ade_sel_plus_Lglobal": float(sel_Lglobal.mean()),
            "along_err_2s_sel": along_sel, "cross_err_2s_sel": cross_sel,
        },
        "PRIMARY_A_minus_L": B(best_A, best_L),
        "secondary": {
            "oracle_fan_minus_LA": B(ade_oracle_fan, best_LA),
            "oracle_fan_minus_L": B(ade_oracle_fan, best_L),
            "oracle_fan_minus_A": B(ade_oracle_fan, best_A),
            "sel_minus_selL": B(ade_sel, sel_L),
            "sel_minus_selA": B(ade_sel, sel_A),
            "selA_minus_selL": B(sel_A, sel_L),
            "sel_minus_selLglobal_LOEO": B(ade_sel, sel_Lglobal),
        },
        "lambda_global_fits_LOEO": lam_fit,
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fans", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [run(p) for p in a.fans]
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    for r in res:
        p, pr = r["point"], r["PRIMARY_A_minus_L"]
        print(f"\n=== {r['arm']}  N={r['n_anchors']}  fail={r['instrument_fail'] or 'none'}")
        print(f"  sel {p['ade_sel']:.4f} | oracle_fan {p['ade_oracle_fan']:.4f} "
              f"| +L {p['ade_oracle_L']:.4f} | +A {p['ade_oracle_A']:.4f} | +LA {p['ade_oracle_LA']:.4f}")
        print(f"  PRIMARY A-L delta {pr['delta']:+.4f} [{pr['lo']:+.4f}, {pr['hi']:+.4f}] "
              f"separated={pr['separated']}")
        s = r["secondary"]["sel_minus_selLglobal_LOEO"]
        print(f"  L-global LOEO on SELECTED: {p['ade_sel']:.4f} -> {p['ade_sel_plus_Lglobal']:.4f} "
              f"delta {s['delta']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] separated={s['separated']}")
