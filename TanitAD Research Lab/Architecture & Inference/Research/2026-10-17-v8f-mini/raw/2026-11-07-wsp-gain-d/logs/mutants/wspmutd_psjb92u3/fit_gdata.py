"""fit_gdata: g_data of PREREG_WSP_GAIN_D.md section 2 -- the OLS coefficients of the REALISED plan speed on (1, v0, a0, kappa0) over the edit's own 20,000 TRAIN fit windows.
Plan-speed definition = the edit code's own (wsp_measure.spd): s = (x30[11] - x30[7]) / 0.8, 0.2 s knots, ego frame at NOW. Realised: x30 := D.gt_future(track[0::2]) (label side; the future is allowed there).
P5's "knots 8-12" is the same pair (1-based); the control below reproduces P5's own coefficients (0.9355 / 1.3841) from P5's window table with THIS function, so the definitions agree to the digit.
Inputs (v0, a0, kappa0) = the model's own kin3 column of the banked TRAIN_MEASURE shards (what the model read at that window).
No eval-141 window is touched. Dev-box CPU, zero training.   usage: fit_gdata.py   -> raw/GDATA.json"""
import os
import sys
import json
import glob
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import wsp_common as W

P5 = W.THEORY + "/p5"
N_BOOT = 1000


def load_shards():
    files = sorted(glob.glob(W.BANK + "/raw/TRAIN_MEASURE_shard[0-9].npz"))
    assert len(files) == 3, files
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    idx = np.concatenate([p["flat_idx"] for p in parts])
    order = np.argsort(idx, kind="stable")
    assert np.array_equal(idx[order], np.arange(20000))
    out = {k: np.concatenate([p[k] for p in parts], 0)[order] for k in parts[0]}
    return out, files


def s_gt_of(D, tracks, clips, t):
    """Realised plan speed (gt[t, 11, 0] - gt[t, 7, 0]) / 0.8 for windows (clips[i], t[i]); one gt_future per clip."""
    s = np.full(len(t), np.nan)
    uniq, inv = np.unique(clips, return_inverse=True)
    for k, c in enumerate(uniq):
        gt, ok = D.gt_future(tracks[str(c)][0::2])
        m = np.flatnonzero(inv == k)
        tt = t[m].astype(int)
        assert ok[tt].all(), f"{c}: a window without a 30-step future"
        s[m] = (gt[tt, W.J_HI, 0].astype(np.float64) - gt[tt, W.J_LO, 0].astype(np.float64)) / W.DT_SEC
    assert np.isfinite(s).all()
    return s


def ols(v0, a0, kap, y, idx=None):
    X = np.stack([np.ones(len(y)), v0, a0, kap], 1)
    if idx is not None:
        X, y = X[idx], y[idx]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta


def r2(v0, a0, kap, y, beta):
    X = np.stack([np.ones(len(y)), v0, a0, kap], 1)
    return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    t0 = time.time()
    got, head = W.assert_card_frozen()
    assert head == W.CARD_SHA, f"REFUSED: card prefix changed ({head})"
    ids_sha = W.sha256_file(W.BANK + "/raw/train_window_ids.txt")
    assert ids_sha == "948a2a1d68e91764965baa408fe5aab309ff5dd85955e8be7f740abae108812b", ids_sha
    R, files = load_shards()
    ids = [ln.strip() for ln in open(W.BANK + "/raw/train_window_ids.txt", encoding="utf-8") if ln.strip()]
    assert len(ids) == 20000
    clips = np.array([i.rsplit(":", 1)[0] for i in ids])
    tt = np.array([int(i.rsplit(":", 1)[1]) for i in ids])
    assert (R["clip"] == clips).all() and (R["t0"] == tt).all(), "shard window order differs from the id list"
    import tanitad.data.v8f_mini_data as D
    assert os.path.abspath(D.__file__).replace("\\", "/").startswith(W.TREE), D.__file__
    tracks = D.load_native_tracks(W.TRACKS)
    kin3 = R["kin3"].astype(np.float64)
    v0, a0, kap = kin3[:, 0], kin3[:, 1], kin3[:, 2]
    assert 3.0 <= v0.min() and v0.max() < 8.0
    s_gt = s_gt_of(D, tracks, clips, tt)
    beta = ols(v0, a0, kap, s_gt)
    # episode-cluster bootstrap of the coefficient (info; the same estimator as P5's next-lever read, B 1,000 seed 0)
    uniq, inv = np.unique(clips, return_inverse=True)
    byep = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(0)
    bs = np.empty((N_BOOT, 4))
    for b in range(N_BOOT):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        bs[b] = ols(v0, a0, kap, s_gt, np.concatenate([byep[p] for p in pick]))
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    # CONTROL 1: this function reproduces P5's own coefficients on P5's own windows (same definition of the realised plan speed)
    tab = np.load(P5 + "/raw/window_table_band.npz")
    rows = np.load(P5 + "/raw/p5_sample_rows.npy")
    pc, pt = tab["clips"][tab["clip_idx"][rows]], tab["t"][rows]
    sp = s_gt_of(D, tracks, pc, pt)
    bp = ols(tab["v0"][rows], tab["a0_model"][rows], tab["kappa0"][rows], sp)
    p5 = json.load(open(P5 + "/raw/P5_NEXT_LEVER.json"))
    c1 = {"p5_banked_coef_v0": p5["data_optimal_ds_da0_given_v0_kappa0"]["coef_on_v0"], "p5_banked_coef_a0": p5["data_optimal_ds_da0_given_v0_kappa0"]["coef_on_a0"],
          "this_function_on_p5_windows_v0": float(bp[1]), "this_function_on_p5_windows_a0": float(bp[2]),
          "max_abs_diff": float(max(abs(bp[1] - p5["data_optimal_ds_da0_given_v0_kappa0"]["coef_on_v0"]), abs(bp[2] - p5["data_optimal_ds_da0_given_v0_kappa0"]["coef_on_a0"]))), "n_windows": int(len(rows))}
    assert c1["max_abs_diff"] < 1e-6, c1          # the P5 table stores float32 columns: 1.7e-8 observed
    # CONTROL 2: the shard kin3 equals the P5 table's (v0, a0_model, kappa0) on the windows both samples hold
    key = {(str(c), int(t)): i for i, (c, t) in enumerate(zip(clips, tt))}
    common = [(i, key[(str(c), int(t))]) for i, (c, t) in enumerate(zip(pc, pt)) if (str(c), int(t)) in key]
    ia = np.array([p for p, _ in common]); ib = np.array([q for _, q in common])
    c2 = {"n_common_windows": int(len(common)),
          "max_abs_v0": float(np.abs(tab["v0"][rows][ia] - v0[ib]).max()), "max_abs_a0": float(np.abs(tab["a0_model"][rows][ia] - a0[ib]).max()), "max_abs_kappa0": float(np.abs(tab["kappa0"][rows][ia] - kap[ib]).max()),
          "max_abs_s_gt": float(np.abs(sp[ia] - s_gt[ib]).max())}
    assert c2["n_common_windows"] > 1000 and c2["max_abs_s_gt"] < 1e-9 and c2["max_abs_v0"] < 1e-4 and c2["max_abs_a0"] < 1e-4, c2
    # CONTROL 3: a deliberately wrong definition (forward-shifted secant, knots 12-16) must move the coefficients (the check can fail)
    def s_wrong():
        s = np.full(len(tt), np.nan)
        for k, c in enumerate(uniq):
            gt, ok = D.gt_future(tracks[str(c)][0::2]); m = byep[k]; q = tt[m]
            s[m] = (gt[q, 15, 0].astype(np.float64) - gt[q, 11, 0].astype(np.float64)) / W.DT_SEC
        return s
    bw = ols(v0, a0, kap, s_wrong())
    c3 = {"wrong_secant_knots_12_16_coef_v0": float(bw[1]), "wrong_secant_coef_a0": float(bw[2]), "moves_a0_coefficient_by": float(abs(bw[2] - beta[2]))}
    assert c3["moves_a0_coefficient_by"] > 0.05, c3
    out = {"card": "PREREG_WSP_GAIN_D.md", "card_prefix_sha256": head, "train_window_ids_sha256": ids_sha, "n_fit_windows": 20000, "n_episodes": int(len(uniq)),
           "shards": {os.path.basename(f): W.sha256_file(f) for f in files},
           "target": "s_gt = (gt_x[11] - gt_x[7]) / 0.8, 0-based knots of D.gt_future(track[0::2]) = the edit code's wsp_measure.spd pair (P5's 'knots 8-12', 1-based); GT future, ego frame at NOW (label side)",
           "regressors": "1, v0, a0, kappa0 = the model's own kin3 columns at the window (shards)", "g_data": {"dsdv0": float(beta[1]), "dsda0": float(beta[2])},
           "coef": {"intercept": float(beta[0]), "v0": float(beta[1]), "a0": float(beta[2]), "kappa0": float(beta[3])},
           "ci95_episode_cluster_bootstrap_B1000_seed0": {"v0": [float(ci[0, 1]), float(ci[1, 1])], "a0": [float(ci[0, 2]), float(ci[1, 2])]},
           "r2": r2(v0, a0, kap, s_gt, beta), "mean_realised_speed": float(s_gt.mean()), "mean_v0": float(v0.mean()),
           "p5_beside": {"dsdv0": p5["data_optimal_ds_da0_given_v0_kappa0"]["coef_on_v0"], "dsda0": p5["data_optimal_ds_da0_given_v0_kappa0"]["coef_on_a0"], "n_windows": p5["n_windows"], "file": "p5/raw/P5_NEXT_LEVER.json"},
           "controls": {"C1_reproduces_P5_on_P5_windows": c1, "C2_shard_kin3_and_s_gt_equal_P5_table_on_common_windows": c2, "C3_wrong_secant_moves_the_coefficient": c3},
           "per_seed": {}, "wall_s": time.time() - t0}
    for s in (0, 1):
        stot = R[f"s{s}.s_tot"].astype(np.float64)
        out["per_seed"][str(s)] = {"model_median_dsdv0": float(np.median(R[f"s{s}.gv"])), "model_median_dsda0": float(np.median(R[f"s{s}.ga"])), "mean_model_plan_speed": float(stot.mean()),
                                   "shift_target_mean_realised_minus_mean_plan": float(s_gt.mean() - stot.mean())}
    json.dump(out, open(W.HERE + "/raw/GDATA.json", "w"), indent=1)
    np.save(W.HERE + "/raw/s_gt_train20k.npy", s_gt)
    print(json.dumps({k: out[k] for k in ("g_data", "coef", "ci95_episode_cluster_bootstrap_B1000_seed0", "r2", "mean_realised_speed", "p5_beside", "per_seed")}, indent=1))
    print("C1", c1); print("C2", c2); print("C3", c3)


if __name__ == "__main__":
    main()
