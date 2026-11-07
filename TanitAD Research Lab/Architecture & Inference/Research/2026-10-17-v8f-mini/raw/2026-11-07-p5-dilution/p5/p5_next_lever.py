"""p5_next_lever: BESIDE, POST HOC, DECIDES NOTHING. P5 FAILS (dilution cannot explain a gain ratio of 0.67 / 0.59), so the card's section 4 leaves 'a learned response to the data, intent or the frontier'.
The cheapest discriminating measurement on the same 20,000 TRAIN windows: the data's own least-squares gain of the realised plan speed on a0 (given v0, kappa0).
s_gt := (gt_x[knot 12] - gt_x[knot 8]) / 0.8 -- the P1/P2 plan-speed definition applied to the GT future (ego frame at NOW), gt = D.gt_future(pose_grid), pose_grid = track[0::2]; label side, future allowed.
If the data-optimal ds/da0 sits near the model's 1.265 / 1.112, the shrinkage is the data's own mean reversion of acceleration (unresolved intent), not noise. Output: raw/P5_NEXT_LEVER.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import p5_lib as L


def main():
    L.assert_card_frozen()
    D = L.setup_tree()
    tab = np.load(L.HERE + "/raw/window_table_band.npz")
    rows = np.load(L.HERE + "/raw/p5_sample_rows.npy")
    tracks = D.load_native_tracks(L.TRACKS)
    clips, ci, t = tab["clips"], tab["clip_idx"][rows], tab["t"][rows]
    s_gt = np.full(len(rows), np.nan)
    for c in np.unique(ci):
        tr = tracks[str(clips[c])]
        gt, ok = D.gt_future(tr[0::2])
        m = np.flatnonzero(ci == c)
        tt = t[m]
        assert ok[tt].all()
        s_gt[m] = (gt[tt, 11, 0] - gt[tt, 7, 0]) / 0.8
    assert np.isfinite(s_gt).all()
    v0, a0, kap, a = tab["v0"][rows], tab["a0_model"][rows], tab["kappa0"][rows], tab["a_pm0.5s"][rows]
    names = tab["clips"][ci]
    uniq, inv = np.unique(names, return_inverse=True)

    def ols(cols, y, idx=None):
        X = np.stack([np.ones(len(y))] + cols, 1)
        if idx is not None:
            X, y = X[idx], y[idx]
        return np.linalg.lstsq(X, y, rcond=None)[0]

    byep = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(0)
    B = 1000
    bs = {"a0": np.empty(B), "a": np.empty(B), "v0_in_a0_fit": np.empty(B)}
    for b in range(B):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([byep[p] for p in pick])
        c1 = ols([v0, a0, kap], s_gt, idx)
        c2 = ols([v0, a, kap], s_gt, idx)
        bs["a0"][b], bs["v0_in_a0_fit"][b], bs["a"][b] = c1[2], c1[1], c2[2]
    c_a0, c_a = ols([v0, a0, kap], s_gt), ols([v0, a, kap], s_gt)
    pct = lambda x: [float(q) for q in np.percentile(x, [2.5, 97.5])]
    res = {"n_windows": int(len(rows)), "n_episodes": int(len(uniq)), "estimator": "episode-cluster bootstrap B 1,000 seed 0 of the OLS coefficient; s_gt ~ 1 + v0 + {a0 | a} + kappa0 on the P5 sample",
           "target": "s_gt = (gt_x[knot 12] - gt_x[knot 8]) / 0.8 (GT future, ego frame at NOW)",
           "data_optimal_ds_da0_given_v0_kappa0": {"coef_on_a0": float(c_a0[2]), "ci95": pct(bs["a0"]), "coef_on_v0": float(c_a0[1]), "coef_on_v0_ci95": pct(bs["v0_in_a0_fit"])},
           "data_optimal_ds_da_(SG_label)_given_v0_kappa0": {"coef_on_a": float(c_a[2]), "ci95": pct(bs["a"])},
           "reference": {"base_kd_x_ds_da0": L.GAIN_BASE, "model_total_ds_da0_s0": L.GAIN_TOTAL["s0"], "model_total_ds_da0_s1": L.GAIN_TOTAL["s1"]},
           "r2_of_s_gt_on_v0_a0_kappa0": float(1 - ((s_gt - np.stack([np.ones(len(s_gt)), v0, a0, kap], 1) @ c_a0) ** 2).sum() / ((s_gt - s_gt.mean()) ** 2).sum()),
           "note": "post hoc, decides nothing; the held-out (eval-141) gains of the model are WSP_READ.md's, this is the TRAIN-window data optimum."}
    json.dump(res, open(L.HERE + "/raw/P5_NEXT_LEVER.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
