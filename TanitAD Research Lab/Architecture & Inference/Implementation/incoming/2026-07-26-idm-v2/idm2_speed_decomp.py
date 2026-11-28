"""IDM-v2 -- WHERE does the speed error live?  (sharpens verdict (c))

The pooled per-clip oracle recalibration is confounded: 94-96 % of speed variance
is BETWEEN clips, so ANY per-clip fit looks spectacular.  The decisive question is
narrower:

   Is the deployed head's speed error a CLIP-LEVEL error (it does not know how
   fast this clip is going -> metric scale), or a WITHIN-CLIP error (it does not
   track the accelerations)?  And if it is clip-level, is that offset
   PROPORTIONAL to the clip's speed (multiplicative = monocular scale) or
   constant (additive = a bias)?

Decomposition, per clip c with n_c windows:
    MSE_total = sum_c n_c[ bias_c^2 + within_c ] / N     where bias_c = mean(p-y)
Then regress bias_c on the clip's mean speed.  A slope significantly < 0 with
bias_c ~ -(1-g)*vbar_c is a GAIN g.

Writes /root/idm2/out/speed_decomp.json
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")
import idm2_lib as L                      # noqa: E402
import idm_head as ih                     # noqa: E402

DEV, K = "cuda", 8


def decomp(p, y, eid, dom, name):
    rows, N = [], p.size
    ssq_b = ssq_w = 0.0
    for tag in np.unique(eid):
        m = eid == tag
        pc, yc = p[m], y[m]
        bias = float((pc - yc).mean())
        within = float(((pc - yc) - bias).__pow__(2).mean())
        ssq_b += m.sum() * bias ** 2
        ssq_w += m.sum() * within
        sd = float(yc.std())
        rows.append({"tag": tag, "dom": str(dom[m][0]), "n": int(m.sum()),
                     "vbar": float(yc.mean()), "pbar": float(pc.mean()),
                     "bias": bias, "within_rmse": within ** 0.5,
                     "gt_std_in_clip": sd,
                     "within_corr": float(np.corrcoef(pc, yc)[0, 1]) if sd > 1e-6 else float("nan")})
    mse = float(((p - y) ** 2).mean())
    vb = np.array([r["vbar"] for r in rows])
    bi = np.array([r["bias"] for r in rows])
    A = np.stack([vb, np.ones(vb.size)], 1)
    c, *_ = np.linalg.lstsq(A, bi, rcond=None)
    r_bias_v = float(np.corrcoef(vb, bi)[0, 1])
    # ceiling of a level-only fix: predict each clip's own GT mean
    clipmean = np.zeros_like(p)
    debias = p.copy()
    for r in rows:
        m = eid == r["tag"]
        clipmean[m] = r["vbar"]
        debias[m] = p[m] - r["bias"]
    out = {
        "arm": name, "mse_total": mse,
        "mse_from_clip_level_bias": float(ssq_b / N),
        "mse_within_clip": float(ssq_w / N),
        "frac_of_mse_that_is_CLIP_LEVEL": float(ssq_b / N / max(mse, 1e-12)),
        "bias_vs_clipspeed": {"slope": float(c[0]), "intercept": float(c[1]),
                              "pearson_r": r_bias_v,
                              "implied_gain_g": float(1.0 + c[0]),
                              "n_clips": int(vb.size)},
        "r2_raw": L.chan_metrics(p, y)["r2"],
        "r2_if_clip_level_removed": L.chan_metrics(debias, y)["r2"],
        "r2_of_clip_mean_oracle": L.chan_metrics(clipmean, y)["r2"],
        "mae_raw": float(np.abs(p - y).mean()),
        "mae_if_clip_level_removed": float(np.abs(debias - y).mean()),
        "mean_within_clip_corr": float(np.nanmean([r["within_corr"] for r in rows])),
        "per_clip": rows}
    print(f"\n--- {name} ---")
    print(f"  R2 {out['r2_raw']:+.4f}  MAE {out['mae_raw']:.3f}")
    print(f"  MSE {mse:.3f} = clip-level {out['mse_from_clip_level_bias']:.3f} "
          f"({100*out['frac_of_mse_that_is_CLIP_LEVEL']:.1f} %) + within-clip "
          f"{out['mse_within_clip']:.3f}")
    print(f"  remove the clip-level term -> R2 {out['r2_if_clip_level_removed']:+.4f} "
          f"MAE {out['mae_if_clip_level_removed']:.3f}")
    print(f"  'know each clip's mean speed' oracle -> R2 "
          f"{out['r2_of_clip_mean_oracle']:+.4f}")
    b = out["bias_vs_clipspeed"]
    print(f"  bias_c vs clip mean speed: slope {b['slope']:+.4f} "
          f"(=> gain g={b['implied_gain_g']:.3f}) intercept {b['intercept']:+.3f} "
          f"r={b['pearson_r']:+.3f} over {b['n_clips']} clips")
    print(f"  mean WITHIN-clip corr(pred, gt) = {out['mean_within_clip_corr']:+.4f}")
    return out


def main():
    tr_tags, va_tags = L.split_tags()
    va = L.build_set(va_tags, k=K, stride=2)
    y = va["S"].numpy().astype(np.float64)[:, 0]
    eid, dom = va["eid"], va["dom"]
    res = {"n_val_windows": int(y.size), "n_val_eps": len(va_tags), "arms": {}}

    d = torch.load("/root/idmval/idm_head_v1.pt", weights_only=False)
    h = ih.IDMHead(**d["config"]["head_kwargs"]).to(DEV)
    h.load_state_dict(d["state_dict"]); h.eval()
    Z = va["Z"][:, K - 4:K + 5].to(DEV).float()
    with torch.no_grad():
        P = torch.cat([h(Z[i:i + 1024])["scalars"].cpu()
                       for i in range(0, Z.shape[0], 1024)]).numpy().astype(np.float64)
    res["arms"]["A0"] = decomp(P[:, 0], y, eid, dom, "A0 (idm_head_v1, deployed)")
    for dd in ("pai", "cm"):
        m = dom == dd
        res["arms"][f"A0_{dd}"] = decomp(P[m, 0], y[m], eid[m], dom[m],
                                         f"A0 on {dd} only")
    L.jdump(res, "/root/idm2/out/speed_decomp.json")


if __name__ == "__main__":
    main()
