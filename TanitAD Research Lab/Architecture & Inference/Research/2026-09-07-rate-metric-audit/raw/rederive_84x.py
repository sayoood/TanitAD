"""RE-DERIVE THE 84x ON THE REAL DUMP, with its horizon and grid stamped.

Source: taniteval/results/fan_refc-base-30k.pt  (the artifact p14_banked_fan.py read)
Grid  : wp_steps = [5, 10, 15, 20] -> ts0 = [0, 0.5, 1.0, 1.5, 2.0] s  -- UNIFORM.

CONTROL that must reproduce, or nothing below is admissible:
    rank_shipped   kmae == 2.30973482131958
    straight_floor kmae == 0.027366388589143753

ZERO GPU.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import four_families as ff

DT = 0.1
d = torch.load(os.path.join(HERE, "fan_refc-base-30k.pt"),
               map_location="cpu", weights_only=False)
fan = d["fan"].float().numpy()
gt = d["gt"].float().numpy()
cv = d["cv"].float().numpy()
logits = d["logits"].float().numpy()
wp = list(d["wp_steps"])
ts = np.array([w * DT for w in wp], dtype=np.float64)
ts0 = np.concatenate([[0.0], ts])
n = gt.shape[0]


# ------------- VERBATIM from p14_banked_fan.py -----------------------------
def path_with_origin(p):
    z = np.zeros(p.shape[:-2] + (1, 2), dtype=p.dtype)
    return np.concatenate([z, p], axis=-2)


def curvature(p, ts):
    d1 = np.gradient(p, ts, axis=-2)
    d2 = np.gradient(d1, ts, axis=-2)
    num = np.abs(d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0])
    den = np.power(d1[..., 0] ** 2 + d1[..., 1] ** 2, 1.5)
    return num / np.maximum(den, 1e-6)
# ---------------------------------------------------------------------------


ship_idx = logits.argmax(axis=1)
straight = np.zeros_like(gt)
straight[..., 0] = np.linalg.norm(gt, axis=-1)      # p14's straight_floor
arms = {
    "rank_shipped": fan[np.arange(n), ship_idx],
    "rank_oracle": fan[np.arange(n),
                       np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1).argmin(1)],
    "cv_floor": cv,
    "straight_floor": straight,
}

kg = curvature(path_with_origin(gt), ts0)
out = {"grid": {"wp_steps": wp, "ts0_s": list(ts0),
                "gaps_s": list(np.diff(ts0)), "uniform": bool(len(set(np.round(np.diff(ts0), 9))) == 1)},
       "n": int(n), "ckpt": str(d["ckpt"]), "ckpt_step": int(d["ckpt_step"])}

rows = {}
for name, traj in arms.items():
    kt = curvature(path_with_origin(traj), ts0)
    per_win = np.abs(kt - kg).mean(axis=-1)
    rows[name] = {
        "kmae_MEAN": float(per_win.mean()),           # the PUBLISHED statistic
        "kmae_MEDIAN": float(np.median(per_win)),
        "kmae_p90": float(np.percentile(per_win, 90)),
        "kmae_p99": float(np.percentile(per_win, 99)),
        "kmae_max": float(per_win.max()),
        "kappa_pred_max": float(np.abs(kt).max()),
        "kappa_pred_median": float(np.median(np.abs(kt))),
        # how concentrated is the mean?
        "share_of_sum_from_top1pct": float(
            np.sort(per_win)[-max(1, n // 100):].sum() / per_win.sum()),
        "share_of_sum_from_top5pct": float(
            np.sort(per_win)[-max(1, n // 20):].sum() / per_win.sum()),
        "n_windows_kmae_gt_1": int((per_win > 1.0).sum()),
        "n_windows_kmae_gt_10": int((per_win > 10.0).sum()),
    }
    rows[name]["_per_win"] = per_win
out["p14_estimator"] = {k: {kk: vv for kk, vv in v.items() if kk != "_per_win"}
                        for k, v in rows.items()}

# ---- CONTROL ---------------------------------------------------------------
out["CONTROL_reproduces_published"] = {
    "rank_shipped_kmae": rows["rank_shipped"]["kmae_MEAN"],
    "rank_shipped_published": 2.30973482131958,
    "rank_shipped_match": bool(abs(rows["rank_shipped"]["kmae_MEAN"] - 2.30973482131958) < 1e-4),
    "straight_floor_kmae": rows["straight_floor"]["kmae_MEAN"],
    "straight_floor_published": 0.027366388589143753,
    "straight_floor_match": bool(abs(rows["straight_floor"]["kmae_MEAN"] - 0.027366388589143753) < 1e-6),
    "ratio_shipped_over_straight": rows["rank_shipped"]["kmae_MEAN"] / rows["straight_floor"]["kmae_MEAN"],
}

# ---- WHERE DOES IT COME FROM? speed of the blown-up windows ---------------
ship = arms["rank_shipped"]
step_len = np.linalg.norm(np.diff(path_with_origin(ship), axis=-2), axis=-1)  # [n,4] m
min_step = step_len.min(axis=1)
pw = rows["rank_shipped"]["_per_win"]
order = np.argsort(pw)[::-1]
out["blowup_anatomy"] = {
    "corr_kmae_vs_inverse_min_step": float(np.corrcoef(pw, 1.0 / np.maximum(min_step, 1e-6))[0, 1]),
    "median_min_step_m_all": float(np.median(min_step)),
    "median_min_step_m_top1pct_kmae": float(np.median(min_step[order[:max(1, n // 100)]])),
    "median_min_step_m_bottom50pct_kmae": float(np.median(min_step[order[n // 2:]])),
    "n_windows_min_step_below_0.25m": int((min_step < 0.25).sum()),
    "n_windows_min_step_below_0.05m": int((min_step < 0.05).sum()),
    "top10_windows": [{"kmae": float(pw[i]), "min_step_m": float(min_step[i]),
                       "v0_mps": float(d["v0"][i]), "gt_speed_mps": float(d["speed"][i])}
                      for i in order[:10]],
}

# ---- THE CANONICAL METRIC on the SAME data (masked, arc-length normalised) --
seq = {}
for name, traj in arms.items():
    P = ff._seq_geometry(torch.as_tensor(traj).float(), 0.5)   # 0.5 s grid, STAMPED
    G = ff._seq_geometry(torch.as_tensor(gt).float(), 0.5)
    m = P["pair_valid"] & G["pair_valid"]
    dk = (P["curvature"] - G["curvature"]).abs()
    seq[name] = {"kmae_masked_MEAN": float(dk[m].mean()),
                 "kmae_masked_MEDIAN": float(dk[m].median()),
                 "n_pairs_kept": int(m.sum()), "n_pairs_total": int(m.numel()),
                 "frac_masked_out": float(1 - m.float().mean())}
out["seq_geometry_masked"] = seq
out["seq_geometry_ratio_shipped_over_straight"] = (
    seq["rank_shipped"]["kmae_masked_MEAN"] / seq["straight_floor"]["kmae_masked_MEAN"])

print(json.dumps(out, indent=1))
