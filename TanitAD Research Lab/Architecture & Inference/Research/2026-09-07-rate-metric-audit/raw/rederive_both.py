"""Re-derive BOTH published p14 fan arms (base-30k and xl-30k) and split the
curvature contrast by ego motion. ZERO GPU."""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import four_families as ff

DT = 0.1


def pwo(p):
    z = np.zeros(p.shape[:-2] + (1, 2), dtype=p.dtype)
    return np.concatenate([z, p], axis=-2)


def kappa_p14(p, ts):
    d1 = np.gradient(p, ts, axis=-2); d2 = np.gradient(d1, ts, axis=-2)
    num = np.abs(d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0])
    den = np.power(d1[..., 0] ** 2 + d1[..., 1] ** 2, 1.5)
    return num / np.maximum(den, 1e-6)


out = {}
for tag, fn in (("refc-base-30k", "fan_refc-base-30k.pt"),
                ("refc-xl-30k", "fan_refc-xl-30k.pt")):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        out[tag] = {"ERROR": "dump not present locally"}
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    fan, gt, cv = d["fan"].float().numpy(), d["gt"].float().numpy(), d["cv"].float().numpy()
    logits = d["logits"].float().numpy()
    ts0 = np.concatenate([[0.0], np.array([w * DT for w in d["wp_steps"]], dtype=np.float64)])
    n = gt.shape[0]
    grid_dt = float(np.diff(ts0)[0])
    uniform = bool(len(set(np.round(np.diff(ts0), 9))) == 1)
    straight = np.zeros_like(gt); straight[..., 0] = np.linalg.norm(gt, axis=-1)
    ship = fan[np.arange(n), logits.argmax(1)]
    v0 = d["v0"].numpy()
    moving = v0 > 0.5
    kg = kappa_p14(pwo(gt), ts0)

    def kmae(traj, keep=None):
        v = np.abs(kappa_p14(pwo(traj), ts0) - kg).mean(-1)
        return v if keep is None else v[keep]

    # canonical masked
    def seq_kmae(traj):
        P = ff._seq_geometry(torch.as_tensor(traj).float(), grid_dt)
        G = ff._seq_geometry(torch.as_tensor(gt).float(), grid_dt)
        m = P["pair_valid"] & G["pair_valid"]
        return float((P["curvature"] - G["curvature"]).abs()[m].mean()), float(1 - m.float().mean())

    s_ship, mask_ship = seq_kmae(ship)
    s_str, mask_str = seq_kmae(straight)
    out[tag] = {
        "ckpt": str(d["ckpt"]), "ckpt_step": int(d["ckpt_step"]), "n_windows": int(n),
        "n_episodes": int(len(set(np.asarray(d["eid"]).tolist()))),
        "wp_steps": list(d["wp_steps"]), "ts0_s": [float(x) for x in ts0],
        "grid_dt_s": grid_dt, "grid_uniform": uniform, "n_anchors": int(d["n_anchors"]),
        "PUBLISHED_p14_unmasked": {
            "shipped_kmae": float(kmae(ship).mean()),
            "cv_floor_kmae": float(kmae(cv).mean()),
            "straight_floor_kmae": float(kmae(straight).mean()),
            "ratio_shipped_over_straight": float(kmae(ship).mean() / kmae(straight).mean()),
        },
        "SPLIT_by_ego_motion": {
            "n_stopped_v0_le_0.5": int((~moving).sum()),
            "frac_stopped": float((~moving).mean()),
            "moving_ratio": float(kmae(ship, moving).mean() / kmae(straight, moving).mean()),
            "stopped_ratio": float(kmae(ship, ~moving).mean() / kmae(straight, ~moving).mean()),
            "moving_shipped": float(kmae(ship, moving).mean()),
            "moving_straight": float(kmae(straight, moving).mean()),
        },
        "CANONICAL_seq_geometry_masked": {
            "shipped_kmae": s_ship, "straight_floor_kmae": s_str,
            "ratio_shipped_over_straight": s_ship / s_str,
            "frac_pairs_masked_shipped": mask_ship,
        },
        "median_kmae_p14": {"shipped": float(np.median(kmae(ship))),
                            "straight_floor": float(np.median(kmae(straight)))},
    }

print(json.dumps(out, indent=1))
