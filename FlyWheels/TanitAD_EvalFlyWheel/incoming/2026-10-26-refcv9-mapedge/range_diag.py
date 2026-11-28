"""range_diag.py -- DESCRIPTIVE (no bar): where along the path do (a) the OWN sidewalk rule's false alarms on HUMAN paths and (b) BASE's
true GT-sidewalk violations happen?  Decides whether a RANGE-GATED own-map rule is worth pre-registering.  CPU, 0 GPU.
Population: S6c windows with a GT map and a valid human path (MAPSOFT's fitting population, all folds)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapsoft as MSOFT  # noqa: E402

MS = MSOFT.MS
L = MS.L
BANDS = ((0, 10), (10, 20), (20, 40), (40, 60), (60, 101))


def band_flags(geom, flag_plane):
    """per path (N): bool [N, n_bands] -- any in-range FP5 point inside band b (by x = ix * 0.1 m) on a flagged cell."""
    ix, iy, ins = geom.ix, geom.iy, geom.inside
    hit = ins & torch.as_tensor(flag_plane)[ix, iy]
    x = ix.float() * 0.1
    return torch.stack([(hit & (x >= a) & (x < b)).flatten(1).any(1) for a, b in BANDS], 1).numpy()


def main():
    tr = MS.load_trace("F:/refcv9_mapsel/trace_s0.npz")
    n = tr["n"]
    keys = list(zip(tr["e_i"].tolist(), tr["ws"].tolist()))
    hp, hv = MS.D.human_plans(keys)
    pop = MSOFT.s6c_mask(tr, n) & np.asarray(tr["gt_label"], bool) & hv
    base = np.load("C:/Users/Admin/refcv8_pf/refcv9_mapsel/result/s0/features_s0.npz")["pick_BASE"]
    rows = []
    for j in np.nonzero(pop)[0]:
        z = np.load(MS.rast_file("F:/refcv9_mapsel/rast_s0", tr["e_i"][j], tr["ws"][j]))
        own_sw = z["psw"] >= L.PSW_MIN_U8
        gt_sw = (z["gt"] == 7)
        plans = torch.from_numpy(np.stack([hp[j], tr["fan"][j, base[j]]]).astype(np.float32))
        g = L.Geom(plans)
        o, t = band_flags(g, own_sw), band_flags(g, gt_sw)
        rows.append((int(tr["e_i"][j]), o[0], t[0], o[1], t[1]))
    hum_own = np.array([r[1] for r in rows]); hum_gt = np.array([r[2] for r in rows])
    b_own = np.array([r[3] for r in rows]); b_gt = np.array([r[4] for r in rows])
    names = [f"{a}-{b}m" for a, b in BANDS]
    out = {"n_windows": len(rows), "n_episodes": len({r[0] for r in rows}), "bands": names,
           "HUMAN_own_flag_rate": dict(zip(names, hum_own.mean(0).round(4).tolist())),
           "HUMAN_own_FALSE_alarm_rate (own flags, GT does not in that band)": dict(zip(names, (hum_own & ~hum_gt).mean(0).round(4).tolist())),
           "HUMAN_gt_sidewalk_rate": dict(zip(names, hum_gt.mean(0).round(4).tolist())),
           "BASE_gt_violation_rate": dict(zip(names, b_gt.mean(0).round(4).tolist())),
           "BASE_own_flag_rate": dict(zip(names, b_own.mean(0).round(4).tolist())),
           "BASE_gt_violation_caught_by_own_in_band": dict(zip(names, [float((b_gt[:, k] & b_own[:, k]).sum() / max(b_gt[:, k].sum(), 1)) for k in range(len(BANDS))])),
           "cumulative_from_0": {}}
    for k in range(1, len(BANDS) + 1):
        ho, hg, bo, bg = hum_own[:, :k].any(1), hum_gt[:, :k].any(1), b_own[:, :k].any(1), b_gt[:, :k].any(1)
        out["cumulative_from_0"][f"0-{BANDS[k - 1][1]}m"] = {"HUMAN_false_alarm": float((ho & ~hg).mean()), "BASE_true_violation": float(bg.mean()),
                                                           "BASE_violation_caught_by_own": float((bg & bo).sum() / max(bg.sum(), 1)),
                                                           "BASE_flagged_by_own": float(bo.mean())}
    out["evidence"] = "MEASURED, DESCRIPTIVE (no bar); raw candidates/human paths (not staged); T1 open-loop eval139; seed 0"
    json.dump(out, open(HERE / "result" / "RANGE_DIAG.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
