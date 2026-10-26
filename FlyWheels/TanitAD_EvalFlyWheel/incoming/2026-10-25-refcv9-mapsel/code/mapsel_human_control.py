"""mapsel_human_control.py -- refcv9 MAP-SELECT K_HUMAN + K_SENS (PREREG_MAPSEL.md sec. 5 as AMENDED by PREREG_MAPSEL_AMENDMENT1.md): the HUMAN trajectory scored by the SAME footprint / path sampler / GT raster as the candidates.  0 GPU.

AMENDED semantics: off-drivable = a check point on a SEEN SIDEWALK cell (code 7); nocls / not-seen = UNKNOWN.  Two controls:
  K_HUMAN (a SANITY BOUND, not a blind control: the class set was chosen after the sealed set read 9.26 %): human off-drivable <= 2 %, parity with the banked S6 `g` < 1e-3 m;
  K_SENS  (BLIND SENSITIVITY): the human path shifted laterally by +3.5 m and by -3.5 m (one lane, rigid shift in the ego frame) must read an off-drivable rate SEPARATED ABOVE the unshifted
          human's for at least one side (paired episode-cluster bootstrap, delta CI lower bound > 0), else the metric cannot see violations and the run is VOID.

The human path = the logged future of the ego at the 8 plan instants (0.5 .. 6 s) in the t0 ego frame (decisions/gt_future_ext; parity with the banked S6 `g` is asserted).  A rate above
`--max-off` (default 2 %) on off-drivable means the footprint / raster / frame geometry is WRONG and the whole experiment STOPS (exit 3, `ok: false` in the JSON).

    python mapsel_human_control.py --out result/K_HUMAN.json [--limit-clips N]
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import mapsel_data as D  # noqa: E402
import mapsel_lib as L  # noqa: E402

torch.set_num_threads(4)
SHIFT_M = 3.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-off", type=float, default=0.02)
    ap.add_argument("--limit-clips", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    import l2_common as C
    E = C.load_eval_pf(0)
    keys = list(zip(E["clip"].tolist(), E["ws"].tolist()))
    hp, valid = D.human_plans(keys)
    par = float(np.abs(hp[:, [1, 3, 4, 5, 6, 7]] - E["g"].astype(np.float32))[valid].max())
    gt = D.GTReader()
    PR = C.stack()["PR"]
    comp = np.asarray(E["compliant"], bool)
    order = sorted(range(len(keys)), key=lambda i: (keys[i][0], keys[i][1]))
    if a.limit_clips:
        keep = set(sorted({keys[i][0] for i in order})[:a.limit_clips])
        order = [i for i in order if keys[i][0] in keep]
    rows = []
    for n, i in enumerate(order):
        ci, ws = keys[i]
        codes = gt.codes(ci, ws)
        if codes is None or not valid[i]:
            continue
        sh = np.stack([hp[i], hp[i] + np.array([0.0, SHIFT_M], np.float32), hp[i] - np.array([0.0, SHIFT_M], np.float32)]).astype(np.float32)
        g = L.Geom(torch.from_numpy(sh))
        mg = L.masks_gt(codes)
        o_, e_, l_ = (x.numpy() for x in g.flags(mg))
        cn, cu = (x.numpy() for x in g.coverage(mg))
        rows.append((i, ci, bool(o_[0]), bool(e_[0]), bool(l_[0]), bool(comp[i]), bool(o_[1]), bool(o_[2]), float(cn[0]), float(cu[0])))
        if n % 500 == 0:
            print(f"[human] {n}/{len(order)} el={time.time() - t0:.0f}s", flush=True)
    R = np.array([(r[2], r[3], r[4], r[5]) for r in rows], bool)
    Rp = np.array([r[6] for r in rows], float)
    Rm = np.array([r[7] for r in rows], float)
    cov = np.array([(r[8], r[9]) for r in rows], float)
    ridx = np.array([r[0] for r in rows])
    eid = np.array([int(E["eid"][r[0]]) for r in rows])
    res = {"tool": "mapsel_human_control.py", "n_windows_gt_valid": int(len(rows)), "n_clips": int(len({r[1] for r in rows})), "parity_vs_banked_S6_g_max_abs_m": par,
           "wall_s": round(time.time() - t0, 1), "max_off_allowed": a.max_off, "limit_clips": a.limit_clips}
    for nm, m in (("all", np.ones(len(rows), bool)), ("S6_compliant", R[:, 3])):
        d = {"n": int(m.sum())}
        for k, col in (("off_drivable", 0), ("edge_contact", 1), ("lane_cross", 2)):
            d[k] = float(R[m, col].mean()) if m.any() else float("nan")
        # per-episode cluster spread (descriptive): the per-episode off-rate quantiles
        if m.any():
            per = [R[m & (eid == e), 0].mean() for e in sorted(set(eid[m].tolist()))]
            d["off_drivable_per_episode_q50_q90_max"] = [float(np.quantile(per, 0.5)), float(np.quantile(per, 0.9)), float(np.max(per))]
        d["coverage_nocls_share"], d["coverage_unseen_share"] = float(cov[m, 0].mean()), float(cov[m, 1].mean())
        res[nm] = d
    # ---- K_SENS: blind lateral-shift sensitivity on the S6-compliant population, paired episode-cluster bootstrap (new = shifted, ref = unshifted human)
    mc = R[:, 3]
    eids = [int(E["eid"][i]) for i in ridx[mc]]
    base_off = R[mc, 0].astype(float)
    sens = {"shift_m": SHIFT_M, "human_off_rate": float(base_off.mean()), "n": int(mc.sum())}
    for nm_, arr in (("plus_left", Rp), ("minus_right", Rm)):
        c = PR.paired_cell(arr[mc], base_off, eids)
        sens[nm_] = {"rate": float(arr[mc].mean()), **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items() if k in ("delta", "lo", "hi", "n_windows", "n_episodes", "separated", "status", "reason")}}
        sens[nm_]["separated_above"] = bool("lo" in c and c["lo"] > 0)
    sens["ok"] = bool(sens["plus_left"]["separated_above"] or sens["minus_right"]["separated_above"])
    res["sensitivity"] = sens
    res["ok"] = bool(res["S6_compliant"]["off_drivable"] <= a.max_off and par < 1e-3 and sens["ok"])
    res["amendment"] = "AMENDMENT 1 class set: off-drivable = seen sidewalk (code 7) only"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_windows_gt_valid", "parity_vs_banked_S6_g_max_abs_m", "S6_compliant", "sensitivity", "ok")}, indent=1), flush=True)
    return 0 if res["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
