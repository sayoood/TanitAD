"""P2 - IS 2.0 s THE RIGHT EARLIEST SPLIT?  (RULE ZERO: the next lever, run.)

P1 MEASURED that on lane-change windows 503 of 506 oracle picks land on the
`t_split = 2.0 s` pair -- the EARLIEST split in the family and the one at the
edge of the grid. An edge-of-grid winner is a hypothesis, not an answer: it may
mean 2.0 s is right, or it may mean the family is truncated.

⛔ AND THERE IS A COST TO GOING EARLIER, stated before the numbers. Every split
at t >= 2.0 s is INVISIBLE on the programme's 2 s scored grid (P1 B5: max |ADE
change| exactly 0.0, 0 windows repick), so the extension provably cannot
contaminate `ade_0_2s`. A split at t < 2.0 s gives that up. This measures what
the trade is worth; it does not assume either answer.

Reported per split point IN ISOLATION (one pair each, +2 candidates) so the
splits are ranked against each other rather than as a bundle.
"""
import argparse
import json
import sys

import numpy as np
import torch

import _env  # noqa: F401
from tanitad.refs import anchor_twoseg as ts    # noqa: E402
import p1_twoseg_supply as P                    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sd = torch.load(P.CKPT, map_location="cpu", weights_only=False)
    C0 = sd.get("model", sd)["core.decoder.anchor_controls"].float()
    n_base = C0.shape[0]

    GT, V0, TAG, n_dump = P_load(a)
    n = len(V0)
    LC, TN = TAG[:, 0] == 1, TAG[:, 1] == 1
    print(f"[grid] {n} scoreable windows, {int(LC.sum())} lane-change, "
          f"{int(TN.sum())} turn (corpus: 141 B1-v7.2 EVAL clips)")

    base_fan = P.roll(C0, V0, P.SLOTS)
    base, _ = P.best(base_fan, GT)
    s2 = [P.SLOTS[i] for i in range(4)]
    base2_fan = P.roll(C0, V0, s2)
    base2, _ = P.best(base2_fan, GT[:, :4])
    print(f"[base] N=117 ALL {base.mean():.4f} LC {base[LC].mean():.4f} "
          f"TURN {base[TN].mean():.4f} | 2 s ALL {base2.mean():.4f}")

    res = {"n_scoreable_windows": int(n), "n_lane_change": int(LC.sum()),
           "n_turn": int(TN.sum()),
           "base": {"ade_all": float(base.mean()),
                    "ade_lane_change": float(base[LC].mean()),
                    "ade_turn": float(base[TN].mean()),
                    "ade_all_2s": float(base2.mean())},
           "single_split": {}, "cumulative": {}}

    print(f"\n{'split':>7}{'n_new':>7}{'LC ADE':>10}{'gain':>9}"
          f"{'ALL':>10}{'gain':>9}{'TURN gain':>11}{'LCpicks':>9}"
          f"{'2s gain':>11}{'2s repick':>11}")
    grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for t in grid:
        ext = ts.extend_controls(C0, P.T_S, t_split_s=(t,))
        fan = P.roll(ext, V0, P.SLOTS)
        ade, idx = P.best(fan, GT)
        f2 = P.roll(ext, V0, s2)
        a2, i2 = P.best(f2, GT[:, :4])
        picks = idx >= n_base
        r = {"n_new": int(ext.shape[0] - n_base),
             "ade_lane_change": float(ade[LC].mean()),
             "gain_lane_change": float(base[LC].mean() - ade[LC].mean()),
             "ade_all": float(ade.mean()),
             "gain_all": float(base.mean() - ade.mean()),
             "gain_turn": float(base[TN].mean() - ade[TN].mean()),
             "lc_pick_rate": float(picks[LC].mean()),
             "lc_picks": int(picks[LC].sum()),
             "gain_all_2s": float(base2.mean() - a2.mean()),
             "n_2s_repick": int((i2 >= n_base).sum())}
        res["single_split"][f"{t:.1f}"] = r
        print(f"{t:>7.1f}{r['n_new']:>7d}{r['ade_lane_change']:>10.4f}"
              f"{r['gain_lane_change']:>9.4f}{r['ade_all']:>10.4f}"
              f"{r['gain_all']:>9.4f}{r['gain_turn']:>11.4f}"
              f"{r['lc_picks']:>9d}{r['gain_all_2s']:>11.6f}"
              f"{r['n_2s_repick']:>11d}")

    print(f"\nCUMULATIVE families (the build decision)")
    print(f"{'family':<34}{'n_new':>7}{'LC ADE':>10}{'gain':>9}{'ALL':>10}"
          f"{'gain':>9}{'TURN gain':>11}{'2s gain':>11}{'2s repick':>11}")
    fams = {
        "2/3/4 s  (SHIPPED)": (2.0, 3.0, 4.0),
        "1.5/2/3/4 s": (1.5, 2.0, 3.0, 4.0),
        "1/1.5/2/3/4 s": (1.0, 1.5, 2.0, 3.0, 4.0),
        "1.5/2/2.5/3/4 s": (1.5, 2.0, 2.5, 3.0, 4.0),
        "2/2.5/3/4 s": (2.0, 2.5, 3.0, 4.0),
        "2/3/4/5 s": (2.0, 3.0, 4.0, 5.0),
    }
    for name, spl in fams.items():
        ext = ts.extend_controls(C0, P.T_S, t_split_s=spl)
        fan = P.roll(ext, V0, P.SLOTS)
        ade, idx = P.best(fan, GT)
        f2 = P.roll(ext, V0, s2)
        a2, i2 = P.best(f2, GT[:, :4])
        r = {"splits": list(spl), "n_new": int(ext.shape[0] - n_base),
             "ade_lane_change": float(ade[LC].mean()),
             "gain_lane_change": float(base[LC].mean() - ade[LC].mean()),
             "ade_all": float(ade.mean()),
             "gain_all": float(base.mean() - ade.mean()),
             "gain_turn": float(base[TN].mean() - ade[TN].mean()),
             "lc_picks": int((idx >= n_base)[LC].sum()),
             "gain_all_2s": float(base2.mean() - a2.mean()),
             "n_2s_repick": int((i2 >= n_base).sum())}
        res["cumulative"][name] = r
        print(f"{name:<34}{r['n_new']:>7d}{r['ade_lane_change']:>10.4f}"
              f"{r['gain_lane_change']:>9.4f}{r['ade_all']:>10.4f}"
              f"{r['gain_all']:>9.4f}{r['gain_turn']:>11.4f}"
              f"{r['gain_all_2s']:>11.6f}{r['n_2s_repick']:>11d}")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {a.out}")
    return 0


def P_load(a):
    """The P1 window/GT build, reused verbatim by calling into its main path."""
    import os
    with open(a.arm_json, encoding="utf-8") as fh:
        man = json.load(fh)["refcv3"]["manifest"]
    clip_of = {int(e["file_index"]): e["clip_id"] for e in man["episodes"]}
    off = int(man["corpus"]["frames"]["provider_to_raw_frame_offset"])
    n_fut = int(round(P.T_S * P.HZ))
    v0s, gts, tag, n_dump = [], [], [], 0
    for fi in sorted(clip_of):
        z = np.load(os.path.join(a.dump, f"ep{fi:03d}.npz"), allow_pickle=True)
        ws, v0 = z["ws"].astype(int), z["v0"].astype(np.float64)
        n_dump += len(ws)
        p = torch.load(os.path.join(P.EPS, clip_of[fi] + ".v2ep.pt"),
                       map_location="cpu",
                       weights_only=False)["poses"].numpy().astype(np.float64)
        for i, w in enumerate(ws):
            j0 = w + off
            if j0 + n_fut >= len(p):
                continue
            seg = p[j0:j0 + n_fut + 1]
            c, s = np.cos(-seg[0, 2]), np.sin(-seg[0, 2])
            dx, dy = seg[:, 0] - seg[0, 0], seg[:, 1] - seg[0, 1]
            lon = c * dx - s * dy
            lat = s * dx + c * dy
            yaw = P.wrap(seg[:, 2] - seg[0, 2])
            gts.append(np.stack([lon[list(P.HORIZONS)],
                                 lat[list(P.HORIZONS)]], -1))
            v0s.append(v0[i])
            dyt, dyp = np.degrees(yaw[-1]), np.degrees(np.abs(yaw).max())
            lc = (P.LANE_LO <= abs(lat[-1]) < P.LANE_HI) and \
                 abs(dyt) <= P.RET_NET_DEG and dyp >= P.RET_PEAK_DEG
            tag.append((1 if lc else 0, 1 if abs(dyt) >= P.TURN_DEG else 0))
    return np.asarray(gts), np.asarray(v0s), np.asarray(tag), n_dump


if __name__ == "__main__":
    raise SystemExit(main())
