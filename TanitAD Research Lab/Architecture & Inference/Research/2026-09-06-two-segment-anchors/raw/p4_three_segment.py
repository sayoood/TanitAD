"""P4 - THE NEXT LEVER, RUN (RULE ZERO): does a THIRD segment recover the
curvature the two-segment family gives away?

P3 REFUTED the kink hypothesis. On the 506 repicked lane-change windows the
curvature degradation is NOT at the flip (segment 2.0-3.0 s is the SMALLEST
delta, +0.001334); it is in the TAIL -- 3-4 s +0.006441, 4-5 s +0.005670,
5-6 s +0.004835, i.e. 72.0 % of the total per-segment degradation sits in the
3-6 s band. ⇒ the candidate is RIGHT IN SHAPE and WRONG IN DURATION: after the
flip it keeps counter-steering for 4 s while the human has finished and gone
straight.

THE LEVER: a THREE-segment schedule `+a_lat` on [0, t1), `-a_lat` on [t1, t2),
`0` after -- one more split, which RETURNS TO STRAIGHT.

⛔ THIS IS A PROBE, NOT A SHIP. It is measured model-free so the decision to
build a 3-segment schedule (a 4-column `controls` and a new
`control_schedule`) is taken on evidence. The 2-segment family shipped in this
turn is the PRE-REGISTERED one and is NOT swapped after seeing these numbers.

Each (t1, t2) is scored as its OWN +2-candidate family, so the pairs are ranked
against each other and against the 2-segment `t_split = 2.0 s` reference
(LC gain 0.1640, curvature +0.003361 on its repicks).
"""
import argparse
import json
import sys

import numpy as np
import torch

import _env  # noqa: F401
import p1_twoseg_supply as P                    # noqa: E402
import p2_split_sweep as S                      # noqa: E402
from tanitad.models.kinematic import rollout_unicycle   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def roll3(a_lon, a_lat, t1, t2, v0, steps=60, dt=0.1):
    """[n, C, 8, 2] three-segment fan: +a_lat, -a_lat, then ZERO."""
    out = np.empty((len(v0), len(a_lat), len(P.SLOTS), 2))
    k = np.arange(steps)                                   # TICK index, exact
    k1 = np.round(np.asarray(t1) / dt).astype(int)
    k2 = np.round(np.asarray(t2) / dt).astype(int)
    sgn = np.where(k[None, :] < k1[:, None], 1.0,
                   np.where(k[None, :] < k2[:, None], -1.0, 0.0))   # [C, T]
    for s in range(0, len(v0), P.CHUNK):
        v = np.asarray(v0[s:s + P.CHUNK], dtype=np.float64)
        b, c = len(v), len(a_lat)
        vv = np.maximum(v, P.ALAT_V_FLOOR) ** 2
        kap = np.clip(np.asarray(a_lat)[None, :] / vv[:, None],
                      -P.KAPPA_CAP, P.KAPPA_CAP)                    # [b, C]
        ck = kap[:, :, None] * sgn[None, :, :]                      # [b, C, T]
        ca = np.broadcast_to(np.asarray(a_lon)[None, :, None], ck.shape)
        ctl = torch.as_tensor(np.stack([ca, ck], -1),
                              dtype=torch.float32).reshape(-1, steps, 2)
        s0 = torch.zeros(b * c, 4, dtype=torch.float32)
        s0[:, 3] = torch.as_tensor(np.repeat(v, c), dtype=torch.float32)
        st = rollout_unicycle(s0, ctl, dt=dt).reshape(b, c, steps, 4)
        out[s:s + P.CHUNK] = st[:, :, P.SLOTS, :2].numpy().astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sd = torch.load(P.CKPT, map_location="cpu", weights_only=False)
    C0 = sd.get("model", sd)["core.decoder.anchor_controls"].float()
    GT, V0, TAG, _ = S.P_load(a)
    n = len(V0)
    LC, TN = TAG[:, 0] == 1, TAG[:, 1] == 1
    fb = P.roll(C0, V0, P.SLOTS)
    ab, ib = P.best(fb, GT)
    pb = fb[np.arange(n), ib]
    print(f"[base] N=117 n={n} LC={int(LC.sum())} ALL {ab.mean():.4f} "
          f"LC {ab[LC].mean():.4f} TURN {ab[TN].mean():.4f}")

    res = {"n_scoreable_windows": int(n), "n_lane_change": int(LC.sum()),
           "base": {"ade_all": float(ab.mean()),
                    "ade_lane_change": float(ab[LC].mean()),
                    "ade_turn": float(ab[TN].mean())},
           "families": {}}

    print(f"\n{'family':<30}{'n_new':>7}{'LC ADE':>10}{'LCgain':>9}"
          f"{'ALLgain':>9}{'TURNgain':>10}{'LCpicks':>9}"
          f"{'curv b':>10}{'curv e':>10}{'d curv':>10}{'cross d':>9}")
    combos = [("2-seg t=2.0 (SHIPPED ref)", 2.0, 6.0),
              ("3-seg 2.0 -> 3.0", 2.0, 3.0),
              ("3-seg 2.0 -> 3.5", 2.0, 3.5),
              ("3-seg 2.0 -> 4.0", 2.0, 4.0),
              ("3-seg 2.0 -> 4.5", 2.0, 4.5),
              ("3-seg 1.5 -> 3.0", 1.5, 3.0),
              ("3-seg 1.5 -> 3.5", 1.5, 3.5),
              ("3-seg 2.5 -> 4.0", 2.5, 4.0),
              ("3-seg 2.5 -> 4.5", 2.5, 4.5),
              ("3-seg 3.0 -> 5.0", 3.0, 5.0)]
    mags = [-0.75, 0.75]
    for name, t1, t2 in combos:
        A_lat = np.array(mags)
        A_lon = np.zeros_like(A_lat)
        T1 = np.full_like(A_lat, t1)
        T2 = np.full_like(A_lat, t2)
        new = roll3(A_lon, A_lat, T1, T2, V0)
        fan = np.concatenate([fb, new], axis=1)
        ade, idx = P.best(fan, GT)
        pk = fan[np.arange(n), idx]
        rep = idx >= 117
        m = LC & rep
        if m.any():
            cb = float(np.nanmean(P.families(pb[m], GT[m],
                                             P.SLOT_DT)["curv_mae_1pm"]))
            ce = float(np.nanmean(P.families(pk[m], GT[m],
                                             P.SLOT_DT)["curv_mae_1pm"]))
            xb = float(np.nanmean(P.families(pb[m], GT[m],
                                             P.SLOT_DT)["cross_mae_m"]))
            xe = float(np.nanmean(P.families(pk[m], GT[m],
                                             P.SLOT_DT)["cross_mae_m"]))
        else:
            cb = ce = xb = xe = float("nan")
        r = {"t1": t1, "t2": t2, "n_new": 2,
             "ade_lane_change": float(ade[LC].mean()),
             "gain_lane_change": float(ab[LC].mean() - ade[LC].mean()),
             "gain_all": float(ab.mean() - ade.mean()),
             "gain_turn": float(ab[TN].mean() - ade[TN].mean()),
             "lc_picks": int(m.sum()),
             "curv_base_on_repicks": cb, "curv_ext_on_repicks": ce,
             "curv_delta_on_repicks": ce - cb,
             "cross_base_on_repicks": xb, "cross_ext_on_repicks": xe,
             "cross_delta_on_repicks": xe - xb}
        res["families"][name] = r
        print(f"{name:<30}{2:>7d}{r['ade_lane_change']:>10.4f}"
              f"{r['gain_lane_change']:>9.4f}{r['gain_all']:>9.4f}"
              f"{r['gain_turn']:>10.4f}{r['lc_picks']:>9d}"
              f"{cb:>10.6f}{ce:>10.6f}{ce-cb:>10.6f}{xe-xb:>9.4f}")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
